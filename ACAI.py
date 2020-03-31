import time
import os
import math
import datetime
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
import torch.nn.functional as F

from data import Pegasus, PegasusSampler
from lib import AutoEncoder, Discriminator

args = {
    "epochs": 75,
    "batch_size": 64,
    "depth": 16,
    "latent": 16,
    "lr": 1e-3,
    "advdepth": 16,
    "advweight": 0.5,
    "reg": 0.2,
    "weight_decay": 1e-5,
    "width": 32,
    "latent_width": 4,
    "device": "cuda"
}

args["scales"] = int(math.log2(args["width"] // args["latent_width"]))
args["advdepth"] = args["advdepth"] or args["depth"]                    # Don't allow advdepth of 0

train_set = Pegasus(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))


def graph(ae_arr, disc_arr, save=False):

    plt.plot(ae_arr, "r", label="ae")
    plt.plot(disc_arr, "b", label="disc")

    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.legend(loc="upper right")

    if save:

        filename = "graphs/" + ",".join([str(x) for x in list(args.values())]) + ".png"

        plt.savefig(filename)

    plt.show()


def calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha):

    gamma = args["reg"]

    loss = F.mse_loss(disc_mix, alpha.squeeze())                                        # || d_omega(x^_alpha) - alpha||^2
    regulariser = torch.mean(discriminator(gamma * x + (1 - gamma) * x_hat)) ** 2       # || d_omega(gamma * x + (1 - gamma) x^) ||^2

    return loss + regulariser

# The figure is consuming memory!

def calc_loss_ae(x, x_hat, disc_mix):

    loss = F.binary_cross_entropy(x_hat, x)                                     # ||x - g_phi(f_theta(x))||^2
    regulariser = args["advweight"] * (torch.mean(disc_mix) ** 2)               # lambda * || d_omega(x^_alpha) ||^2

    return loss + regulariser


ae = AutoEncoder(args["scales"], args["depth"], args["latent"]).to(args["device"])
d = Discriminator(args["scales"], args['advdepth'], args['latent']).to(args['device'])

# Optimiser for autoencoder parameters
opt_ae = Adam(ae.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

# Optimiser for discriminator parameters
opt_d = Adam(d.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])

start_time = time.time()

losses_ae = np.zeros(args["epochs"])
losses_d = np.zeros(args["epochs"])

# Create output directories
dirs = ["x", "x_hat", "d"]
root = "images"

if not os.path.exists(root):

    os.mkdir(root)

run = datetime.datetime.today().strftime("%Y%m%d_%H%M")
os.mkdir(os.path.join(root, run))

for dir in dirs:

    os.mkdir(os.path.join(root, run, dir))


def imsave(tensor, folder, epoch):

    tensor = tensor.detach().cpu()
    img = make_grid(tensor).numpy()

    transposed = np.transpose(img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects (M, N, 3)

    plt.savefig("{}/{}/{}/{}.png".format(root, run, folder, epoch))


def output(tensor, y, alpha, preds, epoch):

    alpha = alpha.detach().cpu().squeeze().numpy()
    preds = preds.detach().cpu().squeeze().numpy()
    tensor = tensor.detach().cpu().numpy()

    transposed = np.transpose(tensor, (0, 2, 3, 1))     # Move colour to last channel

    plt.figure(figsize=(15, 20))

    for i, img in enumerate(transposed):    # A (3, 32, 32) image

        plt.subplot(7, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        label = ""

        if y[i] == 4:

            label += "deer/"

        elif y[i] == 7:

            label += "horse/"

        if y[args["batch_size"] - 1 - i] == 0:

            label += "plane"

        elif y[args["batch_size"] - 1 - i] == 2:

            label += "bird"

        label += " ({:.3f}, {:.3f})".format(alpha[i], preds[i])

        plt.xlabel(label)
        plt.imshow(img, cmap=plt.cm.binary)

    plt.savefig("{}/{}/{}/{}.png".format(root, run, "d", epoch))
    plt.clf()


for epoch in range(args["epochs"]):

    i = 0
    start = time.time()

    for x, y in train_loader:

        x = x.to(args["device"])    # Input images
        z, x_hat = ae(x)            # Encoded and decoded images
        half = args["batch_size"] // 2

        alpha = torch.rand(half, 1, 1, 1).to(args['device']) / 2

        bodies = z[half:]
        wings = z[:half]

        encode_mix = alpha * wings + (1 - alpha) * bodies   # Combined latent space
        decode_mix = ae.decoder(encode_mix)                 # Decoded combined latent space

        disc_mix = d(decode_mix)                # Estimates of alpha

        loss_ae = calc_loss_ae(x, x_hat, disc_mix)
        losses_ae[epoch] += loss_ae.item()

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        loss_d = calc_loss_disc(x, x_hat, d, disc_mix, alpha)
        losses_d[epoch] += loss_d.item()

        opt_d.zero_grad()
        loss_d.backward(retain_graph=True)
        opt_d.step()

        i += 1

        if i == len(train_loader) - 1:

            imsave(x, "x", epoch)
            imsave(x_hat, "x_hat", epoch)

            output(decode_mix, y, alpha, disc_mix, epoch)

    losses_d[epoch] /= len(train_loader)
    losses_ae[epoch] /= len(train_loader)

    print("{}/{}: {:.4f}, {:.4f} ({:.2f}s)".format(epoch + 1, args["epochs"], losses_ae[epoch], losses_d[epoch], time.time() - start))

    if losses_ae[epoch] > 1:

        raise Exception("Unacceptable autoencoder loss")

    else:

        torch.save(ae.state_dict(), "weights/ae_asd.pkl")
        torch.save(d.state_dict(), "weights/d_asd.pkl")
