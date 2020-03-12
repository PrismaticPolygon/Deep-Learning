import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

import time
import math

from data import Pegasus, PegasusSampler
from lib import ACAIAutoEncoder, ACAIDiscriminator


# https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/

args = {
    "epochs": 200,
    "batch_size": 64,
    "depth": 16,
    "latent": 16,
    "lr": 0.0001,
    "advdepth": 16,
    "advweight": 0.5,
    "reg": 0.2,
    "weight_decay": 1e-5,
    "width": 32,
    "latent_width": 4,
    "device": "cuda"
}

args["scales"] = int(math.log2(args["width"] // args["latent_width"]))


transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_set = Pegasus(root='./data', train=True, download=True, transform=transform_train)
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


criterion_disc = nn.MSELoss()


def calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha):
    """
    Calculate the loss of the discriminator. The first term attempts to recover alpha. The second term is not crucial
    but enforces that the critic outputs 0 for non-interpolated data and that the critic is exposed to realistic data
    even when the autoencoder reconstructions are of poor quality.
    :param x: the input. A Tensor of shape (64, 32, 32, 3)
    :param x_hat: x encoded then decoded. A Tensor for shape (64, 32, 32, 3)
    :param discriminator: the discriminator / critic network
    :param disc_mix: discriminator predictions for alpha
    :param alpha: alpha
    :return: L_d
    """

    gamma = args["reg"]

    loss = criterion_disc(disc_mix, alpha.squeeze())                                    # || d_omega(x^_alpha) - alpha||^2
    regulariser = torch.mean(discriminator(gamma * x + (1 - gamma) * x_hat)) ** 2       # || d_omega(gamma * x + (1 - gamma) x^) ||^2

    return loss + regulariser


criterion_ae = nn.BCELoss()


def calc_loss_ae(x, x_hat, disc_mix):
    """
    Calculate the loss of the autoencoder. THe first term attempts to reconstruct the input. The second term tries to
    make the critic network output 0 at all times.
    :param x: the input. A Tensor of shape (64, 32, 32, 3)
    :param x_hat: x encoded then decoded. A Tensor of shape (64, 32, 32, 3)
    :param disc_mix: discriminator predictions for alpha
    :return: L_{f, g}
    """

    loss = criterion_ae(x_hat, x)                                               # ||x - g_phi(f_theta(x))||^2
    regulariser = args["advweight"] * (torch.mean(disc_mix) ** 2)             # lambda * || d_omega(x^_alpha) ||^2

    return loss + regulariser


def imsave(tensor, filename):

    tensor = tensor.detach().cpu()
    img = make_grid(tensor)

    np_img = img.numpy()
    transposed = np.transpose(np_img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects(M, N, 3)

    plt.savefig("images/acai/" + filename + ".png")


ae = ACAIAutoEncoder(args["scales"], args["depth"], args["latent"]).to(args["device"])
discriminator = ACAIDiscriminator(args["scales"], args['advdepth'], args['latent']).to(args['device'])

# Optimiser for autoencoder parameters
optimiser_ae = Adam(
    ae.parameters(),
    lr=args["lr"],
    weight_decay=args["weight_decay"]
)

# Optimiser for discriminator parameters
optimiser_d = Adam(
    discriminator.parameters(),
    lr=args["lr"],
    weight_decay=args["weight_decay"]
)

start_time = time.time()

losses_ae = np.zeros(args["epochs"])
losses_d = np.zeros(args["epochs"])

# Nice.
# Now to add some logging, and we're peachy.
# Ah. Fine. Let's bring it up to date.
# It does seem t

for epoch in range(args["epochs"]):

    i = 0
    start = time.time()
    k =

    for x, y in train_loader:

        x = x.to(args["device"])    # Input images
        z, x_hat = ae(x)            # Encoded and decoded images
        half = args["batch_size"] // 2

        alpha = torch.rand(half, 1, 1, 1).to(args['device']) / 2

        horses = z[half:]
        birds = z[:half]

        encode_mix = alpha * birds + (1 - alpha) * horses   # Combined latent space
        decode_mix = ae.decoder(encode_mix)                 # Decoded combined latent space

        disc_mix = discriminator(decode_mix)                # Estimates of alpha

        loss_ae = calc_loss_ae(x, x_hat, disc_mix)
        losses_ae[epoch] += loss_ae.item()

        optimiser_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        optimiser_ae.step()

        loss_d = calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha)
        losses_d[epoch] += loss_d.item()

        optimiser_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optimiser_d.step()

        i += 1

        if i == len(train_loader) - 1:

            imsave(x, "x/{}".format(epoch))
            imsave(x_hat, "x_hat/{}".format(epoch))
            imsave(decode_mix, "d/{}".format(epoch))

    losses_d[epoch] /= len(train_loader)
    losses_ae[epoch] /= len(train_loader)

    print("{}/{}: {:.4f}, {:.4f} ({:.2f}s)".format(epoch, args["epochs"], losses_ae[epoch], losses_d[epoch], time.time() - start))

    torch.save(ae.state_dict(), "weights/ae_old.pkl")
    torch.save(d.state_dict(), "weights/d_old.pkl")
