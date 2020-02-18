import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

import time
import math

from lib import build_encoder, build_decoder, NormalizeInverse
from data import Pegasus, PegasusSampler

args = {
    "epochs": 6,
    "batch_size": 64,
    "depth": 16,
    "latent": 2,
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


# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

normalize = transforms.Normalize(mean=mean, std=std)
inverse_normalize = NormalizeInverse(mean=mean, std=std)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# So the auto-encoder reconstructs images.
# I need to get these images back out of PyTorch!

train_set = Pegasus(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))

test_set = Pegasus(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))

# No luck. Let's get onto colab, me thinks.

def imshow(img):

    img = inverse_normalize(img)
    np_img = img.cpu().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


class Discriminator(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = build_encoder(scales, depth, latent)

    def forward(self, x):

        x = self.encoder(x)

        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)

        return x


encoder = build_encoder(args["scales"], args['depth'], args['latent']).to(args['device'])
decoder = build_decoder(args["scales"], args['depth'], args['latent']).to(args['device'])

discriminator = Discriminator(args["scales"], args['advdepth'], args['latent']).to(args['device'])

# Optimiser for autoencoder parameters
opt_ae = Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=args["lr"],
    weight_decay=args["weight_decay"]
)

# Optimiser for discriminator parameters
opt_d = Adam(
    discriminator.parameters(),
    lr=args["lr"],
    weight_decay=args["weight_decay"]
)

start_time = time.time()


def graph(ae_arr, disc_arr):

    plt.plot(ae_arr, "r", label="ae")
    plt.plot(disc_arr, "b", label="disc")

    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.legend(loc="upper right")

    plt.show()


def calc_loss_disc(x, ae, discriminator, disc_mix, alpha):

    loss_disc = torch.mean(torch.pow(disc_mix - alpha.reshape(-1), 2))
    loss_disc_real = torch.mean(torch.pow(discriminator(ae + args["reg"] * (x - ae)), 2))

    return loss_disc + loss_disc_real


def calc_loss_ae(x, x_hat, disc_mix):
    """

    :param x: the input. A Tensor of shape (64, 32, 32, 3)
    :param x_hat: x encoded then decoded. A Tensor of shape (64, 32, 32, 3)
    :param disc_mix: discriminator predictions for alpha
    :return: the loss of the AE
    """

    loss = F.mse_loss(x, x_hat)                                         # ||x - g_phi(f_theta(x))||^2
    regularisation = args["advweight"] * (torch.mean(disc_mix) ** 2)    # lambda * || d_omega(x^_alpha) ||^2

    return loss + regularisation

loss_ae_arr = np.zeros(0)
loss_disc_arr = np.zeros(0)

for epoch in range(args["epochs"]):

    i = 0

    print("\nEPOCH {}/{}\n".format(epoch + 1, args["epochs"]))

    train_loss_ae_arr = np.zeros(0)
    train_loss_disc_arr = np.zeros(0)

    for x, y in train_loader:

        x = x.to(args["device"])

        # Via a convex combination. What IS a convex combination?

        z = encoder(x)
        x_hat = decoder(z)

        half = args["batch_size"] // 2

        # Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        bird_half = z[:half] + z[:half]   # If we flip both we train it on the same set of images twice
        horse_half = z[half:] + torch.flip(z[half:], [0])

        both = torch.cat((bird_half, horse_half), 0).to(args["device"])

        encode_mix = alpha * both + (1 - alpha) * both

        decode_mix = decoder(encode_mix)
        disc_mix = discriminator(decode_mix)

        loss_ae = calc_loss_ae(x, x_hat, disc_mix)

        train_loss_ae_arr = np.append(train_loss_ae_arr, loss_ae.item())

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        loss_disc = calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha)

        train_loss_disc_arr = np.append(train_loss_disc_arr, loss_disc.item())

        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        print("{}/156: {:.2f}, {:.2f}".format(i + 1, loss_ae.item(), loss_disc.item()))

        i += 1

    loss_ae_arr = np.append(loss_ae_arr, train_loss_ae_arr.mean())
    loss_disc_arr = np.append(loss_disc_arr, train_loss_disc_arr.mean())

    graph(loss_ae_arr, loss_disc_arr)

