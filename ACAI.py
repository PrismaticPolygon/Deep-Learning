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


train_set = Pegasus(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))

test_set = Pegasus(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))


def imshow(img):

    img = inverse_normalize(img)
    np_img = img.cpu().numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

# I'll think about the Discriminator, then.
# Weirdly simple:




"""return tf.reduce_mean(layers.encoder(x, scales, advdepth, latent, 'disc'), axis=[1, 2, 3])"""


class Discriminator(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = build_encoder(scales, depth, latent)

    def forward(self, x):

        x = self.encoder(x)  # (64, 2, 4, 4)

        return torch.mean(x, [1, 2, 3])  # (64)


def graph(ae_arr, disc_arr):

    plt.plot(ae_arr, "r", label="ae")
    plt.plot(disc_arr, "b", label="disc")

    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.legend(loc="upper right")

    plt.show()

# They use a benchmark. It would be a LOT of effort to crib, so I'm not going to.

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

    loss = F.mse_loss(disc_mix, alpha.squeeze())                                    # || d_omega(x^_alpha) - alpha||^2
    regulariser = torch.mean(discriminator(gamma * x + (1 - gamma) * x_hat)) ** 2   # || d_omega(gamma * x + (1 - gamma) x^) ||^2

    return loss + regulariser


def calc_loss_ae(x, x_hat, disc_mix):
    """
    Calculate the loss of the autoencoder. THe first term attempts to reconstruct the input. The second term tries to
    make the critic network output 0 at all times.
    :param x: the input. A Tensor of shape (64, 32, 32, 3)
    :param x_hat: x encoded then decoded. A Tensor of shape (64, 32, 32, 3)
    :param disc_mix: discriminator predictions for alpha
    :return: L_{f, g}
    """

    loss = F.mse_loss(x, x_hat)                                         # ||x - g_phi(f_theta(x))||^2
    regulariser = args["advweight"] * (torch.mean(disc_mix) ** 2)       # lambda * || d_omega(x^_alpha) ||^2

    return loss + regulariser


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


loss_ae_arr = np.zeros(args["epochs"])
loss_disc_arr = np.zeros(args["epochs"])

for epoch in range(args["epochs"]):

    i = 0

    print("\nEPOCH {}/{}\n".format(epoch + 1, args["epochs"]))

    for x, y in train_loader:

        x = x.to(args["device"])

        # Via a convex combination. What IS a convex combination?
        # A convex combination is a linear combination of points where all coefficients
        # are negative and sum to 1. Definitely not what I've done.
        # A bunch of reshaping, I suspect.

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

        loss_ae_arr[epoch] += loss_ae.item()

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        loss_disc = calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha)

        loss_disc_arr[epoch] += loss_disc.item()

        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        print("{}/156: {:.2f}, {:.2f}".format(i + 1, loss_ae.item(), loss_disc.item()))

        i += 1

    loss_disc_arr[epoch] /= len(train_loader)
    loss_ae_arr[epoch] /= len(train_loader)

    graph(loss_ae_arr[:epoch + 1], loss_disc_arr[:epoch + 1])

