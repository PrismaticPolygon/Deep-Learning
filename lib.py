import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
from functools import partial


def Encoder(scales, depth, latent):

    activation = partial(nn.LeakyReLU, negative_slope=0.2)
    kernel_size = 3
    in_channels = depth

    layers = [
        nn.Conv2d(3, depth, 1, padding=1)
    ]

    for scale in range(scales):

        out_channels = depth << scale

        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            activation(),
            nn.AvgPool2d(2)
        ])

        in_channels = out_channels

    out_channels = depth << scales

    layers.extend([
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        activation(),
        nn.Conv2d(out_channels, latent, kernel_size, padding=1)
    ])

    return nn.Sequential(*layers)


def Decoder(scales, depth, latent):

    activation = partial(nn.LeakyReLU, negative_slope=0.2)
    kernel_size = 3
    in_channels = latent

    layers = []

    for scale in range(scales - 1, -1, -1):     # Descend from 64 to 16

        out_channels = depth << scale

        layers.extend([
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)),
            activation(),
            nn.Upsample(scale_factor=2)
        ])

        in_channels = out_channels

    layers.extend([
        spectral_norm(nn.Conv2d(in_channels, depth, kernel_size, padding=1)),
        activation(),
        spectral_norm(nn.Conv2d(depth, 3, kernel_size, padding=1)),
        nn.Sigmoid()    # To convert output to [0, 1]
    ])

    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = Encoder(scales, depth, latent)
        self.decoder = Decoder(scales, depth, latent)

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class Discriminator(nn.Module):

    def __init__(self, scales, advdepth, latent):

        super().__init__()

        self.encoder = AutoEncoder(scales, advdepth, latent)

    def forward(self, x):

        x, _ = self.encoder(x)  # (64, 2, 4, 4)

        return torch.mean(x, [1, 2, 3])  # (64)


if __name__ == "__main__":

    e = Encoder(3, 16, 16)

    print("\nENCODER\n")

    print(e)

    d = Decoder(3, 16, 16)

    print("\nDECODER\n")

    print(d)
