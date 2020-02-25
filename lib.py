import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils import spectral_norm


def initialiser(layers, slope=0.2):

    for layer in layers:

        if hasattr(layer, 'weight'):

            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)

        if hasattr(layer, 'bias'):

            layer.bias.data.zero_()


def ACAI_Encoder(scales, depth, latent):

    activation = nn.LeakyReLU
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

    initialiser(layers)

    return nn.Sequential(*layers)


def ACAI_Decoder(scales, depth, latent):

    activation = nn.LeakyReLU
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

    initialiser(layers)

    return nn.Sequential(*layers)


class ACAIAutoEncoder(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = ACAI_Encoder(scales, depth, latent)
        self.decoder = ACAI_Decoder(scales, depth, latent)

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


class ACAIDiscriminator(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = ACAIAutoEncoder(scales, depth, latent)

    def forward(self, x):

        x, _ = self.encoder(x)  # (64, 2, 4, 4)

        return torch.mean(x, [1, 2, 3])  # (64)


if __name__ == "__main__":

    e = ACAI_Encoder(3, 16, 16)

    print("\nENCODER\n")

    print(e)

    d = ACAI_Decoder(3, 16, 16)

    print("\nDECODER\n")

    print(d)
