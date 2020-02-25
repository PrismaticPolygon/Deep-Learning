import torch.nn as nn
import numpy as np


def initialiser(layers, slope=0.2):

    for layer in layers:

        if hasattr(layer, 'weight'):

            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)

        if hasattr(layer, 'bias'):

            layer.bias.data.zero_()

# All convolutions are zero-padded.

# THe encoder consists of blocks of two consecutive 3 x 3 convolutional layers followed by 2 x 2 average pooling

# Ah. By the time we pool, we've run out of size. I wonder if they compress... no.


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
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            activation(),
            nn.Upsample(scale_factor=2)
        ])

        in_channels = out_channels

    layers.extend([
        nn.Conv2d(in_channels, depth, kernel_size, padding=1),
        activation(),
        nn.Conv2d(depth, 3, kernel_size, padding=1),
        nn.Sigmoid()    # To convert output to [0, 1]
    ])

    initialiser(layers)

    return nn.Sequential(*layers)


if __name__ == "__main__":

    e = ACAI_Encoder(3, 16, 16)

    print("\nENCODER\n")

    print(e)

    d = ACAI_Decoder(3, 16, 16)

    print("\nDECODER\n")

    print(d)
