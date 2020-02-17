import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np



# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class NormalizeInverse(transforms.Normalize):

    def __init__(self, mean, std):

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv

        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):

        return super().__call__(tensor.clone())


def initialiser(layers, slope=0.2):

    for layer in layers:

        if hasattr(layer, 'weight'):

            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)

        if hasattr(layer, 'bias'):

            layer.bias.data.zero_()


def build_encoder(scales, depth, latent, colors=3):

    activation = nn.LeakyReLU
    kernel_size = 3
    padding = 1
    stride = 1
    in_channels = depth

    layers = [
        nn.Conv2d(colors, depth, 1, stride, padding)
    ]

    for scale in range(scales):

        out_channels = depth << scale

        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation(),
            nn.AvgPool2d(2)
        ])

        in_channels = out_channels

    out_channels = depth << scales

    layers.extend([
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        activation(),
        nn.Conv2d(out_channels, latent, kernel_size, stride, padding)
    ])

    initialiser(layers)

    return nn.Sequential(*layers)


def build_decoder(scales, depth, latent, colors=3):

    activation = nn.LeakyReLU
    kernel_size = 3
    stride = 1
    padding = 1
    in_channels = latent

    layers = []

    for scale in range(scales - 1, -1, -1):

        out_channels = depth << scale

        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            activation(),
            nn.Upsample(scale_factor=2)
        ])

        in_channels = out_channels

    layers.extend([
        nn.Conv2d(in_channels, depth, kernel_size, stride, padding),
        activation(),
        nn.Conv2d(depth, colors, kernel_size, stride, padding)
    ])

    initialiser(layers)

    return nn.Sequential(*layers)


if __name__ == "__main__":

    e = build_encoder(3, 16, 2, 3)

    print("\nENCODER\n")

    print(e)

    d = build_decoder(3, 16, 2, 3)

    print("\nDECODER\n")

    print(d)


