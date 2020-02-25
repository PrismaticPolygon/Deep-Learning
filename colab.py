import time
import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import Sampler
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

from livelossplot import PlotLosses

# https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
class Pegasus(CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super().__init__(root, train, transform, target_transform, download)

        bird_index = 2
        horse_index = 7

        indices = np.arange(len(self.targets))

        horse_indices = np.array([i for i in indices if self.targets[i] == horse_index])
        bird_indices = np.array([i for i in indices if self.targets[i] == bird_index])

        bird_data = np.take(self.data, bird_indices, axis=0)
        horse_data = np.take(self.data, horse_indices, axis=0)

        np.random.shuffle(bird_data)
        np.random.shuffle(horse_data)

        self.data = np.vstack((bird_data, horse_data))
        self.targets = np.zeros((10000, 1), dtype=np.uint8)

        self.targets[:5000] = bird_index
        self.targets[5000:] = horse_index


# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
class PegasusSampler(Sampler):

    def __init__(self, data_source, batch_size):

        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):

        batch_size = self.batch_size // 2

        for n in range(len(self)):  # 156 batches

            batch = [i for i in range(n * batch_size, (n + 1) * batch_size)]
            batch += [len(self.data_source) - i - 1 for i in range(n * batch_size, (n + 1) * batch_size)]

            yield batch

    def __len__(self):

        return len(self.data_source) // self.batch_size


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


def imshow(tensor):

    tensor = tensor.detach().cpu()
    img = make_grid(tensor)

    np_img = img.numpy()
    transposed = np.transpose(np_img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects(M, N, 3)

    plt.show()


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

liveplot = PlotLosses()

for epoch in range(args["epochs"]):

    i = 0
    loss_ae_sum = 0
    loss_d_sum = 0
    start = time.time()

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
        loss_ae_sum += loss_ae.item()

        optimiser_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        optimiser_ae.step()

        loss_d = calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha)
        loss_d_sum += loss_d.item()

        optimiser_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optimiser_d.step()

        i += 1

        if i == len(train_loader) - 1:

            imshow(x)
            imshow(x_hat)
            imshow(decode_mix)

    loss_ae_sum /= len(train_loader)
    loss_d_sum /= len(train_loader)

    liveplot.update({
        'ae_loss': loss_ae_sum,
        'd_loss': loss_d_sum,
    })

    liveplot.draw()

    print("{}/{}: {:.4f}, {:.4f} ({:.2f}s)".format(epoch, args["epochs"],loss_ae_sum, loss_d_sum, time.time() - start))


