import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

from lib import build_encoder, build_decoder, NormalizeInverse
from data import Pegasus, PegasusSampler

import time
import math

args = {
    "epochs": 5,
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

pegasus_set = Pegasus(root='./data', train=True, download=True, transform=transform_train)
pegasus_loader = DataLoader(pegasus_set, batch_sampler=PegasusSampler(pegasus_set, batch_size=args["batch_size"]))


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

i = 0
start_time = time.time()

for epoch in range(args["epochs"]):

    print("\nEPOCH {}/{}\n".format(epoch, args["epochs"]))

    for x, y in pegasus_loader:

        # imshow(make_grid(x))  # Display the current batch.

        x = Variable(x).cuda()

        encode = encoder(x)
        ae = decoder(encode)

        half = args["batch_size"] // 2

        # Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        bird_half = encode[:half] + torch.flip(encode[:half], [0])
        horse_half = encode[half:] + torch.flip(encode[half:], [0])

        both = torch.cat((bird_half, horse_half), 0).to(args["device"])

        encode_mix = alpha * both + (1 - alpha) * both

        decode_mix = decoder(encode_mix)
        disc_mix = discriminator(decode_mix)

        loss_disc = torch.mean(torch.pow(disc_mix - alpha.reshape(-1), 2))
        loss_disc_real = torch.mean(torch.pow(discriminator(ae + args["reg"] * (x - ae)), 2))

        loss_ae = F.mse_loss(x, ae)
        loss_ae_disc = torch.mean(torch.pow(disc_mix, 2))

        loss_ae = loss_ae + args["advweight"] * loss_ae_disc

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        loss_disc = loss_disc + loss_disc_real

        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        print(loss_ae.data, loss_disc.data)
