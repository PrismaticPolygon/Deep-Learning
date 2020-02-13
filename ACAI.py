import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam

from lib import build_encoder, build_decoder

import time
import math

from collections import defaultdict

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

# Taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

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

train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=0, drop_last=True)

test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# pytorch doesn't support negative strides / can't flip tensors
# so instead this function swaps the two halves of a tensor
# That works fine, too.
# It doesn't matter how I do this.
# I may as well shuffle it, to be honest.
def swap_halves(x):

    a, b = x.split(x.shape[0] // 2)

    return torch.cat([b, a])

# torch.lerp only support scalar weight
def lerp(start, end, weights):

    return start + weights * (end - start)

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

def L2(x):

    return torch.mean(x**2)


losses = defaultdict(list)

i = 0
start_time = time.time()

for epoch in range(args["epochs"]):

    print("\nEPOCH {}/{}\n".format(epoch, args["epochs"]))

    # We have x, l, and h

    for x, y in train_loader:

        x = Variable(x).cuda()

        encode = encoder(x)
        # decode = decoder(h)   # What's h?
        ae = decoder(encode)

        # args[batch_size] = x.shape[0]. Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        # Maybe I shouldn't fuck with his implement too much...

        encode_mix = alpha * encode + (1 - alpha) * torch.flip(encode, [0])
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

        print(loss_ae, loss_disc)