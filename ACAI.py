import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam

import time

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
    "device": "cuda"
}

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

# I'm close to leaving for the day.
# Once it's trained, how do we test it?

def Encoder(scales, depth, latent):

    layers = [
        nn.Conv2d(3, depth, 1, padding=1)  # 3-16-1. Hardly the typical 4-2-1
    ]

    kp = depth

    for scale in range(scales):

        k = depth << scale

        layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.LeakyReLU()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), nn.LeakyReLU()])
        layers.append(nn.AvgPool2d(2))

        kp = k

    k = depth << scales

    layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.LeakyReLU()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))

    # Initializer(layers)

    """
        Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        AvgPool2d(kernel_size=2, stride=2, padding=0)
        Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        AvgPool2d(kernel_size=2, stride=2, padding=0)
        Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        AvgPool2d(kernel_size=2, stride=2, padding=0)
        Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        LeakyReLU(negative_slope=0.01)
        Conv2d(128, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    """

    return nn.Sequential(*layers)

def Decoder(scales, depth, latent):

    layers = []
    kp = latent

    for scale in range(scales - 1, -1, -1):

        k = depth << scale

        layers.extend([nn.Conv2d(kp, k, 3, padding=1), nn.LeakyReLU()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), nn.LeakyReLU()])
        layers.append(nn.Upsample(scale_factor=2))

        kp = k

    layers.extend([nn.Conv2d(kp, depth, 3, padding=1), nn.LeakyReLU()])
    layers.append(nn.Conv2d(depth, 3, 3, padding=1))
    # Initializer(layers)

    """
    Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Upsample(scale_factor=2.0, mode=nearest)
    Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Upsample(scale_factor=2.0, mode=nearest)
    Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Upsample(scale_factor=2.0, mode=nearest)
    Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    LeakyReLU(negative_slope=0.01)
    Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    """

    return nn.Sequential(*layers)

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
        self.encoder = Encoder(scales, depth, latent)

    def forward(self, x):

        x = self.encoder(x)

        x = x.reshape(x.shape[0], -1)   # Not sure why this is necessary.
        x = torch.mean(x, -1)

        return x

encoder = Encoder(3, args['depth'], args['latent']).to(args['device'])
decoder = Decoder(3, args['depth'], args['latent']).to(args['device'])
discriminator = Discriminator(3, args['advdepth'], args['latent']).to(args['device'])

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

# Expected object of type cuda but got device type cpu. For encoder(x).

losses = defaultdict(list)

it = 0
start_time = time.time()

for epoch in range(args["epochs"]):

    print("\nEPOCH {}/{}\n".format(epoch, args["epochs"]))

    # We have x, l, and h

    for x, y in train_loader:

        x = Variable(x).cuda()

        encode = encoder(x)
        # decode = decoder(h)   # What's h?
        ae = decoder(encode)
        loss_ae_mse = F.mse_loss(x, ae)

        # args[batch_size] = x.shape[0]. Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        # Maybe I shouldn't fuck with his implement too much...

        encode_mix = alpha * encode + (1 - alpha) * torch.flip(encode, [0])
        decode_mix = decoder(encode_mix)

        loss_disc = F.mse_loss(decode_mix, alpha.reshape(-1))
        loss_disc_real = F.mse_loss()

        disc = discriminator(torch.lerp(ae, x, args['reg']))

        # Ah, I see. This is rather complex.

        z_mix = lerp(encode, torch.flip(encode, [0]), alpha)

        out_mix = decoder(z_mix)

        # Judge them
        disc_mix = discriminator(out_mix)

        loss_ae_L2 = L2(disc_mix) * args['advweight']
        loss_ae = loss_ae_mse + loss_ae_L2

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        # Discriminator losses and optimisation

        # Why are there more losses here?
        # Let's switch to his.

        loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1)) # This is superior.
        loss_disc_L2 = L2(disc)
        loss_disc = loss_disc_mse + loss_disc_L2

        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        # Well, there we go.
        # So I've got this working...

        losses['std(disc_mix)'].append(torch.std(disc_mix).item())
        losses['loss_disc_mse'].append(loss_disc_mse.item())
        losses['loss_disc_l2'].append(loss_disc_L2.item())
        losses['loss_disc'].append(loss_disc.item())
        losses['loss_ae_mse'].append(loss_ae_mse.item())
        losses['loss_ae_l2'].append(loss_ae_L2.item())
        losses['loss_ae'].append(loss_ae.item())


        # Naturally, it will be become more complex.
        # Once we're interpolating images.... let's watch some YouTube and eat.
        # img = img.view(img.size(0), -1)
        # img = Variable(img).cuda()
        # # ===================forward=====================
        # output = model(img)
        # loss = criterion(output, img)
        # # ===================backward====================
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

# Mostly works. That's an odd error, though. I imagine it's due to my cut-off.
# And there's definitely a dataset option for that.

# We expect loss_disc to remain constant, and loss_ae to decrease.