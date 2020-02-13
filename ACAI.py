import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.optim import Adam

from lib import encoder, decoder

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
        self.encoder = encoder(scales, depth, latent)

    def forward(self, x):

        x = self.encoder(x) # I see only an x here.

        x = x.reshape(x.shape[0], -1)   # Not sure why this is necessary.
        x = torch.mean(x, -1)

        return x

e = encoder(args["scales"], args['depth'], args['latent']).to(args['device'])
d = decoder(args["scales"], args['depth'], args['latent']).to(args['device'])

# Fuck. I don't know \textit{enough} about what I'm doing to easily debug yet.
# Fuck Mr. Willcox for thinking I'm going to do anything fancy. Implementating this paper is pretty goddamn fancy
# in of itself!

discriminator = Discriminator(args["scales"], args['advdepth'], args['latent']).to(args['device'])

# Optimiser for autoencoder parameters
opt_ae = Adam(
    list(e.parameters()) + list(d.parameters()),
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

        encode = e(x)

        # print(encode.shape)  # (64, 2, 4, 4)

        # decode = decoder(h)   # What's h?
        ae = d(encode)

        # print(ae.shape)  # (64, 3, 30, 30). Forgot some padding.

        loss_ae_mse = F.mse_loss(x, ae)

        # args[batch_size] = x.shape[0]. Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        # Maybe I shouldn't fuck with his implement too much...

        encode_mix = alpha * encode + (1 - alpha) * torch.flip(encode, [0])

        # print(encode_mix.shape)

        decode_mix = d(encode_mix)

        # print(decode_mix.shape)

        # loss_disc = F.mse_loss(decode_mix, alpha.reshape(-1))
        # loss_disc_real = F.mse_loss()

        disc = discriminator(torch.lerp(ae, x, args['reg']))

        # Ah, I see. This is rather complex.

        z_mix = lerp(encode, torch.flip(encode, [0]), alpha)

        out_mix = d(z_mix)

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

        print(loss_ae, loss_disc)

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