# -*- coding: utf-8 -*-
"""e8ca989584b3b0e6526c0a737ed412f0

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/gist/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
"""

# Commented out IPython magic to ensure Python compatibility.
# %env CUDA_VISIBLE_DEVICES=0

import math
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import clear_output

import torch
from torch import nn, optim
import torchvision
import torch.nn.functional as F

# https://github.com/kylemcdonald/python-utils
# from utils.show_array import *
# from utils.make_mosaic import *
# from utils.pixelated import *

args = {
    'epochs': 100,
    'width': 32,
    'latent_width': 4,
    'depth': 16,
    'advdepth': 16,
    'advweight': 0.5,
    'reg': 0.2,
    'latent': 2,
    'colors': 1,
    'lr': 0.0001,
    'batch_size': 64,
    'device': 'cuda'
}

from sklearn.datasets import fetch_openml

def build_batches(x, n):
    x = np.asarray(x)
    m = (x.shape[0] // n) * n
    return x[:m].reshape(-1, n, *x.shape[1:])

def get_mnist32_batches(batch_size, data_format='channels_first'):
    channel_index = 1 if data_format == 'channels_first' else 3
    mnist = fetch_openml('mnist_784')
    data_x = mnist['data'].reshape(-1,28,28).astype(np.float32) / 255.
    data_x = np.pad(data_x, ((0,0), (2,2), (2,2)), mode='constant')
    data_x = np.expand_dims(data_x, channel_index)
    data_y = mnist['target']
    indices = np.arange(len(data_x))
    np.random.shuffle(indices)
    y_batches = build_batches(data_y[indices], batch_size)
    x_batches = build_batches(data_x[indices], batch_size)
    return x_batches, y_batches

# x_batches = (1093, 64, 1, 32, 32). This is 1093 batches, of size 64, where each element is (1, 32, 32)
# y-batches = (1093, 64). This is 1093 batches, of size 64, where each element is a label.
x_batches, y_batches = get_mnist32_batches(args['batch_size'])

# Convert to a tensor.
x_batches = torch.FloatTensor(x_batches).to(args['device'])

# pytorch doesn't support negative strides / can't flip tensors
# so instead this function swaps the two halves of a tensor
def swap_halves(x):
    a, b = x.split(x.shape[0]//2)
    return torch.cat([b, a])

# But what is it?

# torch.lerp only support scalar weight
def lerp(start, end, weights):

    return start + weights * (end - start)

def L2(x):
    return torch.mean(x**2)

activation = nn.LeakyReLU

# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):

    for layer in layers:

        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1/np.sqrt((1 + slope**2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)

        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()

def Encoder(scales, depth, latent, colors):

    layers = []
    layers.append(nn.Conv2d(colors, depth, 1, padding=1))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.AvgPool2d(2))
        kp = k
    k = depth << scales
    layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
    layers.append(nn.Conv2d(k, latent, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

def Decoder(scales, depth, latent, colors):

    layers = []
    kp = latent
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
        layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])
        layers.append(nn.Upsample(scale_factor=2))
        kp = k
    layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
    layers.append(nn.Conv2d(depth, colors, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, scales, depth, latent, colors):
        super().__init__()
        self.encoder = Encoder(scales, depth, latent, colors)
        
    def forward(self, x):

        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)
        return x

scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
encoder = Encoder(scales, args['depth'], args['latent'], args['colors']).to(args['device'])
decoder = Decoder(scales, args['depth'], args['latent'], args['colors']).to(args['device'])
discriminator = Discriminator(scales, args['advdepth'], args['latent'], args['colors']).to(args['device'])

opt_ae = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args['lr'], weight_decay=1e-5)
opt_d = optim.Adam(discriminator.parameters(), lr=args['lr'], weight_decay=1e-5)

losses = defaultdict(list)

# helper functions for visualizing the status
def reconstruct(x):
    out = decoder(encoder(x))

    return None

    # return make_mosaic(out.cpu().data.numpy().squeeze())

def interpolate_2(x, side=8):
    z = encoder(x)
    z = z.data.cpu().numpy()

    a, b = z[:side], z[-side:]
    z_interp = [a * (1-t) + b * t for t in np.linspace(0,1,side-2)]
    z_interp = np.vstack(z_interp)
    x_interp = decoder(torch.FloatTensor(z_interp).to(args['device']))
    x_interp = x_interp.cpu().data.numpy()

    x_fixed = x.data.cpu().numpy()
    all = []
    all.extend(x_fixed[:side])
    all.extend(x_interp)
    all.extend(x_fixed[-side:])

    return None
    
    # return make_mosaic(np.asarray(all).squeeze())

def interpolate_4(x, side=8):
    z = encoder(x)
    z = z.data.cpu().numpy()
    
    n = side*side
    xv, yv = np.meshgrid(np.linspace(0, 1, side),
                         np.linspace(0, 1, side))
    xv = xv.reshape(n, 1, 1, 1)
    yv = yv.reshape(n, 1, 1, 1)

    z_interp = \
        z[0] * (1-xv) * (1-yv) + \
        z[1] * xv * (1-yv) + \
        z[2] * (1-xv) * yv + \
        z[3] * xv * yv

    x_fixed = x.data.cpu().numpy()
    x_interp = decoder(torch.FloatTensor(z_interp).to(args['device']))
    x_interp = x_interp.data.cpu().numpy()
    x_interp[0] = x_fixed[0]
    x_interp[side-1] = x_fixed[1]
    x_interp[n-side] = x_fixed[2]
    x_interp[n-1] = x_fixed[3]

    return None

    # return make_mosaic(x_interp.squeeze())

# random samples based on a reference distribution
def random_samples(x):

    z = encoder(x_batches[0])
    z = z.data.cpu().numpy()
    z_sample = np.random.normal(loc=z.mean(axis=0), scale=z.std(axis=0), size=z.shape)
    x_sample = decoder(torch.FloatTensor(z_sample).to(args['device']))
    x_sample = x_sample.data.cpu().numpy()

    return None

    # return make_mosaic(x_sample.squeeze())

def status():
    x = x_batches[0]
    chunks = [reconstruct(x), interpolate_2(x), interpolate_4(x), random_samples(x)]
    chunks = [np.pad(e, (0,1), mode='constant', constant_values=255) for e in chunks]

    return None
    # return make_mosaic(chunks)

it = 0
start_time = time.time()

try:

    for epoch in range(args['epochs']):

        for x in x_batches:

            print(x.shape)

            # Encode batch x
            z = encoder(x)
            # Decode z into x'. Why?
            out = decoder(z)

            # I don't understand why we decode it. The discriminator is the critic.
            # BUt isn't that the case? We have no use for the labels? I think that is indeed the case.
            # Because we're not judging how good we are encoding, we're judging how good we are at producing images that look like they've been encoded.


            disc = discriminator(torch.lerp(out, x, args['reg']))

            # Calculate a random alpha of size (64, 1, 1, 1) in range [0, 0.5]
            alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

            # Produce the interpolated images
            z_mix = lerp(z, swap_halves(z), alpha)

            # Decode the interpolated images
            out_mix = decoder(z_mix)

            # Judge them
            disc_mix = discriminator(out_mix)

            # Calculate the MSE of the AE
            loss_ae_mse = F.mse_loss(out, x)
            loss_ae_l2 = L2(disc_mix) * args['advweight']
            loss_ae = loss_ae_mse + loss_ae_l2
            
            opt_ae.zero_grad()
            loss_ae.backward(retain_graph=True)
            opt_ae.step()
            
            loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1))
            loss_disc_l2 = L2(disc)
            loss_disc = loss_disc_mse + loss_disc_l2
            
            opt_d.zero_grad()
            loss_disc.backward()
            opt_d.step()

            losses['std(disc_mix)'].append(torch.std(disc_mix).item())
            losses['loss_disc_mse'].append(loss_disc_mse.item())
            losses['loss_disc_l2'].append(loss_disc_l2.item())
            losses['loss_disc'].append(loss_disc.item())
            losses['loss_ae_mse'].append(loss_ae_mse.item())
            losses['loss_ae_l2'].append(loss_ae_l2.item())
            losses['loss_ae'].append(loss_ae.item())

            if it % 100 == 0:

                img = status()
                
                plt.figure(facecolor='w', figsize=(10,4))

                for key in losses:
                    total = len(losses[key])
                    skip = 1 + (total // 1000)
                    y = build_batches(losses[key], skip).mean(axis=-1)
                    x = np.linspace(0, total, len(y))
                    plt.plot(x, y, label=key, lw=0.5)

                plt.legend(loc='upper right')
                
                # clear_output(wait=True)
                plt.show()

                # show_array(img * 255)
                
                speed = args['batch_size'] * it / (time.time() - start_time)
                print(f'{epoch+1}/{args["epochs"]}; {speed:.2f} samples/sec')

            it += 1
except KeyboardInterrupt:
    pass

# show the distribution of predictions from the discriminator
plt.hist(disc_mix.data.cpu().numpy(), range=[0,0.5], bins=20)
plt.show()
print(disc_mix)

# distribution of each z dimension
z = encoder(x_batches[0])
z = z.data.cpu().numpy().reshape(len(z), -1).T
for dim in z:

    plt.hist(dim, bins=12, alpha=0.1)

plt.show()
