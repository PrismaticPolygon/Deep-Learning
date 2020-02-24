import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

import time
import math

from lib import build_encoder, build_decoder, NormalizeInverse
from data import Pegasus, PegasusSampler

args = {
    "epochs": 10,
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


train_set = Pegasus(root='./data', train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))

test_set = Pegasus(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))


def imshow(tensor, inv=False):

    tensor = tensor.detach().cpu()  # (64, 3, 32, 32) (B, C, W, H)

    img = torchvision.utils.make_grid(tensor)

    if inv:

        img = inverse_normalize(img)

    np_img = img.numpy()

    transposed = np.transpose(np_img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects(M, N, 3)

    plt.show()



"""return tf.reduce_mean(layers.encoder(x, scales, advdepth, latent, 'disc'), axis=[1, 2, 3])"""


class Discriminator(nn.Module):

    def __init__(self, scales, depth, latent):

        super().__init__()

        self.encoder = build_encoder(scales, depth, latent)

    def forward(self, x):

        x = self.encoder(x)  # (64, 2, 4, 4)

        return torch.mean(x, [1, 2, 3])  # (64)


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

    # Wait. Alpha should be a tensor of shape batch_size.
    # Christ this is frustrating. It the same alpha for every element in the batch, right?
    # Wait: no. It's supposed to be random for each. Gotcha.

    gamma = args["reg"]

    loss = F.mse_loss(disc_mix, alpha.squeeze())                                    # || d_omega(x^_alpha) - alpha||^2
    regulariser = torch.mean(discriminator(gamma * x + (1 - gamma) * x_hat)) ** 2   # || d_omega(gamma * x + (1 - gamma) x^) ||^2

    return loss + regulariser


def calc_loss_ae(x, x_hat, disc_mix):
    """
    Calculate the loss of the autoencoder. THe first term attempts to reconstruct the input. The second term tries to
    make the critic network output 0 at all times.
    :param x: the input. A Tensor of shape (64, 32, 32, 3)
    :param x_hat: x encoded then decoded. A Tensor of shape (64, 32, 32, 3)
    :param disc_mix: discriminator predictions for alpha
    :return: L_{f, g}
    """

    loss = F.mse_loss(x, x_hat)                                         # ||x - g_phi(f_theta(x))||^2
    regulariser = args["advweight"] * (torch.mean(disc_mix) ** 2)       # lambda * || d_omega(x^_alpha) ||^2

    return loss + regulariser


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

start_time = time.time()

loss_ae_arr = np.zeros(args["epochs"])
loss_disc_arr = np.zeros(args["epochs"])

for epoch in range(args["epochs"]):

    i = 0

    print("\nEPOCH {}/{}\n".format(epoch + 1, args["epochs"]))

    for x, y in train_loader:

        x = x.to(args["device"])
        half = args["batch_size"] // 2

        # bird_half = torch.cat((x[:half], x[:half]), 0).to(args["device"])  # If we flip both we train it on the same set of graphs twice
        # horse_half = torch.cat((x[half:], torch.flip(x[half:], [0])), 0).to(args["device"])
        #
        # imshow(bird_half, inv=True)
        # imshow(horse_half, inv=True)

        # Shape (64, 1, 1, 1) is broadcastable, allowing elementwise multiplication: (64) x (64, 3, 32, 32)
        alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        # x_mix = alpha * bird_half + (1 - alpha) * horse_half  # Nice.

        # Okay. That works. Fine. Hell, it could be the actual encoder at fault.
        # It is not necessarily the nicest solution.
        #

        # imshow(x_mix, inv=True)

        z = encoder(x)
        x_hat = decoder(z)

        # Generate random alpha of shape (64, 1, 1, 1) in range [0, 0.5]
        # alpha = torch.rand(args['batch_size'], 1, 1, 1).to(args['device']) / 2

        bird_half = torch.cat((z[:half], z[:half]), 0).to(args["device"])   # If we flip both we train it on the same set of graphs twice
        horse_half = torch.cat((z[half:], torch.flip(z[half:], [0])), 0).to(args["device"])

        # print(bird_half.shape)  # That's more like it. Now it's
        # print(horse_half.shape)

        # If my alpha is 1.... It should be all birds.
        # We could check if it works for the plain images.

        encode_mix = alpha * bird_half + (1 - alpha) * horse_half   # Nice.

        # print(encode_mix.shape)

        decode_mix = decoder(encode_mix)                # (64, 3, 32, 32). Image space.

        # I reckon my encode_mix doesn't do what I think it should.
        # Expected deice cpu but got cuda... when I was trying to mix them. Even AFTER I've sent them to the GPU.

        # print(decode_mix.shape)

        # So the problem now is the grey output for my decode mix. IT is just guessing, and this is the first round, so maybe it's fair enough?
        # I'll run it for an epoch and see what happens.

        imshow(decode_mix, inv=True)  # This should be my output images!

        disc_mix = discriminator(decode_mix)

        loss_ae = calc_loss_ae(x, x_hat, disc_mix)

        loss_ae_arr[epoch] += loss_ae.item()

        opt_ae.zero_grad()
        loss_ae.backward(retain_graph=True)
        opt_ae.step()

        loss_disc = calc_loss_disc(x, x_hat, discriminator, disc_mix, alpha)

        loss_disc_arr[epoch] += loss_disc.item()

        opt_d.zero_grad()
        loss_disc.backward()
        opt_d.step()

        print("{}/156: {:.2f}, {:.2f}".format(i + 1, loss_ae.item(), loss_disc.item()))

        i += 1

    loss_disc_arr[epoch] /= len(train_loader)
    loss_ae_arr[epoch] /= len(train_loader)

    if epoch > 0:

        graph(loss_ae_arr[:epoch + 1], loss_disc_arr[:epoch + 1], save=True)

# HOW do I now do images?
# The example I followed output stuff every now and then, right?
# So we can also visualise how well the auto-encoder is doing. Fuck IT. Let's just output something, Dom.

# Nice.
# Now: think

