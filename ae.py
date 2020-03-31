import torch.nn as nn
import os
import torch
from lib import Encoder, Decoder

from torchvision.utils import make_grid
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

import time

from data import Pegasus, PegasusSampler

BATCH_SIZE = 64
EPOCHS = 1000


class ACAIAutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder(3, 16, 16)
        self.decoder = Decoder(3, 16, 16)

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_set = Pegasus(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=BATCH_SIZE))

ae = ACAIAutoEncoder().to("cuda")

criterion = nn.BCELoss().to("cuda")
optimiser = Adam(ae.parameters())

# https://www.jeremyjordan.me/autoencoders/
# https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html
# https://medium.com/@afagarap/implementing-an-autoencoder-in-pytorch-19baa22647d1
# https://medium.com/analytics-vidhya/dimension-manipulation-using-autoencoder-in-pytorch-on-mnist-dataset-7454578b018
# https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/

# This might be a while, of course.
# But now that I'm happy with this... nope. Spectral Normalisation.
# Hell, we could double the size of our dataset.


def imshow(tensor, filename):

    tensor = tensor.detach().cpu()
    img = make_grid(tensor)

    np_img = img.numpy()
    transposed = np.transpose(np_img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects(M, N, 3)

    plt.savefig("images/ae/" + filename + ".png")


losses = np.zeros(EPOCHS)

# I'm not sure what it is. It appears to be performing worse than my old ones, though.

for epoch in range(EPOCHS):

    i = 0
    epoch_loss = 0
    start = time.time()

    for x, y in train_loader:

        x = x.to("cuda")

        z, x_hat = ae(x)

        loss = criterion(x_hat, x)

        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()

        epoch_loss += loss.item()

        if i == 155:    # Last batch

            imshow(x, "x_sn/{}".format(epoch))
            imshow(x_hat, "x_hat_sn/{}".format(epoch))

        i += 1

    epoch_loss = epoch_loss / 156

    print("{}/{}: {:.5f} ({:.2f}s)".format(epoch + 1, EPOCHS, epoch_loss, time.time() - start))

    losses[epoch] = epoch_loss

print("\n", losses)

if not os.path.exists('./weights'):

    os.mkdir('./weights')

torch.save(ae.state_dict(), "./weights/autoencoder_bird.pkl")

np.save("ae/losses.npy", losses)