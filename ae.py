import torch.nn as nn
import os
import torch
from lib import ACAI_Encoder, ACAI_Decoder

from torchvision.utils import make_grid
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

import time

from data import Pegasus, PegasusSampler

BATCH_SIZE = 64
EPOCHS = 100


class ACAIAutoEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = ACAI_Encoder(3, 16, 16)
        self.decoder = ACAI_Decoder(3, 16, 16)

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded


train_set = Pegasus(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=BATCH_SIZE))

ae = ACAIAutoEncoder().to("cuda")

criterion = nn.MSELoss().to("cuda")
optimiser = Adam(ae.parameters())

# https://www.jeremyjordan.me/autoencoders/
# https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html
# https://medium.com/@afagarap/implementing-an-autoencoder-in-pytorch-19baa22647d1
# https://medium.com/analytics-vidhya/dimension-manipulation-using-autoencoder-in-pytorch-on-mnist-dataset-7454578b018
# https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/


def imshow(tensor, filename):

    tensor = tensor.detach().cpu()
    img = make_grid(tensor)

    np_img = img.numpy()
    transposed = np.transpose(np_img, (1, 2, 0))

    plt.imshow(transposed, interpolation='nearest')  # Expects(M, N, 3)

    plt.savefig("images/ae/" + filename + ".png")

    plt.show()


losses = np.zeros(EPOCHS)

for epoch in range(EPOCHS):

    print("\nEPOCH {}/{}\n".format(epoch, EPOCHS))

    i = 0
    epoch_loss = 0

    for x, y in train_loader:

        x = x.to("cuda")

        start = time.time()
        z, x_hat = ae(x)

        loss = criterion(x_hat, x)

        optimiser.zero_grad()
        loss.backward(retain_graph=True)
        optimiser.step()

        print("{}/156: {:.3f} ({:.2f}s)".format(i + 1, loss.item(), time.time() - start))

        epoch_loss += loss.item()

        if i == 155:    # Last batch

            imshow(x, "x/{}".format(epoch))
            imshow(x_hat, "x_hat/".format(epoch))

        i += 1

    losses[epoch] = epoch_loss / 156

    print("\n", losses)

if not os.path.exists('./weights'):

    os.mkdir('./weights')

torch.save(ae.state_dict(), "./weights/autoencoder.pkl")