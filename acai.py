import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import math

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

args = {
    'epochs': 100,      # Number of training epochs
    'width': 32,        # Size of an image
    'latent_width': 4,  # Width of the latent space
    'depth': 16,        # Depth of first forward convolution
    'advdepth': 16,     # Depth for adversary network
    'advweight': 0.5,   # Adversarial weight
    'reg': 0.2,         # Amount of discriminator regularisation
    'latent': 2,
    'colors': 3,        # Greyscale = 1, colour = 3
    'lr': 0.0001,       # Learning rate
    'batch_size': 64,   # Size of a batch
    'device': 'cuda'    # Device to use (cpu or cuda)
}

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
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
train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=0)

test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def Initializer(layers, slope=0.2):

    for layer in layers:

        if hasattr(layer, 'weight'):

            w = layer.weight.data
            std = 1 / np.sqrt((1 + slope ** 2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)

        if hasattr(layer, 'bias'):

            layer.bias.data.zero_()

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

    Initializer(layers)

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
    Initializer(layers)

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


# scales = int(round(math.log(args['width'] // args['latent_width'], 2))) # 3

# encoder = Encoder(scales, args['depth'], args['latent']).to(args['device'])
# decoder = Decoder(scales, args['depth'], args['latent']).to(args['device'])

# We will also need a decoder. No. Let's just encoder first.

class AutoEncoder(nn.Module):

    def __init__(self):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(scales, args['depth'], args['latent']).to(args['device'])
        self.decoder = Decoder(scales, args['depth'], args['latent']).to(args['device'])

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


model = AutoEncoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args["lr"],
    weight_decay=1e-5
)

for epoch in range(args["epochs"]):

    for data in train_loader:

        img, _ = data   # I'm not sure what this does. This is just optimising a simple autoencoder.
                        # Naturally, it will be become more complex.
                        # Once we're interpolating images.... let's watch some YouTube and eat.
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args["epochs"], loss.data[0]))

    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))