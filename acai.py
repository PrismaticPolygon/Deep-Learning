import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

import torchvision
import torchvision.transforms as transforms

args = {
    'epochs': 100,      # Number of training epochs
    'width': 32,        # Size of an image
    'latent_width': 4,  # Width of the latent space
    'depth': 16,        # Depth of first for convolution
    'advdepth': 16,     # Depth for adversary network
    'advweight': 0.5,   # Adversarial weight
    'reg': 0.2,         # Amount of discriminator regularisation
    'latent': 2,
    'colors': 1,
    'lr': 0.0001,       # Learning rate
    'batch_size': 64,   # Size of a batch
    'device': 'cuda'
}

# Taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')