import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# A simple AutoEncoder following https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c

# ToTensor converts a PIL image or np.ndarray of shape (H, W, C) in range [0, 255]
# to a torch.FloatTensor of shape (C, H, W) in range [0.0, 1.0]

# Normalize normalises a tensor image with mean and standard deviations

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
])

trainTransform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
])

trainset = CIFAR10(root='./data', train=True,download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)


class Autoencoder(nn.Module):

    def __init__(self):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

num_epochs = 5 #you can go for more epochs, I am using a mac
batch_size = 128

model = Autoencoder().cpu()
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

# If we get a single epoch running, then I'm happy. I'll merge the two after that, then we've got
# a good foundation to actually begin.

for epoch in range(num_epochs):

    for data in dataloader:

        img, _ = data
        img = Variable(img).cpu()
        # ===================forward=====================
        output = model(img)
        loss = distance(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data()))