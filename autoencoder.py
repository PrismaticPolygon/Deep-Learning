import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

args = {
    "epochs": 5,
    "batch_size": 64,
    "depth": 16,
    "latent": 2,
    "lr": 0.0001,
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
train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, num_workers=0)

test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

class ACAI(nn.Module):

    def __init__(self):

        super(ACAI, self).__init__()

        self.encoder = Encoder(3, args['depth'], args['latent']).to(args['device'])
        self.decoder = Decoder(3, args['depth'], args['latent']).to(args['device'])

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Autoencoder(nn.Module):

    def __init__(self):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True)).to("cuda")

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True)).to("cuda")

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

model = ACAI().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=1e-5)

for epoch in range(args["epochs"]):

    for data in train_loader:

        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, args["epochs"], loss))
