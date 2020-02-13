import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

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


# https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
class Pegasus(CIFAR10):

    def __init__(self, root, classes=None, train=True, transform=None, target_transform=None, download=False):

        super().__init__(root, train, transform, target_transform, download)

        if classes is not None:

            _classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            classes = [_classes.index(x) for x in classes]

            indices = np.arange(len(self.targets))
            indices = np.array([i for i in indices if self.targets[i] in classes])

            self.targets = np.take(self.targets, indices)
            self.data = np.take(self.data, indices, axis=0)


train_set = Pegasus(root='./data', classes=["bird", "horse"], train=True, download=True, transform=transform_train)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# test_set = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=0, drop_last=True)


