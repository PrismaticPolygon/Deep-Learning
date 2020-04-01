import numpy as np
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

import matplotlib.pyplot as plt


# https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
class Pegasus(CIFAR10):

    def __init__(self, root="./data", train=True, transform=None, target_transform=None, download=False):

        super().__init__(root, train, transform, target_transform, download)

        wing_indices = [0]  # 0 = planes, 2 = birds,
        body_indices = [7]  # 4 = deer, 7 = horses

        indices = np.arange(len(self.targets))

        # Get all indices for targets with value in body_indices
        body_indices = np.array([i for i in indices if self.targets[i] in body_indices])

        # Get all indices for targets with value in wing_indices
        wing_indices = np.array([i for i in indices if self.targets[i] in wing_indices])

        # Shuffle indices
        np.random.shuffle(body_indices)
        np.random.shuffle(wing_indices)

        body_data = np.take(self.data, body_indices, axis=0)
        wing_data = np.take(self.data, wing_indices, axis=0)

        body_targets = np.take(self.targets, body_indices, axis=0)
        wing_targets = np.take(self.targets, wing_indices, axis=0)

        # Stack arrays in sequence vertically
        self.data = np.vstack((body_data, wing_data))

        # Stack arrays in sequence horizontally
        self.targets = np.hstack((body_targets, wing_targets))


# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
class PegasusSampler(Sampler):

    def __init__(self, data_source, batch_size=64):

        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):

        batch_size = self.batch_size // 2

        for n in range(len(self)):  # 10000 / 64 = 156 batches

            bodies = np.arange(n * batch_size, (n + 1) * batch_size)
            wings = -bodies + len(self.data_source) - 1

            np.random.shuffle(bodies)
            np.random.shuffle(wings)

            yield np.concatenate([bodies, wings])

    def __len__(self):

        return len(self.data_source) // self.batch_size


if __name__ == "__main__":

    dataset = Pegasus(transform=transforms.ToTensor())
    sampler = PegasusSampler(dataset)
    loader = DataLoader(dataset, batch_sampler=sampler)

    for x, y in loader:

        img = make_grid(x).numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)), interpolation='nearest')
        plt.show()
