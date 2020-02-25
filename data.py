import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import Sampler


# https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
class Pegasus(CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super().__init__(root, train, transform, target_transform, download)

        bird_index = 2
        horse_index = 7
        # plane_index =

        indices = np.arange(len(self.targets))

        horse_indices = np.array([i for i in indices if self.targets[i] == horse_index])
        bird_indices = np.array([i for i in indices if self.targets[i] == bird_index])

        bird_data = np.take(self.data, bird_indices, axis=0)
        horse_data = np.take(self.data, horse_indices, axis=0)

        np.random.shuffle(bird_data)
        np.random.shuffle(horse_data)

        self.data = np.vstack((bird_data, horse_data))
        self.targets = np.zeros((10000, 1), dtype=np.uint8)

        self.targets[5000:] = horse_index
        self.targets[:5000] = bird_index


# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
class PegasusSampler(Sampler):

    def __init__(self, data_source, batch_size):

        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):

        batch_size = self.batch_size // 2

        for n in range(len(self)):  # 156 batches

            batch = [i for i in range(n * batch_size, (n + 1) * batch_size)]
            batch += [len(self.data_source) - i - 1 for i in range(n * batch_size, (n + 1) * batch_size)]

            yield batch

    def __len__(self):

        return len(self.data_source) // self.batch_size
