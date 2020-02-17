import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import Sampler

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

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        super().__init__(root, train, transform, target_transform, download)

        bird_index = 2
        horse_index = 7

        indices = np.arange(len(self.targets))

        horse_indices = np.array([i for i in indices if self.targets[i] == horse_index])
        bird_indices = np.array([i for i in indices if self.targets[i] == bird_index])

        bird_data = np.take(self.data, bird_indices, axis=0)
        horse_data = np.take(self.data, horse_indices, axis=0)

        np.random.shuffle(bird_data)
        np.random.shuffle(horse_data)

        self.data = np.zeros((20000, 32, 32, 3))

        n = 32

        horse_batch = np.empty((1, 32))
        horse_batch[:] = horse_index

        bird_batch = np.empty((1, 32))
        bird_batch[:] = bird_index

        # Fuck. One of my arrays is too small.
        # This is seriously beginning to piss me off, man.

        for i in range((10000 // n) - 1):

            k = 2 * n * i   # Increments of 64

            self.data[k:k + n] = horse_data[n * i:n * (i + 1)]
            self.targets[k:k + n] = horse_batch

            self.data[k + n:k + (2 * n)] = bird_data[n * i:n * (i + 1)]
            self.targets[k:k + n] = bird_batch


class PegasusSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):

        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):

        for n in range(len(self)):

            batch = [n * i for i in range(self.batch_size)] + [len(self) - (n * i) for i in range(self.batch_size)]

            if len(batch) == self.batch_size:

                yield batch
                batch = []

            if len(batch) > 0 and not self.drop_last:

                yield batch

    def __len__(self):

        if self.drop_last:

            return len(self.data_source) // self.batch_size

        else:

            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


p = Pegasus("./data")

horse_set = Pegasus(root='./data', classes=["horse"], train=True, download=True, transform=transform_train)
horse_loader = DataLoader(horse_set, batch_size=args["batch_size"], shuffle=True, num_workers=0, drop_last=True)

print(p)
