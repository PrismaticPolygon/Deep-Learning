import torch
import torch.nn as nn
import torchvision.transforms as transforms


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class NormalizeInverse(transforms.Normalize):

    def __init__(self, mean, std):

        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv

        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):

        return super().__call__(tensor.clone())


def Encoder():

    return nn.Sequential(
        nn.Conv2d(3, 12, 4, stride=2, padding=1),   # [batch, 12, 16, 16]
        nn.ReLU(),
        nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
        nn.ReLU(),
        nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
        nn.ReLU()
    )

def Decoder():

    return nn.Sequential(
        nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
        nn.ReLU(),
        nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
        nn.ReLU(),
        nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
        nn.Sigmoid()
    )


if __name__ == "__main__":

    e = Encoder()

    print("\nENCODER\n")

    print(e)

    d = Decoder()

    print("\nDECODER\n")

    print(d)
