import os
import torch
import json
import uuid

import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from data import Pegasus, PegasusSampler
from lib import AutoEncoder

run = "runs/20200331_2355"

with open(run + "/args.json", "r") as file:

    args = json.load(file)

ae = AutoEncoder(args["scales"], args["depth"], args["latent"]).to(args["device"])
ae.load_state_dict(torch.load(run + "/weights/ae.pkl"))
ae.eval()

for i in range(100):

    train_set = Pegasus(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_sampler=PegasusSampler(train_set, batch_size=args["batch_size"]))

    for x, _ in train_loader:

        x = x.to(args["device"])
        z, _ = ae(x)  # Encoded and decoded images
        half = args["batch_size"] // 2

        alpha = torch.rand(half, 1, 1, 1).to(args['device']) / 2

        bodies = z[half:]
        wings = z[:half]

        encode_mix = alpha * wings + (1 - alpha) * bodies  # Combined latent space
        decode_mix = ae.decoder(encode_mix)  # Decoded combined latent space

        for image in decode_mix:

            image = image.detach().cpu().numpy()
            transposed = np.transpose(image, (1, 2, 0))

            image = Image.fromarray((transposed * 255).astype(np.uint8))
            image.save("candidates/" + str(uuid.uuid4()) + ".png")