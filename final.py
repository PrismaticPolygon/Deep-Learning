import os
import math

from PIL import Image


images = [Image.open("pegasi/" + file) for file in os.listdir("pegasi")]

size = math.ceil(len(images) ** 0.5)
w, h = images[0].size

border = 3
width = w * size + (size - 1) * border
height = h * size + (size - 1) * border

dst = Image.new("RGB", (width, height), color=(255, 255, 255))

for i, image in enumerate(images):

    y = (i // size)
    x = (i % size)

    width = x * w
    height = y * h

    if 0 < y < size :

        height += (y * border)

    if 0 < x < size:

        width += (x * border)

    dst.paste(image, (width, height))

dst.save("report/batch.png")
