import os
import math

from PIL import Image


images = [Image.open("final/" + file) for file in os.listdir("final")]

size = math.ceil(len(images) ** 0.5)
w, h = images[0].size

border = 2
width = w * size + (size - 2) * border
height = h * size + (size - 2) * border

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

dst.save("report/final.png")
