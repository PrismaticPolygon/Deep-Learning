import matplotlib.pyplot as plt
import numpy as np

ae = np.load("runs/20200401_1949/losses_ae.npy")
d = np.load("runs/20200401_1949/losses_d.npy")


def graph(arr, color, label):

    plt.plot(arr, color, label=label)

    plt.ylabel("Loss")
    plt.xlabel("Epochs")

    plt.legend(loc="upper right")

    plt.savefig("report/{}.png".format(label))

    plt.clf()


graph(ae, "r", "ae")
graph(d, "b", "d")
