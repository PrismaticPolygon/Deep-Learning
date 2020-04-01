import os

from PIL import Image, ImageTk
from tkinter import Tk, Frame, Label

from random import shuffle

images = os.listdir("candidates")
shuffle(images)


class Gallery(Frame):

    def __init__(self, master=None):

        Frame.__init__(self, master)
        w, h = 256, 256

        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)

        master.bind('<space>', self.space)
        master.bind('<Right>', self.right)
        master.bind('<Left>', self.left)

        self.pack()

        self.i = 0
        self.label = Label()
        self.image = None
        self.photo_image = None

        self.set()

        self.label.pack()

    def set(self):

        self.image = Image.open("candidates/" + images[self.i])

        self.photo_image = ImageTk.PhotoImage(self.image.resize((256, 256)))
        self.label.configure(image=self.photo_image)
        self.label.image = self.photo_image

        print("{}/{}".format(self.i, len(images)))

    def right(self, event):

        if self.i < len(images):

            self.i += 1

        self.set()

    def left(self, event):

        if self.i >= 1:

            self.i -= 1

        self.set()

    def space(self, event):

        self.image.save(os.path.join("pegasi", self.image.filename.split("/")[1]))
        self.right(event)


root = Tk()
app = Gallery(master=root)
app.mainloop()
