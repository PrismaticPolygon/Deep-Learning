# Deep-Learning
Coursework for Deep Learning (SSA)

CIFAR-10 consists of 60,000 32x32 colour images in 10 classes (airplane, airmobile, bird, cat, deer, dog, frog, horse,
ship, and truck). The classes are completely mutually exclusive.

There are 50,000 training images and 10,000 test images. 


The dataset can be downloaded [here](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

Base code for generating Pegasi can be found [here](https://colab.research.google.com/gist/cwkx/c0e7421f470255bb6536e523dba703b5/coursework-pegasus.ipynb)

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks,
pitting them against each other in order to generate new, synthetic instances of data that
can pass for real data. 

Discriminative models learn the boundary between classes.
Generative models model the distribution of individual classes.

One neural network, the generator, generates new data instances. The other, the discriminator,
evaluates them for authenticity. 

The generator takes in random number and returns an image
The generated image is fed into the discriminator alongside a stream of images
taken from the actual, ground-truth dataset
The discriminator returns probabilities.

So the discriminator is in a feedback loop with the ground truth of the images, which we know.
The generator is in a feedback loop with the discriminator.

Both nets are trying to optimise a different, opposing objective function in a zero-sum game.

Training tips:

When training the discriminator, hold the generator values constant.
When training the generator, hold the discriminator constant.

Pre-training the discriminator against the dataset before training the generator will establish a clearer gradient.

The two neural networks must have a similar skill level.

[Andrew Ng notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf)

[Get rekt](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/)

Current state-of-the-art is a spectral norm GAN, so let's go from there.

An autoencoder learns to map datapoints to a latent code from which the data can be recovered
with minimal information loss. The latent code is lower-dimensional than the data.

By mixing codes in latent space and decoding the result, an autoencoder can produce a semantically meaningul combination
of the corresponding datapoints. 

"Latent space": a representation of compressed data.

If interpolating between two points in latent space produces a smooth semantic warping in data space,
this suggests that nearby points in latent space are semantically similar. 




### Adversarially constrained auto-encoder interpolation (ACAI)

Based on `Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer
(Berthelot, Raffel, Roy, and Goodfellow, 2018)`, available [here](https://arxiv.org/abs/1807.07543). Source code:
* [TensorFlow](https://github.com/anonymous-iclr-2019/acai-iclr-2019)
* [PyTorch](https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0)
* [PyTorch CIFAR10](https://github.com/kuangliu/pytorch-cifar)


A critic network is fed interpolants and reconstructions and tries to predict the interpolation coefficient
`a` corresponding to its input, with `a = 0` for reconstructions. The autoencoder is trained to fool the critic
into outputting `a = 0` for both interpolants. 

With code here: https://github.com/anonymous-iclr-2019/acai-iclr-2019
BOOM

Let's clone this repo fast!
If I can getting it running on my PC...
That'd be swell.
Though he does want a collab thing. Shame. 

And it's TensorFlow, so I'd have to convert to PyTorch. Smart, making it difficult.
It means that I'll actually have to understand!
