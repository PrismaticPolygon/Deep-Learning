# Deep-Learning-Coursework

Implementation of `Adversarially Constrained Autoencoder Interpolation (ACAI)`, proposed in `Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer
(Berthelot, Raffel, Roy, and Goodfellow, 2018)`, available [here](https://arxiv.org/abs/1807.07543). Source code:
* [TensorFlow](https://github.com/anonymous-iclr-2019/acai-iclr-2019)
* [PyTorch](https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0)

#### TO-DO
* Switch back to one dataset, with each batch half horses and the other birds. Will require sorting a tensor by a label,
or writing a custom half-and-half sampler. This is likely the better approach, but locks us in to birds and horses. 
* Improve running with `argparse` for parameters
* Add loss logs for graphing. Should be easy.
* Add image output
* Add alpha tuning. There is likely to be an optimum value for alpha.
* 
