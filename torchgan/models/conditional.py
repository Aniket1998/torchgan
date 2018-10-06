import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcgan import DCGANGenerator, DCGANDiscriminator

__all__ = ['ConditionalGANGenerator', 'ConditionalGANDiscriminator']

class ConditionalGANGenerator(DCGANGenerator):
    r"""Condititional GAN (CGAN) generator based on a DCGAN model from
    `"Conditional Generative Adversarial Nets
    by Mirza et. al. " <https://arxiv.org/abs/1411.1784>`_ paper

    Args:
        num_classes (int) : Total classes present in the dataset.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the CGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    """
    def __init__(self, num_classes, encoding_dims=100, out_channels=3,
                 step_channels=64, batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(ConditionalGANGenerator, self).__init__(encoding_dims + num_classes, out_channels, step_channels,
                                                      batchnorm, nonlinearity, last_nonlinearity)
        self.encoding_dims = encoding_dims
        self.num_classes = num_classes

    def forward(self, z, y):
        # The generator models a joint conditional over the noise sample z and the labels y
        # The simplest way to do it is to concatenate the two variables and then treat it as input to a DCGAN
        # TODO(Aniket1998) : Experiment with other ways of modelling the conditional such as
        # Concatenation at an intermediate layer, or combining using Multilayer Perceptrons
        return super(ConditionalGANGenerator, self).forward(torch.cat((z, y), dim=1))


class ConditionalGANDiscriminator(DCGANDiscriminator):
    r"""Condititional GAN (CGAN) discriminator based on a DCGAN model from
    `"Conditional Generative Adversarial Nets
    by Mirza et. al. " <https://arxiv.org/abs/1411.1784>`_ paper

    Args:
        num_classes (int) : Total classes present in the dataset.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the CGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    """
    def __init__(self, num_classes, in_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None):
        super(ConditionalGANDiscriminator, self).__init__(in_channels + num_classes, step_channels, batchnorm,
                                                          nonlinearity, last_nonlinearity)
        self.input_dims = in_channels
        self.num_classes = num_classes

    def forward(self, x, y):
        # Change dimensions of the label from (N, n_classes) to (N, n_classes, width, height)
        # This is followed by concatenating with the image, as a simple way to model
        # A joint conditional over the images and the labels
        # TODO(Aniket1998) : Eperiment with other ways of modelling the conditional
        # Same as in the generator
        y = y.unsqueeze(2).unsqueeze(3).expand(-1, y.size(1), x.size(2), x.size(3))
        return super(ConditionalGANDiscriminator, self).forward(torch.cat((x, y), dim=1))
