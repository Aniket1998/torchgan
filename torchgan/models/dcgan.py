import torch.nn as nn
import torch.nn.functional as F
from .model import Generator, Discriminator

__all__ = ['DCGANGenerator', 'DCGANDiscriminator',
           'SmallDCGANGenerator', 'SmallDCGANDiscriminator']

class DCGANGenerator(Generator):
    r"""Deep Convolutional GAN (DCGAN) generator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.
    """
    def __init__(self, encoding_dims=100, out_channels=3, step_channels=64,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(DCGANGenerator, self).__init__(encoding_dims)
        self.ch = out_channels
        self.step_ch = step_channels
        use_bias = not batchnorm

        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity

        if batchnorm is True:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(self.encoding_dims, self.step_ch * 8, 4, 1, 0, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 8), nl,
                nn.ConvTranspose2d(self.step_ch * 8, self.step_ch * 4, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 4), nl,
                nn.ConvTranspose2d(self.step_ch * 4, self.step_ch * 2, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 2), nl,
                nn.ConvTranspose2d(self.step_ch * 2, self.step_ch, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch), nl,
                nn.ConvTranspose2d(self.step_ch, self.ch, 4, 2, 1, bias=use_bias), last_nl)
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(self.encoding_dims, self.step_ch * 8, 4, 1, 0, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 8, self.step_ch * 4, 4, 2, 1, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 4, self.step_ch * 2, 4, 2, 1, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 2, self.step_ch, 4, 2, 1, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch, self.ch, 4, 2, 1, bias=use_bias), last_nl)

        self._weight_initializer()

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class DCGANDiscriminator(Discriminator):
    r"""Deep Convolutional GAN (DCGAN) discriminator from
    `"Unsupervised Representation Learning With Deep Convolutional Generative Aversarial Networks
    by Radford et. al. " <https://arxiv.org/abs/1511.06434>`_ paper

    Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior.
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (toch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                      Defaults to sigmoid when None is passed.
    """

    def __init__(self, in_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None):
        super(DCGANDiscriminator, self).__init__(in_channels)
        self.step_ch = step_channels
        self.batchnorm = batchnorm
        use_bias = not batchnorm

        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity

        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Conv2d(self.input_dims, self.step_ch, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch), nl,
                nn.Conv2d(self.step_ch, self.step_ch * 2, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 2), nl,
                nn.Conv2d(self.step_ch * 2, self.step_ch * 4, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 4), nl,
                nn.Conv2d(self.step_ch * 4, self.step_ch * 8, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 8), nl,
                nn.Conv2d(self.step_ch * 8, 1, 4, 1, 0, bias=use_bias), last_nl)
        else:
            self.model = nn.Sequential(
                nn.Conv2d(self.input_dims, self.step_ch, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch, self.step_ch * 2, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 2, self.step_ch * 4, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 4, self.step_ch * 8, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 8, 1, 4, 1, 0, bias=use_bias), last_nl)

        self._weight_initializer()

    def forward(self, x):
        return self.model(x)

class SmallDCGANGenerator(Generator):
    r"""A custom version of Deep Convolutional GAN (DCGAN) generator to support
    training and inference on images of size :math:`32 \times 32`.

    Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.
    """
    def __init__(self, encoding_dims=100, out_channels=3, step_channels=64,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(SmallDCGANGenerator, self).__init__(encoding_dims)
        self.ch = out_channels
        self.step_ch = step_channels
        use_bias = not batchnorm

        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.Tanh() if last_nonlinearity is None else last_nonlinearity

        if batchnorm is True:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(self.encoding_dims, self.step_ch * 8, 4, 1, 0, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 8), nl,
                nn.ConvTranspose2d(self.step_ch * 8, self.step_ch * 4, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 4), nl,
                nn.ConvTranspose2d(self.step_ch * 4, self.step_ch * 2, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 2), nl,
                nn.ConvTranspose2d(self.step_ch * 2, self.ch, 4, 2, 1, bias=use_bias), last_nl)
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(self.encoding_dims, self.step_ch * 8, 4, 1, 0, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 8, self.step_ch * 4, 4, 2, 1, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 4, self.step_ch * 2, 4, 2, 1, bias=use_bias), nl,
                nn.ConvTranspose2d(self.step_ch * 2, self.ch, 4, 2, 1, bias=use_bias), last_nl)

        self._weight_initializer()

    def forward(self, x):
        x = x.view(-1, x.size(1), 1, 1)
        return self.model(x)


class SmallDCGANDiscriminator(Discriminator):
    r"""A custom version of Deep Convolutional GAN (DCGAN) discriminator to support
    training and inference on images of size :math:`32 \times 32`.

    Args:
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior.
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity (toch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                      Defaults to sigmoid when None is passed.
    """

    def __init__(self, in_channels=3, step_channels=64, batchnorm=True,
                 nonlinearity=None, last_nonlinearity=None):
        super(SmallDCGANDiscriminator, self).__init__(in_channels)
        self.step_ch = step_channels
        self.batchnorm = batchnorm
        use_bias = not batchnorm

        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity

        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Conv2d(self.input_dims, self.step_ch, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch), nl,
                nn.Conv2d(self.step_ch, self.step_ch * 2, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 2), nl,
                nn.Conv2d(self.step_ch * 2, self.step_ch * 4, 4, 2, 1, bias=use_bias),
                nn.BatchNorm2d(self.step_ch * 4), nl,
                nn.Conv2d(self.step_ch * 4, self.step_ch * 8, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 8, 1, 2, 2, 0, bias=use_bias), last_nl)
        else:
            self.model = nn.Sequential(
                nn.Conv2d(self.input_dims, self.step_ch, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch, self.step_ch * 2, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 2, self.step_ch * 4, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 4, self.step_ch * 8, 4, 2, 1, bias=use_bias), nl,
                nn.Conv2d(self.step_ch * 8, 1, 2, 2, 0, bias=use_bias), last_nl)

        self._weight_initializer()

    def forward(self, x):
        return self.model(x)
