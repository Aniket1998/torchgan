import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from .dcgan import DCGANGenerator, DCGANDiscriminator

__all__ = ['InfoGANGenerator', 'InfoGANDiscriminator']

class InfoGANGenerator(DCGANGenerator):
    r"""Generator for InfoGAN based on the Deep Convolutional GAN (DCGAN) architecture, from
    `"InfoGAN : Interpretable Representation Learning With Information Maximizing Generative Aversarial Nets
    by Chen et. al. " <https://arxiv.org/abs/1606.03657>`_ paper

    Args:
        dim_dis (int) : Dimension of the discrete latent code sampled from the prior.
        dim_cont (int) : Dimension of the continuous latent code sampled from the prior.
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior.
        out_channels (int, optional) : Number of channels in the output Tensor.
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim z -> d - > 2 * d -> 4 * d - > 8 * d
                                        where d = step_channels.
        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator.
        nonlinearity(torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed.
        last_nonlinearity(torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed.

    Example:
        >>> import torchgan.models as models
        >>> G = models.InfoGANGenerator(...)
        >>> z = ...
        >>> c_cont = ...
        >>> c_dis = ...
        >>> x = G(z, c_cont, c_dis)
    """
    def __init__(self, dim_dis, dim_cont, encoding_dims=100, out_channels=3,
                 step_channels=64, batchnorm=True, nonlinearity=None, last_nonlinearity=None):
        super(InfoGANGenerator, self).__init__(encoding_dims + dim_dis + dim_cont, out_channels,
                                               step_channels, batchnorm, nonlinearity, last_nonlinearity)
        self.encoding_dims = encoding_dims
        self.dim_cont = dim_cont
        self.dim_dis = dim_dis

    def forward(self, z, c_dis=None, c_cont=None):
        z_cat = torch.cat([z, c_dis, c_cont],
                          dim=1) if c_dis is not None and c_cont is not None else z
        return super(InfoGANGenerator, self).forward(z_cat)


class InfoGANDiscriminator(DCGANDiscriminator):
    r"""Discriminator for InfoGAN based on the Deep Convolutional GAN (DCGAN) architecture, from
    `"InfoGAN : Interpretable Representation Learning With Information Maximizing Generative Aversarial Nets
    by Chen et. al. " <https://arxiv.org/abs/1606.03657>`_ paper

    The approximate conditional probability distribution over the latent code Q(c|x) is chosen to be a factored
    Gaussian for the continuous latent code and a Categorical distribution for the discrete latent code

    Args:
        dim_dis (int) : Dimension of the discrete latent code sampled from the prior
        dim_cont (int) : Dimension of the continuous latent code sampled from the prior
        encoding_dims (int, optional) : Dimension of the encoding vector sampled from the noise prior. Default 100
        out_channels (int, optional) : Number of channels in the output Tensor. Default 3
        step_channels (int, optional) : Number of channels in multiples of which the DCGAN steps up
                                        the convolutional features
                                        The step up is done as dim `z -> d - > 2 * d -> 4 * d - > 8 * d`
                                        where d = step_channels. Default 64

        batchnorm (bool, optional) : If True, use batch normalization in the convolutional layers of the generator
                                     Default True

        nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the intermediate convolutional layers
                                                  Defaults to LeakyReLU(0.2) when None is passed. Default None

        last_nonlinearity (torch.nn.Module, optional) : Nonlinearity to be used in the final convolutional layer
                                                       Defaults to tanh when None is passed. Default None
    Example:
        >>> import torchgan.models as models
        >>> D = models.InfoGANDiscriminator(...)
        >>> x = ...
        >>> score, q_categorical, q_gaussian = D(x)
    """
    def __init__(self, dim_dis, dim_cont, in_channels=3, step_channels=64,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None, latent_nonlinearity=None):
        self.dim_cont = dim_cont
        self.dim_dis = dim_dis
        super(InfoGANDiscriminator, self).__init__(in_channels, step_channels, batchnorm,
                                                   nonlinearity, last_nonlinearity)

        self.latent_nl = nn.LeakyReLU(0.2) if latent_nonlinearity is None else latent_nonlinearity
        self.critic = self.model[len(self.model) - 2:len(self.model)]
        if batchnorm is True:
            self.dist_conv = nn.Sequential(nn.Conv2d(self.step_ch * 8, self.step_ch * 8, 4, 1, 0, bias=not batchnorm),
                                           nn.BatchNorm2d(self.step_ch * 8),
                                           self.latent_nl)
        else:
            self.dist_conv = nn.Sequential(nn.Conv2d(self.step_ch * 8, self.step_ch * 8, 4, 1, 0, bias=not batchnorm),
                                           nn.BatchNorm2d(self.step_ch * 8),
                                           self.latent_nl)

        self.dis_categorical = nn.Linear(self.step_ch * 8, self.dim_dis)

        self.cont_mean = nn.Linear(self.step_ch * 8, self.dim_cont)
        self.cont_logvar = nn.Linear(self.step_ch * 8, self.dim_cont)

        del self.model[len(self.model) - 2:len(self.model)]

    def forward(self, x, return_latents=False):
        x = self.model(x)
        critic_score = self.critic(x)
        x = self.dist_conv(x).view(-1, x.size(1))
        dist_dis = distributions.OneHotCategorical(logits=self.dis_categorical(x))
        dist_cont = distributions.Normal(loc=self.cont_mean(x), scale=torch.exp(0.5 * self.cont_logvar(x)))
        return critic_score, dist_dis, dist_cont if return_latents is True else critic_score
