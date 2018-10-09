import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['least_squares_generator_loss', 'least_squares_discriminator_loss',
           'LeastSquaresGeneratorLoss', 'LeastSquaresDiscriminatorLoss']

def least_squares_generator_loss(dgz, c=1.0, reduction='elementwise_mean'):
    return 0.5 * reduce((dgz - c) ** 2, reduction)


def least_squares_discriminator_loss(dx, dgz, a=0.0, b=1.0, reduction='elementwise_mean'):
    return 0.5 * (reduce((dx - b) ** 2, reduction) + reduce((dgz - a) ** 2, reduction))


class LeastSquaresGeneratorLoss(GeneratorLoss):
    r"""Least Squares GAN generator loss from
    `"Least Squares Generative Adversarial Networks
    by Mao et. al." <https://arxiv.org/abs/1611.04076>`_ paper

    The loss can be described as

    .. math:: L(G) = \frac{(D(G(z)) - c)^2}{2}

    where

    - G : Generator
    - D : Disrciminator
    - c : target generator label (default 1)
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', c=1.0):
        super(LeastSquaresGeneratorLoss, self).__init__(reduction)
        self.c = c

    def forward(self, dgz):
        r"""
        Args:
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return least_squares_generator_loss(dgz, self.c, self.reduction)


class LeastSquaresDiscriminatorLoss(DiscriminatorLoss):
    r"""Least Squares GAN discriminator loss from
    `"Least Squares Generative Adversarial Networks
    by Mao et. al." <https://arxiv.org/abs/1611.04076>`_ paper.

    The loss can be described as:

    .. math:: L(G) = \frac{(D(x) - b)^2 + (D(G(z)) - a)^2}{2}

    where

    - G : Generator
    - D : Disrciminator
    - a : Target discriminator label for generated image (default 0)
    - b : Target discriminator label for real image (default 1)

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', a=0.0, b=1.0):
        super(LeastSquaresDiscriminatorLoss, self).__init__(reduction)
        self.a = a
        self.b = b

    def forward(self, dx, dgz):
        r"""
        Args:
            dx (torch.Tensor) : Output of the Discriminator. It must have the dimensions
                                (N, \*) where \* means any number of additional dimensions.
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return least_squares_discriminator_loss(dx, dgz, self.a, self.b, self.reduction)
