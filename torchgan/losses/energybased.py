import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['energy_based_generator_loss', 'energy_based_discriminator_loss',
           'EnergyBasedGeneratorLoss', 'EnergyBasedDiscriminatorLoss']

def energy_based_generator_loss(dgz, reduction):
    return reduce(dgz, reduction)

def energy_based_discriminator_loss(dx, dgz, margin, reduction='elementwise_mean'):
    return reduce(dx + F.relu(-dgz + margin), reduction)

class EnergyBasedGeneratorLoss(GeneratorLoss):
    r"""Energy Based GAN generator loss from
    `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper.

    The loss can be described as:

    .. math:: L(G) = D(G(z))

    where

    - G : Generator
    - D : Discriminator
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean'):
        super(EnergyBasedGeneratorLoss, self).__init__(reduction)

    def forward(self, dgz):
        r"""
        Args:
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return energy_based_generator_loss(dgz, self.reduction)


class EnergyBasedDiscriminatorLoss(DiscriminatorLoss):
    r"""Energy Based GAN generator loss from
    `"Energy Based Generative Adversarial Network
    by Zhao et. al." <https://arxiv.org/abs/1609.03126>`_ paper

    The loss can be described as:

    .. math:: L(D) = D(x) + max(0, m - D(G(z)))

    where

    - G : Generator
    - D : Discriminator
    - m : Margin Hyperparameter (default 80.0)
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """

    def __init__(self, reduction='elementwise_mean', margin=80.0):
        super(EnergyBasedDiscriminatorLoss, self).__init__(reduction)
        self.margin = margin

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
        return reduce(dx + F.relu(-dgz + self.margin), self.reduction)
