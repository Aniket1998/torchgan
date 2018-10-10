import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['minimax_generator_loss', 'minimax_discriminator_loss', 'MinimaxGeneratorLoss', 'MinimaxDiscriminatorLoss']

def minimax_generator_loss(dgz, nonsaturating=True, reduction='elementwise_mean'):
    if nonsaturating:
        target = torch.ones_like(dgz)
        return F.binary_cross_entropy_with_logits(dgz, target,
                                                  reduction=reduction)
    else:
        target = torch.zeros_like(dgz)
        return -1.0 * F.binary_cross_entropy_with_logits(dgz, target,
                                                         reduction=reduction)

def minimax_discriminator_loss(dx, dgz, reduction='elementwise_mean'):
    target_ones = torch.ones_like(dgz)
    target_zeros = torch.zeros_like(dx)
    loss = F.binary_cross_entropy_with_logits(dx, target_ones,
                                              reduction=reduction)
    loss += F.binary_cross_entropy_with_logits(dgz, target_zeros,
                                               reduction=reduction)
    return loss


class MinimaxGeneratorLoss(GeneratorLoss):
    r"""Minimax game generator loss from the original GAN paper
    `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(G) = log(1 - D(G(z)))

    where

    - G : Generator
    - D : Discriminator
    - z : A sample from the noise prior

    The nonsaturating heuristic is also supported:

    .. math:: L(G) = -log(D(G(z)))

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.

        nonsaturating(bool, optional): Specifies whether to use the
            nonsaturating heuristic loss for the generator
    """
    def __init__(self, reduction='elementwise_mean', nonsaturating=True):
        super(MinimaxGeneratorLoss, self).__init__(reduction)
        self.nonsaturating = nonsaturating

    def forward(self, dgz):
        r"""
        Args:
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return minimax_generator_loss(dgz, self.nonsaturating, self.reduction)


class MinimaxDiscriminatorLoss(DiscriminatorLoss):
    r"""Minimax game discriminator loss from the original GAN paper
    `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(G) = -[log(D(x)) + log(1 - D(G(z)))]

    where

    - G : Generator
    - D : Discriminator
    - x : A sample from the data distribution
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """

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
        return minimax_discriminator_loss(dx, dgz, reduction=self.reduction)
