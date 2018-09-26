import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['MinimaxGeneratorLoss', 'MinimaxDiscriminatorLoss']


class MinimaxGeneratorLoss(GeneratorLoss):
    r"""Minimax game generator loss from the original GAN paper
    "Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>

    The loss can be described as:
        L(G) = log(1 - D(G(z)))

    G : Generator
    D : Discriminator
    z : A sample from the noise prior

    The nonsaturating heuristic is also supported:
        L(G) = -log(D(G(z)))

   Args:
        reduction(string, optional): Specifies the reduction to apply
        to the output: 'none' | 'elementwise_mean' | 'sum'.
         'none' : no reduction will be applied,
        'elementwise_mean' : the sum of the elements will be divided
        by the number of elements in the output
        'sum' : the output will be summed. Default 'elementwise_mean'
        Default True

        nonsaturating(bool, optional): Specifies whether to use the
         nonsaturating heuristic loss for the generator :
        L(G) = -log(D(G(z))).
        Default True

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is appliedotherwise (N, *),
          same shape as input

    """
    def __init__(self, reduction='elementwise_mean', nonsaturating=True):
        super(MinimaxGeneratorLoss, self).__init__(reduction)
        self.nonsaturating = nonsaturating

    def forward(self, dgz):
        if self.nonsaturating:
            target = torch.ones_like(dgz)
            return F.binary_cross_entropy(dgz, target,
                                          reduction=self.reduction)
        else:
            target = torch.zeros_like(dgz)
            return -1.0 * F.binary_cross_entropy(dgz, target,
                                                 reduction=self.reduction)


class MinimaxDiscriminatorLoss(DiscriminatorLoss):
    r"""Minimax game discriminator loss from the original GAN paper
    "Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>

    The loss can be described as:
        L(G) = -[log(D(x)) + log(1 - D(G(z)))]

    G : Generator
    D : Discriminator
    x : A sample from the data distribution
    z : A sample from the noise prior

   Args:
        reduction(string, optional): Specifies the reduction to apply
        to the output: 'none' | 'elementwise_mean' | 'sum'.
         'none' : no reduction will be applied,
        'elementwise_mean' : the sum of the elements will be divided
        by the number of elements in the output
        'sum' : the output will be summed. Default 'elementwise_mean'
        Default True

        nonsaturating(bool, optional): Specifies whether to use the
         nonsaturating heuristic loss for the generator :
        L(G) = -log(D(G(z))).
        Default True

    Shape:
        - dx: (N, *) where * means any number of additional dimensions
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is appliedotherwise (N, *),
          same shape as input

    """

    def forward(self, dx, dgz):
        target_ones = torch.ones_like(dgz)
        target_zeros = torch.zeros_like(dx)
        loss = F.binary_cross_entropy(dx, target_ones,
                                      reduction=self.reduction)
        loss += F.binary_cross_entropy(dgz, target_zeros,
                                       reduction=self.reduction)
        return loss
