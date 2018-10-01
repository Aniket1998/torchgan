import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['EnergyBasedGeneratorLoss', 'EnergyBasedDiscriminatorLoss']


class EnergyBasedGeneratorLoss(GeneratorLoss):
    r"""Energy Based GAN generator loss from :
    "Energy Based Generative Adversarial Network
     by Zhao et. al." <https://arxiv.org/abs/1609.03126>

    The loss can be described as:
        L(G) = D(G(z))

    G : Generator
    D : Discriminator
    z : A sample from the noise prior

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """
    def forward(self, dgz):
        return reduce(dgz, self.reduction)


class EnergyBasedDiscriminatorLoss(DiscriminatorLoss):
    r"""Energy Based GAN generator loss from :
    "Energy Based Generative Adversarial Network
     by Zhao et. al." <https://arxiv.org/abs/1609.03126>

    The loss can be described as:
        L(D) = D(x) + max(0,m - D(G(z)))

    G : Generator
    D : Discriminator
    m : Margin Hyperparameter (default 80.0)
    z : A sample from the noise prior

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """

    def __init__(self, reduction='elementwise_mean', margin=80.0):
        super(EnergyBasedGeneratorLoss, self).__init__(reduction)
        self.margin = margin

    def forward(self, dx, dgz):
        return reduce(dx + F.relu(-dgz + self.margin), self.reduction)
