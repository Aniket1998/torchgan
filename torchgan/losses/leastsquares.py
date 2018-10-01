import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['LeastSquaresGeneratorLoss', 'LeastSquaresDiscriminatorLoss']


class LeastSquaresGeneratorLoss(GeneratorLoss):
    r"""Least Squares GAN generator loss from :
    "Least Squares Generative Adversarial Networks
     by Mao et. al." <https://arxiv.org/abs/1611.04076>

    The loss can be described as:
        L(G) = 0.5 * (D(G(z)) - c)**2

    G : Generator
    D : Disrciminator
    c : target generator label (default 1)
    z : A sample from the noise prior

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
         same shape as input

    """
    def __init__(self, reduction='elementwise_mean', c=1.0):
        super(LeastSquaresGeneratorLoss, self).__init__(reduction)
        self.c = c

    def forward(self, dgz):
        return 0.5 * reduce((dgz - self.c) ** 2, self.reduction)


class LeastSquaresDiscriminatorLoss(DiscriminatorLoss):
    r"""Least Squares GAN discriminator loss from :
    "Least Squares Generative Adversarial Networks
     by Mao et. al." <https://arxiv.org/abs/1611.04076>

    The loss can be described as:
        L(G) = 0.5 * [(D(x) - b)**2 + (D(G(z)) - a)**2]

    G : Generator
    D : Disrciminator
    a : Target discriminator label for generated image (default 0)
    b : Target discriminator label for real image (default 1)

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """
    def __init__(self, reduction='elementwise_mean', a=0.0, b=1.0):
        super(LeastSquaresDiscriminatorLoss, self).__init__(reduction)
        self.a = a
        self.b = b

    def forward(self, dx, dgz):
        return 0.5 * (reduce((dx - self.b) ** 2, self.reduction) +
                      reduce((dgz - self.a) ** 2, self.reduction))
