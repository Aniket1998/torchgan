import torch
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['BoundaryEquilibriumGeneratorLoss',
           'BoundaryEquilibriumDiscriminatorLoss']


class BoundaryEquilibriumGeneratorLoss(GeneratorLoss):
    r"""Boundary Equilibrium GAN generator loss from :
    "BEGAN : Boundary Equilibrium Generative Adversarial Networks
     by Berthelot et. al." <https://arxiv.org/abs/1703.10717>

    The loss can be described as:
        L(D) = D(x) - k_t * D(G(z))
        L(G) = D(G(z))
        k_t+1 = k_t + lambda * (gamma * D(x) - D(G(z)))

    G : Generator
    D : Discriminator
    k_t : Running average of the balance point of G and D
    lambda : Learning rate of the running average
    gamma : Goal bias hyperparameter

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """
    def __init__(self, reduction='elementwise_mean',
                 init_k=0.0, lambd=0.001, gamma=0.5):
        super(BoundaryEquilibriumGeneratorLoss, self).__init__(reduction)
        self.lambd = lambd
        self.k = init_k
        self.gamma = gamma

    def set_k(self, k=0.0):
        self.k = k

    def forward(self, dx, dgz):
        if self.reduction == 'elementwise_mean':
            ld = torch.mean(dx - self.k * dgz)
            lg = torch.mean(dgz)
        else:
            ld = torch.sum(dx - self.k * dgz)
            lg = torch.sum(dgz)

        self.k += self.lambd * (self.gamma * ld - lg)
        return lg


class BoundaryEquilibriumDiscriminatorLoss(DiscriminatorLoss):
    r"""Boundary Equilibrium GAN generator loss from :
    "BEGAN : Boundary Equilibrium Generative Adversarial Networks
     by Berthelot et. al." <https://arxiv.org/abs/1703.10717>

    The loss can be described as:
        L(D) = D(x) - k_t * D(G(z))
        L(G) = D(G(z))
        k_t+1 = k_t + lambda * (gamma * D(x) - D(G(z)))

    G : Generator
    D : Discriminator
    k_t : Running average of the balance point of G and D
    lambda : Learning rate of the running average
    gamma : Goal bias hyperparameter

    Shape:
        - dgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """
    def __init__(self, reduction='elementwise_mean',
                 init_k=0.0, lambd=0.001, gamma=0.5):
        super(BoundaryEquilibriumDiscriminatorLoss, self).__init__(reduction)
        self.lambd = lambd
        self.k = init_k
        self.gamma = gamma

    def set_k(self, k=0.0):
        self.k = k

    def forward(self, dx, dgz):
        if self.reduction == 'elementwise_mean':
            ld = torch.mean(dx - self.k * dgz)
            lg = torch.mean(dgz)
        else:
            ld = torch.sum(dx - self.k * dgz)
            lg = torch.sum(dgz)

        self.k += self.lambd * (self.gamma * ld - lg)
        return ld
