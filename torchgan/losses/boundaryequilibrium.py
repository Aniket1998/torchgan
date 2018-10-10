import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['BoundaryEquilibriumLoss']


class BoundaryEquilibriumLoss(GeneratorLoss, DiscriminatorLoss):
    r"""Boundary Equilibrium GAN generator and discriminator loss from
    `"BEGAN : Boundary Equilibrium Generative Adversarial Networks
    by Berthelot et. al." <https://arxiv.org/abs/1703.10717>`_ paper

    The loss can be described as

    .. math:: L(D) = D(x) - k_t \times D(G(z))
    .. math:: L(G) = D(G(z))
    .. math:: k_{t+1} = k_t + \lambda \times (\gamma \times D(x) - D(G(z)))

    where

    - G : Generator
    - D : Discriminator
    - :math:`k_t` : Running average of the balance point of G and D
    - :math:`\lambda` : Learning rate of the running average
    - :math:`\gamma` : Goal bias hyperparameter

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None,
                 init_k=0.0, lambd=0.001, gamma=0.5):
        super(BoundaryEquilibriumLoss, self).__init__(reduction)
        self.lambd = lambd
        self.k = init_k
        self.gamma = gamma
        self.override_train_ops = override_train_ops

    def set_k(self, k=0.0):
        r"""Change the default value of k

        Args:
            k (float, optional) : New value to be set.
        """
        self.k = k

    def forward(self, dx, dgz):
        r"""
        Args:
            dx (torch.Tensor) : Output of the Discriminator. It must have the dimensions
                                (N, \*) where \* means any number of additional dimensions.
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar tuple if reduction is applied else Tensor tuple each with dimensions (N, \*).
        """
        ld = reduce(dx - self.k * dgz, self.reduction)
        lg = reduce(dgz, self.reduction)
        self.k += self.lambd * (self.gamma * ld - lg)
        return lg, ld

    def train_ops(self, generator, discriminator, optimizer_generator, optimizer_discriminator,
                  real_inputs, noise, labels_provided=False):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_generator,
                                           optimizer_discriminator, real_inputs, noise, labels_provided)
        else:
            real = real_inputs if labels_provided is False else real_inputs[0]
            optimizer_discriminator.zero_grad()
            dx = discriminator(real)
            fake = generator(noise)
            dgz = discriminator(fake.detach())
            loss_g, loss_d = self.forward(dx, dgz)
            loss_d.backward()
            optimizer_discriminator.step()
            optimizer_generator.zero_grad()
            loss_g.backward()
            optimizer_generator.step()
            return loss_g.item(), loss_d.item()
