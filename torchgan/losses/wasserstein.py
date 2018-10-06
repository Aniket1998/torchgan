import torch
import torch.autograd as autograd
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['WassersteinGeneratorLoss', 'WassersteinDiscriminatorLoss', 'WassersteinGradientPenalty']


class WassersteinGeneratorLoss(GeneratorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper

    The loss can be described as:

    .. math:: L(G) = -f(G(z))

    where

    - G : Generator
    - f : Critic/Discriminator
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, fgz):
        r"""
        Args:
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        if self.reduction == 'elementwise_mean':
            return torch.mean(fgz) * -1.0
        elif self.reduction == 'sum':
            return torch.sum(fgz) * -1.0
        else:
            return fgz * -1.0


class WassersteinDiscriminatorLoss(DiscriminatorLoss):
    r"""Wasserstein GAN generator loss from
    `"Wasserstein GAN by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>`_ paper

    The loss can be described as:

    .. math:: L(D) = f(x) - f(G(z))

    where

    - G : Generator
    - f : Critic/Discriminator
    - x : A sample from the data distribution
    - z : A sample from the noise prior

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def forward(self, fx, fgz):
        r"""
        Args:
            dx (torch.Tensor) : Output of the Discriminator. It must have the dimensions
                                (N, \*) where \* means any number of additional dimensions.
            dgz (torch.Tensor) : Output of the Generator. It must have the dimensions
                                 (N, \*) where \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        return reduce(fgz - fx, self.reduction)


class WassersteinGradientPenalty(DiscriminatorLoss):
    r"""Gradient Penalty for the Improved Wasserstein GAN discriminator from
    `"Improved Training of Wasserstein GANs
    by Gulrajani et. al." <https://arxiv.org/abs/1704.00028>`_ paper

    The gradient penalty is calculated as:

    .. math: \lambda \times (norm(grad(D(x))) - 1)^2

    The gradient being taken with respect to x

    where

    - G : Generator
    - D : Disrciminator/Critic
    - :math:`\lambda` : Scaling hyperparameter (default 10.0)
    - x : Interpolation term for the gradient penalty

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.

        lambd(float,optional) : Hyperparameter lambda for scaling the gradient penalty.
    """
    def __init__(self, reduction='elementwise_mean', lambd=10.0):
        super(WassersteinGradientPenalty, self).__init__(reduction)
        self.lambd = lambd

    def forward(self, interpolate, d_interpolate):
        r"""
        Args:
            interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                         \* means any number of additional dimensions.
            d_interpolate (torch.Tensor) : It must have the dimensions (N, \*) where
                                           \* means any number of additional dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        # TODO(Aniket1998): Check for performance bottlenecks
        # If found, write the backprop yourself instead of
        # relying on autograd
        grad_outputs = torch.ones_like(d_interpolate)
        gradients = autograd.grad(outputs=d_interpolate, inputs=interpolate,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]

        gradient_penalty = (gradients.norm(2) - 1) ** 2
        return reduce(self.lambd * gradient_penalty, self.reduction)
