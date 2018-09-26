import torch
import torch.autograd as autograd
from .loss import GeneratorLoss, DiscriminatorLoss

__all__ = ['WassersteinGeneratorLoss', 'WassersteinDiscriminatorLoss', 'WassersteinGradientPenalty']


class WassersteinGeneratorLoss(GeneratorLoss):
    r"""Wasserstein GAN generator loss from :
    "Wasserstein GAN
    by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>

    The loss can be described as:
        L(G) = -f(G(z))

    G : Generator
    f : Critic/Discriminator
    z : A sample from the noise prior

    Shape:
        - fgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
         same shape as input

    """
    def forward(self, fgz):
        if self.reduction == 'elementwise_mean':
            return torch.mean(fgz) * -1.0
        elif self.reduction == 'sum':
            return torch.sum(fgz) * -1.0
        else:
            return fgz * -1.0


class WassersteinDiscriminatorLoss(DiscriminatorLoss):
    r"""Wasserstein GAN generator loss from :
    "Wasserstein GAN
    by Arjovsky et. al." <https://arxiv.org/abs/1701.07875>

    The loss can be described as:
        L(D) = f(x) - f(G(z))

    G : Generator
    f : Critic/Discriminator
    x : A sample from the data distribution
    z : A sample from the noise prior

    Shape:
        - fx: (N, *) where * means any number of additional dimensions
        - fgz: (N, *) where * means any number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
         same shape as input

    """
    def forward(self, fx, fgz):
        if self.reduction == 'elementwise_mean':
            return torch.mean(fx - fgz)
        elif self.reduction == 'sum':
            return torch.sum(fx - fgz)
        else:
            return (fx - fgz)


class WassersteinGradientPenalty(DiscriminatorLoss):
    r"""Gradient Penalty
     for the Improved Wasserstein GAN discriminator from :
    "Improved Training of Wasserstein GANs
    by Gulrajani et. al." <https://arxiv.org/abs/1704.00028>

    The gradient penalty is calculated as:
        LAMBDA * (norm(grad(D(x))) - 1)**2
    The gradient being taken with respect to x

    G : Generator
    D : Disrciminator/Critic
    LAMBDA : Scaling hyperparameter (default 10.0)
    x : Interpolation term for the gradient penalty

    Args:
        reduction(string, optional): Specifies the reduction to apply
        to the output: 'none' | 'elementwise_mean' | 'sum'.
        'none' : no reduction will be applied,
        'elementwise_mean' : the sum of the elements will be divided
        by the number of elements in the output
        'sum' : the output will be summed. Default 'elementwise_mean'
        Default True

        lambd(float,optional) : Hyperparameter lambda
                                for scaling the gradient penalty

    Shape:
        - interpolate: (N, *) where * means any
                       number of additional dimensions
        - d_interpolate: (N, *) where * means any
                        number of additional dimensions
        - Output: scalar if reduction is applied otherwise (N, *),
          same shape as input

    """
    def __init__(self, reduction='elementwise_mean', lambd=10.0):
        super(WassersteinGradientPenalty, self).__init__(reduction)
        self.lambd = lambd

    def forward(self, interpolate, d_interpolate):
        # TODO(Aniket1998): Check for performance bottlenecks
        # If found, write the backprop yourself instead of
        # relying on autograd
        grad_outputs = torch.ones_like(d_interpolate)
        gradients = autograd.grad(outputs=d_interpolate, inputs=interpolate,
                                  grad_outputs=grad_outputs,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]

        gradient_penalty = (gradients.norm(2) - 1) ** 2
        if self.reduction == 'elementwise_mean':
            return self.lambd * torch.mean(gradient_penalty)
        elif self.reduction == 'sum':
            return self.lambd * torch.sum(gradient_penalty)
        else:
            return self.lambd * gradient_penalty
