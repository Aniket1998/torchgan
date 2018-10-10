import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['mutual_information_penalty', 'MutualInformationPenalty']

def mutual_information_penalty(c_dis, c_cont, dist_dis, dist_cont, reduction='elementwise_mean'):
    log_probs = torch.Tensor([torch.mean(dist.log_prob(c)) for dist, c in
                             zip((dist_dis, dist_cont), (c_dis, c_cont))])
    return reduce(-1.0 * log_probs, reduction)


class MutualInformationPenalty(GeneratorLoss, DiscriminatorLoss):
    r"""Mutual Information Penalty as defined in
    `"InfoGAN : Interpretable Representation Learning by Information Maximising Generative Adversarial Nets
    by Chen et. al." <https://arxiv.org/abs/1606.03657>`_ paper

    The loss is the variational lower bound of the mutual information between
    the latent codes and the generator distribution and is defined as

    .. math:: L(G,Q) = log(Q|x)

    where

    - x is drawn from the generator distribution G(z,c)
    - c drawn from the latent code prior P(c)

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, lambd=1.0, reduction='elementwise_mean', override_train_ops=None):
        super(MutualInformationPenalty, self).__init__(reduction, override_train_ops)
        self.lambd = lambd

    def forward(self, c_dis, c_cont, dist_dis, dist_cont):
        r"""
        Args:
            c_dis (int) : The discrete latent code sampled from the prior
            c_cont (int) : The continuous latent code sampled from the prior
            dist_dis (torch.distributions.Distribution) : The auxilliary distribution Q(c|x)
                                                          over the discrete latent code output by the discriminator
            dist_cont (torch.distributions.Distribution) : The auxilliary distribution Q(c|x)
                                                           over the continuous latent code output by the discriminator

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \*).
        """
        log_probs = torch.Tensor([torch.mean(dist.log_prob(c)) for dist, c in
                                 zip((dist_dis, dist_cont), (c_dis, c_cont))])
        return reduce(-1.0 * log_probs, self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator,
                  optimizer_discriminator, dis_code, cont_code, noise):
        if self.override_train_ops is not None:
            self.override_train_ops(self, generator, discriminator, optimizer_generator,
                                    optimizer_discriminator, dis_code, cont_code, noise)
        else:
            optimizer_discriminator.zero_grad()
            optimizer_generator.zero_grad()
            fake = generator(noise, dis_code, cont_code)
            _, dist_dis, dist_cont = discriminator(fake, True)
            loss = self.forward(dis_code, cont_code, dist_dis, dist_cont)
            weighted_loss = self.lambd * loss
            weighted_loss.backward()
            optimizer_discriminator.step()
            optimizer_generator.step()
            return weighted_loss.item()
