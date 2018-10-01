import torch
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['MutualInformationPenalty']

class MutualInformationPenalty(GeneratorLoss, DiscriminatorLoss):
    """Mutual Information Penalty as defined in
    "InfoGAN : Interpretable Representation Learning by Information Maximising Generative Adversarial Nets
    by Chen et. al." <https://arxiv.org/abs/1606.03657>

    The loss is the variational lower bound of the mutual information between
    the latent codes and the generator distribution and is defined as

    L(G,Q) = log(Q|x)
    where x is drawn from the generator distribution G(z,c)
    c drawn from the latent code prior P(c)

    Args:
        c_dis(int) : The discrete latent code sampled from the prior
        c_cont(int) : The continuous latent code sampled from the prior
        dist_dis(torch.distributions.Distribution) : The auxilliary distribution Q(c|x)
                                                     over the discrete latent code output by the discriminator
        dist_cont(torch.distributions.Distribution) : The auxilliary distribution Q(c|x)
                                                      over the continuous latent code output by the discriminator
        """
    def forward(self, c_dis, c_cont, dist_dis, dist_cont):
        log_probs = torch.Tensor([torch.mean(dist.log_prob(c)) for dist, c in
                                 zip((dist_dis, dist_cont), (c_dis, c_cont))])
        return reduce(-1.0 * log_probs, self.reduction)
