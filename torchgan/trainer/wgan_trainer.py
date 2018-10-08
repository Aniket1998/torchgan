import torch
import torchvision
from .trainer import Trainer
from ..losses.wasserstein import WassersteinGradientPenalty

__all__ = ['WGANGP_Trainer']

class WGANGP_Trainer(Trainer):
    def __init__(self, ncritic, discriminator_penalty=WassersteinGradientPenalty, *args, **kwargs):
        super(WGANGP_Trainer, self).__init__(*args, **kwargs)
        self.ncritic = ncritic
        if "penalty_discriminator_options" in kwargs:
            self.discriminator_penalty = discriminator_penalty(**kwargs["penalty_discriminator_options"])
        else:
            self.discriminator_penalty = discriminator_penalty()

    def train_stopper(self):
        return self.loss_information["discriminator_iters"] % self.ncritic != 0

    def generator_train_iter(self, **kwargs):
        if self.loss_information["discriminator_iters"] % self.ncritic == 0:
            super(WGANGP_Trainer, self).generator_train_iter(**kwargs)

    def discriminator_train_iter(self, images, labels, **kwargs):
        sampled_noise = torch.randn(self.batch_size, self.generator.encoding_dims, 1, 1, device=self.device)
        epsilon = torch.rand(1).item()
        fake_images = self.generator(sampled_noise)
        interpolate = epsilon * images + (1 - epsilon) * fake_images

        critic_loss = self.discriminator_loss(images, fake_images.detach())
        gradient_penalty = self.discriminator_penalty(interpolate, self.discriminator(interpolate))

        d_loss = critic_loss + gradient_penalty
        d_loss.backward()
        self.loss_information['discriminator_losses'].append(d_loss)
        self.loss_information['discriminator_iters'] += 1
