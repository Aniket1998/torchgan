import torch.nn as nn

__all__ = ['GeneratorLoss', 'DiscriminatorLoss']


class GeneratorLoss(nn.Module):
    r"""Base class for all generator losses

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops

    def train_ops(self, generator, discriminator, optimizer_generator, noise):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator, noise)
        else:
            optimizer_generator.zero_grad()
            dgz = discriminator(generator(noise))
            loss = self.forward(dgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()

class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean', override_train_ops=None):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, noise,
                  labels_provided=False):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_discriminator,
                   real_inputs, noise, labels_provided)
        else:
            real = real_inputs if labels_provided is False else real_inputs[0]
            optimizer_discriminator.zero_grad()
            dx = discriminator(real)
            fake = generator(noise)
            dgz = discriminator(fake.detach())
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            return loss.item()
