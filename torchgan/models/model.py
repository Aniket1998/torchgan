import torch.nn as nn

__all__ = ['Generator', 'Discriminator']


class Generator(nn.Module):
    r"""Base class for all Generator models

    Args:
        encoding_dims (int) : Dimensions of the sample from the noise prior

    """
    # FIXME(Aniket1998): If a user is overriding the default initializer, he must also override the constructor
    # Find an efficient workaround by fixing the initialization mechanism
    def __init__(self, encoding_dims):
        super(Generator, self).__init__()
        self.encoding_dims = encoding_dims

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all generator models.
        Models that require custom weight initialization can override this method"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):
    r"""Base class for all Discriminator models

    Args:
        input_dims (int) : Dimensions of the input

    """
    def __init__(self, input_dims):
        super(Discriminator, self).__init__()
        self.input_dims = input_dims

    # TODO(Aniket1998): Think of better dictionary lookup based approaches to initialization
    # That allows easy and customizable weight initialization without overriding
    def _weight_initializer(self):
        r"""Default weight initializer for all disciminator models.
        Models that require custom weight initialization can override this method"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
