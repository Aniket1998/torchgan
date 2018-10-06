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
    def __init__(self, reduction='elementwise_mean'):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction


class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output.
            If `none` no reduction will be applied. If `elementwise_mean` the sum of
            the elements will be divided by the number of elements in the output. If
            `sum` the output will be summed.
    """
    def __init__(self, reduction='elementwise_mean'):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
