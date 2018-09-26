import torch.nn as nn

__all__ = ['GeneratorLoss', 'DiscriminatorLoss']


class GeneratorLoss(nn.Module):
    r"""Base class for all generator losses

    Args:
        reduction(string, optional): Specifies the reduction to apply
        to the output: 'none' | 'elementwise_mean' | 'sum'.
         'none' : no reduction will be applied,
        'elementwise_mean' : the sum of the elements will be divided
        by the number of elements in the output
        'sum' : the output will be summed. Default 'elementwise_mean'
        Default True
    """
    def __init__(self, reduction='elementwise_mean'):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction


class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses

    Args:
        reduction(string, optional): Specifies the reduction to apply
        to the output: 'none' | 'elementwise_mean' | 'sum'.
         'none' : no reduction will be applied,
        'elementwise_mean' : the sum of the elements will be divided
        by the number of elements in the output
        'sum' : the output will be summed. Default 'elementwise_mean'
        Default True
    """
    def __init__(self, reduction='elementwise_mean'):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
