import torch

def reduce(x, reduction=None):
    if reduction == "elementwise_mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x
