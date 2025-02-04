import torch


def loss_fn(pred: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
    return -(torch.where(missing_mask, -torch.inf, pred).log_softmax(dim = -1).nan_to_num(0) * torch.where(missing_mask, 0, target)).sum(-1).sum(0)