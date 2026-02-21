# losses/geom_losses.py
import torch

def masked_geom_mse(
    geom_pred: torch.Tensor,   # (B, K)
    geom_gt: torch.Tensor,     # (B, K)
    geom_mask: torch.Tensor,   # (B, K) 0/1
    eps: float = 1e-6
) -> torch.Tensor:
    """
    マスク付き回帰損失:
      sum( mask * (pred-gt)^2 ) / sum(mask)
    """
    diff2 = (geom_pred - geom_gt) ** 2
    num = (diff2 * geom_mask).sum()
    den = geom_mask.sum().clamp_min(eps)
    return num / den
