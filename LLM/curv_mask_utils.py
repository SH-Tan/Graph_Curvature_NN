import torch


def _build_unstructured_mask(metric, sparsity_ratio):
    if sparsity_ratio <= 0:
        return torch.zeros_like(metric, dtype=torch.bool)

    num_pruned = int(metric.shape[1] * sparsity_ratio)
    if num_pruned == 0:
        return torch.zeros_like(metric, dtype=torch.bool)

    # topk is MUCH faster than full sort
    indices = torch.topk(metric, num_pruned, dim=1, largest=False).indices

    mask = torch.zeros_like(metric, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def _build_nm_mask(metric, prune_n, prune_m):
    B, D = metric.shape
    mask = torch.zeros_like(metric, dtype=torch.bool)

    if D < prune_m:
        return mask

    # reshape into blocks
    num_blocks = D // prune_m
    trimmed = metric[:, :num_blocks * prune_m]

    blocks = trimmed.view(B, num_blocks, prune_m)

    # find smallest n in each block
    idx = torch.topk(blocks, prune_n, dim=2, largest=False).indices

    # scatter
    base = torch.arange(num_blocks, device=metric.device).view(1, num_blocks, 1) * prune_m
    idx = idx + base

    mask.scatter_(1, idx.view(B, -1), True)
    return mask
