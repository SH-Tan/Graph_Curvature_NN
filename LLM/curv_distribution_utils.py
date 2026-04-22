import numpy as np
import torch


def minmax_per_batch_nonzero_nozero(x, eps=1e-4):
    mask = x != 0

    xmin = x.masked_fill(~mask, float("inf")).amin(dim=-1, keepdim=True)
    xmax = x.masked_fill(~mask, float("-inf")).amax(dim=-1, keepdim=True)

    has_nonzero = mask.any(dim=-1, keepdim=True)
    xmin = torch.where(has_nonzero, xmin, torch.zeros_like(xmin))
    xmax = torch.where(has_nonzero, xmax, torch.ones_like(xmax))

    denom = (xmax - xmin).clamp_min(eps)

    # normalize nonzero values
    norm = torch.zeros_like(x)
    norm = torch.where(mask, (x - xmin) / denom, norm)

    # push nonzero zeros upward a little so smallest nonzero won't be 0
    norm = torch.where(mask & (norm == 0), torch.full_like(norm, eps), norm)
    return norm


def _normalize_node_value_per_sequence(node_val):
    node_val = node_val.abs()
    node_norm = minmax_per_batch_nonzero_nozero(node_val)
    return 1.0 / node_norm


def _resolve_node_name(operations, names):
    for name in names:
        if not name:
            continue
        if name in operations:
            return name
        # if name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        #     return "layer_input"
    return None


def _edge_distribution(dist_row_or_col, alpha):
    if dist_row_or_col is None:
        return np.array([1.0], dtype=np.float64), np.empty((0,), dtype=np.int64)

    probs = np.asarray(dist_row_or_col, dtype=np.float64).reshape(-1).copy()

    non_zero = np.nonzero(probs)[0]
    if non_zero.size == 0:
        return np.array([1.0], dtype=np.float64), np.empty((0,), dtype=np.int64)

    if np.any(probs == -1.0):
        tmp = (1.0 - alpha) / non_zero.size
        probs[probs == -1.0] = tmp

    return np.hstack((probs[non_zero], np.array([alpha], dtype=np.float64))), non_zero


def _min_reduce_blocks(blocks):
    """
    Faster and cleaner helper than repeating np.minimum.reduce([...]) inline.
    """
    if not blocks:
        return None
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=np.float64)
    return np.minimum.reduce([np.asarray(b, dtype=np.float64) for b in blocks])


def _build_node_distribution(node_tensor, node_name, alpha, eps=1e-7):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    node_tensor = node_tensor.to(dtype=torch.float64)
    if node_tensor.dim() == 3:
        if node_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for node tensor, got shape {tuple(node_tensor.shape)}"
            )
        node_tensor = node_tensor.squeeze(0)

    node_tensor = _normalize_node_value_per_sequence(node_tensor)
    
    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    # weights = torch.exp(-(node_tensor)) * valid_mask
    weights = torch.exp(-(node_tensor ** 2)) * valid_mask

    sum_weights = weights.sum(dim=-1, keepdim=True)
    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )

    empty_mask = (sum_weights <= eps).expand_as(valid_mask)
    dist = torch.where(empty_mask & valid_mask, torch.full_like(dist, -1.0), dist)

    dist = dist * valid_mask
    
    return dist.detach().cpu().numpy().astype(np.float32, copy=False)
