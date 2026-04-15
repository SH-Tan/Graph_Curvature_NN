import numpy as np
import torch


def minmax_per_batch(x, eps=1e-12):
    x_min = x.min(dim=-1, keepdim=True).values
    x_max = x.max(dim=-1, keepdim=True).values
    denom = (x_max - x_min).clamp_min(eps)
    normalized = (x - x_min) / denom
    has_zero = (x == 0).any(dim=-1, keepdim=True)
    return normalized + (~has_zero).to(x.dtype) * eps


def _normalize_node_value_per_sequence(node_val, name,):
    node_val = node_val.abs()
    node_norm = minmax_per_batch(node_val)

    return 1./node_norm


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
