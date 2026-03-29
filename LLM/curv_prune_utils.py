
import torch 
import torch.nn as nn 


def _curvature_to_param_tensor(curvature_result, weight_tensor):
    curvature_score = curvature_result.get("curvature_score")
    if curvature_score is None:
        return None

    W = weight_tensor
    out_features, in_features = W.shape
    device = W.device

    input_value = curvature_result.get("input_value")
    output_value = curvature_result.get("output_value")

    # Fast path: no node info
    if input_value is None or output_value is None:
        return torch.full_like(W, float(curvature_score))

    # ---- flatten once, avoid repeated ops ----
    input_score = input_value.reshape(-1)
    output_score = output_value.reshape(-1)

    # ---- trim instead of pad (faster, avoids alloc) ----
    input_score = input_score[:in_features]
    output_score = output_score[:out_features]

    # ---- if smaller, expand via repeat (faster than pad) ----
    if input_score.numel() < in_features:
        repeat = (in_features + input_score.numel() - 1) // input_score.numel()
        input_score = input_score.repeat(repeat)[:in_features]

    if output_score.numel() < out_features:
        repeat = (out_features + output_score.numel() - 1) // output_score.numel()
        output_score = output_score.repeat(repeat)[:out_features]

    # ---- ensure device + dtype once ----
    input_score = input_score.to(device=device, dtype=W.dtype)
    output_score = output_score.to(device=device, dtype=W.dtype)

    # ---- outer product (core cost) ----
    node_factor = output_score.unsqueeze(1) * input_score.unsqueeze(0)

    # ---- normalize (more stable + cheaper) ----
    mean_val = node_factor.mean()
    if mean_val > 1e-8:
        node_factor = node_factor / mean_val

    return node_factor * float(curvature_score)


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



