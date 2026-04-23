import torch

from curv_distribution_utils import _build_node_distribution


def _build_vproj_to_att_out_value_map(
    out_node,       # [batch, seq, hidden_size]
    v_in,           # num_kv_heads * head_dim
    seq_len,
    head_dim,
    repeat,
    batch_idx=0,
):
    """
    Build a CPU tensor map from v_proj local input index -> reachable Att_out node values.

    Returns
    -------
    value_map : torch.Tensor
        Always on CPU.
    """
    if not torch.is_tensor(out_node):
        out_node = torch.as_tensor(out_node)

    out_node = out_node.detach().to("cpu")

    in_nodes = torch.arange(v_in, dtype=torch.long, device="cpu")
    kv_head = torch.div(in_nodes, head_dim, rounding_mode="floor")
    d = in_nodes % head_dim

    # [v_in, repeat]
    q_offsets = torch.arange(repeat, dtype=torch.long, device="cpu").unsqueeze(0)
    outj_map = d.unsqueeze(1) + (kv_head.unsqueeze(1) * repeat + q_offsets) * head_dim

    # [seq, hidden]
    out_slice = out_node[batch_idx]
    # gather -> [seq, v_in, repeat]
    gathered = out_slice[:, outj_map]

    # [s0_r0, s0_r1, ..., s0_rM,s1_r0, s1_r1, ..., s1_rM, ...]
    value_map = gathered.permute(1, 0, 2).contiguous()
    value_map = value_map.view(v_in, seq_len * repeat)

    return value_map


def _build_oproj_to_att_in_value_map(
    in_node,         # [batch, seq, hidden_size]  (typically Att_out / o_proj input)
    v_out,           # num_q_heads * head_dim = hidden_size
    seq_len,
    head_dim,
    repeat,
    batch_idx=0,
):
    """
    Build value map from o_proj input channels -> corresponding value-space channels.

    Returns
    -------
    value_map : torch.Tensor, shape [v_out, seq_len]
        Row j corresponds to o_proj input channel j, and contains the mapped
        value-side channel across all sequence positions.
    """
    if not torch.is_tensor(in_node):
        in_node = torch.as_tensor(in_node)

    in_node = in_node.detach().to("cpu")

    out_nodes = torch.arange(v_out, dtype=torch.long, device="cpu")
    q_head = torch.div(out_nodes, head_dim, rounding_mode="floor")
    kv_head = torch.div(q_head, repeat, rounding_mode="floor")
    d = out_nodes % head_dim

    # local v index for each o_proj input channel
    v_idx_map = kv_head * head_dim + d   # [v_out]

    # [seq, hidden]
    x = in_node[batch_idx]

    # gather mapped value-space channels across all seq
    # result: [seq, v_out] -> transpose to [v_out, seq]
    value_map = x[:, v_idx_map].transpose(0, 1).contiguous()

    return value_map


def masked_value_map_for_seq(
    value_map,
    s,
    seq_len,
    repeat,
):
    """
    Keep same shape, mask invalid nodes to 0.

    value_map:
        [v_in, seq_len * repeat]   if by_seq_then_out
    """
    v_in = value_map.shape[0]

    # reshape to [v_in, seq_len, repeat]
    x = value_map.view(v_in, seq_len, repeat)

    x_masked = x.clone()

    # zero out invalid positions
    x_masked[:, :s, :] = 0

    return x_masked.view(v_in, seq_len * repeat)



def masked_oproj_value_map_for_seq(value_map, s, seq_len):
    """
    Same shape [v_out, seq_len].
    Keep prefix [:s+1], zero suffix [s+1:].
    """
    mask = torch.ones(seq_len, device=value_map.device, dtype=value_map.dtype)
    mask[s + 1:] = 0
    return value_map * mask.unsqueeze(0)


def _safe_inverse_abs(arr):
    arr = torch.as_tensor(arr, dtype=torch.float32)
    arr = arr.abs()
    inv = torch.full_like(arr, float("inf"))
    inv = torch.where(arr != 0, 1.0 / arr, inv)
    return inv.detach().cpu().numpy().astype("float32", copy=False)


def _precompute_vproj_next_distributions(value_map, seq_len, repeat, node_name, alpha):
    out = []
    for s in range(seq_len):
        masked = masked_value_map_for_seq(
            value_map, s, seq_len, repeat
        ) # [v dim, repeat * seq len]
        
        out.append(_build_node_distribution(masked, node_name, alpha))
    return out


def _precompute_vproj_next_distributions(value_map, node_name, alpha):
    out = _build_node_distribution(value_map, node_name, alpha)
    return out



def _precompute_oproj_prev_distributions(value_map, seq_len, node_name, alpha):
    out = []
    for s in range(seq_len):
        masked = masked_oproj_value_map_for_seq(value_map, s, seq_len)
        out.append(_build_node_distribution(masked, node_name, alpha))
    return out


def _build_vproj_to_att_out_cost(a, seq_len, s_in, v_idx, head_dim, repeat):
    kv_head = v_idx // head_dim
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    block = _safe_inverse_abs(a[0, q_start:q_end, :seq_len, s_in])
    return block.T.reshape(-1)


def _build_att_out_to_o_cost(a, s_out, out_idx, head_dim):
    q_head = out_idx // head_dim
    block = _safe_inverse_abs(a[0, q_head, s_out, :])
    return block.reshape(-1)


