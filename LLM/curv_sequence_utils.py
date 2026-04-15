import torch


def _build_vproj_to_att_out_value_map(
    out_node,       # [batch, seq, hidden_size]
    v_in,           # num_kv_heads * head_dim
    seq_len,
    head_dim,
    repeat,
    batch_idx=0,
    flatten_order="by_out_then_seq",
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

    # [r0_s0, r0_s1, ..., r0_sN, r1_s0, r1_s1, ..., r1_sN, ...]
    if flatten_order == "by_out_then_seq":
        # gather -> [seq, v_in, repeat]
        gathered = out_slice[:, outj_map]
        value_map = gathered.permute(1, 2, 0).contiguous()
        value_map = value_map.view(v_in, repeat * seq_len)

    # [s0_r0, s0_r1, ..., s0_rM,s1_r0, s1_r1, ..., s1_rM, ...]
    elif flatten_order == "by_seq_then_out":
        gathered = out_slice[:, outj_map]
        value_map = gathered.permute(1, 0, 2).contiguous()
        value_map = value_map.view(v_in, seq_len * repeat)

    else:
        raise ValueError(f"Unknown flatten_order: {flatten_order}")

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
    flatten_order="by_out_then_seq"
):
    """
    Keep same shape, mask invalid nodes to 0.

    value_map:
        [v_in, repeat * seq_len]   if by_out_then_seq
        [v_in, seq_len * repeat]   if by_seq_then_out
    """
    v_in = value_map.shape[0]

    if flatten_order == "by_out_then_seq":
        # reshape to [v_in, repeat, seq_len]
        x = value_map.view(v_in, repeat, seq_len)

        # clone to avoid modifying original
        x_masked = x.clone()

        # zero out invalid positions (t < s)
        x_masked[:, :, :s] = 0

        return x_masked.view(v_in, repeat * seq_len)

    elif flatten_order == "by_seq_then_out":
        # reshape to [v_in, seq_len, repeat]
        x = value_map.view(v_in, seq_len, repeat)

        x_masked = x.clone()

        # zero out invalid positions
        x_masked[:, :s, :] = 0

        return x_masked.view(v_in, seq_len * repeat)

    else:
        raise ValueError(f"Unknown flatten_order: {flatten_order}")


def masked_oproj_value_map_for_seq(value_map, s, seq_len):
    """
    Same shape [v_out, seq_len].
    Keep prefix [:s+1], zero suffix [s+1:].
    """
    mask = torch.ones(seq_len, device=value_map.device, dtype=value_map.dtype)
    mask[s + 1:] = 0
    return value_map * mask.unsqueeze(0)
