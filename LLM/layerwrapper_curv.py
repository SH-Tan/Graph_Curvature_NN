import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight, attn_weight @ value


def minmax_per_batch(x, eps=1e-8):
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    return (x - x_min) / (x_max - x_min + eps)


# L2 / root{d}
def _sequence_to_node_value(x):
    if x is None:
        return None

    x = x.detach().float().cpu()

    if x.dim() == 3:
        node_val = torch.linalg.vector_norm(x, ord=2, dim=1) / math.sqrt(x.size(1))

    elif x.dim() == 4:
        node_val = torch.linalg.vector_norm(x, ord=2, dim=(1, 2)) / math.sqrt(x.size(1) * x.size(2))

    else:
        raise ValueError(f"Expected (B,S,N) or (B,H,S,S), got {x.shape}")

    # node_norm = minmax_per_batch(node_val)
    
    return node_val


def _reshape_for_heads(x, num_heads, head_dim):
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _resolve_attention_dims(layer, model):
    attn = layer.self_attn
    config = getattr(attn, "config", None) or getattr(model, "config", None)

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = attn.q_proj.out_features // getattr(config, "num_attention_heads", 1)

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = attn.q_proj.out_features // head_dim

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = attn.k_proj.out_features // head_dim

    num_kv_groups = getattr(attn, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = max(num_heads // num_kv_heads, 1)

    return num_heads, num_kv_heads, num_kv_groups, head_dim


def _store_operation(op_bank, name, node, extra=None, keep_raw=False):
    op_bank[name] = {
        "node": _sequence_to_node_value(node),
        "extra": extra.detach().cpu() if extra is not None else {},
        "raw": node.detach().cpu() if keep_raw and node is not None else None,
    }


def collect_layer_data(layer, x, attention_mask, position_ids, model, layer_id=None, keep_raw=False):
    operations = {}

    with torch.no_grad():
        x_in = x
        x_norm = layer.input_layernorm(x_in)
        
        # ---- store shared layer input ----
        _store_operation(operations, "layer_input", x_norm, keep_raw=keep_raw)

        if not hasattr(layer, "_cached_dims"):
            layer._cached_dims = _resolve_attention_dims(layer, model)

        num_heads, num_kv_heads, num_kv_groups, head_dim = layer._cached_dims

        q_linear = layer.self_attn.q_proj(x_norm)
        k_linear = layer.self_attn.k_proj(x_norm)
        v_linear = layer.self_attn.v_proj(x_norm)

        q = _reshape_for_heads(q_linear, num_heads, head_dim) # [1, 32, 8192, 128]
        k = _reshape_for_heads(k_linear, num_kv_heads, head_dim) # [1, 8, 8192, 128]
        v = _reshape_for_heads(v_linear, num_kv_heads, head_dim) # [1, 8, 8192, 128]

        # ---- RoPE ----
        if position_ids is not None:
            cos, sin = model.model.rotary_emb(x_norm, position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            del cos, sin

        
        k = _repeat_kv(k, num_kv_groups) # [1, 32, 8192, 128]
        v = _repeat_kv(v, num_kv_groups) # [1, 32, 8192, 128]

        # (Q*K^T /sqrt(d)) * V
        A, attn_output = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
        ) # [1, 32, 8192, 128]
        
        attn_context = _merge_heads(attn_output) # [1, 8192, 4096]
        
        q_merge = _merge_heads(q)
        k_merge = _merge_heads(k)
        
        # ---- store Q/K/V (with FULL tensors for cost) ----
        _store_operation(operations, "q_proj", q_merge, extra=q_merge, keep_raw=keep_raw)
        # _store_operation(operations, "k_proj", k_merge, extra=k_merge, keep_raw=keep_raw)
        _store_operation(operations, "v_proj", attn_context, keep_raw=keep_raw)
        
        del q_merge, k_merge
        
        # ---- store A as virtual node (cost carrier) ----
        _store_operation(operations, "A_q", A, keep_raw=keep_raw)
        # _store_operation(operations, "A_k", A.transpose(-2, -1), keep_raw=keep_raw)
        
        # ===== O PROJ =====
        o = layer.self_attn.o_proj(attn_context)
        x_res1 = x_in + o
        x_norm2 = layer.post_attention_layernorm(x_res1)
        
        # ---- attention output ----
        _store_operation(operations, "o_proj", x_norm2, keep_raw=keep_raw)

        # ===== MLP =====
        gate = layer.mlp.gate_proj(x_norm2)
        up = layer.mlp.up_proj(x_norm2)
        act = layer.mlp.act_fn(gate)
        mlp_hidden = act * up
        down = layer.mlp.down_proj(mlp_hidden)
        x_out = x_res1 + down
        
        # _store_operation(operations, "gate_proj", act, keep_raw=keep_raw)
        # _store_operation(operations, "up_proj", up, keep_raw=keep_raw)
        # _store_operation(operations, "down_proj", mlp_hidden, keep_raw=keep_raw) # input node

        # ---- cleanup (GPU memory critical) ----
        del q, k, v, A, attn_output
        del q_linear, k_linear, v_linear
        del attn_context, o, x_res1
        del gate, up, act, mlp_hidden, down

    return x_out, operations


def _make_lm_head_op(model, final_hidden, keep_raw=False):
    if hasattr(model.model, "norm") and model.model.norm is not None:
        final_hidden = model.model.norm(final_hidden)

    logits = model.lm_head(final_hidden)
    probs = torch.softmax(logits.float(), dim=-1)

    return {
        "lm_head": {
            "node": probs,
            "extra": {},
            "raw": probs.detach().cpu() if keep_raw else None,
        }
    }
