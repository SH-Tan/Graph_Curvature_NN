import torch

def _weight_from_model(model, short_name, layer_id, device=None):
    if short_name.startswith("prev_"):
        real_name = short_name.replace("prev_", "")
        if layer_id == 0:
            return None
        return _weight_from_model(model, real_name, layer_id - 1, device=device)

    if short_name == "lm_head":
        w = model.lm_head.weight.detach()
    else:
        layer = model.model.layers[layer_id]
        if short_name == "q_proj":
            w = layer.self_attn.q_proj.weight.detach()
        elif short_name == "k_proj":
            w = layer.self_attn.k_proj.weight.detach()
        elif short_name == "v_proj":
            w = layer.self_attn.v_proj.weight.detach()
        elif short_name == "o_proj":
            w = layer.self_attn.o_proj.weight.detach()
        elif short_name == "gate_proj":
            w = layer.mlp.gate_proj.weight.detach()
        elif short_name == "up_proj":
            w = layer.mlp.up_proj.weight.detach()
        elif short_name == "down_proj":
            w = layer.mlp.down_proj.weight.detach()
        else:
            raise KeyError(short_name)

    return w if device is None else w.to(device, non_blocking=True)


def _operation_distance_matrix_torch(model, operations, short_name, layer_id, device):
    weight = _weight_from_model(model, short_name, layer_id, device=device)
    abs_w = weight.abs().float()
    weight_norm = torch.where(abs_w > 0, 1.0 / abs_w, torch.full_like(abs_w, float("inf")))
    return weight_norm.transpose(0, 1).contiguous()

