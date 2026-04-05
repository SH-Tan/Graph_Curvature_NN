
import torch 
import torch.nn as nn 
from layerwrapper_curv import _resolve_attention_dims
import numpy as np
import os

def _curvature_to_param_tensor(curvature_result, weight_tensor):
    if torch.is_tensor(curvature_result):
        if curvature_result.shape == weight_tensor.shape:
            return curvature_result.to(device=weight_tensor.device, dtype=weight_tensor.dtype)
        if curvature_result.T.shape == weight_tensor.shape:
            return curvature_result.T.to(device=weight_tensor.device, dtype=weight_tensor.dtype)
        return None

    curvature_matrix = curvature_result.get("curvature_matrix")
    if curvature_matrix is not None:
        if curvature_matrix.shape == weight_tensor.shape:
            return curvature_matrix.to(device=weight_tensor.device, dtype=weight_tensor.dtype)
        if curvature_matrix.T.shape == weight_tensor.shape:
            return curvature_matrix.T.to(device=weight_tensor.device, dtype=weight_tensor.dtype)

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




def _resolve_attention_dims_for_layer(model, layer_id):
    layer = model.model.layers[layer_id]
    return _resolve_attention_dims(layer, model)


def _weight_from_model(model, short_name, layer_id):
    if short_name.startswith("prev_"):
        real_name = short_name.replace("prev_", "")
        if layer_id == 0:
            return None  # no previous layer

        try:
            return _weight_from_model(model, real_name, layer_id - 1)
        except KeyError:
            return None
    
    if short_name == "lm_head":
        return model.lm_head.weight.detach().cpu()

    layer = model.model.layers[layer_id]
    if short_name == "q_proj":
        return layer.self_attn.q_proj.weight.detach().cpu()
    if short_name == "k_proj":
        return layer.self_attn.k_proj.weight.detach().cpu()
    if short_name == "v_proj":
        return layer.self_attn.v_proj.weight.detach().cpu()
    if short_name == "o_proj":
        return layer.self_attn.o_proj.weight.detach().cpu()
    if short_name == "gate_proj":
        return layer.mlp.gate_proj.weight.detach().cpu()
    if short_name == "up_proj":
        return layer.mlp.up_proj.weight.detach().cpu()
    if short_name == "down_proj":
        return layer.mlp.down_proj.weight.detach().cpu()
    raise KeyError(f"Unknown op short_name: {short_name}")



def _expand_weight_for_gqa(weight, model, layer_id):
    num_q_heads, num_kv_heads, _, head_dim = _resolve_attention_dims_for_layer(model, layer_id)
    if num_q_heads == num_kv_heads:
        return weight

    group = num_q_heads // num_kv_heads

    out_dim, in_dim = weight.shape
    weight = weight.view(num_kv_heads, head_dim, in_dim)

    weight = weight.unsqueeze(1).repeat(1, group, 1, 1)
    weight = weight.view(num_q_heads, head_dim, in_dim)

    return weight.reshape(num_q_heads * head_dim, in_dim)


def _weight_to_cost_matrix(weight):
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(weight.shape)}")

    # Model weights are (out_features, in_features).
    # Graph costs should be indexed as (in_node, out_node).
    return (1.0 / torch.abs(weight)).transpose(0, 1).contiguous()


def _operation_distance_matrix_torch(model, operations, short_name, layer_id, device):
    weight = _weight_from_model(model, short_name, layer_id)
    if short_name in {"k_proj", "v_proj"}:
        weight = _expand_weight_for_gqa(weight, model, layer_id)

    return _weight_to_cost_matrix(weight)



def adaptive_chunksize(max_chunk=512):
    if not torch.cuda.is_available():
        return max_chunk, max_chunk

    free_mem, total_mem = torch.cuda.mem_get_info()
    gb_free = free_mem / (1024**3)
    if gb_free < 10:
        return 64, 64
    elif gb_free < 20:
        return 128, 128
    elif gb_free < 30:
        return 256, 256
    else:
        return max_chunk, max_chunk
    
    

def _to_cpu_numpy(tensor):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return np.asarray(tensor, dtype=np.float64)
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)



def _sample_rows_with_endpoints(total_rows, num_random_rows, seed):
    if total_rows <= 0:
        return torch.empty(0, dtype=torch.long)

    if total_rows == 1:
        return torch.zeros(1, dtype=torch.long)

    endpoint_rows = torch.tensor([0, total_rows - 1], dtype=torch.long)
    num_middle_rows = max(total_rows - endpoint_rows.numel(), 0)
    num_random_rows = min(max(int(num_random_rows), 0), num_middle_rows)

    if num_random_rows == 0:
        return endpoint_rows

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    sampled_rows = torch.randperm(total_rows - 2, generator=generator)[:num_random_rows] + 1
    row_indices = torch.cat((endpoint_rows, sampled_rows))
    row_indices, _ = torch.sort(row_indices)
    return row_indices


def _format_sampled_row_label(row_prefix, row_idx, last_idx):
    tags = []
    if int(row_idx) == 0:
        tags.append("BOS")
    if int(row_idx) == int(last_idx):
        tags.append("last")

    suffix = f" ({', '.join(tags)})" if tags else ""
    return f"{row_prefix}_{row_idx}{suffix}"


def _sampled_row_style(row_idx, last_idx):
    if int(row_idx) == 0:
        return {
            "color": "#ff1493",
            "linewidth": 2.4,
            "alpha": 1.0,
            "zorder": 4,
        }

    if int(row_idx) == int(last_idx):
        return {
            "color": "#00bcd4",
            "linewidth": 2.0,
            "alpha": 0.95,
            "zorder": 3,
        }

    return {
        "linewidth": 1.0,
        "alpha": 0.85,
        "zorder": 2,
    }


def _featurewise_l2_curve(row_feature_matrix):
    if row_feature_matrix is None or row_feature_matrix.numel() == 0 or row_feature_matrix.shape[0] == 0:
        return None
    return torch.linalg.vector_norm(row_feature_matrix, ord=2, dim=0) / torch.sqrt(
        torch.tensor(float(row_feature_matrix.shape[0]))
    )


def _build_l2_variant_specs(raw_curve):
    if raw_curve is not None:
        raw_curve = raw_curve.detach().float().cpu()
        num_rows = raw_curve.shape[0]

        all_curve = _featurewise_l2_curve(raw_curve)
        minus_zero_curve = _featurewise_l2_curve(raw_curve[1:])
    else:
        all_curve = None
        minus_zero_curve = None
        num_rows = 0

    return [
        {
            "label": "all tokens",
            "rows_used": num_rows,
            "curve": all_curve,
            "color": "#202020",
            "linestyle": "-",
            "linewidth": 2.0,
        },
        {
            "label": "all - BOS",
            "rows_used": max(num_rows - 1, 0),
            "curve": minus_zero_curve,
            "color": "#ff8c00",
            "linestyle": "--",
            "linewidth": 1.9,
        },
    ]


def _reduce_curve_for_plot(curve, max_points=192):
    if curve is None:
        return None, None

    curve = curve.detach().float().cpu()
    num_points = curve.numel()
    if num_points == 0:
        return None, None

    if num_points <= max_points:
        return torch.arange(num_points, dtype=torch.float32), curve

    bin_edges = torch.linspace(0, num_points, max_points + 1)
    x_vals = []
    y_vals = []

    for idx in range(max_points):
        start = int(bin_edges[idx].item())
        end = int(bin_edges[idx + 1].item())
        if end <= start:
            end = min(start + 1, num_points)
        segment = curve[start:end]
        if segment.numel() == 0:
            continue
        x_vals.append((start + end - 1) / 2.0)
        y_vals.append(segment.mean().item())

    if not x_vals:
        return torch.arange(num_points, dtype=torch.float32), curve

    return torch.tensor(x_vals, dtype=torch.float32), torch.tensor(y_vals, dtype=torch.float32)


def _plot_l2_comparison_axis(ax, title, curves_to_plot):
    for curve_spec in curves_to_plot:
        plot_x, plot_y = _reduce_curve_for_plot(curve_spec["curve"])
        if plot_x is None or plot_y is None:
            continue

        ax.plot(
            plot_x,
            plot_y,
            color=curve_spec["color"],
            linestyle=curve_spec["linestyle"],
            linewidth=curve_spec["linewidth"],
            label=f"{curve_spec['label']} (rows={curve_spec['rows_used']})",
        )

    ax.set_xlabel("hidden feature index")
    ax.set_ylabel("L2 across token rows")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def save_layer_input_curves(
    x,
    layer_idx,
    save_dir,
    sample_idx=0,
    num_rows=20,
    seed=0,
    mark = "ori"
):
    if save_dir is None:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save layer input curves") from exc

    if x.dim() == 3:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got input shape {tuple(x.shape)}")
        x = x.squeeze(0)
    elif x.dim() != 2:
        raise ValueError(f"Expected (1, seq, hidden) or (seq, hidden), got {tuple(x.shape)}")

    x_cpu = x.detach().float().cpu()
    seq_len, hidden_size = x_cpu.shape
    if seq_len == 0:
        raise ValueError("Cannot plot layer input with empty sequence dimension")

    row_indices = _sample_rows_with_endpoints(
        total_rows=seq_len,
        num_random_rows=num_rows,
        seed=int(seed) + int(layer_idx) * 1009 + int(sample_idx),
    )

    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))
    x_axis = torch.arange(hidden_size)
    for row_idx in row_indices.tolist():
        plot_style = _sampled_row_style(row_idx, seq_len - 1)
        ax.plot(
            x_axis,
            x_cpu[row_idx],
            label=_format_sampled_row_label("seq_row", row_idx, seq_len - 1),
            **plot_style,
        )

    ax.set_xlabel("hidden feature index")
    ax.set_ylabel("|node value|")
    ax.set_title(
        f"Layer {layer_idx} input curves (sample {sample_idx}, {num_rows} random + index 0 + index last)"
    )
    ax.grid(True, alpha=0.3)
    if len(row_indices) <= 22:
        ax.legend(fontsize=8, ncol=2)

    save_path = os.path.join(save_dir, f"layer_{layer_idx:03d}_sample_{sample_idx:03d}_input_{mark}.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path


def _prepare_op_raw_curve(raw_tensor):
    if raw_tensor is None or not torch.is_tensor(raw_tensor):
        return None, None, None

    raw_tensor = raw_tensor.detach().float().cpu().abs()
    if raw_tensor.dim() == 3:
        if raw_tensor.shape[0] != 1:
            raw_tensor = raw_tensor[:1]
        return raw_tensor.squeeze(0), "sequence row", "feature index"

    if raw_tensor.dim() == 4:
        if raw_tensor.shape[0] != 1:
            raw_tensor = raw_tensor[:1]
        # Average heads so each sampled query row becomes one curve over the key axis.
        return raw_tensor.mean(dim=1).squeeze(0), "query row", "key index"

    return None, None, None


def save_layer_op_curves(
    operations,
    layer_idx,
    save_dir,
    sample_idx=0,
    num_rows=20,
    seed=0,
):
    if save_dir is None:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save layer op curves") from exc

    os.makedirs(save_dir, exist_ok=True)
    save_paths = []

    for op_name, op_data in operations.items():
        if op_name.startswith("prev_"):
            continue

        raw_curve, row_label, x_label = _prepare_op_raw_curve(op_data.get("raw"))
        node_value = op_data.get("node")
        if node_value is None or not torch.is_tensor(node_value):
            continue

        comparison_specs = [
            {
                "title": f"{op_name}: L2(all) vs L2(all-BOS)",
                "indices": (0, 1),
            },
        ]

        num_subplots = len(comparison_specs) + (1 if raw_curve is not None else 0)
        fig_height = 2.4 + (2.0 if raw_curve is not None else 0.0)
        fig, axes = plt.subplots(num_subplots, 1, figsize=(7.0, fig_height))
        if num_subplots == 1:
            axes = [axes]
        else:
            axes = list(axes)

        next_axis_idx = 0

        if raw_curve is not None:
            ax = axes[next_axis_idx]
            next_axis_idx += 1
            row_indices = _sample_rows_with_endpoints(
                total_rows=raw_curve.shape[0],
                num_random_rows=num_rows,
                seed=int(seed) + int(layer_idx) * 1009 + int(sample_idx) + sum(ord(c) for c in op_name),
            )

            x_axis = torch.arange(raw_curve.shape[1])
            for row_idx in row_indices.tolist():
                plot_style = _sampled_row_style(row_idx, raw_curve.shape[0] - 1)
                ax.plot(
                    x_axis,
                    raw_curve[row_idx],
                    label=_format_sampled_row_label(row_label, row_idx, raw_curve.shape[0] - 1),
                    **plot_style,
                )
            ax.set_xlabel(x_label)
            ax.set_ylabel("|node value|")
            ax.set_title(f"{op_name}: {num_rows} random curves + index 0 + index last")
            ax.grid(True, alpha=0.3)
            if len(row_indices) <= 12:
                ax.legend(fontsize=6, ncol=2)

        l2_variants = _build_l2_variant_specs(raw_curve)

        for comparison in comparison_specs:
            ax = axes[next_axis_idx]
            next_axis_idx += 1
            curves_to_plot = [
                l2_variants[idx]
                for idx in comparison["indices"]
                if l2_variants[idx]["curve"] is not None
            ]
            if not curves_to_plot:
                ax.text(0.5, 0.5, "curve not available", ha="center", va="center")
                ax.set_axis_off()
                continue

            _plot_l2_comparison_axis(
                ax=ax,
                title=comparison["title"],
                curves_to_plot=curves_to_plot,
            )

        fig.suptitle(f"Layer {layer_idx} op {op_name} (sample {sample_idx})")
        fig.tight_layout()

        save_path = os.path.join(
            save_dir,
            f"layer_{layer_idx:03d}_{op_name}_sample_{sample_idx:03d}_node.png",
        )
        fig.savefig(save_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        save_paths.append(save_path)

    return save_paths
