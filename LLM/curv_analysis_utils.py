import os

import numpy as np
import torch


EPSILON = 1e-7


def _analysis_file_path(layer_id, short_name):
    analysis_dir = os.path.join(os.path.dirname(__file__), "curv_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return os.path.join(analysis_dir, f"layer_{int(layer_id):03d}_{short_name}_seq_analysis.txt")


def start_curvature_analysis(layer_id, short_name, curvature_shape, total_edges, seq_len):
    analysis_path = _analysis_file_path(layer_id, short_name)
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(f"layer_id: {layer_id}\n")
        f.write(f"op_name: {short_name}\n")
        f.write(f"curvature_shape: {tuple(curvature_shape)}\n")
        f.write(f"total_edges_per_seq: {int(total_edges)}\n")
        f.write(f"seq_len: {int(seq_len)}\n")
        f.write("\n")
        f.write("Per-sequence curvature change summary\n")
        f.write("=" * 80 + "\n")
    return analysis_path


def append_seq_curvature_analysis(analysis_path, seq_idx, prev_vals, new_vals, v_idx, u_idx, topk=10):
    delta = prev_vals - new_vals
    changed_mask = torch.isfinite(delta) & (delta > 0)
    changed_count = int(changed_mask.sum().item())
    total_count = int(delta.numel())
    finite_mask = torch.isfinite(new_vals)
    positive_count = int(((new_vals > 0) & finite_mask).sum().item())
    negative_count = int(((new_vals < 0) & finite_mask).sum().item())
    zero_count = int(((new_vals == 0) & finite_mask).sum().item())

    if finite_mask.any():
        finite_vals = new_vals[finite_mask]
        min_curvature = float(finite_vals.min().item())
        max_curvature = float(finite_vals.max().item())
    else:
        min_curvature = float("nan")
        max_curvature = float("nan")

    if changed_count > 0:
        changed_delta = delta[changed_mask]
        mean_delta = float(changed_delta.mean().item())
        max_delta = float(changed_delta.max().item())
        min_delta = float(changed_delta.min().item())
        rel_delta = changed_delta / prev_vals[changed_mask].abs().clamp_min(EPSILON)
        mean_rel_delta = float(rel_delta.mean().item())

        changed_pos = torch.nonzero(changed_mask, as_tuple=False).flatten()
        top_count = min(topk, changed_count)
        top_vals, top_pos = torch.topk(changed_delta, k=top_count)
        top_indices = changed_pos[top_pos]
    else:
        mean_delta = 0.0
        max_delta = 0.0
        min_delta = 0.0
        mean_rel_delta = 0.0
        top_vals = torch.empty(0, dtype=torch.float32)
        top_indices = torch.empty(0, dtype=torch.long)

    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write(f"seq {seq_idx}\n")
        f.write(
            f"changed_params: {changed_count}/{total_count}, "
            f"mean_abs_delta: {mean_delta:.8f}, "
            f"mean_rel_delta: {mean_rel_delta:.8f}, "
            f"min_delta: {min_delta:.8f}, "
            f"max_delta: {max_delta:.8f}\n"
        )
        f.write(
            f"positive_curvatures: {positive_count}, "
            f"negative_curvatures: {negative_count}, "
            f"zero_curvatures: {zero_count}, "
            f"min_curvature: {min_curvature:.8f}, "
            f"max_curvature: {max_curvature:.8f}\n"
        )

        if changed_count > 0:
            f.write("top_changed_params:\n")
            for rank, (flat_idx, delta_val) in enumerate(zip(top_indices.tolist(), top_vals.tolist()), start=1):
                f.write(
                    f"  {rank}. param[out={int(v_idx[flat_idx])}, in={int(u_idx[flat_idx])}] "
                    f"delta={float(delta_val):.8f}, "
                    f"prev={float(prev_vals[flat_idx]):.8f}, "
                    f"new={float(new_vals[flat_idx]):.8f}\n"
                )
        f.write("-" * 80 + "\n")


def summarize_cost_matrix(cost):
    if torch.is_tensor(cost):
        cost = cost.detach().cpu().numpy()
    else:
        cost = np.asarray(cost)

    finite_mask = np.isfinite(cost)
    finite_vals = cost[finite_mask]

    return {
        "entry_count": int(cost.size),
        "finite_count": int(finite_vals.size),
        "inf_count": int(np.isinf(cost).sum()),
        "max": float(np.max(cost)) if cost.size else float("nan"),
        "finite_min": float(np.min(finite_vals)) if finite_vals.size else float("nan"),
        "finite_mean": float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
        "finite_max": float(np.max(finite_vals)) if finite_vals.size else float("nan"),
    }


def aggregate_cost_stats(cost_stats_list):
    total_entries = 0
    total_finite = 0
    total_inf = 0
    finite_sum = 0.0
    finite_min = float("inf")
    finite_max = float("-inf")
    overall_max = float("-inf")

    for stats in cost_stats_list:
        total_entries += stats["entry_count"]
        total_finite += stats["finite_count"]
        total_inf += stats["inf_count"]
        if np.isfinite(stats["finite_min"]):
            finite_min = min(finite_min, stats["finite_min"])
        if np.isfinite(stats["finite_max"]):
            finite_max = max(finite_max, stats["finite_max"])
        if np.isfinite(stats["finite_mean"]) and stats["finite_count"] > 0:
            finite_sum += stats["finite_mean"] * stats["finite_count"]
        overall_max = max(overall_max, stats["max"])

    finite_mean = (finite_sum / total_finite) if total_finite > 0 else float("nan")
    if total_entries == 0:
        overall_max = float("nan")
    if total_finite == 0:
        finite_min = float("nan")
        finite_max = float("nan")

    return {
        "entry_count": total_entries,
        "finite_count": total_finite,
        "inf_count": total_inf,
        "max": overall_max,
        "finite_min": finite_min,
        "finite_mean": finite_mean,
        "finite_max": finite_max,
    }


def append_seq_cost_analysis(analysis_path, seq_idx, cost_stats):
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write(
            f"seq {seq_idx} cost_stats: "
            f"entries={cost_stats['entry_count']}, "
            f"finite_entries={cost_stats['finite_count']}, "
            f"inf_entries={cost_stats['inf_count']}, "
            f"max={cost_stats['max']:.8f}, "
            f"finite_min={cost_stats['finite_min']:.8f}, "
            f"finite_mean={cost_stats['finite_mean']:.8f}, "
            f"finite_max={cost_stats['finite_max']:.8f}\n"
        )


def append_layer_cost_analysis(analysis_path, cost_stats):
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("Layer cost summary\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"entries={cost_stats['entry_count']}, "
            f"finite_entries={cost_stats['finite_count']}, "
            f"inf_entries={cost_stats['inf_count']}, "
            f"max={cost_stats['max']:.8f}, "
            f"finite_min={cost_stats['finite_min']:.8f}, "
            f"finite_mean={cost_stats['finite_mean']:.8f}, "
            f"finite_max={cost_stats['finite_max']:.8f}\n"
        )


def append_cur_dist_analysis(analysis_path, curr_dist):
    curr_dist_stats = summarize_cost_matrix(curr_dist)
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("cur_dist summary\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"cur_dist_shape={tuple(curr_dist.shape)}, "
            f"entries={curr_dist_stats['entry_count']}, "
            f"finite_entries={curr_dist_stats['finite_count']}, "
            f"inf_entries={curr_dist_stats['inf_count']}, "
            f"max={curr_dist_stats['max']:.8f}, "
            f"finite_min={curr_dist_stats['finite_min']:.8f}, "
            f"finite_mean={curr_dist_stats['finite_mean']:.8f}, "
            f"finite_max={curr_dist_stats['finite_max']:.8f}\n"
        )
        f.write("\n")
