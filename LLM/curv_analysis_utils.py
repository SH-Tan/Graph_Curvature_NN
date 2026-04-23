import os

import numpy as np
import torch


def _analysis_file_path(layer_id, short_name, sample_idx, seq_len, dataset_name):
    analysis_dir = os.path.join(
        os.path.dirname(__file__),
        "curv_analysis",
        f"seq_len_{int(seq_len)}",
        str(dataset_name),
    )
    os.makedirs(analysis_dir, exist_ok=True)
    safe_dataset_name = str(dataset_name).replace("/", "_").replace(" ", "_")
    return os.path.join(
        analysis_dir,
        f"layer_{int(layer_id):03d}_{short_name}_"
        f"{safe_dataset_name}_seq_{int(seq_len):04d}_sample_{int(sample_idx):03d}_"
        "min_curv_summary.txt",
    )


def _summarize_curvatures(curvature):
    curvature = np.asarray(curvature, dtype=np.float32)
    finite_mask = np.isfinite(curvature)
    finite_vals = curvature[finite_mask]

    if finite_vals.size == 0:
        return {
            "finite_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "min_curv": float("nan"),
            "max_curv": float("nan"),
        }

    return {
        "finite_count": int(finite_vals.size),
        "positive_count": int((finite_vals > 0).sum()),
        "negative_count": int((finite_vals < 0).sum()),
        "zero_count": int((finite_vals == 0).sum()),
        "min_curv": float(finite_vals.min()),
        "max_curv": float(finite_vals.max()),
    }


def start_curvature_analysis(layer_id, short_name, sample_idx, curvature_shape, seq_len, dataset_name):
    analysis_path = _analysis_file_path(layer_id, short_name, sample_idx, seq_len, dataset_name)
    header = (
        f"layer_id: {int(layer_id)}\n"
        f"op_name: {short_name}\n"
        f"dataset_name: {dataset_name}\n"
        f"sample_idx: {int(sample_idx)}\n"
        f"curvature_shape: {tuple(curvature_shape)}\n"
        f"seq_len: {int(seq_len)}\n"
        "\n"
        "Per-edge curvature details\n"
        + "=" * 80
        + "\n"
        "Fields: sample_idx, seq_idx, edge=[u_idx,v_idx], W_dist, sp_uv, curvature, "
        "len(mu), len(nu), cost_has_inf, cost_inf_count\n"
        + "\n"
        "Final minimum curvature results for each example\n"
        + "=" * 80
        + "\n"
    )

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(header)
    return analysis_path


def start_sample_section(analysis_path, sample_idx):
    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write(f"Sample {int(sample_idx)}\n")
        f.write("-" * 80 + "\n")


def append_edge_curvature_details(analysis_path, sample_idx, edge_results):
    if not edge_results:
        return

    with open(analysis_path, "a", encoding="utf-8") as f:
        for edge_res in edge_results:
            f.write(
                f"sample_idx={int(sample_idx)}, "
                f"seq_idx={int(edge_res['seq_idx'])}, "
                f"edge=[{int(edge_res['u_idx'])}, {int(edge_res['v_idx'])}], "
                f"W_dist={float(edge_res['w_dist']):.8f}, "
                f"sp_uv={float(edge_res['sp_uv']):.8f}, "
                f"curvature={float(edge_res['curv']):.8f}, "
                f"len(mu)={int(edge_res['mu_len'])}, "
                f"len(nu)={int(edge_res['nu_len'])}, "
                f"cost_has_inf={bool(edge_res['cost_has_inf'])}, "
                f"cost_inf_count={int(edge_res['cost_inf_count'])}\n"
            )


def append_final_min_curvature_summary(
    analysis_path,
    sample_idx,
    curvature,
    avg_mu_len,
    avg_nu_len,
    curr_dist_finite_edges,
    curr_dist_infinite_edges,
):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    summary = _summarize_curvatures(curvature)

    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("Sample summary\n")
        f.write(
            f"sample_idx={int(sample_idx)}, "
            f"curr_dist_finite_edges={int(curr_dist_finite_edges)}, "
            f"curr_dist_infinite_edges={int(curr_dist_infinite_edges)}, "
            f"finite_entries={summary['finite_count']}, "
            f"positive_edges={summary['positive_count']}, "
            f"negative_edges={summary['negative_count']}, "
            f"zero_edges={summary['zero_count']}, "
            f"avg_len(mu)={float(avg_mu_len):.8f}, "
            f"avg_len(nu)={float(avg_nu_len):.8f}, "
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
        )
        f.write("-" * 80 + "\n")
