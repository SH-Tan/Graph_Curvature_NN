import os
import gc
import pickle
import re
from collections import defaultdict

### ----------------------------------------------------
### Helper Functions (replace with your actual versions)
### ----------------------------------------------------

def extract_label_id(filename):
    match = re.search(r"label(\d+)_id(\d+)", filename)
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)


def get_top_c(curvature, prefix_dims):
    """Your optimized version goes here."""
    pass


def aggregate_cnn_weight_curvature(edges, edge_to_weight):
    """Your optimized version goes here."""
    pass


def build_cnn_edge_weight_map(*args, **kwargs):
    """Your existing function."""
    pass


def count_edge_frequency(edges):
    """Your existing function."""
    pass


def count_weight_frequency(weights, model_dims, para_dims):
    """Your existing function."""
    pass


### ----------------------------------------------------
### Main Processing + Incremental Saving
### ----------------------------------------------------

def process_and_save(
    data_path,
    save_prefix,
    model_dims,
    para_dims,
    prefix_dims,
    selected_classes,
    sample_size,
    metric,
    dataset,
):
    """
    Process files and save intermediate pickle results when label_counts hits:
    1, 2, 5, 10 samples per label.
    """

    save_checkpoints = {1, 2, 5, 10}  # change if needed

    prefix = f"{save_prefix}_{metric}_{dataset}_label"
    suffix = ".pkl"

    # --- Collect files ---
    all_files = [
        f for f in os.listdir(data_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    all_files = sorted(all_files, key=lambda f: extract_label_id(f)[1])
    print(f"[INFO] Found {len(all_files)} files.")

    # --- Precompute CNN edge → weight maps ---
    cnn_edge_to_weight_map = {}
    for layer in range(len(model_dims) - 2):
        layer_info = model_dims[layer + 2]
        if layer_info["name"] == "cnn":
            pre_dim = model_dims[layer + 1]["dim"]
            cur_dim = layer_info["dim"]
            cnn_edge_to_weight_map[layer] = build_cnn_edge_weight_map(
                pre_dim["channel"],
                pre_dim["out_size"],
                cur_dim["channel"],
                cur_dim["kernel"],
                cur_dim["stride"],
                cur_dim["padding"],
                layer,
                prefix_dims,
            )

    # --- State ---
    label_counts = {l: 0 for l in selected_classes}

    collected_fc_edges = []
    collected_weights = []

    # --- Main loop ---
    for f in all_files:
        label, sample_id = extract_label_id(f)
        if label not in selected_classes:
            continue
        if label_counts[label] >= sample_size:
            continue

        file_path = os.path.join(data_path, f)
        print(f"[LOAD] {file_path}")

        with open(file_path, "rb") as fp:
            curvature_raw = pickle.load(fp)

        # Extract edges per-layer
        neg_e, pos_e, cnn_e = get_top_c(curvature_raw, prefix_dims)

        # CNN processing
        for layer, edges in cnn_e.items():
            layer_info = model_dims[layer + 2]
            if layer_info["name"] == "cnn":
                pos_w, neg_w = aggregate_cnn_weight_curvature(
                    edges, cnn_edge_to_weight_map[layer]
                )

                # store weight stats
                collected_weights.extend([
                    (w, c, f, para_dims[w[0]])
                    for w, (c, f, _) in pos_w.items()
                ])
                collected_weights.extend([
                    (w, c, f, para_dims[w[0]])
                    for w, (c, f, _) in neg_w.items()
                ])

        # FC processing
        for layer, edges in neg_e.items():
            layer_info = model_dims[layer + 2]
            if layer_info["name"] != "cnn":
                collected_fc_edges.extend(edges)

        for layer, edges in pos_e.items():
            layer_info = model_dims[layer + 2]
            if layer_info["name"] != "cnn":
                collected_fc_edges.extend(edges)

        label_counts[label] += 1
        print(f"[LABEL {label}] count = {label_counts[label]}")

        # --- Checkpoint save ---
        if label_counts[label] in save_checkpoints:
            save_path = f"{save_prefix}_label{label}_count{label_counts[label]}.pkl"
            save_data = {
                "label": label,
                "count": label_counts[label],
                "fc_edges": collected_fc_edges,
                "weights": collected_weights,
            }

            with open(save_path, "wb") as sf:
                pickle.dump(save_data, sf)

            print(f"[SAVE] {save_path}")

        # Clean memory
        del curvature_raw, neg_e, pos_e, cnn_e
        gc.collect()

        # Stop if all labels done
        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("[DONE] Finished processing all labels.")
