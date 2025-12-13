import os
import gc
import pickle
import re
from collections import defaultdict
import numpy as np

from .e2w_utils_new import *

### ----------------------------------------------------
### Helper Functions (replace with your actual versions)
### ----------------------------------------------------


model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 2, "stride": 2, "padding":0, "out_size": 16}}, 
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 14}},   # After conv1_2 
    
    4: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 12}},  
    5: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 2, "padding":0, "out_size": 5}},   # After conv2_2 
    
    6: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 3}},
    7: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 1}},

    8: {"name": "fc", "dim": {"out_size": 512}},  # Flatten(512×2×2) → 1024
    9: {"name": "fc", "dim": {"out_size": 128}},
    10: {"name": "fc", "dim": {"out_size": 10}}
}

model_dims_small = model_dims
selected_classes = list(range(10))


def extract_label_id(filename):
    match = re.search(r"label(\d+)_id(\d+)", filename)
    return (int(match.group(1)), int(match.group(2))) if match else (-1, -1)


def get_top_c(curvature, b, prefix_dims):
    neg_e = defaultdict(list)
    pos_e = defaultdict(list)
    cnn_e = defaultdict(list)

    # Convert once
    curv = np.asarray(curvature)
    edges = curv.copy()
    edges[:, :2] = edges[:, :2].astype(int)

    # ----- PRECOMPUTE LAYER OF EACH UNIQUE NODE -----
    i_nodes = edges[:, 0].astype(int)
    j_nodes = edges[:, 1].astype(int)
    unique_nodes = np.unique(np.concatenate([i_nodes, j_nodes]))

    # Compute layer for each unique node only once
    unique_layers = np.searchsorted(prefix_dims, unique_nodes, side="right") - 1

    # Convert to a dictionary or array lookup
    node_to_layer = dict(zip(unique_nodes, unique_layers))

    # ----- PROCESS EDGES -----
    for i, j, curr in edges:
        i = int(i)
        j = int(j)

        i_layer = node_to_layer[i]
        j_layer = node_to_layer[j]

        # Adjust curvature only for some layers
        if 0 < i_layer < 8 and abs(curr - 1.0) < 1e-6:
            curr = 2.0

        # CNN edges (non-FC)
        if i_layer not in (6, 7, 8):
            cnn_e[i_layer].append((i, j, curr))
            continue

        # FC edges only between adjacent layers
        elif j_layer == i_layer + 1:
            if curr < 0:
                neg_e[i_layer].append((i, j, curr))
            else:
                pos_e[i_layer].append((i, j, curr))

    return neg_e, pos_e, cnn_e



def fc_results(min_fc, count_fc):
    out = []
    for (i, j), cmin in min_fc.items():
        out.append((i, j, count_fc[(i,j)], cmin))
    return out


### ----------------------------------------------------
### Main Processing + Incremental Saving
### ----------------------------------------------------

def process_and_save(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
    para_dims,
    res_path
):
    """
    Process files and save intermediate pickle results when label_counts hits:
    1, 2, 5, 10 samples per label.
    """

    save_checkpoints = [1, 2, 5, 10]  # change if needed
    
    idx = 0

    prefix = f"{model_full_n}_{metric}_{dataset}_label"
    suffix = ".pkl"

    # --- Collect files ---
    all_files = [
        f for f in os.listdir(data_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    # sort by (sample_id, label)
    all_files = sorted(
        all_files,
        key=lambda f: (extract_label_id(f)[1], extract_label_id(f)[0])
    )

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

    # selected_classes = [0,1,2,3,4]
    # --- State ---
    label_counts = {l: 0 for l in selected_classes}

    min_fc = defaultdict(lambda: float("inf"))
    count_fc = Counter()

    min_cnn = defaultdict(lambda: float("inf"))
    count_cnn = Counter()

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
        neg_e, pos_e, cnn_e = get_top_c(curvature_raw, 1, prefix_dims)

        for layer, edges in cnn_e.items():
            layer_info = model_dims[layer + 2]
            if layer_info["name"] == "cnn":
                pos_w, neg_w = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                # Positive curvature weights
                for w, (c, f, z) in pos_w.items():
                    if c < min_cnn[w]:
                        min_cnn[w] = c
                    count_cnn[w] += f

                # Negative curvature weights
                for w, (c, f, z) in neg_w.items():
                    if c < min_cnn[w]:
                        min_cnn[w] = c
                    count_cnn[w] += f

        
        # --- FC edges (direct edge curvature) ---
        for edge_dict in (neg_e, pos_e):
            for layer, edges in edge_dict.items():
                layer_info = model_dims[layer + 2]

                if layer_info["name"] == "cnn":
                    continue

                for (i, j, c) in edges:
                    key = tuple(sorted((i, j)))
                    if c < min_fc[key]:
                        min_fc[key] = c
                    count_fc[key] += 1
                    
        label_counts[label] += 1
        print(f"[LABEL {label}] count = {label_counts[label]}")

        # --- Checkpoint save ---
        if all(label_counts[l] == save_checkpoints[idx] for l in selected_classes):
            save_name = f"{model_full_n}_{metric}_{dataset}_{label_counts[0]}_combined.pkl"
            save_path = os.path.join(res_path, save_name)
            freq_fc = fc_results(min_fc, count_fc)
            freq_cnn = cnn_results(min_cnn, count_cnn, model_dims, para_dims)
            
            p_all = (
                [("edge", i, j, f, c) for (i, j, f, c) in freq_fc] +
                [("weight", w, None, f, c) for (w, f, c, p) in freq_cnn]
            )

            neg_freq_edges_sorted = sorted(p_all, key=lambda x: (x[4]))  # by frequency desc, curvature asc
            pos_freq_edges_sorted = sorted(p_all, key=lambda x: (-x[4])) # by frequency desc, curvature desc
    
            save_data = {
                'neg': neg_freq_edges_sorted,
                'pos': pos_freq_edges_sorted
            }

            with open(save_path, "wb") as sf:
                pickle.dump(save_data, sf)
                
            idx += 1

            print(f"[SAVE] {save_path}")

        # Clean memory
        del curvature_raw, neg_e, pos_e, cnn_e
        gc.collect()

        # Stop if all labels done
        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("[DONE] Finished processing all labels.")


def cal_dims(model_dims):
    dims = []
    layer_num = len(model_dims)
    
    for i in range(1, layer_num + 1):
        cur_name = model_dims[i]["name"]
        cur_dim = model_dims[i]["dim"]
        cur_size = cur_dim['out_size']

        cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)
            
        dims.append(cur_nodes)
    
    return dims


def cal_parameters(model_dims):
    edges = []
    layer_num = len(model_dims)
    
    for i in range(2, layer_num + 1):
        cur_name = model_dims[i]["name"]
        cur_dim = model_dims[i]["dim"]
        cur_size = cur_dim['out_size']

        pre_name = model_dims[i-1]["name"]
        pre_dim = model_dims[i-1]["dim"]
        pre_size = pre_dim['out_size']

        if cur_name == "cnn":
            k = cur_dim['kernel']
            pool = cur_dim.get('pool', False)
            if pool:
                cur_size *= 2
            pre_channel = 1 if (pre_name == "fc") else pre_dim['channel']
            cur_edges = pre_channel * k**2 * cur_dim['channel']
        else:
            pre_nodes = pre_size if (pre_name == "fc") else pre_dim['channel']*(pre_size**2)
            cur_edges = cur_size * pre_nodes
            
        edges.append(cur_edges)
    
    return edges



def process_data(args):

    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    activation = args.activation
    data_path = args.mnist_data_path
    
    if activation.lower() == "relu":
        # from tools.vgg16_custom_relu_new_small_bn import VGG16_CIFAR10_small_BN
        from tools.vgg9_custom_relu import VGG9_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg16_custom_tanh import VGG16_CIFAR10
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "vgg9_100_"
    elif model_pre_name == 'adv':
        model_name = "vgg16_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg16_wd_"
        
    model_name = model_name + activation + "_s2.pth"
    
    print(model_name)
    para_dims = cal_parameters(model_dims_small)

    total_para = sum(para_dims) # - sum(para_dims[0:2]) # - sum(para_dims[:11]) # - sum(para_dims[-3:])
    
    print(para_dims)
    print(total_para)
    
    process_and_save(
        data_path,
        model_full_n,
        metric,
        dataset,
        sample_size,
        prefix_dims,
        para_dims = para_dims,
        res_path=res_path
    )