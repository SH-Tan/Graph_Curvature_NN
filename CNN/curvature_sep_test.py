import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import torch.nn as nn
from collections import defaultdict
import copy

import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import gc
import re

from .e2w_utils_new import *

import sys
sys.path.append("..")

import tools.utils as utils

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

transform_train = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train)
data_test = CIFAR10('./data/cifar10', train=False, download=True, transform=transform_test)


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
selected_classes = [0,1,2,3,4,5,6,7,8,9]


def standard_PGD(model, images, labels, device, eps=11/255, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images




def test_clean(n, loader, device = 'cuda'):
    n.eval()
    total_correct = 0.
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    acc = float(total_correct) / len(loader.dataset)
    return acc



def test_adversarial(net, loader, eps=.1, alpha=.1, iters=100, device = 'cuda'):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    correct = 0.

    for data, target in loader:

        data = standard_PGD(net, data, target, device, eps=eps, alpha=alpha, iters=iters)
        data, target = data.to(device), target.to(device)

        output = net(data)
        pred = output.detach().max(1)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    
    return float(correct) / len(loader.dataset)



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


def cal_edges(model_dims):
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
            cur_edges = pre_channel * k**2 * cur_size**2 * cur_dim['channel']
        else:
            pre_nodes = pre_size if (pre_name == "fc") else pre_dim['channel']*(pre_size**2)
            cur_edges = cur_size * pre_nodes
            
        edges.append(cur_edges)
    
    return edges


from collections import Counter
def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(list)

    # for edge_set in edge_sets:
    for i, j, c in edge_sets:
        key = tuple(sorted((i, j)))  # normalize direction for undirected edges
        freq[key] += 1
        curvature_sum[key].append(c)

    results = []
    for key in freq:
        avg_curv = np.min(curvature_sum[key])
        results.append((key[0], key[1], freq[key], avg_curv))

    return results


# extract label + id from filename
def extract_label_id(f):
    match = re.search(r'label(\d+)_id(\d+)', f)
    if match:
        return int(match.group(1)), int(match.group(2))
    return -1, -1
    

def process_single_label(
    label,
    all_files,
    data_path,
    prefix_dims,
    para_dims,
    sample_size,
    cnn_edge_to_weight_map
):
    label_count = 0
    neg_weight_sets = []
    pos_weight_sets = []
    edge_sets = []

    for f in all_files:
        l, sample_id = extract_label_id(f)
        if l != label:
            continue
        if label_count >= sample_size:
            break

        with open(os.path.join(data_path, f), "rb") as file:
            ricci = pickle.load(file)

        neg_e, pos_e, cnn_e = get_top_c(ricci, b=1, prefix_dims=prefix_dims)

        # ---- CNN layers ----
        for layer, edges in cnn_e.items():
            layer_info = model_dims[layer + 2]
            if layer_info["name"] != "cnn":
                continue

            pos_w, neg_w = aggregate_cnn_weight_curvature(
                edges, cnn_edge_to_weight_map[layer]
            )

            pos_weight_sets.append([
                (w, c, f, para_dims[w[0]])
                for w, (c, f, _) in pos_w.items()
            ])
            neg_weight_sets.append([
                (w, c, f, para_dims[w[0]])
                for w, (c, f, _) in neg_w.items()
            ])

        # ---- FC layers ----
        for layer, edges in neg_e.items():
            if model_dims[layer + 2]["name"] != "cnn":
                edge_sets.extend(edges)

        for layer, edges in pos_e.items():
            if model_dims[layer + 2]["name"] != "cnn":
                edge_sets.extend(edges)

        label_count += 1
        del ricci, neg_e, pos_e, cnn_e
        gc.collect()
        
    freq_fc = count_edge_frequency(edge_sets)
    
    all_weight_sets = pos_weight_sets + neg_weight_sets

    freq_cnn = count_weight_frequency(all_weight_sets, model_dims, para_dims)
    

    p_all = (
        [("edge", i, j, f, c) for (i, j, f, c) in freq_fc] +
        [("weight", w, None, f, c) for (w, f, c, p) in freq_cnn]
    )

    # neg_freq_edges_sorted = sorted(p_all, key=lambda x: (x[4]))  # by frequency desc, curvature asc
    pos_freq_edges_sorted = sorted(p_all, key=lambda x: (-x[4])) # by frequency desc, curvature desc

    return {
        "label": label,
        "pos_weight_sets": pos_freq_edges_sorted,
    }



def process_batches_memory_efficient(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
    select_l,
    para_dims,
    res_path
):
    # prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    prefix = f"{model_full_n}_{metric}_{dataset}_label"
    suffix = ".pkl"

    all_files = sorted(
        [f for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith(suffix)],
        key=lambda f: extract_label_id(f)[1]
    )

    print(f"Found files: {all_files}")
    
    # Precompute edge->weight mapping per CNN layer
    cnn_edge_to_weight_map = {}
    for layer in range(len(model_dims)-2):  # skip input/output placeholder layers
        layer_info = model_dims[layer+2]
        if layer_info["name"] == "cnn":
            pre_dim = model_dims[layer+1]["dim"]
            pre_ch = pre_dim["channel"]
            in_size = pre_dim["out_size"]
            cur_dim = layer_info["dim"]
            cur_ch = cur_dim["channel"]
            kernel = cur_dim["kernel"]
            stride = cur_dim["stride"]
            padding = cur_dim["padding"]

            cnn_edge_to_weight_map[layer] = build_cnn_edge_weight_map(
                pre_ch, in_size, cur_ch, kernel, stride, padding, layer, prefix_dims
            )

    for label in select_l:
        save_path = os.path.join(res_path, f"curv_label{label}.pkl")
        if os.path.exists(save_path):
            print(f"[Skip] Label {label} already processed.")
            continue

        print(f"[Processing] Label {label}")

        result = process_single_label(
            label=label,
            all_files=all_files,
            data_path=data_path,
            prefix_dims=prefix_dims,
            para_dims=para_dims,
            sample_size=sample_size,
            cnn_edge_to_weight_map=cnn_edge_to_weight_map
        )

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

        print(f"[Saved] {save_path}")
        

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



def extract_negative_params(label_data):
    """
    Returns:
      param_id -> curvature (most negative wins)

    param_id:
      ("weight", w)        for CNN
      ("edge", i, j)       for FC
    """
    neg = {}

    for item in label_data["pos_weight_sets"]:
        kind, a, b, _, c = item   # ignore freq, use curvature only

        # if c >= 0:
        #     continue

        if kind == "weight":
            param_id = ("weight", a)      # a = w
        elif kind == "edge":
            param_id = ("edge", a, b)     # a=i, b=j
        else:
            continue

        # keep the most negative curvature
        if param_id not in neg or c < neg[param_id]:
            neg[param_id] = c

    return neg



def aggregate_rest_curvature(all_label_neg, target_label):
    rest = defaultdict(list)

    for l, negs in all_label_neg.items():
        if l == target_label:
            continue
        for p, c in negs.items():
            rest[p].append(c)

    return {p: np.min(cs) for p, cs in rest.items()}



def one_vs_rest_curvature(
    all_label_neg,
    target_label,
    margin=0.0
):
    target = all_label_neg[target_label]
    rest = aggregate_rest_curvature(all_label_neg, target_label)

    S_target_only = []
    S_rest_only = []
    S_shared = []

    for p, c_t in target.items():
        if p not in rest:
            S_target_only.append((p, c_t))
        else:
            c_r = rest[p]
            if (c_t < c_r - margin) and (c_r >= -0.5):
                S_target_only.append((p, c_t))
            else:
                # you can choose which curvature to sort shared by
                S_shared.append((p, c_t))

    for p, c_r in rest.items():
        if p not in target:
            S_rest_only.append((p, c_r))

    # sort by curvature: high → low
    S_target_only.sort(key=lambda x: x[1], reverse=True)
    S_rest_only.sort(key=lambda x: x[1], reverse=True)
    S_shared.sort(key=lambda x: x[1], reverse=True)

    # if you only want the parameter keys (not curvature values):
    S_target_only = [p for p, _ in S_target_only]
    S_rest_only   = [p for p, _ in S_rest_only]
    S_shared      = [p for p, _ in S_shared]

    return S_target_only, S_rest_only, S_shared



def plot_accuracy_and_change(results, removal_ratios, save_dir, title):
    """
    For each target label:
      - Left subplot: absolute accuracy vs % removed
      - Right subplot: delta accuracy vs % removed (relative to step 0)
    """
    os.makedirs(save_dir, exist_ok=True)

    for target_label, label_dict in results.items():
        x = removal_ratios[target_label]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # ---------- Left: absolute accuracy ----------
        ax = axes[0]
        for l, acc_list in label_dict.items():
            name = f"Label {l}"
            if l == target_label:
                name += " (target)"
            ax.plot(x, acc_list, marker="o", label=name)

        ax.set_xlabel("Parameters removed (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs Removal\n(target = {target_label})")
        ax.grid(True)
        ax.legend()

        # ---------- Right: delta accuracy ----------
        ax = axes[1]
        for l, acc_list in label_dict.items():
            delta = np.array(acc_list) - acc_list[0]
            name = f"Label {l}"
            if l == target_label:
                name += " (target)"
            ax.plot(x, delta, marker="o", label=name)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Parameters removed (%)")
        ax.set_ylabel("Δ Accuracy")
        ax.set_title("Accuracy Change (relative to 0%)")
        ax.grid(True)
        ax.legend()
        
        
        # ---------- Global title ----------
        fig.suptitle(
            f"Curvature-Based Parameter Removal (Target Label = {target_label}): " + title,
            fontsize=14,
        )
        
        # Reserve top space for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f"acc_and_delta_target_{target_label}.png"
        )
        plt.savefig(save_path)
        plt.close(fig)



def curv_single_label(args):
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=100, valid_num=5000)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=100)
    
    eps = [1,2,3,5]
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
        from tools.vgg9_custom_tanh import VGG9_CIFAR10
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "vgg9_10_ori_"
    elif model_pre_name == 'adv':
        model_name = "vgg9_10_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg9_10_wd_"
        
    model_name = model_name + activation + "_s2.pth"
    
    net_H = VGG9_CIFAR10(model_dims, None, device, prefix_dims)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    para_dims = cal_parameters(model_dims_small)

    total_para = sum(para_dims) # - sum(para_dims[0:2]) # - sum(para_dims[:11]) # - sum(para_dims[-3:])
    
    print(para_dims)
    print(total_para)
    
    # Load all labels → build per-label negative sets 
    all_label_data = {}
    all_label_neg = {}

    for label in selected_classes:
        save_path = os.path.join(res_path, f"curv_label{label}.pkl")
        
        if os.path.exists(save_path):
            continue
                
        else:
            print(f"[Processing] Label")
            
            process_batches_memory_efficient(
                data_path,
                model_full_n,
                metric,
                dataset,
                sample_size,
                prefix_dims,
                select_l = selected_classes,
                para_dims = para_dims,
                res_path=res_path
            )
            
            break
        
    for label in selected_classes:
        save_path = os.path.join(res_path, f"curv_label{label}.pkl")
    
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"[Loaded] Label {label}")

        all_label_data[label] = data
        all_label_neg[label] = extract_negative_params(data)
    
    margins = [0, 0.05, 0.2, 0.5, 0.7, 1.]
    
    log_path = os.path.join(res_path, "curvature_forgetting_log_target_all_nol8_rest>=-05.txt")

    with open(log_path, "a") as log_f:   # use "w" if you want to overwrite
        
        for m in margins:
            title=f"All, target only parameters (no layer 8), margin = {m}"
            results = {}
            removal_ratios = {}   # store % removed for x-axis
    
            log_f.write(title + ": \n")
            for target_label in selected_classes:
                results[target_label] = {}
                for l in selected_classes:
                    results[target_label][l] = []
                
                S_target, S_rest, S_shared = one_vs_rest_curvature(
                    all_label_neg,
                    target_label,
                    margin=m
                )
                
                # helper: get layer for a node index
                def get_layer(idx, prefix_dims):
                    return np.searchsorted(prefix_dims, idx, side="right") - 1

                neg_filtered_edges = [
                    p for p in S_target
                    if not (
                        p[0] == "edge"
                        and get_layer(p[1], prefix_dims) == 8
                    )
                ]
                
                out_edges = [
                    p for p in S_target
                    if (
                        p[0] == "edge"
                        and get_layer(p[1], prefix_dims) == 8
                    )
                ]
                
                header = (
                    "\n" + "=" * 60 + "\n"
                    f"[Target label: {target_label}]\n"
                    f"  Target-only params: {len(S_target)}, output layer parameters: {len(out_edges)}\n"
                    f"  Rest-only params:   {len(S_rest)}\n"
                    f"  Shared params:      {len(S_shared)}\n"
                    + "=" * 60
                )

                print(header)
                log_f.write(header + "\n")

                neg_edges_only = [
                    (p[0], p[1]) if p[0] == "weight" else p[:3]
                    for p in neg_filtered_edges
                ]


                neg_total = len(neg_edges_only)
                neg_remove_num = np.linspace(0, neg_total, num=10, dtype=int)
                
                removal_ratios[target_label] = [
                    100.0 * r / max(1, neg_total) for r in neg_remove_num
                ]

                sched_msg = f"[Removal schedule] {neg_remove_num.tolist()} / {neg_total} params"
                print(sched_msg)
                log_f.write(sched_msg + "\n")

                for step_idx, rem_f in enumerate(neg_remove_num):
                    step_header = (
                        "\n" + "-" * 50 + "\n"
                        f"[Target {target_label}] "
                        f"Removal step {step_idx + 1}/{len(neg_remove_num)} | "
                        f"Removing {rem_f}/{neg_total} params "
                        f"({100.0 * rem_f / max(1, neg_total):.1f}%)"
                    )

                    print(step_header)
                    log_f.write(step_header + "\n")

                    net_neg = copy.deepcopy(net_H)
                    net_neg.__build_remove_mask__(neg_edges_only, rem_f)

                    acc_clean_neg = test_clean(net_neg, test_loader)
                    acc_msg = f"  ▶ Overall clean accuracy: {acc_clean_neg:.4f}"
                    print(acc_msg)
                    log_f.write(acc_msg + "\n")

                    for l in selected_classes:
                        acc = test_clean(net_neg, sep_dataloader[l])
                        results[target_label][l].append(acc)
                        
                        tag = "⬅ target" if l == target_label else ""
                        line = f"    - Label {l}: acc = {acc:.4f} {tag}"

                        print(line)
                        log_f.write(line + "\n")

                log_f.write("\n")   # blank line between targets

            plot_accuracy_and_change(results, removal_ratios, save_dir=os.path.join(res_path, f"plots/all_nol8_rest>=-05/m_{m}/"), title=title)


