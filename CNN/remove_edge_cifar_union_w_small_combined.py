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

# model_dims = {
#     1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

#     2: {"name": "cnn", "dim": {"channel": 64, "kernel": 4, "stride": 1, "padding":0, "out_size": 29}}, 
#     3: {"name": "cnn", "dim": {"channel": 64, "kernel": 4, "stride": 1, "padding":0, "out_size": 26}},   # After conv1_2 
    
#     4: {"name": "cnn", "dim": {"channel": 128, "kernel": 4, "stride": 1, "padding":0, "out_size": 23}},  
#     5: {"name": "cnn", "dim": {"channel": 128, "kernel": 4, "stride": 1, "padding":0, "out_size": 20}},   # After conv2_2 
    
#     6: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 18}},
#     7: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 16}},
#     8: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 14}},   # After conv3_3
    
#     9: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 12}},   # After conv4_1
#     10: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 10}},   # After conv4_2
#     11: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 8}},   # After conv4_3

#     12: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 6}},   # conv5_1
#     13: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 4}},   # conv5_2 
#     14: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 2}},   # conv5_3 

#     15: {"name": "fc", "dim": {"out_size": 512}},  # Flatten(512×2×2) → 1024
#     16: {"name": "fc", "dim": {"out_size": 512}},
#     17: {"name": "fc", "dim": {"out_size": 10}}
# }



# model_dims = {
#     1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

#     2: {"name": "cnn", "dim": {"channel": 64, "kernel": 4, "stride": 1, "padding":0, "out_size": 29}}, 
#     3: {"name": "cnn", "dim": {"channel": 64, "kernel": 4, "stride": 1, "padding":0, "out_size": 26}},   # After conv1_2 
    
#     4: {"name": "cnn", "dim": {"channel": 128, "kernel": 4, "stride": 1, "padding":0, "out_size": 23}},  
#     5: {"name": "cnn", "dim": {"channel": 128, "kernel": 4, "stride": 1, "padding":0, "out_size": 20}},   # After conv2_2 
    
#     6: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 18}},
#     7: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 16}},
#     8: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 14}},   # After conv3_3
    
#     9: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 12}},   # After conv4_1
#     10: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 10}},   # After conv4_2
#     11: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 8}},   # After conv4_3

#     12: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 6}},   # conv5_1
#     13: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 4}},   # conv5_2 
#     14: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 2}},   # conv5_3 

#     15: {"name": "fc", "dim": {"out_size": 256}},  # Flatten(512×2×2) → 1024
#     16: {"name": "fc", "dim": {"out_size": 512}},
#     17: {"name": "fc", "dim": {"out_size": 10}}
# }

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
all_classes = [0,1,2,3,4,5,6,7,8,9]



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



def test(n, loader, eps, alpha, iters, device):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
 
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, device, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(adv_pred))
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(adv_pred))

            succ_pair[l].append(images[succ_l].cpu())
            robust_pair[l].append(images[robust_l].cpu())

    return succ_pair, robust_pair




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

    
def compute_removal_mapping(summary, total_edges=1, reversed=True):
    """
    Given summary = list of tuples (_, _, freq, _, freq_ratio),
    build mapping: (count, freq, count_str, ratio_str)
    """

    freqs = sorted({round(item[4], 2) for item in summary}, reverse=reversed)

    mapping = []
    for freq_threshold in freqs:
        if reversed:
            count = sum(1 for item in summary if item[4] >= freq_threshold)
        else:
            count = sum(1 for item in summary if item[4] <= freq_threshold)

        mapping.append((count, freq_threshold, str(count), f"{freq_threshold:.2f}"))

    # Ensure sorted by count ascending
    mapping = sorted(mapping, key=lambda x: x[0])

    return mapping

def match_frequencies(remove_counts, freq_map):
    """
    Always return a label for each removal count by:
      - Using the smallest ratio if rc < min count
      - Using the largest ratio if rc > max count
      - Otherwise the first count >= rc
    """

    # Extract the count and ratio_float_str columns
    counts = [c for (c, _, _, _) in freq_map]
    ratios = [ratio for (_, _, _, ratio) in freq_map]

    labels = []
    for rc in remove_counts:
        if rc <= counts[0]:
            labels.append(ratios[0])
        elif rc >= counts[-1]:
            labels.append(ratios[-1])
        else:
            # normal case: find the first count >= rc
            for c, _, _, ratio in freq_map:
                if c >= rc:
                    labels.append(ratio)
                    break

    return labels

    
    
def plot_curve(
    neg_clean_acc, pos_clean_acc,
    neg_remove_num, pos_remove_num,
    label, res_path,
    neg_freq_labels=None, pos_freq_labels=None, x_axis=None
):
    # Colors
    neg_color = '#00A3E0'
    pos_color = '#EC008C'
    text_color = "#041E91FF"

    plt.figure(figsize=(11, 7))

    # Plot lines
    plt.plot(neg_remove_num, neg_clean_acc, label='Negative parameters removed first',
             marker='o', linestyle='--', linewidth=3.5, markersize=13, color=neg_color)

    plt.plot(pos_remove_num, pos_clean_acc, label='Positive parameters removed first',
             marker='x', linestyle='-', linewidth=3.5, markersize=13, color=pos_color)

    # Annotate frequencies BELOW points
    if neg_freq_labels:
        for i, (x, y, r) in enumerate(zip(neg_remove_num, neg_clean_acc, neg_freq_labels)):
            if (i % 4 == 0):
                plt.annotate(r, (x, y), textcoords='offset points',
                            xytext=(-10, -25), ha='left', fontsize=24, color='#000000')
    flag = 0
    
    if pos_freq_labels:
        for i, (x, y, r) in enumerate(zip(pos_remove_num, pos_clean_acc, pos_freq_labels)):
            if (i % 5 == 0):
                plt.annotate(r, (x, y), textcoords='offset points',
                    xytext=(0, -15), ha='center', fontsize=27, color=text_color)
                
            # elif (r < 0) and (~flag):
            #     plt.annotate(r, (x, y), textcoords='offset points',
            #         xytext=(0, 15), ha='center', fontsize=28, color=pos_color)
            #     flag = 1

    # Labels and title
    plt.xlabel('Number of Parameters Removed', fontsize=30, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=31, fontweight='semibold')
    
    # plt.title('Accuracy vs. Edge Removal Count', fontsize=28, fontweight='semibold')
    plt.ylim(0.0, 1.0)

    # Set scientific notation on x-axis
    ax = plt.gca()
    
    # Override the x-axis ticks/labels if `x_axis` is given
    if x_axis is not None:
        # Compute exponent (e.g., 1e+3, 1e+4) based on the max value
        exponent = int(np.floor(np.log10(max(x_axis))))
        scale = 10 ** exponent

        # Scale values and format tick labels as mantissas only
        scaled_ticks = [x / scale for x in x_axis]
        mantissa_labels = [f"{v:.1f}" for v in scaled_ticks]

        # Set the ticks and the scaled mantissa labels
        plt.xticks(ticks=x_axis, labels=mantissa_labels, fontsize=26, fontweight='semibold')

        # Add scientific scale as offset text (e.g., ×1e4) to the end of the x-axis
        ax.annotate(
            f"×1e{exponent}",
            xy=(1.0, 0.0), xycoords='axes fraction',  # Right end of x-axis
            xytext=(10, -35), textcoords='offset points',  # Just below and slightly to the left
            ha='right', va='top',
            fontsize=18, fontweight='semibold'
        )
    else:
        plt.xticks(fontsize=22, fontweight='semibold')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(24)
        ax.xaxis.get_offset_text().set_fontweight('semibold')

    # Ticks
    plt.xticks(fontsize=24, fontweight='semibold')
    plt.yticks(fontsize=26, fontweight='semibold')
    
    # === Axis borders ===
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_color('black')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=24, loc=0)  # create the legend
    for text in legend.get_texts():
        text.set_fontweight('semibold')  # or 'bold'

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{label}_curve_all_para_combined_min2.png'), dpi=400, bbox_inches="tight")
    plt.close()
    
    
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




def process_batches_memory_efficient(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
    total_example,
    para_dims
):
    # prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    prefix = f"{model_full_n}_{metric}_{dataset}_label"
    suffix = ".pkl"

    # extract label + id from filename
    def extract_label_id(f):
        match = re.search(r'label(\d+)_id(\d+)', f)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    all_files = [
        f for f in os.listdir(data_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    # Sort by label then id
    all_files = sorted(all_files, key=lambda f: extract_label_id(f)[1])

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

    label_counts = {l: 0 for l in selected_classes}
    neg_weight_sets = []
    pos_weight_sets = []
    edge_sets = []

    for f in all_files:
        label, sample_id = extract_label_id(f)
        if label not in selected_classes:
            continue
        if label_counts[label] >= sample_size:
            continue
        
        file_path = os.path.join(data_path, f)
        print(f'file_path: {file_path}')
        
        with open(file_path, 'rb') as file:
            batch_data = pickle.load(file)

        new_data = batch_data
        # available = sample_size - label_counts[l]
        use_data = [new_data]

        for ricci in use_data:
            neg_e, pos_e, cnn_e = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
            
            for layer, edges in cnn_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    pos_curv_weights, neg_curv_weights = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    # all_weight_sets.append([(w, c, f, pos_f, neg_f) for w, (c, f, pos_f, neg_f) in weight_curv.items()])
                    neg_weight_sets.append([(w, c, f, para_dims[w[0]]) for w, (c, f, z) in neg_curv_weights.items()])
                    pos_weight_sets.append([(w, c, f, para_dims[w[0]]) for w, (c, f, z) in pos_curv_weights.items()])

         
            # === Aggregate CNN edges → per-weight curvatures ===
            for layer, edge in neg_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    print("ERROR!!")
                    continue
                    weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    neg_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                else:
                    # FC layer — keep per-edge
                    edge_sets.extend(edge)

            for layer, edge in pos_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    print("ERROR!!")
                    continue
                    weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    pos_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                else:
                    edge_sets.extend(edge)
            del ricci, neg_e, pos_e

        label_counts[label] += len(use_data)

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")
    

    freq_fc = count_edge_frequency(edge_sets)
    
    all_weight_sets = pos_weight_sets + neg_weight_sets

    freq_cnn = count_weight_frequency(all_weight_sets, model_dims, para_dims)
    

    p_all = (
        [("edge", i, j, f, c) for (i, j, f, c) in freq_fc] +
        [("weight", w, None, f, c) for (w, f, c, p) in freq_cnn]
    )

    neg_freq_edges_sorted = sorted(p_all, key=lambda x: (x[4]))  # by frequency desc, curvature asc
    pos_freq_edges_sorted = sorted(p_all, key=lambda x: (-x[4])) # by frequency desc, curvature desc

    return neg_freq_edges_sorted, pos_freq_edges_sorted



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



def remove_edge_cifar_union_w_small_combined(args):
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

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=200, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=128)
    
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

    net_full = copy.deepcopy(net_H)
    
    save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}_combined_min2.pkl"
    save_path = os.path.join(res_path, save_name)
    
    print(model_name)
    edge_dims_small = cal_edges(model_dims_small) 
    para_dims = cal_parameters(model_dims_small)

    total_para = sum(para_dims) # - sum(para_dims[0:2]) # - sum(para_dims[:11]) # - sum(para_dims[-3:])
    
    print(para_dims)
    print(total_para)
    
    total = sample_size * len(selected_classes)
    
    test_cleanacc = test_clean(net_full, test_loader)
    
    print(f'Finish Test..')

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data_loaded = pickle.load(f)
            # Access the contents
            neg_freq_dict = data_loaded.get('neg', [])
            pos_freq_dict = data_loaded.get('pos', [])
        print("File loaded successfully!")
    else:

        with open(res_path + "edge_cnn_" + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            
            neg_freq_dict, pos_freq_dict = process_batches_memory_efficient(
                data_path,
                model_full_n,
                metric,
                dataset,
                sample_size,
                prefix_dims,
                total_example = total,
                para_dims = para_dims
            )

            data_to_save = {
                'neg': neg_freq_dict,
                'pos': pos_freq_dict
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)  # use dict to avoid defaultdict issues
    
    print(f'It has {len(neg_freq_dict)} negative curvature edges, {len(pos_freq_dict)} positive curvature egdes.. \n')
    
    neg_acc_clean = []
    pos_acc_clean = []

    # Compute which layer each edge (i) belongs to
    layers_i = [
        np.searchsorted(prefix_dims, i, side='right') - 1
        if item == "edge" else None
        for (item, i, j, f, c) in pos_freq_dict
    ]

    # Filter: keep all weights, and only edges not in layer 9
    pos_filtered_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(pos_freq_dict, layers_i)
        # if f == total
        if (not ((item == "weight") and (i[0] in [1])))
    ]
    
    # Compute which layer each edge (i) belongs to
    layers_i = [
        np.searchsorted(prefix_dims, i, side='right') - 1
        if item == "edge" else None
        for (item, i, j, f, c) in neg_freq_dict
    ]

    # Filter: keep all weights, and only edges not in layer 9
    neg_filtered_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(neg_freq_dict, layers_i)
        # if f == total
        if (not ((item == "weight") and (i[0] in [1])))
    ]
    
    # neg_freq_dict = sorted(neg_freq_dict, key=lambda x: (-(x[3]+x[5]), x[4]))  # by frequency desc, curvature asc
    # pos_freq_dict = sorted(pos_freq_dict, key=lambda x: (-(x[3]+x[5]), -x[4])) # by frequency desc, curvature desc


    neg_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3] for item in neg_freq_dict
    ]
    pos_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3] for item in pos_freq_dict
    ]

    neg_total = len(neg_edges_only)
    pos_total = len(pos_edges_only)
    
    # Filter: keep all weights, and only edges not in layer 9
    neg_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(neg_freq_dict, layers_i)
        if c < 0
    ]
    
    
    neg_total = len(neg_edges_only)
    pos_total = len(pos_edges_only)
    
    print(f'Combined: It has {neg_total} negative curvature edges, {pos_total} positive curvature egdes .. \n')

    
    # neg parts
    neg_len = len(neg_edges)

    # First segment: 3 points from 0 to len(neg_edges)
    part1 = np.linspace(0, neg_len, num=12, dtype=int)

    # Second segment: 5 points from len(neg_edges) to neg_total
    part2 = np.linspace(neg_len, neg_total, num=6, dtype=int)

    # Combine, but avoid duplicate at the boundary
    neg_remove_num = list(part1[:-1]) + list(part2)
    
    # # define split points
    # split1 = int(0.3 * pos_total)
    # split2 = int(0.8 * pos_total)

    # # stage 1: first 40% (coarse)
    # part1 = np.linspace(0, split1, num=4, dtype=int)

    # # stage 2: next 40% (medium)
    # part2 = np.linspace(split1, split2, num=8, dtype=int)

    # # stage 3: last 20% (fine)
    # part3 = np.linspace(split2, pos_total, num=3, dtype=int)

    # # combine, removing duplicates at boundaries
    # pos_remove_num = np.unique(np.concatenate((part1, part2, part3))).tolist()
    
    pos_remove_num = list(np.linspace(0, pos_total, num=30, dtype=int))
        
    remove_num = list(np.linspace(0, total_para, num=10, dtype=int))

    # Build frequency mappings
    neg_freq_map = compute_removal_mapping(neg_freq_dict, reversed=False, total_edges=total)
    pos_freq_map = compute_removal_mapping(pos_freq_dict, total_edges=total)

    neg_freq_labels = match_frequencies(neg_remove_num, neg_freq_map)
    pos_freq_labels = match_frequencies(pos_remove_num, pos_freq_map)

    # Step 2: Choose thresholds — you can just use them all or downsample if too many
    # neg_max_freq = max(freq for (_, _, freq, _) in neg_freq_edges_sorted)
    # neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

    # pos_max_freq = max(freq for (_, _, freq, _) in pos_freq_edges_sorted)
    # pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]
    
    # # Step 3: For each threshold, count how many edges would be removed
    # neg_remove_num = [sum(1 for (_, _, freq, _) in neg_freq_edges_sorted if freq >= t) for t in neg_freq_thresholds]
    # pos_remove_num = [sum(1 for (_, _, freq, _) in pos_freq_edges_sorted if freq >= t) for t in pos_freq_thresholds]
        
    # start remove
    for index, rem_f in enumerate(neg_remove_num):
        # ff.write(f'Remove edge number {rem_f}: \n')

        # remove second layer negative curvature edges
        net_neg = copy.deepcopy(net_H)
        net_neg.__build_remove_mask__(neg_edges_only, rem_f)
        # test acc
        acc_clean_neg = test_clean(net_neg, test_loader)
        neg_acc_clean.append(acc_clean_neg)
        
        # ff.write(f'After remove {rem_f} negative edges, the acc is {acc_clean_neg}\n')

    for index, rem_f in enumerate(pos_remove_num):
        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(pos_edges_only, rem_f)
        # test acc
        acc_clean_pos = test_clean(net_pos, test_loader)
        pos_acc_clean.append(acc_clean_pos)
        # ff.write(f'After remove {rem_f} negative edges, the acc is {acc_clean_pos}\n')
        
    # pack into a dictionary
    data = {
        "neg_clean_acc": neg_acc_clean,
        "pos_clean_acc": pos_acc_clean,
        "neg_remove_num": neg_remove_num,
        "pos_remove_num": pos_remove_num,
    }

    # save to pickle file
    with open(res_path+"results.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Saved variables to results.pkl")
    
    # # Plot
    plot_curve(
        neg_clean_acc=neg_acc_clean,
        pos_clean_acc=pos_acc_clean,
        neg_remove_num=neg_remove_num,
        pos_remove_num=pos_remove_num,
        label=sample_size,
        res_path=res_path,
        neg_freq_labels=neg_freq_labels,
        pos_freq_labels=pos_freq_labels,
        x_axis = remove_num
    )
        # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)


                
                
            
                       
