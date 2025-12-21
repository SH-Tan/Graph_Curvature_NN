import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

import pickle
import time
import pandas as pd

from .e2w_utils_new import *

import sys
sys.path.append("..")

import tools.utils as utils


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

data_train = MNIST('./data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))



selected_classes = [0,1,2,3,4,5,6,7,8,9]


nodes_num = 2118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12, "padding": 0}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4, "padding": 0}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}



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





# def get_top_c(curvature, b, prefix_dims):
#     neg_e = set()
#     pos_e = set()
    
#     for batch in range(b):
#         ricci_curv = np.array(curvature[batch])  # shape (N, 3)

#         # Filter values with valid curvature (<= 1)
#         valid = ricci_curv[ricci_curv[:, 2] <= 1]

#         # Convert to int for indexing
#         valid[:, 0:2] = valid[:, 0:2].astype(int)

#         for i, j, curr in valid:
#             i, j = int(i), int(j)
#             # i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
#             if curr < 0:
#                 neg_e.add((i, j, curr))
#             elif curr >= 0:
#                 pos_e.add((i, j, curr))

#     return neg_e, pos_e



def get_top_c(curvature, b, prefix_dims):
    neg_e = defaultdict(list)
    pos_e = defaultdict(list)
    cnn_e = defaultdict(list)

    # Precompute a fast index-to-layer map
    def find_layer(index):
        return np.searchsorted(prefix_dims, index, side='right') - 1

    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        
        # Filter out large curvature values
        valid = ricci_curv[ricci_curv[:, 2] < 20.0]
        valid[:, :2] = valid[:, :2].astype(int)
        
        for i, j, curr in valid:
            # curr = min(curr, 1.0)  # clip curvature
            i = int(i)
            j = int(j)

            i_layer = find_layer(i)
            j_layer = find_layer(j)
            
            if ((i_layer > 0) and (i_layer < 4)):
                if (abs(curr - 1.00000) < 1e-6):
                    curr = 2.0
            
            if i_layer in [0,1]:
                cnn_e[i_layer].append((i,j,curr))

            # Only keep edges between adjacent layers (excluding input and first hidden)
            elif j_layer == i_layer + 1:
                if curr < 0:
                    neg_e[i_layer].append((i, j, curr))
                elif curr >= 0:
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


    
def compute_removal_mapping(summary, total_edges=1, reversed=True):
    """
    Given summary = list of tuples (_, _, freq, _, freq_ratio),
    build mapping: (count, freq, count_str, ratio_str)
    """

    freqs = sorted({round(item[4], 1) for item in summary}, reverse=reversed)

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
            if (i % 2 == 0):
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





def remove_edge_cnn_union_w_combined(args):
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)

    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    data_path = args.mnist_data_path
    activation = args.activation
    
    if activation.lower() == "relu":
        from tools.LeNet5_custom_small_w import LeNet_custom_v2
    elif activation.lower() == "tanh":
        from tools.LeNet5_custom_small_tanh_w import LeNet_custom_v2
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    edge_dims = cal_edges(model_dims)
    para_dims = cal_parameters(model_dims)
    
    total_edge = sum(edge_dims)
    total_para = sum(para_dims) #- para_dims[0]
    
    print(edge_dims)
    print(para_dims)
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "cnn_ori_"
    elif model_pre_name == 'adv':
        model_name = "cnn_adv_"
    elif model_pre_name == 'wd':
        model_name = "cnn_wd_"
        
    model_name = model_name + activation + ".pth"
    
    net_H = LeNet_custom_v2(model_dims, None, device, prefix_dims)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = []
  
    test_cleanacc = test_clean(net_full, test_loader)
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)   

    save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}_para_combined_min2.pkl"
    save_path = os.path.join(res_path, save_name)
    
    print(save_path)
    
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data_loaded = pickle.load(f)
            # Access the contents
            neg_freq_edges_sorted = data_loaded.get('neg', [])
            pos_freq_edges_sorted = data_loaded.get('pos', [])
        print("File loaded successfully!")
    else:
    
        with open(res_path + "edge_cnn_" + ".txt", "a+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            # print(f'Current label {l}: \n')
            # ff.write(f'Current label {l}: \n')
            
            correct_suffix_res = dataset + "_res_correct.pkl" 
            res_name = model_full_n + metric + '_' + correct_suffix_res

            with open(data_path + res_name, 'rb') as file:
                res_dict = pickle.load(file) 

            neg_weight_sets = []
            pos_weight_sets = []
            edge_sets = []
            
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

            
            for l in selected_classes:
                idx = 0
                for (ricci, batch, dim, node) in res_dict[l]:
                    neg_e, pos_e, cnn_e = get_top_c(ricci, 1, prefix_dims)
                    
                    for layer, edges in cnn_e.items():
                        layer_info = model_dims[layer + 2]
                        if layer_info["name"] == "cnn":
                            pos_curv_weights, neg_curv_weights = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                            # all_weight_sets.append([(w, c, f, pos_f, neg_f) for w, (c, f, pos_f, neg_f) in weight_curv.items()])
                            neg_weight_sets.append([(w, c, f, z) for w, (c, f,z) in neg_curv_weights.items()])
                            pos_weight_sets.append([(w, c, f, z) for w, (c, f,z) in pos_curv_weights.items()])

                    # === Aggregate CNN edges → per-weight curvatures ===
                    for layer, edges in neg_e.items():
                        layer_info = model_dims[layer + 2]
                        if layer_info["name"] == "cnn":
                            weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                            neg_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                        else:
                            # FC layer — keep per-edge
                            edge_sets.extend(edges)

                    for layer, edges in pos_e.items():
                        layer_info = model_dims[layer + 2]
                        if layer_info["name"] == "cnn":
                            weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                            pos_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                        else:
                            edge_sets.extend(edges)

                    idx += 1
                    if idx >= sample_size:
                        break
                    

            freq_fc = count_edge_frequency(edge_sets)
            
            all_weight_sets = pos_weight_sets + neg_weight_sets

            freq_cnn = count_weight_frequency(all_weight_sets, model_dims, para_dims)

            p_all = (
                [("edge", i, j, f, c) for (i, j, f, c) in freq_fc] +
                [("weight", w, None, f, c) for (w, f, c, p) in freq_cnn]
            )

            neg_freq_edges_sorted = sorted(p_all, key=lambda x: (x[4]))  # by frequency desc, curvature asc
            pos_freq_edges_sorted = sorted(p_all, key=lambda x: (-x[4])) # by frequency desc, curvature desc

            data_to_save = {
                'neg': neg_freq_edges_sorted,
                'pos': pos_freq_edges_sorted
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)  # use dict to avoid defaultdict issues

            ff.write(f'\nIt has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
            # ff.write(f'The average minimum c is {np.mean(mini_c_list)}, median = {np.median(mini_c_list)}\n\n')
    
    print(f'It has {len(neg_freq_edges_sorted)} negative curvature parameters, {len(pos_freq_edges_sorted)} positive curvature parameters .. \n')
        
    neg_acc_clean = []
    pos_acc_clean = []
    
    
    total = sample_size * len(selected_classes)
    
    # Compute which layer each edge (i) belongs to
    layers_i = [
        np.searchsorted(prefix_dims, i, side='right') - 1
        if item == "edge" else None
        for (item, i, j, f, c) in pos_freq_edges_sorted
    ]

    # Filter: keep all weights, and only edges not in layer 9
    pos_filtered_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(pos_freq_edges_sorted, layers_i)
        if c >= 0
    ]
    
    # Compute which layer each edge (i) belongs to
    layers_i = [
        np.searchsorted(prefix_dims, i, side='right') - 1
        if item == "edge" else None
        for (item, i, j, f, c) in neg_freq_edges_sorted
    ]

    # Filter: keep all weights, and only edges not in layer 9
    neg_filtered_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(neg_freq_edges_sorted, layers_i)
        if c < 0
    ]
    
    
    neg_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3] for item in neg_freq_edges_sorted
    ]
    pos_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3] for item in pos_freq_edges_sorted
    ]
  
    neg_total = len(neg_edges_only)
    pos_total = len(pos_edges_only)
    
    print(f'New pos {pos_total}, neg {neg_total}')
    
    # Filter: keep all weights, and only edges not in layer 9
    neg_edges = [
        (item, i, j, f, c)
        for (item, i, j, f, c), layer in zip(neg_freq_edges_sorted, layers_i)
        if c < 0
    ]

    # neg parts
    neg_len = len(neg_edges)

    # First segment: 3 points from 0 to len(neg_edges)
    part1 = np.linspace(0, neg_len, num=8, dtype=int)

    # Second segment: 5 points from len(neg_edges) to neg_total
    part2 = np.linspace(neg_len, neg_total, num=4, dtype=int)

    # Combine, but avoid duplicate at the boundary
    neg_remove_num = np.unique(np.concatenate((part1, part2))).tolist()   
    
    # # define split points
    # split1 = int(0.5 * pos_total)
    # split2 = int(0.9 * pos_total)

    # # stage 1: first 40% (coarse)
    # part1 = np.linspace(0, split1, num=6, dtype=int)

    # # stage 2: next 40% (medium)
    # part2 = np.linspace(split1, split2, num=10, dtype=int)

    # # stage 3: last 20% (fine)
    # part3 = np.linspace(split2, pos_total, num=3, dtype=int)

    # # combine, removing duplicates at boundaries
    # pos_remove_num = np.unique(np.concatenate((part1, part2, part3))).tolist()   
    
    pos_remove_num = list(np.linspace(0, pos_total, num=30, dtype=int))
    
    remove_num = list(np.linspace(0, total_para, num=10, dtype=int))
    
    print(neg_remove_num)
    print(pos_remove_num)

    
    # Build frequency mappings
    neg_freq_map = compute_removal_mapping(neg_freq_edges_sorted, reversed=False, total_edges=total)
    pos_freq_map = compute_removal_mapping(pos_freq_edges_sorted, total_edges=total)

    neg_freq_labels = match_frequencies(neg_remove_num, neg_freq_map)
    pos_freq_labels = match_frequencies(pos_remove_num, pos_freq_map)
    
    # # Step 2: Choose thresholds — you can just use them all or downsample if too many
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

    for index, rem_f in enumerate(pos_remove_num):
        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(pos_edges_only, rem_f)
        # test acc
        acc_clean_pos = test_clean(net_pos, test_loader)
        pos_acc_clean.append(acc_clean_pos)
        
        
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
    
        

    # Plot
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
                
