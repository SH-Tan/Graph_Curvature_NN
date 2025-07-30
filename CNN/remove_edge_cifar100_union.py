import torch
from torchvision.datasets.cifar import CIFAR100
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

data_train = CIFAR100('./data/cifar10', train=True, download=True, transform=transform_train)
data_test = CIFAR100('./data/cifar10', train=False, download=True, transform=transform_test)

model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 3, "stride": 1, "padding":1, "out_size": 16}},   # After conv1_2 + pool
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":1, "out_size": 8}},   # After conv2_2 + pool
    4: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":1, "out_size": 4}},   # After conv3_3 + pool
    5: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # After conv4_3 + pool

    6: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # conv5_1
    7: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":True, "out_size": 1}},   # conv5_2
    8: {"name": "cnn", "dim": {"channel": 512, "kernel": 1, "stride": 1, "padding":0, "pool":False, "out_size": 1}},   # conv5_3 + pool

    9: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    10: {"name": "fc", "dim": {"out_size": 512}},
    11: {"name": "fc", "dim": {"out_size": 100}}
}



model_dims_small = {
    1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 1}},   # conv5_2
    2: {"name": "cnn", "dim": {"channel": 512, "kernel": 1, "stride": 1, "padding":0, "pool":False, "out_size": 1}},   # conv5_3 + pool

    3: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    4: {"name": "fc", "dim": {"out_size": 512}},
    5: {"name": "fc", "dim": {"out_size": 100}}
}

selected_classes = list(range(100))

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
    neg_e = set()
    pos_e = set()
    mini_c = 0.
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])  # shape (N, 3)

        # Filter values with valid curvature (<= 1)
        valid = ricci_curv[ricci_curv[:, 2] <= 1]

        # Convert to int for indexing
        valid[:, 0:2] = valid[:, 0:2].astype(int)

        for i, j, curr in valid:
            i, j = int(i), int(j)
            if curr < 0:
                mini_c = min(mini_c, curr)
                neg_e.add((i, j, curr))
            elif curr > 0:
                pos_e.add((i, j, curr))

    return neg_e, pos_e, mini_c



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



# def plot_curve(
#     neg_acc_clean, pos_acc_clean,
#     neg_freq_ratios, pos_freq_ratios,
#     neg_freq_thresholds, pos_freq_thresholds,
#     label, save_path
# ):
#     # CMYK-like colors (safe RGB approximations)
#     neg_colors = ['#00A3E0', '#6CACE4']  # Cyan, Blue-gray
#     pos_colors = ['#EC008C', '#FF6F61']  # Magenta, Warm red

#     plt.figure(figsize=(10, 6))

#     # Negative Edge Plot
#     plt.plot(
#         neg_freq_ratios, neg_acc_clean,
#         label='Negative Edge Removal',
#         marker='o',
#         linestyle='--',
#         linewidth=2,
#         markersize=6,
#         color=neg_colors[0]
#     )

#     # Positive Edge Plot
#     plt.plot(
#         pos_freq_ratios, pos_acc_clean,
#         label='Positive Edge Removal',
#         marker='s',
#         linestyle='-',
#         linewidth=2,
#         markersize=6,
#         color=pos_colors[0]
#     )

#     # Annotate frequencies
#     for x, y, freq in zip(neg_freq_ratios, neg_acc_clean, neg_freq_thresholds):
#         plt.annotate(
#             f"{freq}",
#             (x, y),
#             textcoords="offset points",
#             xytext=(0, 10),
#             ha='center',
#             fontsize=9,
#             color=neg_colors[1]
#         )

#     for x, y, freq in zip(pos_freq_ratios, pos_acc_clean, pos_freq_thresholds):
#         plt.annotate(
#             f"{freq}",
#             (x, y),
#             textcoords="offset points",
#             xytext=(0, -15),
#             ha='center',
#             fontsize=9,
#             color=pos_colors[1]
#         )

#     # Axes and title
#     plt.xlabel("Edge Frequency Threshold (ratio × max frequency)", fontsize=22)
#     plt.ylabel("Clean Accuracy", fontsize=22)
#     plt.title(f"Clean Accuracy vs. Frequency Ratio (Label {label})", fontsize=22)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     plt.ylim(0.0, 1.0)
#     plt.gca().invert_xaxis()

#     # Legend and grid
#     plt.legend(fontsize=20, loc='best')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()

#     # Save
#     filename = os.path.join(save_path, f'{label}_freq_curve.png')
#     plt.savefig(filename, dpi=300)
#     plt.close()
    
    
def compute_removal_mapping(summary, total_edges):
    """
    Given summary = list of tuples (_, _, freq, _),
    compute mapping of frequency thresholds to removal counts,
    with ratio = freq / total_edges.

    Returns list of (count, freq, count_str, ratio_str)
    """
    freqs = sorted({freq for (_, _, freq, _) in summary}, reverse=True)
    mapping = []
    for freq_threshold in freqs:
        count = sum(1 for (_, _, freq, _) in summary if freq >= freq_threshold)
        ratio = freq_threshold / total_edges
        mapping.append((count, freq_threshold, str(count), f"{ratio:.2f}"))
    return mapping

# 2. Match removal counts to closest frequency thresholds
def match_frequencies(remove_counts, freq_map):
    """
    For each removal count, find the ratio string from freq_map where count >= remove_count.

    freq_map is a list of tuples (count, freq, ratio_str, ratio_float_str).

    Returns list of strings (ratio_float_str), or 'N/A' if no match found.
    """
    labels = []
    for rc in remove_counts:
        matched_ratio = next((ratio for (count, _, _, ratio) in freq_map if count >= rc), None)
        labels.append(matched_ratio if matched_ratio is not None else 'N/A')
    return labels

    
    
def plot_curve(
    neg_clean_acc, pos_clean_acc,
    neg_remove_num, pos_remove_num,
    label, res_path,
    neg_freq_labels=None, pos_freq_labels=None
):
    # Colors
    neg_color = '#00A3E0'
    pos_color = '#EC008C'

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(neg_remove_num, neg_clean_acc, label='Negative edges removed first',
             marker='o', linestyle='--', linewidth=3., markersize=13, color=neg_color)

    plt.plot(pos_remove_num, pos_clean_acc, label='Positive edges removed first',
             marker='x', linestyle='-', linewidth=3., markersize=13, color=pos_color)

    # Annotate frequencies BELOW points
    if neg_freq_labels:
        for x, y, r in zip(neg_remove_num, neg_clean_acc, neg_freq_labels):
            plt.annotate(r, (x, y), textcoords='offset points',
                         xytext=(-10, -25), ha='left', fontsize=18, color='#000000')

    if pos_freq_labels:
        for x, y, r in zip(pos_remove_num, pos_clean_acc, pos_freq_labels):
            plt.annotate(r, (x, y), textcoords='offset points',
                         xytext=(0, -15), ha='center', fontsize=18, color=pos_color)

    # Labels and title
    plt.xlabel('Number of Edges Removed', fontsize=33, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=33, fontweight='semibold')
    
    # plt.title('Accuracy vs. Edge Removal Count', fontsize=28, fontweight='semibold')
    plt.ylim(0.0, 1.0)

    # Set scientific notation on x-axis
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) 
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('semibold')

    # Ticks
    plt.xticks(fontsize=22, fontweight='semibold')
    plt.yticks(fontsize=22, fontweight='semibold')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=22, loc='best')  # create the legend
    for text in legend.get_texts():
        text.set_fontweight('semibold')  # or 'bold'

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{label}_curve_all.pdf'), dpi=300)
    plt.close()
    
    
def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(float)

    for edge_set in edge_sets:
        for i, j, c in edge_set:
            # Normalize undirected edge direction efficiently
            key = (min(i, j), max(i, j))
            freq[key] += 1
            curvature_sum[key] += c

    # Use list comprehension for speed and clarity
    results = [
        (i, j, count, curvature_sum[(i, j)] / count)
        for (i, j), count in freq.items()
    ]

    return results




def process_batches_memory_efficient(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims
):
    prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    suffix = ".pkl"

    def extract_batch_num(f):
        match = re.search(r'batch(\d+)', f)
        return int(match.group(1)) if match else -1

    all_files = sorted([
        f for f in os.listdir(data_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ], key=extract_batch_num)

    print(f"Found files: {all_files}")

    label_counts = {l: 0 for l in selected_classes}
    neg_edge_sets = []
    pos_edge_sets = []
    mini_c_list = []

    for f in all_files:
        file_path = os.path.join(data_path, f)
        print(f'file_path: {file_path}')
        
        with open(file_path, 'rb') as file:
            batch_data = pickle.load(file)

        for l in selected_classes:
            if label_counts[l] >= sample_size:
                continue

            new_data = batch_data.get(l, [])
            available = sample_size - label_counts[l]
            use_data = new_data[:available]

            for ricci in use_data:
                neg_e, pos_e, mini_c = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
                neg_edge_sets.append(neg_e)
                pos_edge_sets.append(pos_e)
                mini_c_list.append(mini_c)
                del ricci, neg_e, pos_e

            label_counts[l] += len(use_data)

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")

    neg_freq = count_edge_frequency(neg_edge_sets)
    pos_freq = count_edge_frequency(pos_edge_sets)

    return neg_freq, pos_freq, mini_c_list



def remove_edge_cifar100_union(args):
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
        from tools.vgg16_custom_relu import VGG16_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg16_custom_tanh import VGG16_CIFAR10
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "vgg16_100_ori_"
    elif model_pre_name == 'adv':
        model_name = "vgg16_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg16_wd_"
        
    model_name = model_name + activation + ".pth"
    
    net_H = VGG16_CIFAR10(model_dims, None, device, num_classes=100)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)
    
    edge_dims_small = cal_edges(model_dims_small)
    
    total_edge = sum(edge_dims_small) - edge_dims_small[0]
    print(total_edge)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = []
      
    test_cleanacc = test_clean(net_full, test_loader)
    
    print(f'Finish Test..')

    freq_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    
    print(f'Start removal..')
    
    save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}.pkl"
    save_path = os.path.join(res_path, save_name)
    
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data_loaded = pickle.load(f)
            # Access the contents
            neg_freq_edges_sorted = data_loaded.get('neg', [])
            pos_freq_edges_sorted = data_loaded.get('pos', [])
        print("File loaded successfully!")
    else:

        with open(res_path + "edge_cnn_" + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')

            neg_freq_dict, pos_freq_dict, mini_c_list = process_batches_memory_efficient(
                data_path,
                model_full_n,
                metric,
                dataset,
                sample_size,
                prefix_dims
            )
            
            mini_c_list = np.array(mini_c_list)
            
            neg_freq_edges_sorted = sorted(neg_freq_dict, key=lambda x: (-x[2], x[3]))
            pos_freq_edges_sorted = sorted(pos_freq_dict, key=lambda x: (-x[2], -x[3]))
            
            data_to_save = {
                'neg': neg_freq_edges_sorted,
                'pos': pos_freq_edges_sorted
            }
            
            save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}.pkl"
            save_path = os.path.join(res_path, save_name)

            with open(save_path, 'wb') as f:
                pickle.dump(data_to_save, f)  # use dict to avoid defaultdict issues

    print(f'It has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
    # ff.write(f'\nIt has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
    # ff.write(f'The average minimum c is {np.mean(mini_c_list)}, median = {np.median(mini_c_list)}\n\n')
    
    neg_acc_clean = []
    pos_acc_clean = []
    
    neg_edges_only = [(i, j) for (i, j, _, _) in neg_freq_edges_sorted]
    pos_edges_only = [(i, j) for (i, j, _, _) in pos_freq_edges_sorted]

    neg_total = len(neg_edges_only)
    pos_total = len(pos_edges_only)

    # Generate uniformly spaced points (including 0 and total) for each list
    neg_remove_num = list(np.linspace(0, neg_total, num=6, dtype=int))
    pos_remove_num = list(np.linspace(0, pos_total, num=2, dtype=int))
    # remove_num = list(np.linspace(0, total_edge, num=20, dtype=int))
    # neg_remove_num = []
    # pos_remove_num = []
    
    total = sample_size * len(selected_classes)
        
    # Build frequency mappings
    neg_freq_map = compute_removal_mapping(neg_freq_edges_sorted, total_edges=total)
    pos_freq_map = compute_removal_mapping(pos_freq_edges_sorted, total_edges=total)

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
        
        # neg_remove_num.append(rem_f)
        # if (rem_f > neg_total):
        #     break
        
        # ff.write(f'After remove {rem_f} negative edges, the acc is {acc_clean_neg}\n')

    for index, rem_f in enumerate(pos_remove_num):
        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(pos_edges_only, rem_f)
        # test acc
        acc_clean_pos = test_clean(net_pos, test_loader)
        pos_acc_clean.append(acc_clean_pos)
        # ff.write(f'After remove {rem_f} negative edges, the acc is {acc_clean_pos}\n')
        
        # pos_remove_num.append(rem_f)
        # if (rem_f > pos_total):
        #     break

    # Plot
    plot_curve(
        neg_clean_acc=neg_acc_clean,
        pos_clean_acc=pos_acc_clean,
        neg_remove_num=neg_remove_num,
        pos_remove_num=pos_remove_num,
        label=sample_size,
        res_path=res_path,
        neg_freq_labels=neg_freq_labels,
        pos_freq_labels=pos_freq_labels
    )
        # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)


                
                
            
                       
