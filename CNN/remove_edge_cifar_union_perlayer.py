import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import time

import re
import gc
import pickle
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils
from tools.graph_curvature import graph_curvature_main_torch
# from tools.vgg16_custom_relu import VGG16_CIFAR10

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

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

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 3, "stride": 1, "padding":1, "out_size": 16}},   # After conv1_2 + pool
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":1, "out_size": 8}},   # After conv2_2 + pool
    4: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":1, "out_size": 4}},   # After conv3_3 + pool
    5: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # After conv4_3 + pool

    6: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # conv5_1
    7: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":True, "out_size": 1}},   # conv5_2
    8: {"name": "cnn", "dim": {"channel": 512, "kernel": 1, "stride": 1, "padding":0, "pool":False, "out_size": 1}},   # conv5_3 + pool

    9: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    10: {"name": "fc", "dim": {"out_size": 512}},
    11: {"name": "fc", "dim": {"out_size": 10}}
}



model_dims_small = {
    1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 1}},   # conv5_2
    2: {"name": "cnn", "dim": {"channel": 512, "kernel": 1, "stride": 1, "padding":0, "pool":False, "out_size": 1}},   # conv5_3 + pool

    3: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    4: {"name": "fc", "dim": {"out_size": 512}},
    5: {"name": "fc", "dim": {"out_size": 10}}
}



selected_classes = [0,1,2,3,4,5,6,7,8,9]


def process_batches_memory_efficient(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
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

    # Tracking how many samples per label we’ve used
    label_counts = {l: 0 for l in selected_classes}

    # Accumulate edge frequency and curvature directly
    neg_edge_acc = defaultdict(lambda: defaultdict(list))
    pos_edge_acc = defaultdict(lambda: defaultdict(list))

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
            
            # print(len(use_data))

            for ricci in use_data:
                t1 = time.time()
                neg_e, pos_e = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
                t2 = time.time()

                # Accumulate stats layer-wise
                for layer, edges in neg_e.items():
                    for (i, j, curv) in edges:
                        neg_edge_acc[layer][(i, j)].append(curv)
                for layer, edges in pos_e.items():
                    for (i, j, curv) in edges:
                        pos_edge_acc[layer][(i, j)].append(curv)
                        
                t3 = time.time()

                del ricci, neg_e, pos_e
                
                # print(f'1: {t3-t2} - {t2-t1} - {t3-t1}')

            label_counts[l] += len(use_data)

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")

    # Convert accumulated stats into sorted output
    def reduce_and_sort(edge_acc, sort_desc=False):
        # Convert accumulated stats to expected format
        edge_sets = []
        for layer in edge_acc:
            edge_dict = {layer: [(i, j, curv) for (i, j), curvs in edge_acc[layer].items() for curv in curvs]}
            edge_sets.append(edge_dict)
        return count_edge_frequency_and_sort(edge_sets, sort_curvature_desc=sort_desc)

    neg_freq_dict = reduce_and_sort(neg_edge_acc, sort_desc=False)
    pos_freq_dict = reduce_and_sort(pos_edge_acc, sort_desc=True)
    
    return neg_freq_dict, pos_freq_dict


    

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
    neg_e = defaultdict(list)  # i_layer -> list of (i, j, curvature)
    pos_e = defaultdict(list)
    seen_edges = set()

    # Optional: cache for searchsorted results
    layer_cache = {}

    def get_layer(node_idx):
        if node_idx not in layer_cache:
            layer_cache[node_idx] = np.searchsorted(prefix_dims, node_idx, side='right') - 1
        return layer_cache[node_idx]

    # Step 1: Collect valid curvature edges
    for batch in range(b):
        for i, j, curr in curvature[batch]:
            if curr > 1:
                continue

            i, j = int(i), int(j)
            seen_edges.add((i, j))  # optional: use (min(i, j), max(i, j)) for undirected

            i_layer = get_layer(i)
            j_layer = get_layer(j)

            if i_layer >= 7 and j_layer == i_layer + 1:
                if curr < 0:
                    neg_e[i_layer].append((i, j, curr))
                elif curr > 0:
                    pos_e[i_layer].append((i, j, curr))

    return neg_e, pos_e



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
    neg_color = "#00E016"
    pos_color = "#EC6600"

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
    plt.savefig(os.path.join(res_path, f'{label}_curve_perlayer.pdf'), dpi=300)
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




from collections import Counter
def count_edge_frequency_and_sort(edge_sets, sort_curvature_desc=False):
    """
    edge_sets: list of dicts. Each dict maps layer -> list of (i, j, curvature)

    Returns:
        sorted_edges_by_layer: dict mapping layer -> list of (i, j, freq, avg_curvature), sorted
    """
    freq_dict = defaultdict(lambda: defaultdict(int))      # layer -> (i, j) -> count
    curv_dict = defaultdict(lambda: defaultdict(list))     # layer -> (i, j) -> list of curvatures

    for edge_dict in edge_sets:
        for layer, edges in edge_dict.items():
            for (i, j, curv) in edges:
                key = (i, j)
                freq_dict[layer][key] += 1
                curv_dict[layer][key].append(curv)

    sorted_edges_by_layer = dict()

    for layer in freq_dict:
        edge_stats = []
        for (i, j), count in freq_dict[layer].items():
            curv_list = curv_dict[layer][(i, j)]
            avg_curv = sum(curv_list) / len(curv_list)
            edge_stats.append((i, j, count, avg_curv))

        # Sort by freq descending, then avg curvature ascending or descending
        if sort_curvature_desc:
            edge_stats.sort(key=lambda x: (-x[2], -x[3]))  # freq ↓, curvature ↓
        else:
            edge_stats.sort(key=lambda x: (-x[2], x[3]))   # freq ↓, curvature ↑

        sorted_edges_by_layer[layer] = edge_stats

    return sorted_edges_by_layer





def remove_edge_cifar_union_perlayer(args):
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

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=128, valid_num=50)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03]
    dims = cal_dims(model_dims)
    
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
        model_name = "vgg16_ori_"
    elif model_pre_name == 'adv':
        model_name = "vgg16_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg16_wd_"
        
    model_name = model_name + activation + ".pth"
        
    net_H = VGG16_CIFAR10(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = []
  
    test_cleanacc = test_clean(net_full, test_loader)
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)
    print(f'Finish test..')

    freq_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0]
    
    # print(f'Finish read pickle file...')
    
    with open(res_path + "edge_cnn_" + ".txt", "a+") as ff:
        ff.write(f'For model {model_name}: \n')
        ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
        # print(f'Current label {l}: \n')
        # ff.write(f'Current label {l}: \n')

        neg_freq_dict, pos_freq_dict = process_batches_memory_efficient(
            data_path,
            model_full_n,
            metric,
            dataset,
            sample_size,
            prefix_dims
        )
        print(neg_freq_dict.keys())

        for layer in sorted(neg_freq_dict.keys() | pos_freq_dict.keys()):
            neg_acc_clean = []
            pos_acc_clean = []
            
            neg_summary = neg_freq_dict.get(layer, [])
            pos_summary = pos_freq_dict.get(layer, [])

            # Extract edges: (i, j) from (i, j, freq, avg_curv)
            neg_edges = [(i, j) for (i, j, _, _) in neg_freq_dict.get(layer, [])]
            pos_edges = [(i, j) for (i, j, _, _) in pos_freq_dict.get(layer, [])]

            neg_total = len(neg_edges)
            pos_total = len(pos_edges)

            neg_remove_num = list(np.linspace(0, neg_total, num=3, dtype=int))
            pos_remove_num = list(np.linspace(0, pos_total, num=6, dtype=int))
            
            print(f"\nLayer {layer}:")
            print(f"  Negative edges: {neg_total}")
            print(f"  Positive edges: {pos_total}")
            # print(f"  Overlapping edges: {overlap_count}")
            print(f"  Neg remove nums: {neg_remove_num}")
            print(f"  Pos remove nums: {pos_remove_num}")

            ff.write(f"\nLayer {layer}:\n")
            ff.write(f"  Negative edges: {neg_total}\n")
            ff.write(f"  Positive edges: {pos_total}\n")
            # ff.write(f"  Overlapping edges: {overlap_count}\n")
            ff.write(f"  Neg remove nums: {neg_remove_num}\n")
            ff.write(f"  Pos remove nums: {pos_remove_num}\n")
            
            total = sample_size * len(selected_classes)
                
            # Build frequency mappings
            neg_freq_map = compute_removal_mapping(neg_summary, total_edges=total)
            pos_freq_map = compute_removal_mapping(pos_summary, total_edges=total)

            neg_freq_labels = match_frequencies(neg_remove_num, neg_freq_map)
            pos_freq_labels = match_frequencies(pos_remove_num, pos_freq_map)

            # start remove
            for index, rem_f in enumerate(neg_remove_num):
                ff.write(f'Remove edge number {rem_f}: \n')

                # remove second layer negative curvature edges
                net_neg = copy.deepcopy(net_H)
                net_neg.__build_remove_mask__(neg_edges, rem_f)
                # test acc
                acc_clean_neg = test_clean(net_neg, test_loader)
                neg_acc_clean.append(acc_clean_neg)

            for index, rem_f in enumerate(pos_remove_num):
                # remove positive curvature edges
                net_pos = copy.deepcopy(net_H)
                net_pos.__build_remove_mask__(pos_edges, rem_f)
                # test acc
                acc_clean_pos = test_clean(net_pos, test_loader)
                pos_acc_clean.append(acc_clean_pos)  

            # Plot
            plot_curve(
                neg_clean_acc=neg_acc_clean,
                pos_clean_acc=pos_acc_clean,
                neg_remove_num=neg_remove_num,
                pos_remove_num=pos_remove_num,
                label=str(sample_size) + '_' + str(layer),
                res_path=res_path,
                neg_freq_labels=neg_freq_labels,
                pos_freq_labels=pos_freq_labels
            )


            # plot_curve(neg_acc_clean, pos_acc_clean, neg_remove_num, pos_remove_num, neg_total, pos_total, str(layer) + '_' + str(sample_size), res_path)
            # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)
                
