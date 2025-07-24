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
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4}},
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





def get_top_c(curvature, b, prefix_dims):
    neg_e = set()
    pos_e = set()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])  # shape (N, 3)

        # Filter values with valid curvature (<= 1)
        valid = ricci_curv[ricci_curv[:, 2] <= 1]

        # Convert to int for indexing
        valid[:, 0:2] = valid[:, 0:2].astype(int)

        for i, j, curr in valid:
            i, j = int(i), int(j)
            i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
            if i_layer >= 2: 
                if curr < 0:
                    neg_e.add((i, j, curr))
                elif curr > 0:
                    pos_e.add((i, j, curr))

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


from collections import Counter
def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(float)

    for edge_set in edge_sets:
        for i, j, c in edge_set:
            key = tuple(sorted((i, j)))  # normalize direction for undirected edges
            freq[key] += 1
            curvature_sum[key] += c

    results = []
    for key in freq:
        avg_curv = curvature_sum[key] / freq[key]
        results.append((key[0], key[1], freq[key], avg_curv))

    return results




def remove_edge_cnn_union(args):
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

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

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
        from tools.LeNet5_custom_small import LeNet_custom_v2
    elif activation.lower() == "tanh":
        from tools.LeNet5_custom_small_tanh import LeNet_custom_v2
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
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
    
    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = []
  
    test_cleanacc = test_clean(net_full, test_loader)
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

    correct_suffix_res = dataset + "_res_correct.pkl" 
    res_name = model_full_n + metric + '_' + correct_suffix_res

    with open(data_path + res_name, 'rb') as file:
        res_dict = pickle.load(file)    

    freq_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    
    with open(res_path + "edge_cnn_" + ".txt", "a+") as ff:
        ff.write(f'For model {model_name}: \n')
        ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
        # print(f'Current label {l}: \n')
        # ff.write(f'Current label {l}: \n')

        neg_acc_clean = []
        pos_acc_clean = []
        neg_edge_sets = []
        pos_edge_sets = []
        for l in selected_classes:
            idx = 0
            for (ricci, batch, dim, node) in res_dict[l]:
                neg_e_other, pos_e = get_top_c(ricci, 1, prefix_dims)
                neg_edge_sets.append(neg_e_other)
                pos_edge_sets.append(pos_e)
                idx += 1
                if idx >= sample_size:
                    break

        neg_freq_dict = count_edge_frequency(neg_edge_sets)
        pos_freq_dict = count_edge_frequency(pos_edge_sets)
        neg_freq_edges_sorted = sorted(neg_freq_dict, key=lambda x: (-x[2], x[3]))
        pos_freq_edges_sorted = sorted(pos_freq_dict, key=lambda x: (-x[2], -x[3]))

        print(f'It has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
        ff.write(f'\nIt has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
            
        neg_edges_only = [(i, j) for (i, j, _, _) in neg_freq_edges_sorted]
        pos_edges_only = [(i, j) for (i, j, _, _) in pos_freq_edges_sorted]

        # Compute overlap
        # Convert to sets for fast overlap calculation
        neg_set = set(neg_edges_only)
        pos_set = set(pos_edges_only)
        overlap = neg_set & pos_set
        overlap_count = len(overlap)
        ff.write(f"\nNumber of overlapping edges: {overlap_count}\n")
        
        neg_total = len(neg_edges_only)
        pos_total = len(pos_edges_only)

        # remove_num = [0, 5000, (int)(len(neg_edges_only)*0.3), (int)(len(neg_edges_only)*0.5), (int)(len(neg_edges_only)*0.7), len(neg_edges_only), (int)(len(pos_edges_only)*0.7), (int)(len(pos_edges_only)*0.9), (int)(len(pos_edges_only))]
        # Generate uniformly spaced points (including 0 and total) for each list
        neg_remove_num = list(np.linspace(0, neg_total, num=6, dtype=int))
        pos_remove_num = list(np.linspace(0, pos_total, num=8, dtype=int))
        
        total = sample_size * len(selected_classes)
        
        # Build frequency mappings
        neg_freq_map = compute_removal_mapping(neg_freq_edges_sorted, total_edges=total)
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
            ff.write(f'Remove edge number {rem_f}: \n')

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
            
        
        data_to_save = {
            'neg': neg_freq_edges_sorted,
            'pos': pos_freq_edges_sorted
        }
        
        save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}.pkl"
        save_path = os.path.join(res_path, save_name)

        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)  # use dict to avoid defaultdict issues

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
                
