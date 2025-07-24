import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import seaborn as sns
import copy
import pickle
import matplotlib.pyplot as plt
import re
import gc

from collections import Counter

import sys
sys.path.append("..")

import tools.utils as utils
from tools.edge_remove import Edge_Remove

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


layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    21: [784, 200, 150, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]



def process_batches_memory_efficient(
    data_path: str,
    model_full_n: str,
    metric: str,
    dataset: str,
    sample_size: int,
    prefix_dims
):

    prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    suffix = ".pkl"

    def extract_batch_num(filename: str) -> int:
        match = re.search(r'batch(\d+)', filename)
        return int(match.group(1)) if match else -1

    all_files = sorted(
        [f for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith(suffix)],
        key=extract_batch_num
    )

    print(f"Found files: {all_files}")

    label_counts = {l: 0 for l in selected_classes}
    neg_edges_all = []  # list of (i, j, curvature)
    pos_edges_all = []
    mini_c_list = []

    for filename in all_files:
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f)
            
        print(f'File path: {file_path}')

        for l in selected_classes:
            if label_counts[l] >= sample_size:
                continue

            new_data = batch_data.get(l, [])
            available = sample_size - label_counts[l]
            use_data = new_data[:available]

            for ricci in use_data:
                neg_e, pos_e, mini_c = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
                neg_edges_all.extend(neg_e)
                pos_edges_all.extend(pos_e)
                mini_c_list.append(mini_c)
                label_counts[l] += 1

            del new_data, use_data
            gc.collect()

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")

    # === Frequency + curvature summarization ===
    def count_edge_frequency(edge_list):
        freq = Counter()
        curvature_sum = defaultdict(float)
        for i, j, c in edge_list:
            key = tuple(sorted((i, j)))  # undirected
            freq[key] += 1
            curvature_sum[key] += c
        results = [
            (i, j, freq[(i, j)], curvature_sum[(i, j)] / freq[(i, j)])
            for (i, j) in freq
        ]
        return results

    neg_summary = sorted(count_edge_frequency(neg_edges_all), key=lambda x: x[3])      # sort by avg curvature ↑
    pos_summary = sorted(count_edge_frequency(pos_edges_all), key=lambda x: -x[3])     # sort by avg curvature ↓

    return neg_summary, pos_summary, mini_c_list


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

    # Step 1: Collect existing curvature edges
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_edge_fc_union(args):
    set_seed(59)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03]

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
        from tools.small_model_relu import FC_MD
    elif activation.lower() == "tanh":
        from tools.small_model_tanh import FC_MD
    
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    layer = [2]
    if model_type.lower() == "fc" and "big" not in model_pre_name.lower():
        layer = [2,4]

    for layer_num in layer:
        # build model
        dims = model_zoo[layer_num]
        
        if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
            model_name = "best_ori_10l_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == "adv":
            model_name = "pgdtrain_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == 'big_adv':
            model_name = "big_adv_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_ori':
            model_name = "big_ori_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd':
            model_name = "big_wd_"
            dims = model_zoo[21]
        else:
            raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
        
        model_name = model_name + activation + ".pth"
        
        prefix_dims = np.cumsum([0] + dims).tolist()
        
        print(f'Now for model {model_path + model_name}....\n')

        net_H = FC_MD(dims, layer_num)

        net_H.load_state_dict(torch.load(model_path + model_name))
        net_H = net_H.to(device)

        net_full = copy.deepcopy(net_H)
        test_cleanacc = test_clean(net_full, test_loader)

        # Define proportional thresholds
        freq_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0]

        # for l in selected_classes:
        with open(res_path + "edge_fc_" + str(layer_num) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            # print(f'Current label {l}: \n')
            # ff.write(f'Current label {l}: \n')
            
            neg_acc_clean = []
            pos_acc_clean = []
            
            neg_summary, pos_summary, mini_c_list = process_batches_memory_efficient(
                data_path,
                model_full_n,
                metric,
                dataset,
                sample_size,
                prefix_dims
            )
            
            mini_c_list = np.array(mini_c_list)

            print(f'It has {len(neg_summary)} negative curvature edges, {len(pos_summary)} positive curvature edges .. \n')
            ff.write(f'\nIt has {len(neg_summary)} negative curvature edges, {len(pos_summary)} positive curvature edges .. \n')
            ff.write(f'The average minimum c is {np.mean(mini_c_list)}, mean = {np.mean(mini_c_list)}\n\n')
            
            
            neg_edges_only = [(i, j) for (i, j, _, _) in neg_summary]
            pos_edges_only = [(i, j) for (i, j, _, _) in pos_summary]

            neg_total = len(neg_edges_only)
            pos_total = len(pos_edges_only)

            neg_remove_num = list(np.linspace(0, neg_total, num=6, dtype=int))
            pos_remove_num = list(np.linspace(0, pos_total, num=8, dtype=int))
            
            total = sample_size * len(selected_classes)
            
            # Build frequency mappings
            neg_freq_map = compute_removal_mapping(neg_summary, total_edges=total)
            pos_freq_map = compute_removal_mapping(pos_summary, total_edges=total)

            neg_freq_labels = match_frequencies(neg_remove_num, neg_freq_map)
            pos_freq_labels = match_frequencies(pos_remove_num, pos_freq_map)

            # Step 2: Choose thresholds — you can just use them all or downsample if too many
            # neg_max_freq = max(freq for (_, _, freq, _) in neg_summary)
            # neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

            # pos_max_freq = max(freq for (_, _, freq, _) in pos_summary)
            # pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]

            # # Step 3: For each threshold, count how many edges would be removed
            # neg_remove_num = [sum(1 for (_, _, freq, _) in neg_summary if freq >= t) for t in neg_freq_thresholds]
            # pos_remove_num = [sum(1 for (_, _, freq, _) in pos_summary if freq >= t) for t in pos_freq_thresholds]

            # start remove
            for index, rem_f in enumerate(neg_remove_num):
                print(f'Remove edge number {rem_f}:')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f)

                # remove negative curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(neg_edges_only)), res_path)
                edge_r.e_remove(neg_edges_only, cur_n + "other_neg.pth")
                
                # test acc
                net_neg = FC_MD(dims, layer_num)

                net_neg.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
                net_neg = net_neg.to(device)
                os.remove(res_path + cur_n + "other_neg.pth")

                acc_clean_neg_other = test_clean(net_neg, test_loader)
                neg_acc_clean.append(acc_clean_neg_other)

            
            for index, rem_f in enumerate(pos_remove_num):
                # ff.write(f'Remove edge number {rem_f}: \n')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) 
                # remove positive curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(pos_edges_only)), res_path)
                edge_r.e_remove(pos_edges_only, cur_n + "pos.pth")
                
                # test acc
                net_pos = FC_MD(dims, layer_num)

                net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                net_pos = net_pos.to(device)
                os.remove(res_path + cur_n + "pos.pth")

                acc_clean_pos = test_clean(net_pos, test_loader)
                pos_acc_clean.append(acc_clean_pos) 
                
            ff.write(f'\n\n')
            
            data_to_save = {
                'neg': neg_summary,
                'pos': pos_summary
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



    
