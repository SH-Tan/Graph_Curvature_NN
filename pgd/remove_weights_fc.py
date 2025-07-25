import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import seaborn as sns
import copy

# from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import simps
import pickle
import time
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils

from tools.FC_linear import FC_Linear
from tools.graph_curvature import graph_curvature_main_torch
from tools.draw_net import DrawNN
from tools.edge_remove import Edge_Remove
from tools.get_c import get_c

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



def get_all_edges_sorted_by_weight(dims, weights, device='cuda'):
    """
    Returns:
        - sorted_edges: [2, total_edges], LongTensor, globally indexed
        - sorted_weights: [total_edges], FloatTensor, sorted descending
    """
    batch_idx = 0
    weight_idx = 0
    global_node_offset = 0

    all_edges = []
    all_weights = []

    for i in range(len(dims) - 1):
        src_size, dst_size = dims[i], dims[i+1]
        num_edges = src_size * dst_size

        # Extract weights for this layer
        direct_dist = weights[batch_idx, weight_idx:weight_idx + num_edges]

        # Global node indices
        src_nodes = torch.arange(src_size, device=device) + global_node_offset
        dst_nodes = torch.arange(dst_size, device=device) + global_node_offset + src_size
        global_node_offset += src_size

        src_grid, dst_grid = torch.meshgrid(src_nodes, dst_nodes, indexing='ij')
        edges = torch.stack([src_grid.flatten(), dst_grid.flatten()])  # [2, num_edges]

        all_edges.append(edges)
        all_weights.append(direct_dist.flatten())

        weight_idx += num_edges

    # Combine all
    all_edges = torch.cat(all_edges, dim=1)      # [2, total_edges]
    all_weights = torch.cat(all_weights, dim=0)  # [total_edges]

    # Sort by descending weight
    sorted_indices = torch.argsort(all_weights, descending=True)
    sorted_edges_high = all_edges[:, sorted_indices]
    # sorted_weights = all_weights[sorted_indices]
    
    # Sort by ascending weight
    sorted_indices = torch.argsort(all_weights, descending=False)
    sorted_edges_low = all_edges[:, sorted_indices]
    # sorted_weights = all_weights[sorted_indices]

    return sorted_edges_high, sorted_edges_low


def separate_edges_by_layer_in_order(sorted_edges, prefix_dims):
    layer_to_edges = defaultdict(list)
    src_nodes = sorted_edges[0]
    dst_nodes = sorted_edges[1]

    for i_raw, j_raw in zip(src_nodes, dst_nodes):
        i = i_raw
        j = j_raw

        src_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        dst_layer = np.searchsorted(prefix_dims, j, side='right') - 1

        # Detect layer transition from src_layer → dst_layer
        if dst_layer == src_layer + 1:
            layer_to_edges[src_layer].append((i_raw, j_raw))  # keep global indices

    return layer_to_edges



def plot_curve(high_clean_acc, low_clean_acc, remove_num, res_path, name):
    # Colors
    high_color = "#29E000" 
    low_color = "#1F00EC"   

    # Sort by remove_num
    combined = sorted(zip(remove_num, high_clean_acc, low_clean_acc), key=lambda x: x[0])
    remove_sorted, high_sorted, low_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(remove_sorted, high_sorted, label='High weight removed first',
             marker='o', linestyle='--', linewidth=3., markersize=13, color=high_color)

    plt.plot(remove_sorted, low_sorted, label='Low weight removed first',
             marker='x', linestyle='-', linewidth=3., markersize=13, color=low_color)

    # Labels and title
    plt.xlabel('Number of Edges Removed', fontsize=33, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=33, fontweight='semibold')
    plt.ylim(0.0, 1.0)

    # Scientific x-axis
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('semibold')

    # Ticks
    plt.xticks(fontsize=22, fontweight='semibold')
    plt.yticks(fontsize=22, fontweight='semibold')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=22, loc='best')
    for text in legend.get_texts():
        text.set_fontweight('semibold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{name}_remove_w_curve.pdf'), dpi=300)
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


def remove_w_fc(args):
    seed = 59
    set_seed(seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    activation = args.activation
    
    if activation.lower() == "relu":
        from tools.small_model_relu import FC_MD
    elif activation.lower() == "tanh":
        from tools.small_model_tanh import FC_MD
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    layers = [2,4]
    if 'big' in model_pre_name.lower():
        layers = [2]

    remove_num = []
    
    # build model
    for layer_num in layers:
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
        
        neural_list = []
        nodes_num = 0
        edges_num = 0
        i = 0
        for p in net_H.parameters():
            if i == 0:
                nodes_num += p.shape[1]
            if i%2 == 0:
                nodes_num += p.shape[0]
                edges_num += (p.shape[0] * p.shape[1])
                neural_list.append(p.shape[0])
            i += 1
            
        img = None
        for count, (images, labels) in enumerate(train_loader):
            img = images.to(device)
            break
        
        edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))
        weights = output.detach().clone().to(device) 
        sorted_edges_high, sorted_edges_low = get_all_edges_sorted_by_weight(dims, weights, device) 
        
        # Step 1: Convert tensors to numpy BEFORE separating by layer
        sorted_edges_high_np = sorted_edges_high.cpu().numpy().T
        sorted_edges_low_np = sorted_edges_low.cpu().numpy().T
        
        # Step 2: Define total number of edges and removal schedule
        neg_total = len(sorted_edges_low_np)
        pos_total = len(sorted_edges_high_np)

        low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
        high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
        
        high_acc_clean = []
        low_acc_clean = []

        # Remove high-weight (positive) edges and test
        for rem_f in high_remove_num:
            cur_n = model_full_n + '_global_' + str(rem_f)

            net_H.load_state_dict(torch.load(model_path + model_name))
            edge_r = Edge_Remove(net_H, dims, min(rem_f, len(sorted_edges_high_np)), res_path)
            edge_r.e_remove(sorted_edges_high_np, cur_n + "_high.pth")

            net_neg = FC_MD(dims, layer_num)
            net_neg.load_state_dict(torch.load(res_path + cur_n + "_high.pth"))
            net_neg = net_neg.to(device)
            os.remove(res_path + cur_n + "_high.pth")

            acc_high = test_clean(net_neg, test_loader)
            high_acc_clean.append(acc_high)

        # Remove low-weight (negative) edges and test
        for rem_f in low_remove_num:
            cur_n = model_full_n + '_global_' + str(rem_f)

            net_H.load_state_dict(torch.load(model_path + model_name))
            edge_r = Edge_Remove(net_H, dims, min(rem_f, len(sorted_edges_low_np)), res_path)
            edge_r.e_remove(sorted_edges_low_np, cur_n + "_low.pth")

            net_pos = FC_MD(dims, layer_num)
            net_pos.load_state_dict(torch.load(res_path + cur_n + "_low.pth"))
            net_pos = net_pos.to(device)
            os.remove(res_path + cur_n + "_low.pth")

            acc_low = test_clean(net_pos, test_loader)
            low_acc_clean.append(acc_low)

        # Step 2: Separate edges by layer
        # sorted_edges_high_by_layer = separate_edges_by_layer_in_order(sorted_edges_high_np, prefix_dims)
        # sorted_edges_low_by_layer = separate_edges_by_layer_in_order(sorted_edges_low_np, prefix_dims)
        # print(sorted_edges_low_by_layer.keys())
        
        # Step 3: Per-layer analysis
        # for layer in sorted(sorted_edges_low_by_layer.keys() | sorted_edges_high_by_layer.keys()):
        #     high_acc_clean = []
        #     low_acc_clean = []
            
        #     # These are just lists of (i, j), not 4-tuples
        #     neg_edges = sorted_edges_low_by_layer.get(layer, [])
        #     pos_edges = sorted_edges_high_by_layer.get(layer, [])

        #     neg_total = len(neg_edges)
        #     pos_total = len(pos_edges)

        #     low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
        #     high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
            
        #     print(f"Layer {layer}:")
        #     print(f"  low total = {neg_total}, high total = {pos_total}")
            
        #     print(f"  Low remove nums: {low_remove_num}")
        #     print(f"  High remove nums: {high_remove_num}")
            
        #     # start remove
        #     for index, rem_f in enumerate(high_remove_num):
        #         cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f)

        #         # remove high weight edges
        #         net_H.load_state_dict(torch.load(model_path + model_name))
        #         edge_r = Edge_Remove(net_H, dims, min(rem_f, len(sorted_edges_high)), res_path)
        #         edge_r.e_remove(sorted_edges_high, cur_n + "other_neg.pth")
                
        #         # test acc
        #         net_neg = FC_MD(dims, layer_num)

        #         net_neg.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
        #         net_neg = net_neg.to(device)
        #         os.remove(res_path + cur_n + "other_neg.pth")

        #         acc_high = test_clean(net_neg, test_loader)
        #         high_acc_clean.append(acc_high)
                
        #     for index, rem_f in enumerate(low_remove_num):
        #         # remove low weight edges
        #         net_H.load_state_dict(torch.load(model_path + model_name))
        #         edge_r = Edge_Remove(net_H, dims, min(rem_f, len(sorted_edges_low)), res_path)
        #         edge_r.e_remove(sorted_edges_low, cur_n + "pos.pth")
                
        #         # test acc
        #         net_pos = FC_MD(dims, layer_num)

        #         net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
        #         net_pos = net_pos.to(device)
        #         os.remove(res_path + cur_n + "pos.pth")

        #         acc_low = test_clean(net_pos, test_loader)
        #         low_acc_clean.append(acc_low)
                    
        plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, model_full_n+activation)                
                            
  
   
            