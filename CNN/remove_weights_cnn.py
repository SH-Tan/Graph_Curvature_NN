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
import torch.nn.functional as F

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



def get_last_three_layer_edges_sorted_cnn(model_dims, weights, prefix_dims, device='cuda'):
    """
    Extracts and sorts edges from layers 3→4, 4→5, and 5→6 using global indexing.
    Supports both CNN→FC and FC→FC transitions.
    """
    batch_idx = 0
    weight_idx = 0
    all_edges = []
    all_weights = []

    num_layers = len(model_dims)  # model_dims[1] to model_dims[6]

    for i in range(num_layers - 1):  # transitions: i → i+1
        l = i + 1
        current_layer = model_dims[l + 1]
        src_size = prefix_dims[i+1] - prefix_dims[i]
        dst_size = prefix_dims[i+2] - prefix_dims[i+1]

        # Only keep transitions from 3→4, 4→5, 5→6
        if not (i in [2, 3]):
            # Still skip over weights
            if current_layer['name'] == 'fc':
                weight_idx += src_size * dst_size
            elif current_layer['name'] in ['cnn', 'pooling']:
                k = current_layer['dim']['kernel']
                s = current_layer['dim']['stride']
                in_size = model_dims[l]['dim']['out_size']
                pre_ch = model_dims[l]['dim'].get('channel', 1)
                cur_ch = current_layer['dim']['channel']
                dummy = torch.zeros(1, pre_ch, in_size, in_size, device=device)
                patches = F.unfold(dummy, kernel_size=k, stride=s).shape[-1]
                weight_idx += patches * cur_ch * (k**2 * pre_ch)
            continue

        # Process desired transition
        if current_layer['name'] == 'fc':
            # FC layer
            direct_w = weights[batch_idx, weight_idx:weight_idx + src_size * dst_size].view(src_size, dst_size)

            src_idx = torch.arange(src_size, device=device) + prefix_dims[i]
            dst_idx = torch.arange(dst_size, device=device) + prefix_dims[i+1]
            src_grid, dst_grid = torch.meshgrid(src_idx, dst_idx, indexing='ij')
            edges = torch.stack([src_grid.flatten(), dst_grid.flatten()])
            edge_weights = direct_w.flatten()

            all_edges.append(edges)
            all_weights.append(edge_weights)

            weight_idx += src_size * dst_size

        elif current_layer['name'] in ['cnn', 'pooling']:
            # CNN layer (layer 3→4)
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']

            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()
            unfolded = F.unfold(dummy, kernel_size=k, stride=s).transpose(1, 2).int()
            patches = unfolded.shape[1]
            step = k ** 2

            n = 0
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0, p].tolist()
                    global_src = torch.tensor(cur_idx, device=device) + prefix_dims[i]
                    global_dst = torch.tensor([prefix_dims[i+1] + n] * len(cur_idx), device=device)

                    edges = torch.stack([global_src, global_dst])
                    edge_weights = weights[batch_idx, weight_idx:weight_idx + len(cur_idx)]

                    all_edges.append(edges)
                    all_weights.append(edge_weights)

                    weight_idx += len(cur_idx)
                    n += 1

    # Stack and sort
    all_edges = torch.cat(all_edges, dim=1)
    all_weights = torch.cat(all_weights, dim=0)
    
    abs_weights = torch.abs(all_weights)

    # Sort by descending |weight|
    sorted_indices_high = torch.argsort(abs_weights, descending=True)
    sorted_edges_high = all_edges[:, sorted_indices_high]

    # Sort by ascending |weight|
    sorted_indices_low = torch.argsort(abs_weights, descending=False)
    sorted_edges_low = all_edges[:, sorted_indices_low]

    # # Sort by descending weight
    # sorted_indices = torch.argsort(all_weights, descending=True)
    # sorted_edges_high = all_edges[:, sorted_indices]
    # # sorted_weights = all_weights[sorted_indices]
    
    # # Sort by ascending weight
    # sorted_indices = torch.argsort(all_weights, descending=False)
    # sorted_edges_low = all_edges[:, sorted_indices]
    # # sorted_weights = all_weights[sorted_indices]

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




def plot_curve(high_clean_acc, low_clean_acc, remove_num, res_path, name):
    # Colors
    high_color = "#29E000" 
    low_color = "#1F00EC"   

    # Sort by remove_num
    combined = sorted(zip(remove_num, high_clean_acc, low_clean_acc), key=lambda x: x[0])
    remove_sorted, high_sorted, low_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(remove_sorted, high_sorted, label='Large weight removed first',
             marker='o', linestyle='--', linewidth=3., markersize=13, color=high_color)

    plt.plot(remove_sorted, low_sorted, label='Small weight removed first',
             marker='x', linestyle='-', linewidth=3., markersize=13, color=low_color)

    # Labels and title
    plt.xlabel('Number of Edges Removed', fontsize=33, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=33, fontweight='semibold')
    plt.ylim(0.0, 1.0)

    # Scientific x-axis
    ax = plt.gca()
    
    # Compute exponent (e.g., 1e+3, 1e+4) based on the max value
    exponent = int(np.floor(np.log10(max(remove_num))))
    scale = 10 ** exponent

    # Scale values and format tick labels as mantissas only
    scaled_ticks = [x / scale for x in remove_num]
    mantissa_labels = [f"{v:.1f}" for v in scaled_ticks]

    # Set the ticks and the scaled mantissa labels
    plt.xticks(ticks=remove_num, labels=mantissa_labels, fontsize=22, fontweight='semibold')

    # Add scientific scale as offset text (e.g., ×1e4) to the end of the x-axis
    ax.annotate(
            f"×1e{exponent}",
            xy=(1.0, 0.0), xycoords='axes fraction',  # Right end of x-axis
            xytext=(10, -35), textcoords='offset points',  # Just below and slightly to the left
            ha='right', va='top',
            fontsize=18, fontweight='semibold'
        )
    
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # ax.xaxis.get_offset_text().set_fontsize(20)
    # ax.xaxis.get_offset_text().set_fontweight('semibold')

    # # Ticks
    # plt.xticks(fontsize=22, fontweight='semibold')
    plt.yticks(fontsize=22, fontweight='semibold')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=22, loc='best')
    for text in legend.get_texts():
        text.set_fontweight('semibold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{name}_remove_w_curve_perlayer.pdf'), dpi=300)
    plt.close()
    
    
def plot_tensor_hist(tensor, bins=1000, title="Histogram", log=False, save_path=None):
    """
    Plot a histogram of a PyTorch tensor.
    - tensor: torch.Tensor (can be on CPU or GPU)
    - bins: number of histogram bins
    - log: set True for a log-scaled y-axis
    - save_path: if provided, save the figure to this path
    """
    # Detach, move to CPU, flatten, and filter finite values
    t = tensor.detach().float().flatten().cpu()
    finite_mask = torch.isfinite(t)
    t = t[finite_mask]
    if t.numel() == 0:
        print("No finite values to plot.")
        return

    # Convert to numpy for matplotlib
    arr = t.numpy()

    plt.figure(figsize=(7,4))
    plt.hist(arr, bins=bins, log=log)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, f'Histogram_remove_w_curve.pdf'), dpi=300)
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
    

def remove_w_cnn(args):
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

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
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
    
    img = None
    for count, (images, labels) in enumerate(train_loader):
        img = images.to(device)
        break
    
    edge_array, nodes_ori, output = net_full.NN_info_batch(img)
    weights = output.detach().clone().to(device)  
    # plot_tensor_hist(torch.abs(weights), bins=100, title="Output Weights Histogram", save_path=res_path) 
    # print(f'Finish Histgram..')
    
    sorted_edges_high, sorted_edges_low = get_last_three_layer_edges_sorted_cnn(model_dims, weights, prefix_dims, device) 
    sorted_edges_high = sorted_edges_high.cpu().numpy().T
    sorted_edges_low = sorted_edges_low.cpu().numpy().T
    
    # Step 2: Define total number of edges and removal schedule
    neg_total = len(sorted_edges_low)
    pos_total = len(sorted_edges_high)

    low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
    high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
    high_acc_clean = []
    low_acc_clean = []
    
    # start remove
    for index, rem_f in enumerate(high_remove_num):
        # remove second layer negative curvature edges
        net_neg = copy.deepcopy(net_H)
        net_neg.__build_remove_mask__(sorted_edges_high, rem_f, mask='local')
        # test acc
        acc = test_clean(net_neg, test_loader)
        high_acc_clean.append(acc)
        
    for index, rem_f in enumerate(low_remove_num):
        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(sorted_edges_low, rem_f, mask='local')
        # test acc
        acc_low = test_clean(net_pos, test_loader)
        low_acc_clean.append(acc_low)
        
    plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, model_full_n+activation)   
    
    # pack into a dictionary
    data = {
        "neg_clean_acc": low_acc_clean,
        "pos_clean_acc": high_acc_clean,
        "neg_remove_num": low_remove_num,
        "pos_remove_num": low_remove_num,
    }

    # save to pickle file
    with open(res_path+"results_weight_local.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Saved variables to results.pkl")
    
    # Step 2: Separate edges by layer
    # sorted_edges_high_by_layer = separate_edges_by_layer_in_order(sorted_edges_high, prefix_dims)
    # sorted_edges_low_by_layer = separate_edges_by_layer_in_order(sorted_edges_low, prefix_dims)
    # print(sorted_edges_low_by_layer.keys())
    
    # # Step 3: Per-layer analysis
    # for layer in sorted(sorted_edges_low_by_layer.keys() | sorted_edges_high_by_layer.keys()):
    #     high_acc_clean = []
    #     low_acc_clean = []
        
    #     # These are just lists of (i, j), not 4-tuples
    #     neg_edges = sorted_edges_low_by_layer.get(layer, [])
    #     pos_edges = sorted_edges_high_by_layer.get(layer, [])

    #     neg_total = len(neg_edges)
    #     pos_total = len(pos_edges)

    #     low_remove_num = list(np.linspace(0, neg_total, num=6, dtype=int))
    #     high_remove_num = list(np.linspace(0, pos_total, num=8, dtype=int))

    #     print(f"Layer {layer}:")
    #     # print(f"  low total = {neg_total}, high total = {pos_total}, overlap = {overlap_count}")
        
    #     print(f"  Low remove nums: {low_remove_num}")
    #     print(f"  High remove nums: {high_remove_num}")
        
    #     # start remove
    #     for index, rem_f in enumerate(high_remove_num):
    #         # remove second layer negative curvature edges
    #         net_neg = copy.deepcopy(net_H)
    #         net_neg.__build_remove_mask__(pos_edges, rem_f)
    #         # test acc
    #         acc = test_clean(net_neg, test_loader)
    #         high_acc_clean.append(acc)
            
    #     for index, rem_f in enumerate(low_remove_num):
    #         # remove positive curvature edges
    #         net_pos = copy.deepcopy(net_H)
    #         net_pos.__build_remove_mask__(neg_edges, rem_f)
    #         # test acc
    #         acc_low = test_clean(net_pos, test_loader)
    #         low_acc_clean.append(acc_low)
        
    #     plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, model_full_n+activation+str(layer))   
        
    
    
