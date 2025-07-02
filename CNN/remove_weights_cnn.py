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
        if not (i in [2, 3, 4]):
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

    # Sort by descending weight
    sorted_indices = torch.argsort(all_weights, descending=True)
    sorted_edges_high = all_edges[:, sorted_indices]
    # sorted_weights = all_weights[sorted_indices]
    
    # Sort by ascending weight
    sorted_indices = torch.argsort(all_weights, descending=False)
    sorted_edges_low = all_edges[:, sorted_indices]
    # sorted_weights = all_weights[sorted_indices]

    return sorted_edges_high, sorted_edges_low





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
    # Zip, sort, and unzip to reorder all lists by remove_numbers
    combined = sorted(zip(remove_num, high_clean_acc, low_clean_acc), key=lambda x: x[0])
    remove_sorted, high_sorted, low_sorted = zip(*combined)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(remove_sorted, high_sorted, label='High Weight Edge Clean Acc', marker='o', linestyle='--')
    plt.plot(remove_sorted, low_sorted, label='Low Weight Edge Clean Acc', marker='x', linestyle='-')

    plt.xlabel('Remove Number')
    plt.ylabel('Clean Accuracy')
    plt.title('Clean Accuracy vs Remove Number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_path + name + f'_remove_w_curve.png')
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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    activation = args.activation
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)
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
    
    img = None
    for count, (images, labels) in enumerate(train_loader):
        img = images.to(device)
        break
    
    edge_array, nodes_ori, output = net_full.NN_info_batch(img)
    weights = output.detach().clone().to(device)  
    sorted_edges_high, sorted_edges_low = get_last_three_layer_edges_sorted_cnn(model_dims, weights, prefix_dims, device) 
    sorted_edges_high = sorted_edges_high.cpu().numpy().T
    sorted_edges_low = sorted_edges_low.cpu().numpy().T
    
    remove_num = [0, 1000, 3000, (int)(len(sorted_edges_high)*0.05), (int)(len(sorted_edges_high)*0.1), (int)(len(sorted_edges_high)*0.2), (int)(len(sorted_edges_high)*0.3), (int)(len(sorted_edges_high)*0.5), (int)(len(sorted_edges_high)*0.7), (int)(len(sorted_edges_high))]
    high_acc_clean = []
    low_acc_clean = []
    
    # start remove
    for index, rem_f in enumerate(remove_num):
        # remove second layer negative curvature edges
        net_neg = copy.deepcopy(net_H)
        net_neg.__build_remove_mask__(sorted_edges_high, rem_f)
        # test acc
        acc = test_clean(net_neg, test_loader)
        high_acc_clean.append(acc)

        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(sorted_edges_low, rem_f)
        # test acc
        acc_low = test_clean(net_pos, test_loader)
        low_acc_clean.append(acc_low)
        
    plot_curve(high_acc_clean, low_acc_clean, remove_num, res_path, model_full_n+activation)   
        
    
    
