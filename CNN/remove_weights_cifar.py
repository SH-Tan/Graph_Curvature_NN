import torch
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

import pandas as pd
# from tools.vgg16_custom_mnist import VGG16_CIFAR10

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

# model_dims_small = {
#     1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":False, "out_size": 1}},   # conv5_3 + pool

#     2: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
#     3: {"name": "fc", "dim": {"out_size": 512}},
#     4: {"name": "fc", "dim": {"out_size": 10}}
# }

# model_dims = {
#     1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},   # Input image

#     2: {"name": "cnn", "dim": {"channel": 64, "kernel": 3, "stride": 1, "padding": 1, "out_size": 14}},  # After conv1_2 + pool
#     3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding": 1, "out_size": 7}},   # After conv2_2 + pool
#     4: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding": 1, "out_size": 3}},   # After conv3_3 + pool
#     5: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "out_size": 1}},   # After conv4_2 + pool

#     6: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "out_size": 1}},   # conv5_1

#     7: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
#     8: {"name": "fc", "dim": {"out_size": 512}},
#     9: {"name": "fc", "dim": {"out_size": 10}},
# }

# model_dims_small = {
#     1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "out_size": 1}}, 
#     2: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "out_size": 1}},   # conv5_1

#     3: {"name": "fc", "dim": {"out_size": 1024}},
#     4: {"name": "fc", "dim": {"out_size": 512}},
#     5: {"name": "fc", "dim": {"out_size": 10}},
# }



# data_train = MNIST('./data/mnist',
#                   train=True,
#                   download=True,
#                   transform=transforms.Compose([
#                       # transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))

# data_test = MNIST('./data/mnist',
#                   train=False,
#                   download=True,
#                   transform=transforms.Compose([
#                       # transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))

selected_classes = [0,1,2,3,4,5,6,7,8,9]



def get_last_three_layer_edges_sorted_cnn(model_dims, weights, prefix_dims, device='cuda', pre_n=0):
    """
    Extracts and sorts edges from layers 3→4, 4→5, and 5→6 using global indexing.
    Supports both CNN→FC and FC→FC transitions.
    Applies a global node offset `pre_n` to all node indices.
    """
    batch_idx = 0
    weight_idx = 0
    all_edges = []
    all_weights = []

    num_layers = len(model_dims)  # model_dims[1] to model_dims[6]

    for i in range(num_layers - 1):  # transitions: i → i+1
        l = i + 1
        current_layer = model_dims[l + 1]
        src_size = prefix_dims[i + 1] - prefix_dims[i]
        dst_size = prefix_dims[i + 2] - prefix_dims[i + 1]

        # Only keep transitions from 3→4, 4→5, 5→6
        if not (i in [1, 2, 3]):
            # Still skip over weights
            if current_layer['name'] == 'fc':
                weight_idx += src_size * dst_size
            elif current_layer['name'] in ['cnn', 'pooling']:
                k = current_layer['dim']['kernel']
                s = current_layer['dim']['stride']
                p = current_layer['dim']['padding']
                in_size = model_dims[l]['dim']['out_size']
                pre_ch = model_dims[l]['dim'].get('channel', 1)
                cur_ch = current_layer['dim']['channel']
                dummy = torch.zeros(1, pre_ch, in_size, in_size, device=device)
                patches = F.unfold(dummy, kernel_size=k, stride=s, padding=p).shape[-1]
                weight_idx += patches * cur_ch * (k**2 * pre_ch)
            continue

        # Process desired transition
        if current_layer['name'] == 'fc':
            direct_w = weights[batch_idx, weight_idx:weight_idx + src_size * dst_size].view(src_size, dst_size)

            src_idx = torch.arange(src_size, device=device) + prefix_dims[i] + pre_n
            dst_idx = torch.arange(dst_size, device=device) + prefix_dims[i + 1] + pre_n
            src_grid, dst_grid = torch.meshgrid(src_idx, dst_idx, indexing='ij')
            edges = torch.stack([src_grid.flatten(), dst_grid.flatten()])
            edge_weights = direct_w.flatten()

            all_edges.append(edges)
            all_weights.append(edge_weights)

            weight_idx += src_size * dst_size

        elif current_layer['name'] in ['cnn', 'pooling']:
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
                    global_src = torch.tensor(cur_idx, device=device) + prefix_dims[i] + pre_n
                    global_dst = torch.tensor([prefix_dims[i + 1] + n + pre_n] * len(cur_idx), device=device)

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
    sorted_indices_high = torch.argsort(all_weights, descending=True)
    sorted_edges_high = all_edges[:, sorted_indices_high]

    # Sort by ascending weight
    sorted_indices_low = torch.argsort(all_weights, descending=False)
    sorted_edges_low = all_edges[:, sorted_indices_low]

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
    

def remove_w_cifar(args):
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
    
    if activation.lower() == "relu":
        from tools.vgg16_custom_relu import VGG16_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg16_custom_tanh import VGG16_CIFAR10
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=100, valid_num=500)
    dims_full = cal_dims(model_dims)
    dims = cal_dims(model_dims_small)
    prefix_dims = np.cumsum([0] + dims).tolist()
    prefix_dims_full = np.cumsum([0] + dims_full).tolist()
    pre_n=(np.sum(dims_full)-np.sum(dims))
    
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

    # acc_clean = test_clean(net_H, test_loader)
    # print(acc_clean)
    
    net_full = copy.deepcopy(net_H)

    print(model_name)
  
    img = None
    for count, (images, labels) in enumerate(train_loader):
        img = images.to(device)
        break
    
    edge_array, nodes_ori, output = net_full.NN_info_batch(img)
    del img, edge_array, nodes_ori
    
    weights = output.detach().to(device)  
    sorted_edges_high, sorted_edges_low = get_last_three_layer_edges_sorted_cnn(model_dims_small, weights, prefix_dims, device, pre_n=pre_n) 

    # Step 1: Convert tensors to numpy BEFORE separating by layer
    sorted_edges_high_np = sorted_edges_high.cpu().numpy()
    sorted_edges_low_np = sorted_edges_low.cpu().numpy()

    # Step 2: Separate edges by layer
    sorted_edges_high_by_layer = separate_edges_by_layer_in_order(sorted_edges_high_np, prefix_dims_full)
    sorted_edges_low_by_layer = separate_edges_by_layer_in_order(sorted_edges_low_np, prefix_dims_full)
    print(sorted_edges_low_by_layer.keys())
    
    # Step 3: Per-layer analysis
    for layer in sorted(sorted_edges_low_by_layer.keys() | sorted_edges_high_by_layer.keys()):
        high_acc_clean = []
        low_acc_clean = []

        # These are just lists of (i, j), not 4-tuples
        neg_edges = sorted_edges_low_by_layer.get(layer, [])
        pos_edges = sorted_edges_high_by_layer.get(layer, [])
        
        print(neg_edges[:10], pos_edges[:10])

        neg_set = set(neg_edges)
        pos_set = set(pos_edges)

        overlap = neg_set & pos_set
        overlap_count = len(overlap)

        neg_total = len(neg_edges)
        pos_total = len(pos_edges)

        low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
        high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))

        print(f"Layer {layer}:")
        print(f"  low total = {neg_total}, high total = {pos_total}, overlap = {overlap_count}")
        
        print(f"  Low remove nums: {low_remove_num}")
        print(f"  High remove nums: {high_remove_num}")

        # start remove
        for index, rem_f in enumerate(high_remove_num):
            # remove second layer negative curvature edges
            net_neg = copy.deepcopy(net_H)
            net_neg.__build_remove_mask__(pos_edges, rem_f)
            # test acc
            acc = test_clean(net_neg, test_loader)
            high_acc_clean.append(acc)

        for index, rem_f in enumerate(low_remove_num):
            # remove positive curvature edges
            net_pos = copy.deepcopy(net_H)
            net_pos.__build_remove_mask__(neg_edges, rem_f)
            # test acc
            acc_low = test_clean(net_pos, test_loader)
            low_acc_clean.append(acc_low)
            
        plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, model_full_n+activation+str(layer))   
            
        
    
