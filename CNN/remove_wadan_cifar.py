import torch
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
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



selected_classes = [0,1,2,3,4,5,6,7,8,9]


def load_dataset_from_disk(path, batch_size=128, shuffle=True):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    return dataset






def wanda_score_all_fc(model, dataloader, device, nsamples=10, offset = 0):
    model.eval()
    model.to(device)

    # FC layers to analyze
    fc_layers = ["fc1", "fc2", "fc3"]
    activations = {name: [] for name in fc_layers}

    # Register hooks for each FC layer
    handles = []
    for name in fc_layers:
        layer = getattr(model, name)
        def make_hook(layer_name):
            def hook_fn(module, inp, out):
                activations[layer_name].append(inp[0].detach().to("cpu"))
            return hook_fn
        handles.append(layer.register_forward_hook(make_hook(name)))

    # Collect activations from calibration data
    with torch.no_grad():
        for l in selected_classes:
            for i, (x, _) in enumerate(dataloader[l]):
                if i >= nsamples:
                    break
                x = x.to(device)
                # x = model.normalize(x)
                _ = model(x)

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute Wanda score for each layer and assign to edges
    all_scores = []
    all_edges = []

    for name in fc_layers:
        layer = getattr(model, name)
        act = torch.cat(activations[name], dim=0)
        scaler_row = torch.sqrt((act ** 2).mean(0))
        W = layer.weight.data.cpu()
        W_metric = torch.abs(W) * scaler_row.reshape(1, -1)

        # Convert to edge list: (src, dst, score)
        num_out, num_in = W_metric.shape
        src_nodes = np.arange(num_in) + offset
        dst_nodes = np.arange(num_out) + offset + num_in

        edges = np.array(np.meshgrid(src_nodes, dst_nodes)).T.reshape(-1, 2)
        edge_scores = W_metric.flatten().numpy()

        all_edges.append(edges)
        all_scores.append(edge_scores)

        offset += num_in  # shift indices for next layer

    all_edges = np.vstack(all_edges)
    all_scores = np.concatenate(all_scores)

    # Sort edges by Wanda score
    sorted_idx_high = np.argsort(-all_scores)  # high→low
    sorted_idx_low = np.argsort(all_scores)    # low→high

    sorted_edges_high = all_edges[sorted_idx_high]
    sorted_edges_low = all_edges[sorted_idx_low]

    return sorted_edges_high, sorted_edges_low, all_scores




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
    plt.savefig(os.path.join(res_path, f'{name}_remove_w_curve.pdf'), dpi=300)
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
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def remove_wadan_cifar(args):
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
        from tools.vgg16_custom_relu import VGG16_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg16_custom_tanh import VGG16_CIFAR10
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    val_set = load_dataset_from_disk("./data/CIFAR10_val", batch_size=64, shuffle=False)
    sep_dataloader = utils.sep_label(val_set, selected_classes, bs=32)
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=100, valid_num=500)

    node_dims = cal_dims(model_dims)
    print(node_dims)
    offset = np.sum(node_dims[:-4])
    
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
    
    # Compute Wanda scores
    sorted_edges_high, sorted_edges_low, all_scores = wanda_score_all_fc(net_full, sep_dataloader, device, nsamples=10, offset = offset)
  
    # print(sorted_edges_high.shape)
    
    # Step 1: Convert tensors to numpy BEFORE separating by layer
    sorted_edges_high_np = sorted_edges_high
    sorted_edges_low_np = sorted_edges_low
    
    # Step 2: Define total number of edges and removal schedule
    neg_total = len(sorted_edges_low_np)
    pos_total = len(sorted_edges_high_np)
    
    low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
    high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
    
    high_acc_clean = []
    low_acc_clean = []
    
    # start remove
    for index, rem_f in enumerate(high_remove_num):
        # remove second layer negative curvature edges
        net_neg = copy.deepcopy(net_H)
        net_neg.__build_remove_mask__(sorted_edges_high_np, rem_f)
        # test acc
        acc = test_clean(net_neg, test_loader)
        high_acc_clean.append(acc)

    for index, rem_f in enumerate(low_remove_num):
        # remove positive curvature edges
        net_pos = copy.deepcopy(net_H)
        net_pos.__build_remove_mask__(sorted_edges_low_np, rem_f)
        # test acc
        acc_low = test_clean(net_pos, test_loader)
        low_acc_clean.append(acc_low)
        
    plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, model_full_n+activation)


    # # Step 2: Separate edges by layer
    # sorted_edges_high_by_layer = separate_edges_by_layer_in_order(sorted_edges_high_np, prefix_dims_full)
    # sorted_edges_low_by_layer = separate_edges_by_layer_in_order(sorted_edges_low_np, prefix_dims_full)
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
            
        
    
