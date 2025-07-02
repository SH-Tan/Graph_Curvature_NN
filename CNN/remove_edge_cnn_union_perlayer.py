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
from tools.graph_curvature import graph_curvature_main_torch


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
    c = []
    neg_e = defaultdict(list)  # layer -> list of (i, j, curvature)
    pos_e = defaultdict(list)
    seen_edges = set()
    noseen = 0
    zero = 0.

    # Step 1: Collect existing curvature edges
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                curr = 1.0
            i1, j1 = int(i), int(j)
            c.append((i1, j1, curr))
            seen_edges.add((i1, j1))

    # Step 2: Assign curvature edges to per-layer groups
    c.sort(key=lambda x: x[2])
    for (i, j, curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        j_layer = np.searchsorted(prefix_dims, j, side='right') - 1

        if i_layer >= 2 and j_layer == i_layer + 1:  # ensure valid edge between adjacent FC layers
            if curr < 0:
                neg_e[i_layer].append((i, j, curr))
            else:
                pos_e[i_layer].append((i, j, curr))
                if curr == 0:
                    zero += 1

    # Step 3: Add missing FC edges with default curvature = 1
    fc_layers = [i for i in sorted(model_dims.keys()) if model_dims[i]["name"] == "fc"]
    fc_indices = [list(model_dims.keys()).index(i) for i in fc_layers]
    

    # Generate all possible FC edges
    all_fc_edges = set()
    for l in range(fc_indices[0]-1, fc_indices[-1]):
        i_layer = l
        j_layer = l + 1
        start_i, end_i = prefix_dims[i_layer], prefix_dims[i_layer + 1]
        start_j, end_j = prefix_dims[j_layer], prefix_dims[j_layer + 1]

        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                all_fc_edges.add((i, j))
                if (i, j) not in seen_edges:
                    pos_e[i_layer].append((i, j, 1.0))  # default curvature
                    noseen += 1

    return neg_e, pos_e, noseen




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



def plot_curve(neg_clean_acc, pos_clean_acc, neg_remove_num, pos_remove_num, neg_end, pos_end, label, res_path):
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(neg_remove_num, neg_clean_acc, label='Negative Edge Clean Acc', marker='o', linestyle='--')
    plt.plot(pos_remove_num, pos_clean_acc, label='Positive Edge Clean Acc', marker='x', linestyle='-')

    # Vertical lines
    plt.axvline(x=neg_end, color='red', linestyle=':', label=f'Neg End ({neg_end})')
    plt.axvline(x=pos_end, color='green', linestyle=':', label=f'Pos End ({pos_end})')

    plt.xlabel('Remove Number')
    plt.ylabel('Clean Accuracy')
    plt.title('Clean Accuracy vs Remove Number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_path + f'{label}_curve_all.png')
    plt.close()



# def plot_curve(neg_acc_clean, pos_acc_clean, neg_freq_ratios, pos_freq_ratios, neg_freq_thresholds, pos_freq_thresholds, label, save_path):
#     plt.figure(figsize=(8, 6))

#     plt.plot(neg_freq_ratios, neg_acc_clean, 'r-o', label='Negative Edge Removal')
#     plt.plot(pos_freq_ratios, pos_acc_clean, 'b-o', label='Positive Edge Removal')

#     for x, y, freq in zip(neg_freq_ratios, neg_acc_clean, neg_freq_thresholds):
#         plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, 10),
#                      ha='center', fontsize=8, color='red')

#     for x, y, freq in zip(pos_freq_ratios, pos_acc_clean, pos_freq_thresholds):
#         plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, -15),
#                      ha='center', fontsize=8, color='blue')

#     plt.xlabel("Edge Frequency Threshold (ratio × max frequency)")
#     plt.ylabel("Accuracy")
#     plt.title(f"Accuracy vs Frequency Ratio for Label {label}")
#     plt.xticks(neg_freq_ratios)  # or freq_ratios if shared
#     plt.gca().invert_xaxis()
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'_fre_curve_label_{label}.png'))
#     plt.close()



from collections import Counter
def count_edge_frequency(edge_sets):
    # edge_sets: list of dicts, each dict maps layer -> list of (i, j, curvature)
    freq_dict = defaultdict(lambda: defaultdict(int))  # layer -> (i,j) -> count
    curv_dict = defaultdict(lambda: defaultdict(list)) # layer -> (i,j) -> list of curvature values

    for edge_dict in edge_sets:  # one sample
        for layer, edges in edge_dict.items():
            for (i, j, curv) in edges:
                freq_dict[layer][(i, j)] += 1
                curv_dict[layer][(i, j)].append(curv)

    return freq_dict, curv_dict




def remove_edge_cnn_union_perlayer(args):
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
    misclassified_suffix_res = dataset + "_res_misclassified.pkl"    
    res_name = model_full_n + metric + '_' + correct_suffix_res
    misres_name = model_full_n + metric + '_' + misclassified_suffix_res

    with open(data_path + res_name, 'rb') as file:
        res_dict = pickle.load(file)    

    freq_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0]
    
    with open(res_path + "edge_cnn_" + ".txt", "a+") as ff:
        ff.write(f'For model {model_name}: \n')
        ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
        # print(f'Current label {l}: \n')
        # ff.write(f'Current label {l}: \n')

        neg_edge_sets = []
        pos_edge_sets = []
        noseen_num = []
        for l in selected_classes:
            idx = 0
            for (ricci, batch, dim, node) in res_dict[l]:
                neg_e_other, pos_e, noseen = get_top_c(ricci, 1, prefix_dims)
                noseen_num.append(noseen)
                neg_edge_sets.append(neg_e_other)
                pos_edge_sets.append(pos_e)
                idx += 1
                if idx >= sample_size:
                    break

        neg_freq_dict, _ = count_edge_frequency(neg_edge_sets)
        pos_freq_dict, _ = count_edge_frequency(pos_edge_sets)
        neg_freq_edges_sorted = {
            layer: sorted(edges.items(), key=lambda x: x[1], reverse=True)
            for layer, edges in neg_freq_dict.items()
        }
        pos_freq_edges_sorted = {
            layer: sorted(edges.items(), key=lambda x: x[1], reverse=True)
            for layer, edges in pos_freq_dict.items()
        }

        for layer in sorted(neg_freq_edges_sorted.keys() | pos_freq_edges_sorted.keys()):
            neg_acc_clean = []
            pos_acc_clean = []
        
            neg_edges = [(i, j) for (i, j), _ in neg_freq_edges_sorted.get(layer, [])]
            pos_edges = [(i, j) for (i, j), _ in pos_freq_edges_sorted.get(layer, [])]

            neg_set = set(neg_edges)
            pos_set = set(pos_edges)

            overlap = neg_set & pos_set
            overlap_count = len(overlap)

            neg_total = len(neg_edges)
            pos_total = len(pos_edges)

            neg_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
            pos_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
            
            print(f"\nLayer {layer}:")
            print(f"  Negative edges: {neg_total}")
            print(f"  Positive edges: {pos_total}")
            print(f"  Overlapping edges: {overlap_count}")
            print(f"  Neg remove nums: {neg_remove_num}")
            print(f"  Pos remove nums: {pos_remove_num}")

            ff.write(f"\nLayer {layer}:\n")
            ff.write(f"  Negative edges: {neg_total}\n")
            ff.write(f"  Positive edges: {pos_total}\n")
            ff.write(f"  Overlapping edges: {overlap_count}\n")
            ff.write(f"  Neg remove nums: {neg_remove_num}\n")
            ff.write(f"  Pos remove nums: {pos_remove_num}\n")

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

            plot_curve(neg_acc_clean, pos_acc_clean, neg_remove_num, pos_remove_num, neg_total, pos_total, layer, res_path)
            # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)
                
