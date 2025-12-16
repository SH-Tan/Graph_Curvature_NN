import torch
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import copy

import pickle
import time
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils
from tools.graph_curvature_cnn_threshold_optimized import graph_curvature_main_torch


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 2, "stride": 2, "padding":0, "out_size": 16}}, 
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 14}},   # After conv1_2 
    
    4: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 12}},  
    5: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 2, "padding":0, "out_size": 5}},   # After conv2_2 
    
    6: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 3}},
    7: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 1}},

    8: {"name": "fc", "dim": {"out_size": 512}},  # Flatten(512×2×2) → 1024
    9: {"name": "fc", "dim": {"out_size": 128}},
    10: {"name": "fc", "dim": {"out_size": 10}}
}


model_dims_small = model_dims

# model_dims_small = {
#     1: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 10}},   # After conv4_2
#     2: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 8}},   # After conv4_3

#     3: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 6}},   # conv5_1
#     4: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 4}},   # conv5_2 
#     5: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "pool":False, "out_size": 2}},   # conv5_3 

#     6: {"name": "fc", "dim": {"out_size": 512}},  # Flatten(512×2×2) → 2048
#     7: {"name": "fc", "dim": {"out_size": 256}},
#     8: {"name": "fc", "dim": {"out_size": 10}}
# }

selected_classes = [0,1,2,3,4,5,6,7,8,9]


def load_dataset_from_disk(path, batch_size=128, shuffle=True):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    return dataset



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



def test(n, loader, device):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
 
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            robust_l = pred.eq(labels.view_as(pred))
            succ_l = ~pred.eq(labels.view_as(pred))

            succ_pair[l].append((images[succ_l].cpu(), pred[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), pred[robust_l].cpu()))

    return succ_pair, robust_pair



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




def community_check_cifar_vgg9(args):
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

    # train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    val_set = load_dataset_from_disk("./data/CIFAR10_val", batch_size=64, shuffle=False)
    sep_dataloader = utils.sep_label(val_set, selected_classes, bs=32)
    
    dims_full = cal_dims(model_dims)
    dims = cal_dims(model_dims_small)
    edge_dims = cal_edges(model_dims_small)

    print(dims_full)
    print(dims)
    print(edge_dims)

    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    hops = args.hops
    sample_size = args.sample_num
    activation = args.activation
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    if activation.lower() == "relu":
        from tools.vgg9_custom_relu import VGG9_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg9_custom_tanh import VGG9_CIFAR10

    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "vgg9_10_ori_"
    elif model_pre_name == 'adv':
        model_name = "vgg9_10_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg9_10_wd_"
        
    model_name = model_name + activation + "_s2_new.pth"
    
    net_H = VGG9_CIFAR10(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)

    succ_pair, robust_pair = test(net_H, sep_dataloader, device=device)
    res_l = defaultdict(list)

    print("Finished loading model and test data..")
    
    # Track how many samples have been processed for each class
    label_progress = defaultdict(int)

    # Prepare iterators for each label's data
    data_iterators = {l: iter(robust_pair[l]) for l in selected_classes}
    finished_labels = set()
    round_id = 1  # Track how many full batches have been saved

    while len(finished_labels) < len(selected_classes):
        finished_l = 0
        for l in selected_classes:
            finished_l += 1
            if label_progress[l] >= sample_size:
                finished_labels.add(l)
                continue

            num_needed = min(1, sample_size - label_progress[l])  # Process up to 10 per round
            current_count = 0

            try:
                while current_count < num_needed:
                    images, labels = next(data_iterators[l])

                    for idx in range(images.shape[0]):
                        if label_progress[l] >= sample_size:
                            finished_labels.add(l)
                            break

                        if current_count >= num_needed:
                            break

                        if (label_progress[l] % 1 == 0):
                            print(f'Label {l}: finish {label_progress[l]} examples...')

                        with torch.no_grad():
                            net_full.eval()
                            img = images[idx].to(device, non_blocking=True)
                            edge_array, nodes_ori, output, node_before, node_alpha = net_full.NN_info_batch(img.unsqueeze(0))

                            weights = output.detach().cpu().to(device) 
                            nodes_ori = nodes_ori.detach().clone().to(device) 
                            edge_array = edge_array.detach().clone().cpu()
                            node_alpha = node_alpha.detach().clone().cpu()
                            node_before = node_before.detach().clone().cpu()
                            
                            # a = nodes_ori[:,3072:-1280]
                            # b = node_alpha[:,3072:-1280]
                            # a[b < 0.5] = 0
                            # nodes_ori[:,3072:-1280] = a
                            # nodes_ori[node_alpha <= 0.5] = 0.
                            
                            del output, img
                            torch.cuda.empty_cache()
                            
                            if metric.lower() == "w1":
                                weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims, model_dims_small)
                                weights_inv = weights_inv1.detach()
                                weights_inv2 = weights_inv2.detach()
                                ricci_curvature = graph_curvature_main_torch(
                                    dims, weights_inv, device=device,
                                    model_dims=model_dims_small,
                                    probability_w=weights_inv2, alpha=alpha
                                )
                            elif metric.lower() == "w3":
                                weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims, model_dims_small)
                                weights_inv = weights_inv1.detach()
                                weights_inv2 = weights_inv2.detach()
   
                                ricci_curvature = graph_curvature_main_torch(
                                    dims, weights_inv, device=device,
                                    model_dims=model_dims_small,
                                    probability_w=weights_inv2, alpha=alpha,
                                    pre_n=(np.sum(dims_full) - np.sum(dims)),
                                    layers_to_process=[1,2,3]
                                )
                            elif metric.lower() == "w4":
                                weights_inv1, weights_inv2= net_full.normalization_weight_w4(nodes_ori, weights, dims, model_dims_small)
                                
                                # Move back to CPU immediately to save GPU RAM
                                weights_inv = weights_inv1.detach().cpu()
                                weights_inv_p = weights_inv2.detach().cpu()
                                node_abs = torch.abs(nodes_ori)
                                edge_array_abs = torch.abs(edge_array)

                                del weights, weights_inv1, weights_inv2
                                torch.cuda.empty_cache()

                                weight_idx = 0
                                start = 0
                                combined = []
                                for l_key in [0,1,2,3,4,5,[6,7,8]]:  # loop variable is l_key
                                    print(f'Current label {l} - l_key {l_key}.....')
                                    # Determine weight index slice
                                    weight_idx = min(l_key) - 1 if isinstance(l_key, list) else l_key - 1
                                    weight_idx = max(0, weight_idx)
                                    start = np.sum(edge_dims[0:weight_idx]) if weight_idx > 0 else 0

                                    # Determine number of edges to include
                                    num_edges = len(l_key) if isinstance(l_key, list) else 1
                                    end = start + np.sum(edge_dims[weight_idx: weight_idx + num_edges + 2])
                                    
                                    # Move only the current slice to GPU
                                    w_inv_slice = weights_inv[:, start:end].to(device, non_blocking=True)
                                    w_inv2_slice = weights_inv_p[:, start:end].to(device, non_blocking=True)
                                    edge_slice = edge_array[:, start:end].to(device, non_blocking=True)

                                    # Compute Ricci curvature for current layer(s)
                                    ricci_results = graph_curvature_main_torch(
                                        dims,
                                        w_inv_slice,
                                        device=device,
                                        model_dims=model_dims_small,
                                        probability_w=w_inv2_slice,
                                        alpha=alpha,
                                        pre_n=(np.sum(dims_full) - np.sum(dims)),
                                        nodes=node_abs,  # stays on CPU, passed as reference
                                        edge_value=edge_slice,
                                        threshold=0.,
                                        layers_to_process=list(l_key) if isinstance(l_key, list) else [l_key],
                                        nodes_alpha = abs(node_alpha),
                                    )
                                    
                                    # with open(res_path + f"output_{l_key}.txt", "a+") as f:
                                    #     for i, (batch_key, triples) in enumerate(ricci_results.items()):
                                    #         for i, j, lm, ln, m, sp, curv in triples:
                                    #             f.write(f'edge {i}-{j}: \n')
                                    #             f.write(f"in-neighbor number: {lm}, out-neighbor number {ln}\n")
                                    #             f.write(f"m = {m:.6f}, sp = {sp:.6f}, {curv:.6f}\n")
                                    #         f.write("\n")
                                            
                                    # print("finished")
                                            
                                    
                                    for batch_key, triples in ricci_results.items():
                                        combined.extend(triples)   # append all (i, j, val) tuples
                                        
                                     # Free per-loop tensors
                                    del w_inv_slice, w_inv2_slice, edge_slice, ricci_results
                                    torch.cuda.empty_cache()
                            else:
                                raise Exception("Invalid graph metric, should be {w1, w3, w4}!")

                            # === Save one sample per file ===
                            save_name = f"{model_full_n}_{metric}_{dataset}_label{l}_id{label_progress[l]}_round{round_id}.pkl"
                            save_path = os.path.join(res_path, save_name)

                            with open(save_path, 'wb') as f:
                                pickle.dump(combined, f)

                            print(f"[Saved] Label {l}, Example {label_progress[l]} → {save_name}")

                            # Final cleanup
                            del edge_array, nodes_ori, node_abs, edge_array_abs, combined, node_before, node_alpha
                            torch.cuda.empty_cache()

                            label_progress[l] += 1
                            current_count += 1

            except StopIteration:
                finished_labels.add(l)
                continue

        # === Check if all labels have collected 10 new samples ===
        if all(len(res_l[l]) == 10 for l in selected_classes if label_progress[l] < sample_size) or finished_l >= len(selected_classes):
            round_id += 1