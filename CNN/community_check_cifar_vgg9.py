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
from .e2w_utils_new import *

import sys
sys.path.append("..")

import tools.utils as utils
from tools.graph_curvature_cnn_threshold_combinemag import graph_curvature_main_torch


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


def build_cnn_map(
    prefix_dims,
):
    # Precompute edge->weight mapping per CNN layer
    cnn_edge_to_weight_map = {}
    for layer in range(len(model_dims)-2):  # skip input/output placeholder layers
        layer_info = model_dims[layer+2]
        if layer_info["name"] == "cnn":
            pre_dim = model_dims[layer+1]["dim"]
            pre_ch = pre_dim["channel"]
            in_size = pre_dim["out_size"]
            cur_dim = layer_info["dim"]
            cur_ch = cur_dim["channel"]
            kernel = cur_dim["kernel"]
            stride = cur_dim["stride"]
            padding = cur_dim["padding"]

            cnn_edge_to_weight_map[layer] = build_cnn_edge_weight_map(
                pre_ch, in_size, cur_ch, kernel, stride, padding, layer, prefix_dims
            )
    return cnn_edge_to_weight_map



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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    val_set = load_dataset_from_disk("./data/CIFAR10_val", batch_size=64, shuffle=False)
    sep_dataloader = utils.sep_label(val_set, selected_classes, bs=32, seed=seed)
    
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
        
    model_name = model_name + activation + "_s2.pth"
    
    net_H = VGG9_CIFAR10(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
    
    # model_name_pruned = "vgg9_10_ori_" + activation + "_s2_pruned.pth"
    # net_H_pruned = VGG9_CIFAR10(model_dims, None, device)
    # net_H_pruned.load_state_dict(torch.load(model_path + model_name_pruned))
    # net_H_pruned = net_H_pruned.to(device)
    # net_H_pruned.init_remove_mask()
 
    prefix_dims = np.cumsum([0] + dims_full).tolist()
    
    # remove_mask = net_H_pruned.remove_mask
    # cnn_edge_to_weight_map = build_cnn_map(prefix_dims)
    
    # # Precompute allowed edges for each layer
    # # edge_allowed[layer][src, dst] = True if weight is non-zero, False if masked
    # edge_allowed = {}

    # for layer in range(len(dims)-1):
    #     model_dim_j = model_dims[layer+2]

    #     if remove_mask is None or cnn_edge_to_weight_map is None:
    #         edge_allowed[layer] = None
    #         continue

    #     if model_dim_j["name"] == "fc":
    #         # Move once to CPU and convert to boolean array
    #         edge_allowed[layer] = remove_mask[layer+1].cpu().numpy().astype(bool)
    #     elif model_dim_j["name"] == "cnn":
    #         # Move CNN mask to CPU once
    #         cnn_m_cpu = remove_mask[layer+1].cpu()
    #         allowed = {}
    #         for (global_src, global_dst), (l, oc, ic, kh, kw) in cnn_edge_to_weight_map[layer].items():
    #             # Access CPU tensor and convert to Python bool
    #             allowed[(global_src, global_dst)] = bool(cnn_m_cpu[oc, ic, kh, kw].item())
    #         edge_allowed[layer] = allowed
            
    # # Clear references
    # remove_mask = None
    # cnn_edge_to_weight_map = None

    # If they were on GPU, free GPU memory
    torch.cuda.empty_cache()

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

    log_file = res_path + "time_profile.txt"
    sp_dict = None
    
    with open(log_file, "a+") as f:
        t_start = time.time()
        while len(finished_labels) < len(selected_classes):
            finished_l = 0
            for order, l in enumerate(selected_classes):
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
                                net_H.eval()
                                img = images[idx].to(device, non_blocking=True)
                                edge_array, nodes_ori, output, nodes_before = net_H.NN_info_batch(img.unsqueeze(0))
                                # net_H_pruned.eval()
                                # _, _, output1, _ = net_H_pruned.NN_info_batch(img.unsqueeze(0))
                                
                                weights = output.detach().clone().to(device)
                                nodes_ori = nodes_ori.detach().clone().to(device)
                                # node_before = nodes_before.detach().clone().cpu()
                                # weights1 = output1.detach().clone().to(device)
                                
                                # weights[abs(weights)<=1e-30] = 1e-30
                                
                                # print(len(weights[weights==0]), len(weights[0]), torch.min(weights))
    
                                edge_array = edge_array.detach().clone().cpu()
                                
                                del output, img, nodes_before,output1
                                torch.cuda.empty_cache()
                                
                                if metric.lower() == "w1":
                                    weights_inv1, weights_inv2 = net_H.normalization_weight_w1(nodes_ori, weights, dims, model_dims_small)
                                    weights_inv = weights_inv1.detach()
                                    weights_inv2 = weights_inv2.detach()
                                    ricci_curvature = graph_curvature_main_torch(
                                        dims, weights_inv, device=device,
                                        model_dims=model_dims_small,
                                        probability_w=weights_inv2, alpha=alpha
                                    )
                                elif metric.lower() == "w3":
                                    weights_inv1, weights_inv2 = net_H.normalization_weight_w3(nodes_ori, weights, dims, model_dims_small)
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
                                    weights_inv1, weights_inv2 = net_H.normalization_weight_w4(nodes_ori, weights, dims, model_dims_small)
                                    
                                    # Move back to CPU immediately to save GPU RAM
                                    weights_inv = weights_inv1.detach().cpu()
                                    weights_inv_p = weights_inv2.detach().cpu()
                                    node_abs = torch.abs(nodes_ori)
                                    # edge_array_abs = torch.abs(edge_array)

                                    del weights, weights_inv1, weights_inv2, nodes_ori
                                    torch.cuda.empty_cache()
                                    
                                    weight_idx = 0
                                    start = 0
                                    combined = []
                                    
                                    torch.cuda.synchronize(device)
                                    t1 = time.time()
                                    f.write("\n===== New Run =====\n")
                                    f.write("layer | total time(s) | dist time(s) | matrix time(s) | curv time(s) | alloc(MB) | reserved(MB) | peak(MB)\n")

                                    if order == 0:
                                        sp_dict = utils.cnn_layerwise_shortest_path_torch(model_dims, weights_inv.to(device), prefix_dims, device='cuda')
                                        sp_dict = {k: v.cpu() for k, v in sp_dict.items()}
                                    torch.cuda.empty_cache()
                                    
                                    dis_time = time.time() - t1
                                    
                                    print(f'Finish distance matrix')
                                     
                                    for l_key in [0,1,2,3,4,5,6,7,8]:  # loop variable is l_key
                                        print(f'Current label {l} - l_key {l_key}.....')
                                        
                                        # 🔹 reset peak stats for this layer
                                        torch.cuda.reset_peak_memory_stats(device)
                                        torch.cuda.synchronize(device)
                                        iter_start = time.time()
    
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
                                        # output1_slice = weights1[:, start:end].to(device, non_blocking=True)
                                        # edge_slice_noninv = edge_array[:, start:end].to(device, non_blocking=True)

                                        # Compute Ricci curvature for current layer(s)
                                        ricci_results, matrix_t, curv_t = graph_curvature_main_torch(
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
                                            # edge_allowed = torch.abs(output1_slice), 
                                            sp_dict = sp_dict,
                                        )
                        
                                        for batch_key, triples in ricci_results.items():
                                            combined.extend(triples)   # append all (i, j, val) tuples
                                            
                                        # Free per-loop tensors
                                        del w_inv_slice, w_inv2_slice, edge_slice, ricci_results, output1_slice
                                        torch.cuda.empty_cache()
                                            
                                        # 🔹 timing end
                                        torch.cuda.synchronize(device)
                                        iter_time = time.time() - iter_start

                                        # 🔹 memory stats
                                        alloc = torch.cuda.memory_allocated(device) / 1024**2
                                        reserved = torch.cuda.memory_reserved(device) / 1024**2
                                        peak = torch.cuda.max_memory_allocated(device) / 1024**2

                                        # 🔹 write per-layer log
                                        f.write(f"{l_key:5d} | {iter_time:13.3f} | {dis_time:13.3f} | {matrix_t:13.3f} | {curv_t:13.3f} | {alloc:9.1f} | {reserved:11.1f} | {peak:8.1f}\n")
       
                                    # ✅ total time
                                    torch.cuda.synchronize(device)
                                    t2 = time.time()
                                    total_time = t2 - t1

                                    f.write(f"\nTOTAL TIME for ONE EXAMPLES: {total_time:.3f} s, label = {l}\n")
                                else:
                                    raise Exception("Invalid graph metric, should be {w1, w3, w4}!")

                                # === Save one sample per file ===
                                save_name = f"{model_full_n}_{metric}_{dataset}_label{l}_id{label_progress[l]}_round{round_id}.pkl"
                                save_path = os.path.join(res_path, save_name)

                                with open(save_path, 'wb') as ff:
                                    pickle.dump(combined, ff)

                                print(f"[Saved] Label {l}, Example {label_progress[l]} → {save_name}")

                                # Final cleanup
                                del edge_array, node_abs, combined, weights1
                                torch.cuda.empty_cache()

                                label_progress[l] += 1
                                current_count += 1

                except StopIteration:
                    finished_labels.add(l)
                    continue

            # === Check if all labels have collected 10 new samples ===
            if all(len(res_l[l]) == 10 for l in selected_classes if label_progress[l] < sample_size) or finished_l >= len(selected_classes):
                round_id += 1
                t_end = time.time()
                total_time = t_end - t_start

                f.write(f"\nTOTAL TIME for 10 EXAMPLES: {total_time:.3f} s\n")