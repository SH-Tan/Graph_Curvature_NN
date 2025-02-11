import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import time

import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import community as community_louvain
import statsmodels.api as sm
from scipy.integrate import simps
import pickle

import sys
sys.path.append("..")

import tools.utils as utils
from RicciCurvature.OllivierRicci import OllivierRicci
from tools.LeNet5_custom_small import LeNet_custom_v2 as LeNet_custom_v2
from tools.graph_curvature import graph_curvature_main_torch


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



def show_results(G, curvature="ricciCurvature"):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[0], reverse=False) # ascending
    curvatures = []
    neg_e = 0
    edge_set = set()
    remain_edges = set()
    
    for (n1,n2,c) in sorted_edge:
        
        weights.append(c["weight"])
        curvatures.append(c[curvature])
        # edge_set.add((n1, n2))

        if (c[curvature] < 0):
            neg_e += 1
            edge_set.add((n1, n2))
        else:
            remain_edges.add((n1, n2))


    return edge_set, remain_edges, weights, curvatures


def standard_PGD(model, images, labels, device, eps=11/255, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


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
            
            adv_img = standard_PGD(n, images, labels, device,  eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(pred)
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
            
            succ_pair[l].append((images[succ_l].cpu(), adv_img[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), adv_img[robust_l].cpu()))
            
        print(f'Finish label {l}....')
    return succ_pair, robust_pair

def get_fraction(curvature, b):
    c = []
    neg = []
    total_e = []
    for i in range(b):
        curr = np.array(curvature[i])
        neg.append(len(curr[curr<0]))
        total_e.append(len(curr))
    return np.array(neg), np.array(total_e), curr


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

    

def build_cnn_adj(nodes_num, model_dims, net, im, device, metric):
    # raw value: edge weights
    net.eval()
    d = im.to(device)
    
    edge_array, nodes = net.edge_w_batch(d)
    edge_array = edge_array.cpu().detach().numpy() 
    
    if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
        output = net.get_weights(d)
        output = output.cpu().detach().numpy() 
        output[edge_array == 0] = 0.
        
    elif metric.lower() == "q_exp":
        output = edge_array
    else:
        raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
    
    w_avg = np.mean(output, axis=0)
    
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

    current_l = 1
    start_col = 0
    cur_s_col = 0
    nodes_total = 0

    layer_num = len(model_dims)

    for i in range(1, layer_num):
        current_l = i
        next_l= i + 1

        cur_name = model_dims[current_l]["name"]
        cur_dim = model_dims[current_l]["dim"]
        cur_size = cur_dim['out_size']

        cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)

        nxt_name = model_dims[next_l]["name"]
        nxt_dim = model_dims[next_l]["dim"]
        out_size = nxt_dim['out_size']

        # cnn layer
        if (nxt_name == "cnn" or nxt_name == "pooling"):
            k = nxt_dim['kernel']
            s = nxt_dim['stride']
            c = nxt_dim['channel']
            
            step = k**2      
            n = 0
            tensor_2d = torch.arange(cur_nodes).reshape(1,cur_channel,cur_size,cur_size).float()
            
            if (nxt_name == "cnn"):
                indices = F.unfold(tensor_2d, (k,k), stride = s).transpose(1,2).int()
                end_col = start_col + step*cur_channel
                for ur_c in range(c): 
                    for l in range(indices.shape[1]):
                        cur_idx = indices[0,l] + nodes_total 
                        cur_idx = cur_idx.tolist()
                        assert(len(cur_idx) == step*cur_channel)
                        adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]
            
                        start_col = end_col
                        end_col = start_col + step*cur_channel
                        n += 1
            else:
                indices = F.unfold(tensor_2d, (k,k), stride = s)
                i_unf = indices.view(1, cur_channel, k*k, -1).transpose(2,3)
                indices = i_unf.reshape(i_unf.shape[0], i_unf.shape[1]*i_unf.shape[2], i_unf.shape[3]).int()
                
                for l in range(indices.shape[1]):
                    end_col = start_col + step
                    cur_idx = indices[0,l] + nodes_total 
                    cur_idx = cur_idx.tolist()
                    assert(len(cur_idx) == step)
                    adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]

                    start_col = end_col
                    end_col = start_col + step
                    n += 1
                    
            nodes_total += cur_nodes
            # print(start_col)  

        # fc layer
        elif (nxt_name == "fc"):  
            cur_s_col = nodes_total + cur_nodes
            cur_e_col = nodes_total + cur_nodes + out_size

            end_col = start_col + out_size

            for node in range(nodes_total, nodes_total + cur_nodes, 1):
                # print(f'i : {node}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
                adjacent_m[node, cur_s_col : cur_e_col] = w_avg[start_col : end_col]        
                
                start_col = end_col
                end_col = end_col + out_size            
                    
            nodes_total += cur_nodes
        # print(nodes_total)
        
    return adjacent_m, nodes



'''
parameters:
    @ q_NGR, q_INV, q_EXP     --G_Def
    @ model type/name: fc, cnn   --model_type
    @ model path   --model_path
    @ result path  --res_path
    @ example num    --sample_num

'''
def cnn_main(args):
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
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, valid_num = 2000, test_bs=1)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    sample_size = args.sample_num
    alpha = args.alpha
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    # build model
    model_name= "mnist_relu_small.pth"
    
    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
    
    print(model_name)
    
    eps = [0.03, 0.07, 0.1, 0.2]
    Q = [1]
    # eps = [0.1]
    
    for q in Q:
        for e in eps:
            succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)
    
            robust_c = defaultdict(list)
            nonrobust_c = defaultdict(list)
            non_fraction = defaultdict(list)
            rob_fraction = defaultdict(list)
            
            for l in selected_classes:
                print(f'For label {l}....\n')
                # nonrobust images
                count = 0
                for (ori_im, adv_im) in succ_pair[l]:
                    for im in ori_im:
                        img = im.to(device)
                        edge_array, nodes_ori, output = net_H.NN_info_batch(img.unsqueeze(0))
                        
                        if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
                            weights = output.detach().clone().to(device)                   
                            weights[edge_array == 0] = 0.
                            
                        elif metric.lower() == "q_exp":
                            weights = edge_array.detach().clone().to(device)  
                        
                        if metric.lower() == "q_ngr":
                            weights_inv, weights_inv2 = net_H.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
                            weights_inv = weights_inv.detach()
                            weights_inv2 = weights_inv2.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, model_dims=model_dims, alpha=alpha)
                            
                        elif metric.lower() == "q_inv":
                            weights_inv = net_H.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)
            
                        elif metric.lower() == "q_exp":
                            weights_inv = net_H.normalization_weight_w6(nodes_ori, weights, dims, model_dims, q)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)
                        else:
                            raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                        
                        neg_num, total_edge, c = get_fraction(ricci_curvature, weights_inv.shape[0])
                    
                        nonrobust_c[l].append(c)
                        non_fraction[l].append(neg_num/total_edge)
                        
                        count += 1
                        if (count % 10 == 0):
                            print(f'Finish {count} graphs....')
                            
                        if (count >= sample_size):
                            break
          
                count = 0
                for (ori_im, adv_im) in robust_pair[l]:
                    for im in ori_im:
                        img = im.to(device)
                        edge_array, nodes_ori, output = net_H.NN_info_batch(img.unsqueeze(0))
                        
                        if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
                            weights = output.detach().clone().to(device)                   
                            weights[edge_array == 0] = 0.
                            
                        elif metric.lower() == "q_exp":
                            weights = edge_array.detach().clone().to(device)  
                        
                        if metric.lower() == "q_ngr":
                            weights_inv, weights_inv2 = net_H.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
                            weights_inv = weights_inv.detach()
                            weights_inv2 = weights_inv2.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, model_dims=model_dims, alpha=alpha)
                            
                        elif metric.lower() == "q_inv":
                            weights_inv = net_H.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)
            
                        elif metric.lower() == "q_exp":
                            weights_inv = net_H.normalization_weight_w6(nodes_ori, weights, dims, model_dims, q)
                            weights_inv = weights_inv.detach()
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)
                        else:
                            raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
    
                        neg_num, total_edge, c = get_fraction(ricci_curvature, weights_inv.shape[0])
        
                        robust_c[l].append(c)
                        rob_fraction[l].append(neg_num/total_edge)
                        
                        count += 1
                        if (count % 10 == 0):
                            print(f'Finish {count} graphs....')
                            
                        if (count >= sample_size):
                            break
            
            with open(res_path + model_full_n + str(e) + metric + str(q) + "frac_robust_cnn.pkl", 'wb') as file:
                pickle.dump(rob_fraction, file)
            with open(res_path + model_full_n + str(e) + metric + str(q) +  "frac_norobust_cnn.pkl", 'wb') as file:
                pickle.dump(non_fraction, file)
                
            with open(res_path + model_full_n + str(e) + metric + str(q) + "curv_robust_cnn.pkl", 'wb') as file:
                pickle.dump(robust_c, file)
            with open(res_path + model_full_n + str(e) + metric + str(q) + "curv_norobust_cnn.pkl", 'wb') as file:
                pickle.dump(nonrobust_c, file)
