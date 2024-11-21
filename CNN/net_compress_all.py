import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import random
import os
import pandas as pd
from matplotlib import pyplot as plt

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import time

from PGD_test import pgd_test
import pickle

import sys
sys.path.append("..")

import tools.utils as utils
import tools.cnn_adj_matrix as build_cnn_adj
from tools.LeNet5 import LeNet as LeNet
from tools.LeNet5_custom_v2 import LeNet_custom_v2 as LeNet_custom_v2


os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



data_train = MNIST('../data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))


file_path = "edge_v/"
model_path = "models/"
res_path = "res/"
graph_file = "graph/"

# model_path = "advtrain/models/"
# res_path = "advtrain/"

nodes_num = 9118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 5, "stride": 1, "out_size": 28}},
    3: {"name": "pooling", "dim": {"channel": 6, "kernel": 2, "stride": 2, "out_size": 14}},
    4: {"name": "cnn", "dim": {"channel": 16, "kernel": 5, "stride": 1, "out_size": 10}},
    5: {"name": "pooling", "dim": {"channel": 16, "kernel": 2, "stride": 2, "out_size":5}},
    6: {"name": "fc", "dim": {"out_size": 120}},
    7: {"name": "fc", "dim": {"out_size": 84}},
    8: {"name": "fc", "dim": {"out_size": 10}}
}


def show_results(G, curvature="ricciCurvature"):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0), reverse=True) # ascending

    curvatures = []
    # edge = list(np.array(sorted_edge)[:, 0:2])
    # edge_set = {(n1,n2) for (n1,n2) in edge}
    neg_e = 0
    edge_set = []
    
    for (n1,n2,c) in sorted_edge:
        weights.append(c["weight"])
        curvatures.append(c[curvature])
        # edge_set.add((n1, n2))
        if (c[curvature] > 0):
            neg_e += 1
            edge_set.append((n1, n2))
        else:
            break
            
    top_10_percent_count = int(len(edge_set) * 0.02)
    edge_set = edge_set[:top_10_percent_count]
            
    # np_curvature = np.array(curvatures)

    # avg_w = weights/neg_e if neg_e > 0 else 0.
    # avg_c = curvatures/neg_e if neg_e > 0 else 0.

    return edge_set


def test(n, loader):
    n.eval()
    total_correct = 0
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    
    acc = float(total_correct) / len(loader.dataset)
    return acc


# def cal_partition(partition, f):
#     communities = set(partition.values())
#     n = len(communities)
#     f.write(f'Total {n} partitions...\n')

#     nodes_per_c_n = np.zeros((n), dtype=np.int32)

#     nodes_per_c_dict = {}
#     for i in range(n):
#         nodes_per_c_dict[i] = []
        
#     for k, v in partition.items():
#         nodes_per_c_n[v] += 1
#         nodes_per_c_dict[v].append(k)
        
#     for i in range(n):
#         if nodes_per_c_n[i] > 1:
#             f.write(f'Partition {i} has {nodes_per_c_n[i]} nodes.\n')
#             f.write(f'Partition {i} contains nodes {nodes_per_c_dict[i]}.\n\n')
#     f.write('\n')


# for directed graph
def cal_partition(partition, f):
    communities = set(partition)
    n = len(communities)
    f.write(f'Total {n} partitions...\n')

    for com in communities:
        if (len(com) > 5):
            f.write(f'This partition has {len(com)} nodes...\n')
            max_partition = np.array(list(com))
            cur_l = 1
            pre_n = 0
            
            while cur_l <= len(model_dims):
                cur_name = model_dims[cur_l]["name"]
                cur_dim = model_dims[cur_l]["dim"]
                cur_size = cur_dim['out_size']

                cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

                if cur_name == "input":
                    cur_nodes = cur_channel * cur_size**2
                else:
                    cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)
                
                cur_l_n = max_partition[max_partition >= pre_n]
                cur_l_n = cur_l_n[cur_l_n < pre_n+cur_nodes]
                
                f.write(f'The current layer is {cur_l}, it has {len(cur_l_n)} nodes...\n')
                cur_l += 1
                pre_n += cur_nodes
            
            f.write('\n')


if __name__ == '__main__':
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, valid_num=5000)

    # valid_dataloader = utils.sep_label(valid_dataset, selected_classes)
    # sep_dataloader = utils.sep_label(test_dataset, selected_classes)

    
    # build model
    # origin model
    model_name= "mnist_relu.pth"
    # model_name= "pgdtrain_lenet.pth"
    # model_name= "mnist_tanh.pth"
    model = LeNet()
    model.load_state_dict(torch.load(model_path + model_name))
    model = model.to(device)
  
    with open(res_path + "dirnet_comp_relu_1over.txt", "w+") as f:     
        real_acc = []
        adv_acc = []
        sub_real_acc = []
        sub_adv_acc = []
        model_size = []
        var_weight = []
        var_cur = []
        
        # use for remove edge test
        net_H = LeNet_custom_v2(model_dims, None, device)
        net_H.load_state_dict(torch.load(model_path + model_name))
        net_H = net_H.to(device)

        ori_acc = test(model, test_loader)
        flag = 1
        
        # original model
        acc_full, sub_real, sub_adversary, full_adv = pgd_test(model, model, test_loader, f, eps=.1, alpha=.1, iters=100)
        
        real_acc.append(acc_full)
        adv_acc.append(full_adv)
        sub_real_acc.append(sub_real)
        sub_adv_acc.append(sub_adversary)
        model_size.append(round(1, 2))
        
        # file_n = file_path + "full_edge_v.csv"
        
        # net_H.load_state_dict(torch.load(res_path + model_name))
        adjacent_m, total_e,_ = build_cnn_adj.build_cnn_adj(nodes_num, model_dims, net_H, valid_loader, device)   
        
        # indices = np.where(adjacent_m == 0)

        # # Convert indices to list of tuples for better readability
        # zero_edges = set(list(zip(indices[0], indices[1])))
        # # zero_edges = set([(x, y) for x, y in index_tuples if x < y])
        
        # # add remove zero edges
        # net_H.__build_remove_mask__(zero_edges)
        # print()
        
        # cur_acc = test(net_H, test_loader)
        
        # original model
        acc_full, sub_real, sub_adversary, full_adv = pgd_test(model, net_H, test_loader, f, eps=.1, alpha=.1, iters=100)
        
        real_acc.append(acc_full)
        adv_acc.append(full_adv)
        sub_real_acc.append(sub_real)
        sub_adv_acc.append(sub_adversary)
        model_size.append(1)

        f.write(f'The origin acc is {ori_acc:.3f}...')
        f.write('\n')
        
        total_edge = total_e
        
        while(flag):            
            # Create network object
            # adjacent_m[adjacent_m == -1] = 0.
            G = nx.from_numpy_array(adjacent_m, create_using=nx.DiGraph)

            orf = OllivierRicci(G, alpha=0., verbose="ERROR")
            # orf.compute_ricci_flow(iterations=100)
            orf.compute_ricci_curvature()

            G1 = orf.G.copy()
            f.write(f'Start with {G1}.... Total edges are {total_e}... \n\n')
            print(G1)
            
            partition = nx.community.greedy_modularity_communities(G1)
            cal_partition(partition, f)
            
            # with open(res_path + graph_file + 'graph' + str(step) + '.pkl', 'wb') as file:
            #     pickle.dump(G1, file)
            
            cur_edge = len(G1.edges())
            
            edge_set = show_results(G1, "ricciCurvature")

            # add remove edge set
            net_H.__build_remove_mask__(edge_set)
            
            cur_acc = test(net_H, test_loader)

            f.write(f'Test Accuracy after remove {len(edge_set)} edges: {cur_acc:.3f}...\n')
            f.write("\n") 
            
            # or (len(sub_adv_acc)> 0 and sub_adversary < sub_adv_acc[-1]) 
            
            if len(edge_set)  == 0 or (cur_acc < 0.7):
                flag = 0
            else:
                adjacent_m, total_e,_ = build_cnn_adj.build_cnn_adj(nodes_num, model_dims, net_H, valid_loader, device) 
                
                # indices = np.where(adjacent_m == 0)

                # # Convert indices to list of tuples for better readability
                # zero_edges1 = set(list(zip(indices[0], indices[1])))
                
                # # add remove zero edges
                # removed_e = zero_edges1 - zero_edges - edge_set
                # net_H.__build_remove_mask__(removed_e)
                # zero_edges = zero_edges1
                # print()
                
            f.write('\n')
            f.write('*'*50 + 'PGD test' + '*'*50)
            f.write('\n')
            f.write('\n')
                
            acc_full, sub_real, sub_adversary, full_adv = pgd_test(model, net_H, test_loader, f, eps=.1, alpha=.1, iters=100)

            real_acc.append(acc_full)
            adv_acc.append(full_adv)
            sub_real_acc.append(sub_real)
            sub_adv_acc.append(sub_adversary)
            model_size.append(round((cur_edge - len(edge_set))/total_edge, 3))

            # f.write('\n')
            # f.write('='*100)
            # f.write('\n')
            # f.write('\n')
        # f.write('\n')
        # f.write(f'{model_size}')
        
        f.write('\n')
        f.write('='*100 + '\n\n')
        utils.plot_acc(real_acc, adv_acc, "LeNet5_w0_dir_relu_1over_0.1", res_path, sub_real_acc, sub_adv_acc, model_size)
                    
                        
                    