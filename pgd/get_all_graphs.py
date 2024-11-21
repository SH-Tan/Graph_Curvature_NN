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
from tools.small_model import FC_MD

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx



def get_adj_m(dims, nodes_num, abs_weights):
    adjacent_m_l = []

    for (l, sca_w) in abs_weights:
        # build adjacent matrix
        adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

        layer_num = len(dims)
        cur_s_col = dims[0]
        cur_e_col = dims[0] + dims[1]

        cur_layer = 1
        start_col = 0
        end_col = dims[1]

        for i in range(nodes_num - dims[layer_num-1]):
            # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
            adjacent_m[i, cur_s_col : cur_e_col] = sca_w[start_col : end_col]
            
            if (cur_layer < layer_num-1 and i == cur_s_col - 1):
                cur_layer += 1
                start_col = end_col
                end_col = end_col + dims[cur_layer]
                cur_s_col = cur_e_col
                cur_e_col = cur_e_col + dims[cur_layer]
            else:
                start_col = end_col
                end_col = end_col + dims[cur_layer]
                
        adjacent_m_l.append((l, adjacent_m))
        
    return adjacent_m_l



def show_results(G, num, curvature="ricciCurvature", thre = 0.95):
    
    edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get('ricciCurvature', 0), reverse = True)
    edge = list(np.array(edge)[:, 0:2])
    # my_set = {(n1,n2) for (n1,n2) in edge}

    edge_set = set()
    
    # Print the first five results
    index = 0
    for (n1,n2) in edge:
        if (G[n1][n2][curvature] > thre):
            edge_set.add((n1, n2))
            index += 1
            if (index > num):
                break
    return edge_set


# build mask
def build_mask(mask, edge_set, dims):
    
    layer_num = len(dims) - 1
    
    for (n1, n2) in edge_set:
        cur_l = 0
        index = 0
        
        while (cur_l < layer_num and (n1 > (dims[cur_l]-1))):
            n1 -= dims[cur_l]
            n2 -= dims[cur_l]
            cur_l += 1
            index += (dims[cur_l] * dims[cur_l-1])
            
        assert(n2 >= 0)
        while (cur_l < layer_num and (n2 > (dims[cur_l]-1))):
            n2 -= dims[cur_l]
            cur_l += 1
            
        index += (n1*dims[cur_l] + n2)
        mask[:, index] = 1.
    
    return mask



def get_graph(net, dims, selected_classes, file_path):
    neural_list = []
    nodes_num = 0
    edges_num = 0
    i = 0
    for p in net.parameters():
        if i == 0:
            nodes_num += p.shape[1]
        if i%2 == 0:
            nodes_num += p.shape[0]
            edges_num += (p.shape[0] * p.shape[1])
            neural_list.append(p.shape[0])
        i += 1
        
    print(f'Total nodes are {nodes_num}, total edges are {edges_num}.')
    
    
    avg_edge_w = []
    layer_num = len(dims) - 2

    for l in selected_classes:
        file_n = "full_edge_v" + str(l) +".csv"
        # raw value: edge weights
        edge_weights = pd.read_csv(file_path + file_n, index_col=0)
        # cols = edge_weights.columns # number of neurals
        e_weights = edge_weights.values
        w_avg = np.mean(e_weights, axis=0)
        avg_edge_w.append((l, w_avg))
        
        
    abs_weights = []
    for (l, w_avg) in avg_edge_w:
        w_avg = np.abs(w_avg) 
        min_w = min(w_avg)
        max_w = max(w_avg)
        print(f"Min: {min_w:.2f}, Max: {max_w:.2f}")
        
        # w_avg[w_avg < 0.01] = 0.

        abs_weights.append((l, w_avg))
        
    adjacent_m_l = get_adj_m(dims, nodes_num, abs_weights)
    
    
    G_l = []
    for (l, adjacent_m) in adjacent_m_l:
        # Create network object
        G = nx.from_numpy_array(adjacent_m)

        orf = OllivierRicci(G, alpha=0.5, verbose="TRACE")
        # orf.compute_ricci_flow(iterations=100)
        orf.compute_ricci_curvature()

        G1 = orf.G.copy()
        print(G1)
        G_l.append((l, G, G1))
        
    return G_l, edges_num

def get_mask(G_l, edges_num, dims, num = 1000, thre = 0.95):
    e_l = []
    IOU = None
    for (l, G, G1) in G_l:
        # edge = sorted(G1.edges(data=True), key=lambda edge: edge[2].get("ricciCurvature", 0), reverse = True)
        edge_set = show_results(G1, num, "ricciCurvature", thre)
        IOU = edge_set if IOU == None else (IOU & edge_set)
        e_l.append((edge_set, l))
        print(f'for label {l}, have {len(edge_set)} edges after threshold...')
        
        
    # build mask
    mask_dict = dict()
    for (edge_set, l) in e_l:
        mask = torch.zeros((1, edges_num), dtype=torch.int32)
        mask = build_mask(mask, edge_set, dims)
        mask_dict[l] = mask
        
    return mask_dict