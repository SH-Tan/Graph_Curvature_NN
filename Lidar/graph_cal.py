
import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict
import networkx as nx

from Lidar.RICCICURVATURE.OllivierRicci import OllivierRicci
from Lidar.RICCICURVATURE.q_exponential import q_exponential


q = 1


# change -10000 to 0 
def q_adjust(adj, nodes, dims):
    q_exp = q_exponential(q)
    node_v = nodes.cpu()
    ind = torch.where(node_v <= 0)[0]
    
    new_adj = np.zeros_like(adj)
    adj[:, ind[ind >= dims[0]]] *= -1 
    new_adj = 1.0/abs(q_exp.q_exponential_series(adj))
    new_adj[adj == -10000] = 0.
    return new_adj


# dims, nodes_num, edges_num, NN_w, edge_v, nodes
def build_adjm(nodes_num, dims, edge_v, nodes, NN_w):
    NN_w[edge_v == 0] = 0.
    w_avg = torch.mean(NN_w, axis=0) # (edge num,)
    # build adjacent matrix
    adjacent_m = torch.zeros((nodes_num, nodes_num), dtype=torch.float32)
    # adjacent_m *= -10000

    d_num = len(dims)
    cur_s_col = dims[0]
    cur_e_col = dims[0] + dims[1]

    cur_layer = 1
    start_col = 0
    end_col = dims[1]

    for i in range(nodes_num - dims[d_num-1]):
        # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
        adjacent_m[i, cur_s_col : cur_e_col] = w_avg[start_col : end_col]
        
        if (cur_layer < d_num-1 and i == cur_s_col - 1):
            cur_layer += 1
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            cur_s_col = cur_e_col
            cur_e_col = cur_e_col + dims[cur_layer]
        else:
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            
    return adjacent_m



def show_results(G, curvature="ricciCurvature"):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0), reverse=False) # ascending
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


def cal_curvature(adj, nodes, dims, nodes_ori):
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # print(G)
    
    orf = OllivierRicci(G, alpha=0., method="OTD")
    orf.recal_graph_weight_tanh(nodes)
    orf.compute_ricci_curvature()
    G1 = orf.G.copy()

    edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature")

    ratio = len(edge_set)/(len(remain_edges)+len(edge_set))
    
    return np.array(c), ratio, G1