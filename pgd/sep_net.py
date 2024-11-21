import torch
import numpy as np

from tools.small_model import FC_MD

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

from tools.edge_remove import Edge_Remove
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def build_adjm(loader, net, nodes_num, dims, device):
    net.eval()
    for i, (d, label) in enumerate(loader):    
        d = d.to(device)
        output = net.edge_w_batch(d)
        
        output = output.cpu().detach().numpy() 
        if (i == 0):
            data = output
        else:
            data = np.vstack((data,output))
    
    w_avg = np.mean(data, axis=0)
    w_avg = np.abs(w_avg) 


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
        adjacent_m[i, cur_s_col : cur_e_col] = w_avg[start_col : end_col]
        
        if (cur_layer < layer_num-1 and i == cur_s_col - 1):
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
    
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0)) # ascending
    
    edge = list(np.array(sorted_edge)[:, 0:2])
    edge_set = {(n1,n2) for (n1,n2) in edge}

    return edge_set



def sep_net(full_net, net_pos, dims, layer_num, loader, nodes_num, res_path, device):

    # edge_weights = pd.read_csv(model_path, index_col=0)
    # # cols = edge_weights.columns # number of neurals
    # e_weights = edge_weights.values
    # w_avg = np.mean(e_weights, axis=0)
    # w_avg = np.abs(w_avg) 
    
    adjacent_m = build_adjm(loader, net_pos, nodes_num, dims, device)
    
    # Create network object
    G = nx.from_numpy_array(adjacent_m)

    orf = OllivierRicci(G, alpha=0.5, verbose="TRACE")
    # orf.compute_ricci_flow(iterations=100)
    orf.compute_ricci_curvature()

    G1 = orf.G.copy()
    print(G1)

    edge_set = show_results(G1, "ricciCurvature")
    
    edge_r = Edge_Remove(full_net, dims, 28, G1, "ricciCurvature", res_path)
    edge_r.e_remove(edge_set, "complement.pth")
    
    return G1
