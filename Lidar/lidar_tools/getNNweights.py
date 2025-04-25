import torch
import torch.nn as nn

import numpy as np


# calculate edge weights, NN weights, node value (before/ after ReLU)
def getNN_info(model, input, device):
    dims = []
    dims.append(input.shape[1])
    
    nodes_num = input.shape[1]
    edges_num = 0
    
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []
    
    NN_w = None
    edge_v = None
    
    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = torch.tensor(model['weights'][layer], dtype = torch.float64).to(device)
        offsets[layer] = torch.tensor(model['offsets'][layer], dtype = torch.float64).to(device)
        
        nodes_num += weights[layer].shape[0]
        edges_num += weights[layer].shape[0] * dims[-1]
        dims.append(weights[layer].shape[0])
        
        activations.append(model['activations'][layer])
        
        layerCount += 1
    
    curNeurons = input.T.to(device)  # 21*batch
    nodes = input.view(-1, input.shape[1]).to(device)
    # nodes_ori = torch.zeros_like(nodes) # without tanh
    x_tmp = input.view(-1, input.shape[1]).to(device)
    ones_tmp = torch.ones_like(x_tmp)
    
    for layer in range(layerCount-1):
        cur_shape = weights[layer+1].shape[0] * weights[layer+1].shape[1]

        curNeurons = weights[layer+1]@curNeurons + offsets[layer+1].reshape(len(offsets[layer+1]),1)
        
        x_tmp = torch.cat([torch.reshape(weights[layer+1].T * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in x_tmp], axis=0)
        ones_tmp = torch.cat([torch.reshape(weights[layer+1].T * x1[np.newaxis,:].T, (1, cur_shape)) for x1 in ones_tmp], axis=0)

        edge_v = x_tmp if edge_v == None else torch.cat((edge_v, x_tmp), axis=1)
        NN_w = ones_tmp if NN_w == None else torch.cat((NN_w, ones_tmp), axis=1)
        
        # nodes_ori = torch.cat((nodes_ori, curNeurons.T.view(-1, curNeurons.shape[0])), axis=1)
        
        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + torch.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = torch.tanh(curNeurons)

        x_tmp = curNeurons.T
        nodes = torch.cat((nodes, curNeurons.T.view(-1, curNeurons.shape[0])), axis=1)
        ones_tmp = torch.ones_like(x_tmp)

    return dims, nodes_num, edges_num, NN_w, edge_v, nodes
    

