import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd



def cal_edge_v(net, loader, device):
                
    net.eval()

    for i, (d, label) in enumerate(loader):    
        d = d.to(device)
        output = net.edge_w_batch(d)
        
        output = output.cpu().detach().numpy() 
        if (i == 0):
            data1 = output
        else:
            data1 = np.vstack((data1,output))

    w_avg = np.mean(data1, axis=0)
    return w_avg
    

def build_cnn_adj(nodes_num, model_dims, net, loader, device, file_n = None, flag = False):
    eps = 10e9
    # raw value: edge weights
    if (file_n != None):
        edge_weights = pd.read_csv(file_n, index_col=0)
        e_weights = edge_weights.values.squeeze()
    else:
        e_weights = cal_edge_v(net, loader, device)
    w_avg = np.abs(e_weights) 
    
    w_avg[w_avg != 0] = 1/w_avg[w_avg != 0]
    if flag:
        w_avg[w_avg == 0] = eps
    # w_avg = np.where(w_avg_abs != 0, 1/w_avg_abs, 0)
    
    total_e = w_avg.shape[0]
    
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)
    # adjacent_m = np.full(adjacent_m.shape, -1.)

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
        
    # ind = np.where(adjacent_m < 0)

    # i = ind[0]
    # j = ind[1]

    # adjacent_m[j,i] = -adjacent_m[i,j]
    # adjacent_m[adjacent_m<0] = 0
    
    # adjacent_m[adjacent_m < 0] = 0.
        
    return adjacent_m, total_e, w_avg
            