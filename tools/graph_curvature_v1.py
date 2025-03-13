import torch
import numpy as np
import ot
import multiprocessing as mp
from multiprocessing import get_context
from functools import partial
import torch.nn.functional as F
import time
from collections import defaultdict
import sys

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=sys.maxsize)

EPSILON = 1e-7
proc = mp.cpu_count()

_dims = []
_prefix_dims = []
_sp_dict = {}
_distribution_in = {}
_distribution_out = {}
_alpha = 0.


# For CNN
def cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda'):
    batch_size, weight_num = weights.shape
    layers = sorted(model_dims.items(), key=lambda x: x[0])
    num_layers = len(layers)
    shortest_paths = {}
    inf = torch.tensor(float('inf'), device=device)
    
    
    weight_idx = 0
    for i in range(num_layers - 1):
        l = i + 1
        current_layer = model_dims[l+1]
        src_size = prefix_dims[i+1] - prefix_dims[i]
        dst_size = prefix_dims[i+2] - prefix_dims[i+1]
        
        if current_layer['name'] == 'fc':
            # FC layer handling
            direct_dist = weights[:, weight_idx:weight_idx+src_size*dst_size]
            direct_dist = direct_dist.view(batch_size, src_size, dst_size)
            
            shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
            weight_idx += src_size * dst_size
            
            # f.write(f'{i}-{i+1}: {shortest_paths[(i, i+1)]}\n')
            
        elif current_layer['name'] in ['cnn', 'pooling']:
            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            
            # CNN layer handling
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            # Generate receptive field indices
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()

            # Unfold operation to get receptive field indices
            unfolded = F.unfold(dummy, kernel_size=k, stride=s).transpose(1, 2).int()
            patches = unfolded.shape[1]

            step = k**2 
            
            # Create indices matrix
            n = 0
            end_col = weight_idx + step*pre_ch
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0,p].tolist()
                    adjacent_m[:, cur_idx, n] = weights[:, weight_idx : end_col]
                    
                    weight_idx = end_col
                    end_col = weight_idx + step*pre_ch
                    n += 1
            
            shortest_paths[(i, i+1)] = torch.where(adjacent_m > 0, adjacent_m, inf)

    def min_plus_mult(a, b):
        return (a.unsqueeze(3) + b.unsqueeze(1)).min(dim=2)[0]

    # Dynamic programming approach remains the same
    for d in range(2, num_layers):
        for i in range(num_layers - d):
            j = i + d
            current_min = torch.full((batch_size, prefix_dims[i+1]-prefix_dims[i], 
                                    prefix_dims[j+1]-prefix_dims[j]), float('inf'), device=device)
            
            for k in range(i+1, j):
                if (i, k) in shortest_paths and (k, j) in shortest_paths:
                    current_min = torch.minimum(current_min, 
                                              min_plus_mult(shortest_paths[(i, k)], 
                                                          shortest_paths[(k, j)]))
            
            if (i, j-1) in shortest_paths and (j-1, j) in shortest_paths:
                current_min = torch.minimum(current_min,
                                          min_plus_mult(shortest_paths[(i, j-1)],
                                                      shortest_paths[(j-1, j)]))
            
            shortest_paths[(i, j)] = current_min
            
    return shortest_paths


# For FC
def layerwise_shortest_path_torch(dims, weights, device='cuda'):
    batch_size, edge_num = weights.shape
    num_layers = len(dims)
    shortest_paths = {}

    weight_idx = 0
    for i in range(num_layers - 1):
        src_size, dst_size = dims[i], dims[i+1]
        direct_dist = weights[:, weight_idx:weight_idx+src_size*dst_size]
        direct_dist = direct_dist.view(batch_size, src_size, dst_size)
        inf = torch.tensor(float('inf'), device=device)
        shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
        weight_idx += src_size * dst_size

    def min_plus_mult(a, b):
        return (a.unsqueeze(3) + b.unsqueeze(1)).min(dim=2)[0]

    for d in range(2, num_layers):
        for i in range(num_layers - d):
            j = i + d
            current_min = torch.full((batch_size, dims[i], dims[j]), float('inf'), device=device)
            
            for k in range(i+1, j):
                if (i, k) in shortest_paths and (k, j) in shortest_paths:
                    current_min = torch.minimum(current_min, 
                                              min_plus_mult(shortest_paths[(i, k)], 
                                                          shortest_paths[(k, j)]))
            
            if (i, j-1) in shortest_paths and (j-1, j) in shortest_paths:
                current_min = torch.minimum(current_min,
                                          min_plus_mult(shortest_paths[(i, j-1)],
                                                      shortest_paths[(j-1, j)]))
            
            shortest_paths[(i, j)] = current_min

    return shortest_paths


def cnn_adjacent_layer(model_dims, weights, prefix_dims, device='cuda'):
    batch_size, weight_num = weights.shape
    layers = sorted(model_dims.items(), key=lambda x: x[0])
    num_layers = len(layers)
    shortest_paths = {}
    inf = torch.tensor(float('inf'), device=device)
    
    
    weight_idx = 0
    for i in range(num_layers - 1):
        l = i + 1
        current_layer = model_dims[l+1]
        src_size = prefix_dims[i+1] - prefix_dims[i]
        dst_size = prefix_dims[i+2] - prefix_dims[i+1]
        
        if current_layer['name'] == 'fc':
            # FC layer handling
            direct_dist = weights[:, weight_idx:weight_idx+src_size*dst_size]
            direct_dist = direct_dist.view(batch_size, src_size, dst_size)
            
            shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
            weight_idx += src_size * dst_size
            
            # f.write(f'{i}-{i+1}: {shortest_paths[(i, i+1)]}\n')
            
        elif current_layer['name'] in ['cnn', 'pooling']:
            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            
            # CNN layer handling
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            # Generate receptive field indices
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()

            # Unfold operation to get receptive field indices
            unfolded = F.unfold(dummy, kernel_size=k, stride=s).transpose(1, 2).int()
            patches = unfolded.shape[1]

            step = k**2 
            
            # Create indices matrix
            n = 0
            end_col = weight_idx + step*pre_ch
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0,p].tolist()
                    adjacent_m[:, cur_idx, n] = weights[:, weight_idx : end_col]
                    
                    weight_idx = end_col
                    end_col = weight_idx + step*pre_ch
                    n += 1
            
            shortest_paths[(i, i+1)] = torch.where(adjacent_m > 0, adjacent_m, inf)
            
    return shortest_paths


def fc_adjacent_layer(dims, weights, device='cuda'):
    batch_size, edge_num = weights.shape
    num_layers = len(dims)
    shortest_paths = {}

    weight_idx = 0
    for i in range(num_layers - 1):
        src_size, dst_size = dims[i], dims[i+1]
        direct_dist = weights[:, weight_idx:weight_idx+src_size*dst_size]
        direct_dist = direct_dist.view(batch_size, src_size, dst_size)
        inf = torch.tensor(float('inf'), device=device)
        shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
        weight_idx += src_size * dst_size
        
    return shortest_paths


def get_layer_path(sp_dict, prefix_dims, b, i, j):     
    i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
    j_layer = np.searchsorted(prefix_dims, j, side='right') - 1
    
    i_idx = i - prefix_dims[i_layer]
    j_idx = j - prefix_dims[j_layer]
    return sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item()


def process_edge(b, edge):
    i, j = edge
    i_layer = np.searchsorted(_prefix_dims, i, side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, j, side='right') - 1
    
    if j_layer != i_layer + 1:
        return (b, i, j, 2.0)
    
    if (i_layer, j_layer) not in _sp_dict:
        return (b, i, j, 2.0)
    
    i_idx = i - _prefix_dims[i_layer]
    j_idx = j - _prefix_dims[j_layer]
    sp = _sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item()

    # In-neighbors distribution
    if i_layer == 0:
        mu = np.array([1.0])
        in_neigh = [i]
    else:
        mu = _distribution_in[i_layer][b, :, i - _prefix_dims[i_layer]]
        if len(np.nonzero(mu)[0]) == 0:   
            mu = np.array([1.0])
            in_neigh = [i]
        else:
            if (np.any(mu == -1.)):
                tmp = (1.0 - _alpha) / len(np.nonzero(mu)[0])
                mu[mu==-1] = tmp
            non_zero = np.nonzero(mu)[0]
            in_neigh = np.array(range(_prefix_dims[i_layer-1], _prefix_dims[i_layer]))
            in_neigh = list(in_neigh[non_zero]) + [i]
            mu = np.hstack((mu[non_zero], np.array(_alpha)))

    # Out-neighbors distribution
    nu = np.array([1.0])
    out_neigh = [j]
    
    # if j_layer == len(_dims)-1:
    #     nu = np.array([1.0])
    #     out_neigh = [j]
    # else:
    #     nu = _distribution_out[j_layer][b, j - _prefix_dims[j_layer], :]
    #     if len(np.nonzero(nu)[0]) == 0:     
    #         nu = np.array([1.0])
    #         out_neigh = [j]
    #     else:
    #         if (np.any(nu == -1.)):
    #             tmp = (1.0 - _alpha) / len(np.nonzero(nu)[0])
    #             nu[nu==-1] = tmp
    #         non_zero = np.nonzero(nu)[0]
    #         out_neigh = np.array(range(_prefix_dims[j_layer+1], _prefix_dims[j_layer+2]))
    #         out_neigh = list(out_neigh[non_zero]) + [j]
    #         nu = np.hstack((nu[non_zero], np.array(_alpha)))

    # Get submatrix for neighbors
    d_np = np.zeros((len(in_neigh), len(out_neigh)))
    for m_idx, m in enumerate(in_neigh):
        for n_idx, n in enumerate(out_neigh):
            if (m == n):
                d_np[m_idx, n_idx] = 0.
            else:
                d_np[m_idx, n_idx] = get_layer_path(_sp_dict, _prefix_dims, b, m, n)
    
    if d_np.size == 0 or np.isinf(d_np).all():
        return (b, i, j, 2.0)

    m = ot.emd2(mu, nu, d_np)
    
    return (b, i, j, 1.0 - m/sp)



def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return process_edge(*stuff)


def graph_curvature_main_torch(dims, weights, model_dims = None, device='cuda', probability_w = None, alpha = 0.):
    global _dims 
    global _prefix_dims 
    global _sp_dict 
    global _distribution_in 
    global _distribution_out
    global _alpha
    
    _alpha = alpha

    weights = weights.to(device)
    batch_size = weights.shape[0]
    prefix_dims = np.cumsum([0] + dims).tolist()

    _dims = dims
    _prefix_dims = np.array(prefix_dims)

    # Compute shortest paths
    if model_dims:
        sp_dict = cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda')
        if probability_w != None:
            sp1 = cnn_adjacent_layer(model_dims, probability_w, prefix_dims, device='cuda')
    else:
        sp_dict = layerwise_shortest_path_torch(dims, weights, device)
        if probability_w != None:
            sp1 = fc_adjacent_layer(dims, probability_w, device)
            
        
    _sp_dict = {k: v.cpu().numpy() for k, v in sp_dict.items()}
    
    if probability_w != None:
        dis_w = sp1
    else:
        dis_w = sp_dict
    
    # Precompute distributions using dictionary
    distribution_in, distribution_out = {}, {}
    for layer in range(1, len(dims)):
        if (layer-1, layer) in dis_w:
            path_sub = dis_w[(layer-1, layer)]
            mask = (path_sub != float('inf'))
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
            sum_weights = weights_layer.sum(dim=1)
            
            dist_prev = ((1.0 - _alpha) * weights_layer) / sum_weights[:, None, :]
            
            indices = torch.where(sum_weights <= EPSILON)[1]
            mask1 = (path_sub[:,:,indices] != float('inf'))
            dist_prev[:,:,indices] = -1
            dist_prev[:,:,indices] *= mask1
            dist_prev *= mask
            
            distribution_in[layer] = dist_prev.cpu().numpy()

    for layer in range(len(dims)-1):
        if (layer, layer+1) in dis_w:
            path_sub = dis_w[(layer, layer+1)]
            mask = (path_sub != float('inf'))
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
            sum_weights = weights_layer.sum(dim=2)
            
            dist_next = ((1.0 - _alpha) * weights_layer) / sum_weights[:, :, None]
  
            indices = torch.where(sum_weights <= EPSILON)[1]
            mask1 = (path_sub[:,indices,:] != float('inf'))
            dist_next[:,indices,:] = -1
            dist_next[:,indices,:] *= mask1
            dist_next *= mask
     
            distribution_out[layer] = dist_next.cpu().numpy()
            
    _distribution_in = distribution_in
    _distribution_out = distribution_out

    # Generate edges from original weights
    edges = []
    for layer in range(len(dims)-1):
        sp_array = sp_dict[(layer, layer+1)]
        
        for b in range(batch_size):
            non_inf = torch.nonzero(~torch.isinf(sp_array[b])).cpu().numpy()
            for src, dst in non_inf:
                global_src = prefix_dims[layer] + src
                global_dst = prefix_dims[layer+1] + dst
                edges.append((b, (global_src, global_dst)))

    args = [(b, edge) for b, edge in edges]
   
    # Process edges in parallel
    ricci_results = defaultdict(list)
    with get_context('fork').Pool(processes=proc) as pool:
        
        chunksize, extra = divmod(len(args), proc * 4)
        if extra:
            chunksize += 1
        
        results = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()
    

    for b, i, j, val in results:
        ricci_results[b].append(val)

    return ricci_results


if __name__ == '__main__':
    # dims = [2, 3, 1]
    # weights = torch.tensor([
    #     [1, 0, 0.5, 1.5, 1, 2, 0.5, 0.2, 0.3],
    #     [1.2, 2, 0.5, 1.58, 1, 1.8, 0.5, 0.7, 0.8]
    # ], device='cuda')
    
    model_dims = {
        1: {"name": "input", "dim": {"channel": 1, "out_size": 3}},
        2: {"name": "cnn", "dim": {"channel": 2, "kernel": 2, "stride": 1, "out_size": 2}},
        3: {"name": "fc", "dim": {"out_size": 2}},
        4: {"name": "fc", "dim": {"out_size": 1}}
    }
    
    dims = [9, 8, 2, 1]
    
    edge_num = 32 + 32*2 + 2
    weights = torch.rand(1, edge_num)
    
    print(weights)

    ricci_curvature = graph_curvature_main_torch(dims, weights, model_dims=model_dims)
    print("Ricci Curvature Results:")
    for b in range(weights.shape[0]):
        print(f"\nBatch {b}:")
        print(ricci_curvature[b])