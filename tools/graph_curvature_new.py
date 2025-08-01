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
import random
import os
import math

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
_pre_n = 0
_W = dict()


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
            # CNN layer handling
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            padding = current_layer['dim'].get('padding', 0)
            pool = current_layer['dim'].get('pool', False)
            if pool:
                dst_size = dst_size * 2 * 2
            else:
                dst_size = dst_size
            
            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            
            # Generate receptive field indices
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()

            # Unfold operation to get receptive field indices
            unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1, 2).int()
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
            
            # if (i, j-1) in shortest_paths and (j-1, j) in shortest_paths:
            #     current_min = torch.minimum(current_min,
            #                               min_plus_mult(shortest_paths[(i, j-1)],
            #                                           shortest_paths[(j-1, j)]))
            
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
            padding = current_layer['dim'].get('padding', 0)
            # Generate receptive field indices
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()

            # Unfold operation to get receptive field indices
            unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1, 2).int()
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


# def get_layer_path(sp_dict, prefix_dims, b, i, j):     
#     i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
#     j_layer = np.searchsorted(prefix_dims, j, side='right') - 1
    
#     i_idx = i - prefix_dims[i_layer]
#     j_idx = j - prefix_dims[j_layer]
#     return sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item()


def compute_full_path_matrix(b, in_neigh, out_neigh):
    d_np = np.full((len(in_neigh), len(out_neigh)), np.inf)

    if len(in_neigh) > 1 and len(out_neigh) > 1:
        fill_shortest_paths(d_np,  b, in_neigh[:-1], out_neigh[:-1])
    if len(in_neigh) > 0 and len(out_neigh) > 1:
        fill_shortest_paths(d_np, b, [in_neigh[-1]], out_neigh[:-1], row_offset=len(in_neigh) - 1, col_offset=0)
    if len(in_neigh) > 1 and len(out_neigh) > 0:
        fill_shortest_paths(d_np, b, in_neigh[:-1], [out_neigh[-1]], row_offset=0, col_offset=len(out_neigh) - 1)
    return d_np




def fill_shortest_paths(d_np, b, in_neigh, out_neigh, row_offset=0, col_offset=0):
    if len(in_neigh) == 0 or len(out_neigh) == 0:
        return

    in_neigh = np.atleast_1d(np.array(in_neigh))
    out_neigh = np.atleast_1d(np.array(out_neigh))

    i_layer = np.searchsorted(_prefix_dims, in_neigh[0], side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, out_neigh[0], side='right') - 1

    assert np.all(np.searchsorted(_prefix_dims, in_neigh, side='right') - 1 == i_layer), "in_neigh not in same layer"
    assert np.all(np.searchsorted(_prefix_dims, out_neigh, side='right') - 1 == j_layer), "out_neigh not in same layer"

    in_idx = in_neigh - _prefix_dims[i_layer]
    out_idx = out_neigh - _prefix_dims[j_layer]

    sp_tensor = _sp_dict[(i_layer, j_layer)][b]
    submat = sp_tensor[np.ix_(in_idx, out_idx)]
    
    # Correctly insert into the right region of d_np
    d_np[row_offset:row_offset + len(in_neigh), col_offset:col_offset + len(out_neigh)] = submat



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
    
    if i_layer > 0 and j_layer < len(_dims)-1:
        m = _W.get(i_layer)
        if m is not None:
            return (b, i, j, 1.0 - m/sp)

    # In-neighbors distribution
    if i_layer == 0:
        mu = np.array([1.0])
        in_neigh = [i]
    else:
        mu = _distribution_in[i_layer][b, :, i_idx]
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
    if j_layer == len(_dims)-1:
        nu = np.array([1.0])
        out_neigh = [j]
    else:
        nu = _distribution_out[j_layer][b, j_idx, :]
        if len(np.nonzero(nu)[0]) == 0:     
            nu = np.array([1.0])
            out_neigh = [j]
        else:
            if (np.any(nu == -1.)):
                tmp = (1.0 - _alpha) / len(np.nonzero(nu)[0])
                nu[nu==-1] = tmp
            non_zero = np.nonzero(nu)[0]
            out_neigh = np.array(range(_prefix_dims[j_layer+1], _prefix_dims[j_layer+2]))
            out_neigh = list(out_neigh[non_zero]) + [j]
            nu = np.hstack((nu[non_zero], np.array(_alpha)))

    
    # Get submatrix for neighbors
    # d_np = np.full((len(in_neigh), len(out_neigh)), np.inf)
    assert(in_neigh[-1] == i and out_neigh[-1] == j)
    d_np = compute_full_path_matrix(0, in_neigh, out_neigh)
    
    d_np[-1, -1] = sp
    if np.isinf(d_np).any():
        print(i_layer, np.isinf(d_np).sum(), np.isnan(d_np).sum())

    if d_np.size == 0 or np.isinf(d_np).all():
        return (b,i, j, 2.0)
    
    m = ot.emd2(mu, nu, d_np)
    # print(f'{i} - {j}: {len(mu)} - {len(nu)} - W = {m}')

    return (b, i, j, 1.0 - m/sp)



def process_edge_input(b, edge, W):
    i, j = edge
    i_layer = np.searchsorted(_prefix_dims, i, side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, j, side='right') - 1
    
    res = []
    
    if j_layer != i_layer + 1:
        return []
    
    if (i_layer, j_layer) not in _sp_dict:
        return []
    
    i_idx = i - _prefix_dims[i_layer]
    sp_row = _sp_dict[(i_layer, j_layer)][b, i_idx]
    # Dsts where the cost is finite
    dst_idx = np.nonzero(np.isfinite(sp_row))[0]

    for dst in dst_idx.tolist():
        sp = float(sp_row[dst])
        # score = 1 - W/sp, guard sp==0 just in case
        score = 1.0 - (W / sp) if sp != 0.0 else (-np.inf if W > 0 else 1.0)

        src_global = i  # starts from i only
        dst_global = _prefix_dims[j_layer] + int(dst)
        res.append((b, src_global, dst_global, float(score)))

    return res



def getW(b, i, j, i_layer, j_layer):
    i_idx = i - _prefix_dims[i_layer]
    j_idx = j - _prefix_dims[j_layer]
    sp = _sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item()

    # In-neighbors distribution
    if i_layer == 0:
        mu = np.array([1.0])
        in_neigh = [i]
    else:
        mu = _distribution_in[i_layer][b, :, i_idx]
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
    if j_layer == len(_dims)-1:
        nu = np.array([1.0])
        out_neigh = [j]
    else:
        nu = _distribution_out[j_layer][b, j_idx, :]
        if len(np.nonzero(nu)[0]) == 0:     
            nu = np.array([1.0])
            out_neigh = [j]
        else:
            if (np.any(nu == -1.)):
                tmp = (1.0 - _alpha) / len(np.nonzero(nu)[0])
                nu[nu==-1] = tmp
            non_zero = np.nonzero(nu)[0]
            out_neigh = np.array(range(_prefix_dims[j_layer+1], _prefix_dims[j_layer+2]))
            out_neigh = list(out_neigh[non_zero]) + [j]
            nu = np.hstack((nu[non_zero], np.array(_alpha)))

    
    # Get submatrix for neighbors
    # d_np = np.full((len(in_neigh), len(out_neigh)), np.inf)
    assert(in_neigh[-1] == i and out_neigh[-1] == j)
    d_np = compute_full_path_matrix(0, in_neigh, out_neigh)
    
    d_np[-1, -1] = sp
    if np.isinf(d_np).any():
        print(i_layer, np.isinf(d_np).sum(), np.isnan(d_np).sum())

    m = ot.emd2(mu, nu, d_np)

    return m
    



def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return process_edge(*stuff)


def _wrap_compute_single_edge_input(stuff):
    """Wrapper for args in multiprocessing."""
    return process_edge_input(*stuff)


def graph_curvature_main_torch(dims, weights, model_dims = None, device='cuda', probability_w = None, alpha = 0., pre_n=0, layers_to_process=None):
    import gc
    global _dims 
    global _prefix_dims 
    global _sp_dict 
    global _distribution_in 
    global _distribution_out
    global _alpha
    global _pre_n
    
    _alpha = alpha
    _pre_n = pre_n

    weights = weights.to(device)
    batch_size = weights.shape[0]
    prefix_dims = np.cumsum([0] + dims).tolist()

    _dims = dims
    _prefix_dims = np.array(prefix_dims)
    
    layers = layers_to_process or list(range(len(dims)-1))

    # Compute shortest paths
    if model_dims:
        sp_dict = cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda')
        if probability_w != None:
            probability_w = probability_w.to(device)
            sp1 = cnn_adjacent_layer(model_dims, probability_w, prefix_dims, device='cuda')
    else:
        sp_dict = layerwise_shortest_path_torch(dims, weights, device)
        if probability_w != None:
            probability_w = probability_w.to(device)
            sp1 = fc_adjacent_layer(dims, probability_w, device)
            
    _sp_dict = {k: v.cpu().numpy() for k, v in sp_dict.items()}
    
    # print(f'Finish shorest path')

    dis_w = sp1 if probability_w is not None else sp_dict
        
    del weights
    del probability_w
    torch.cuda.empty_cache()
        
    # print(dis_w)
        
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
            
            # Clean up
            del path_sub, mask, weights_layer, sum_weights, dist_prev, mask1
            torch.cuda.empty_cache()

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
            
            # Clean up
            del path_sub, mask, weights_layer, sum_weights, dist_next, mask1
            torch.cuda.empty_cache()
       
    _distribution_in = distribution_in
    _distribution_out = distribution_out
    
    ricci_results = defaultdict(list)
    
    # input layer
    # W_cache = {}
    # edges = []   # keep same structure: list[(b, (i_global, j_global))]

    # layer = 0
    # i_layer = layer
    # j_layer = layer + 1

    # sp_array = sp_dict[(i_layer, j_layer)]  # shape: (B, N_in, N_out)

    # for b in range(batch_size):
    #     # Get all finite (src, dst) pairs, row-major ordered (src asc, then dst asc)
    #     non_inf = torch.nonzero(torch.isfinite(sp_array[b]), as_tuple=False).cpu().numpy()

    #     seen_src = set()
    #     for src, dst in non_inf:
    #         if src in seen_src:
    #             continue  # we only want the first edge from this src
    #         seen_src.add(src)

    #         # Map local (layer) indices to global node ids
    #         i_global = prefix_dims[i_layer] + int(src)
    #         j_global = prefix_dims[j_layer] + int(dst)

    #         # Compute and store W for this first edge
    #         W_val = getW(b, i_global, j_global, i_layer, j_layer)
    #         # W_cache[(b, i_global)] = W_val

    #         # Keep your original edges list with exactly one edge per input node now
    #         edges.append((b, (i_global, j_global), W_val))

    # # Build args like you had before
    # args = [(b, edge, W) for b, edge, W in edges]
    
    # with get_context('fork').Pool(processes=proc) as pool:
        
    #     chunksize = max(1, (len(args) + (proc * 4 - 1)) // (proc * 4))

    #     results = pool.imap_unordered(_wrap_compute_single_edge_input, args, chunksize=chunksize)
    #     pool.close()
    #     pool.join()
        
    # # print(f'Finish..')
    
    # for res in results:
    #     for b, i, j, val in res:
    #         ricci_results[b].append((i+_pre_n, j+_pre_n, val))
            
    # print(f'Finish input layer')
    
    
    # middle layer
    # count = 0
    # for layer in layers:
    #     if count > 10:
    #         break
    #     if 1 < layer < len(dims) - 2:
    #         sp_array = sp_dict[(layer, layer + 1)]
    #         layer_W = _W.get(layer, None)

    #         for b in range(batch_size):
    #             non_inf = torch.nonzero(torch.isfinite(sp_array[b]), as_tuple=False).cpu().numpy()
    #             for src, dst in non_inf:
    #                 i_global = prefix_dims[layer] + int(src)
    #                 j_global = prefix_dims[layer + 1] + int(dst)

    #                 if layer_W is None:
    #                     layer_W = getW(b, i_global, j_global, layer, layer + 1)
    #                     _W[layer] = layer_W  # cache for rest of this (and future) layers

    #                 sp = sp_array[b, src, dst].item()
    #                 ricci_results[b].append((i_global + _pre_n, j_global + _pre_n, 1.0 - layer_W / sp))
    #                 count += 1
    #                 print(f'W = {layer_W}')
    #                 if count > 10:
    #                     break
                    
    # print(f'Finish middle layer')
        
    # output layer
    # Generate edges from original weights
    edges = []
    # layer = layers[-1]

    for layer in layers:
        # if layer <= 1 or layer == len(dims) - 2:
        sp_array = sp_dict[(layer, layer+1)]
        
        for b in range(batch_size):
            non_inf = torch.nonzero(torch.isfinite(sp_array[b]), as_tuple=False).cpu().numpy()
            for src, dst in non_inf:
                global_src = prefix_dims[layer] + src
                global_dst = prefix_dims[layer+1] + dst
        
                # print(f'{src} - {dst}: {sp_array[b][src][dst]} {_sp_dict[(layer, layer+1)][b][src][dst]} - {global_src}:{global_dst}')
                edges.append((b, (global_src, global_dst)))
                if (len(edges) > 10):
                    break

    args = [(b, edge) for b, edge in edges]
    
    # print(len(args))
    
    del sp_dict
    torch.cuda.empty_cache()

    # print(len(args))
    # Process edges in parallel
    
    with get_context('fork').Pool(processes=proc) as pool:
        
        chunksize, extra = divmod(len(args), proc * 4)
        if extra:
            chunksize += 1

        results = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()
        
    # print(f'Finish..')
    
    for b, i, j, val in results:
        ricci_results[b].append((i+_pre_n, j+_pre_n, val))
        
    # Final GPU cleanup
    torch.cuda.empty_cache()
    gc.collect()

    return ricci_results


if __name__ == '__main__':
    dims = [4, 3, 3, 2]
    # weights = torch.tensor([
    #     [1, 0, 0.5, 1.5, 1, 2, 0.5, 0.2, 0.3],
    #     [1.2, 2, 0.5, 1.58, 1, 1.8, 0.5, 0.7, 0.8]
    # ], device='cuda')

    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    model_dims = {
        1: {"name": "input", "dim": {"channel": 1, "out_size": 3}},
        2: {"name": "cnn", "dim": {"channel": 2, "kernel": 2, "stride": 1, "out_size": 2}},
        3: {"name": "fc", "dim": {"out_size": 2}},
        4: {"name": "fc", "dim": {"out_size": 1}}
    }
    
    # dims = [9, 8, 2, 1]
    
    # edge_num = 32 + 8*2 + 2
    edge_num = 27
    weights = torch.rand(1, edge_num)
    # weights[0,0] = float('inf')
    node = torch.rand(1, edge_num)
    
    print(weights)
    print(1./weights)
    w = 1./weights
    
    # for i in range(4,7):
    #     print(f'Edge {i} - {7}: {w[0,(i-4)*3]}')
    #     print(f'Edge {i} - {8}: {w[0,(i-4)*3+1]}')
    #     print(f'Edge {i} - {9}: {w[0,(i-4)*3+2]}')
    
    # print(node)
    # print(1./node)

    ricci_curvature = graph_curvature_main_torch(dims, weights,probability_w=node.to())

    print("Ricci Curvature Results:")
    for b in range(weights.shape[0]):
        print(f"\nBatch {b}:")
        for (x,y,c) in ricci_curvature[b]:
            print(x,y,c)