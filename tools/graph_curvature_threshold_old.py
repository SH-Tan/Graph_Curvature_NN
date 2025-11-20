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
import gc

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
_nodes_value = None
_edge_value = None



def out_distribution(model_dims, sp1, device='cuda', thre = 0.5, dist = None):
    out_dist_matrices = {}
    inf = torch.tensor(float('inf'), device=device)

    # Sort by key so we process (i,j) pairs in order
    for (i, j), cur_sp in sorted(sp1.items()):
        batch_size, src_size, dst_size = cur_sp.shape
        cur_dist = dist[(i, j)]      # some distance/importance measure
        # initialize
        out_matrix = torch.full_like(cur_sp, inf, device=device, dtype=torch.float32)
        
        cur_layer = model_dims[j+1]
        cur_n = cur_layer["name"]

        # global offset for destination nodes
        dst_prefix = _prefix_dims[j]
        
        for src_idx in range(src_size):
            
            if cur_n == "fc":
                out_neighbors = torch.arange(_prefix_dims[j], _prefix_dims[j+1], device=device)
                node_slice = torch.abs(_nodes_value[:, out_neighbors])
                
                if thre <= 0:
                    out_matrix[:, src_idx, :] = 1./node_slice
                    continue
                
                # corresponding weights (importance)
                weight_v = 1.0 / (cur_dist[:, src_idx, :])
                
                # --- ✅ mask invalid (inf or 0) entries ---
                valid_mask = node_slice > 0 
                valid_weights = weight_v[valid_mask]
                
                if valid_weights.numel() == 0:
                    # no valid edges → skip
                    out_matrix[:, src_idx, :] = inf
                    continue

                threshold = torch.quantile(valid_weights, thre)
                top_mask = (weight_v >= threshold) & valid_mask
                
                # --- assign normalized inverse node values ---
                inv_vals = torch.where(top_mask, 1.0 / (node_slice), inf)
                out_matrix[:, src_idx, :] = inv_vals
                
            else:
                # mask valid edges (not inf)
                mask = (cur_sp[:, src_idx, :] != 0) # [dst_size]
                
                # if (i > 0):
                #     prev_layer = model_dims[i]
                #     prev_n = prev_layer["name"]
                    
                #     if prev_n == "input":
                #         mask = (cur_sp[:, src_idx, :] != inf) & (cur_sp[:, src_idx, :] > 0)

                if not torch.any(mask):
                    # print(i,j,src_idx)
                    continue

                valid_idx = torch.nonzero(mask[0], as_tuple=True)[0]  # assuming batch dim = 1
                inv_vals = torch.full_like(cur_sp[:, src_idx, :], inf, device=device, dtype=torch.float32)

                # global destination node index
                global_idx = dst_prefix + valid_idx
                node_slice = torch.abs(_nodes_value[:, global_idx])  # [B, dst_size] or [1, dst_size]
                
                # --- Step 1: normalize to [0, 1] ---
                # min_vals = node_slice.min(dim=1, keepdim=True)[0]
                # max_vals = node_slice.max(dim=1, keepdim=True)[0]
                
                # node_slice = torch.where(
                #     max_vals > min_vals,
                #     (node_slice - min_vals) / (max_vals - min_vals),
                #     node_slice,
                # )

                if thre <= 0:
                    # Compute reciprocal, avoid division by zero
                    node_inv = torch.where(node_slice != 0, 1.0 / node_slice, torch.zeros_like(node_slice))

                    # assign node value to all valid incoming edges
                    out_matrix[:, src_idx, valid_idx]= node_inv
                    continue
                
                # --- ✅ mask invalid (inf or 0) entries ---
                valid_mask = node_slice > 0 
                
                if not torch.any(valid_mask):
                    continue
                
                # corresponding weights (importance)
                weight_v = 1.0 / (cur_dist[:, src_idx, valid_idx])
                valid_weights = weight_v[valid_mask]
                
                threshold = torch.quantile(valid_weights, thre)
                top_mask = (weight_v >= threshold) & valid_mask
                
                # --- Assign normalized node values ---
                inv_vals[:,valid_idx] = torch.where(top_mask, 1.0 / (node_slice), inf)
            
                out_matrix[:, src_idx, :] = inv_vals

        out_dist_matrices[(i, j)] = torch.where(out_matrix > 0, out_matrix, inf)

    return out_dist_matrices

            


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
                    
                    w = weights[:, weight_idx : end_col]
                    adjacent_m[:, cur_idx, n] = w
                    
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


def cnn_adjacent_layer(model_dims, weights, prefix_dims, device='cuda', thre = 0.5):
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
            
            # shortest_paths[(i, i + 1)] = torch.where(direct_dist > 0, direct_dist, inf)
            # weight_idx += src_size * dst_size

            edge_slice = _edge_value[:,weight_idx:weight_idx+src_size*dst_size]  # [batch_size, dst_size]
            edge_slice = edge_slice.view(batch_size, src_size, dst_size)
            
            if thre <= 0:
                shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
                weight_idx += src_size * dst_size
                continue
            
            # Initialize adjacency with inf
            adjacent_m = torch.full_like(direct_dist, inf, dtype=torch.float32, device=device)
            
            # Process each destination neuron (per column)
            for col in range(dst_size):
                valid_vals = edge_slice[:,:, col]
                nonzero_mask = valid_vals != 0
                if not torch.any(nonzero_mask):
                    continue

                # Select valid (nonzero) edge values
                valid_vals = valid_vals[nonzero_mask]

                # Compute threshold among valid entries
                threshold_val = torch.quantile(valid_vals, thre)

                # Select top edges (>= threshold)
                top_mask = (edge_slice[:, :, col] >= threshold_val) & nonzero_mask

                # Copy weights and mask non-top edges
                top_weights = direct_dist[:, :, col].clone()
                top_weights[~top_mask] = inf
                
                # Assign to adjacency matrix
                adjacent_m[:, :, col] = top_weights
                
             # Store as shortest path matrix
            shortest_paths[(i, i + 1)] = torch.where(adjacent_m > 0, adjacent_m, inf)
            weight_idx += src_size * dst_size
            
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
                    cur_idx = unfolded[0, p].tolist()
                    
                    # Slice edge values and weights
                    edge_slice = _edge_value[:, weight_idx:end_col]       # [batch_size, num_edges]
                    weight_slice = weights[:, weight_idx:end_col]        # [batch_size, num_edges]
     
                    # Get non-zero indices (since batch_size=1, we can drop batch dimension)
                    nonzero_mask = edge_slice != 0
                    
                    if thre <= 0:
                        adjacent_m[:, cur_idx, n] = weight_slice

                    else:
                        # Values of valid edges
                        valid_vals = edge_slice[nonzero_mask]
                        
                        if not torch.any(nonzero_mask):
                            continue

                        # Compute quantile threshold among valid entries
                        threshold_val = torch.quantile(valid_vals, thre)
                        
                        # Select top edges (>= threshold)
                        top_mask = (edge_slice >= threshold_val)
                        
                        # Set all non-top edges to inf explicitly
                        weight_slice[~top_mask] = inf
                        weight_slice[~nonzero_mask] = 0

                        # Assign to adjacency matrix
                        adjacent_m[:, cur_idx, n] = weight_slice

                    # Update indexing
                    weight_idx = end_col
                    end_col = weight_idx + step * pre_ch
                    n += 1

            # Save result
            shortest_paths[(i, i + 1)] = adjacent_m
            
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
    
    # if (_nodes_value[:,i] == 0):
    #     return (b, i, j, 1.0)
    
    i_idx = i - _prefix_dims[i_layer]
    j_idx = j - _prefix_dims[j_layer]
    sp = _sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item()
    
    # sp *= (1./_nodes_value[:,i])

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
            
    # nu = np.array([1.0])
    # out_neigh = [j]
        
    # Out-neighbors distribution
    if j_layer == len(_dims)-1:
        nu = np.array([1.0])
        out_neigh = [j]
    else:
        nu = _distribution_out[j_layer][b, j_idx, :]
        out_neigh = np.array(range(_prefix_dims[j_layer + 1], _prefix_dims[j_layer + 2]))
        
        if len(np.nonzero(nu)[0]) == 0:     
            nu = np.array([1.0])
            out_neigh = [j]
        else:
            if (np.any(nu == -1.)):
                tmp = (1.0 - _alpha) / len(np.nonzero(nu)[0])
                nu[nu==-1] = tmp
            non_zero = np.nonzero(nu)[0]
            # out_neigh = np.array(range(_prefix_dims[j_layer+1], _prefix_dims[j_layer+2]))
            out_neigh = list(out_neigh[non_zero]) + [j]
            nu = np.hstack((nu[non_zero], np.array(_alpha)))

    
    # Get submatrix for neighbors
    assert(in_neigh[-1] == i and out_neigh[-1] == j)
    d_np = compute_full_path_matrix(0, in_neigh, out_neigh)
    
    d_np[-1, -1] = sp

    if d_np.size == 0 or np.isinf(d_np).all():
        return (b,i, j, 2.0)
    
    try:
        m = ot.emd2(mu, nu, d_np)
    except:
        print(_nodes_value[:,i],_nodes_value[:,j])
    
    # if np.isinf(d_np).any():
    #     print(i_layer, np.isinf(d_np).sum(), np.isnan(d_np).sum(), m)
    
    # === Debug print section for layer 4 ===
    # if i_layer == 3 and (sp <= 1./0.36 or sp >= 1./ 0.00001):
    #     print(_prefix_dims)
    #     print(f"\nEdge ({i} → {j}), sp = {sp}")
    #     print("Node values:")
    #     print("  i node:", _nodes_value[:, i])
    #     print("  j node:", _nodes_value[:, j])
        
    #     # Sort and print top 10 μ
    #     mu_sorted_idx = np.argsort(mu)[::-1][:10]
    #     print("\nTop 10 μ values:")
    #     for rank, idx in enumerate(mu_sorted_idx):
    #         node_id = in_neigh[idx] if idx < len(in_neigh) else None
    #         node_val = _nodes_value[:, node_id] if node_id is not None else None
    #         print(f"  {rank+1}. μ[{idx}] = {mu[idx]:.4f}, node = {node_id}, value = {node_val}")

    #     # Sort and print top 10 ν
    #     nu_sorted_idx = np.argsort(nu)[::-1][:10]
    #     print("\nTop 10 ν values:")
    #     for rank, idx in enumerate(nu_sorted_idx):
    #         node_id = out_neigh[idx] if idx < len(out_neigh) else None
    #         node_val = _nodes_value[:, node_id] if node_id is not None else None
    #         print(f"  {rank+1}. ν[{idx}] = {nu[idx]:.4f}, node = {node_id}, value = {node_val}")

    return (b, i, j, 1.0 - m/sp)



def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return process_edge(*stuff)


def graph_curvature_main_torch(dims, weights, model_dims = None, device='cuda', probability_w = None, alpha = 0., pre_n=0, layers_to_process=None, nodes = None, edge_value = None, threshold = 0.5):
    global _dims 
    global _prefix_dims 
    global _sp_dict 
    global _distribution_in 
    global _distribution_out
    global _alpha
    global _pre_n
    global _nodes_value
    global _edge_value
    
    _alpha = alpha
    _pre_n = pre_n
    _nodes_value = nodes
    _edge_value = edge_value

    weights = weights.to(device)
    batch_size = weights.shape[0]
    prefix_dims = np.cumsum([0] + dims).tolist()

    _dims = dims
    _prefix_dims = np.array(prefix_dims)
    
    layers = layers_to_process or list(range(len(dims)-1))
    inf = torch.tensor(float('inf'), device=device)

    # Compute shortest paths
    if model_dims:
        sp_dict = cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda')
        # _sp_dict = {k: v.cpu().numpy() for k, v in sp_dict.items()}
        if probability_w != None:
            sp1 = cnn_adjacent_layer(model_dims, probability_w[0].to(device), prefix_dims, device='cuda', thre = threshold)
            sp2 = out_distribution(model_dims, sp1, device='cuda', thre = threshold, dist = sp_dict)
            # sp3 = cnn_adjacent_layer(model_dims, probability_w[1].to(device), prefix_dims, device='cuda', thre = threshold)
    else:
        sp_dict = layerwise_shortest_path_torch(dims, weights, device)
        # _sp_dict = {k: v.cpu().numpy() for k, v in sp_dict.items()}
        if probability_w != None:
            probability_w = probability_w.to(device)
            sp1 = fc_adjacent_layer(dims, probability_w, device)
            
    _sp_dict = {k: v.cpu().numpy() for k, v in sp_dict.items()}
    
    # print(_sp_dict)
    
    # print("sp1")
    # for (i, j), cur_sp in sorted(sp1.items()):
    #     print(i, j, cur_sp[:, 0:2, 0:10])
    #     input()
        
    # print("sp2")
    # for (i, j), cur_sp in sorted(sp2.items()):
    #     print(i, j, cur_sp[:, 0:2, 0:10])
    #     input()
    
    # print("sp3")
    # for (i, j), cur_sp in sorted(sp3.items()):
    #     print(i, j, cur_sp[:, 0:2, 0:10])
    #     input()

    if probability_w != None:
        dis_w_in = sp1
        dis_w_out = sp2
    else:
        dis_w_in = sp_dict
        dis_w_out = sp_dict
        
    # print(dis_w)
    
    del weights
    del probability_w
    del sp1, sp2
    torch.cuda.empty_cache()
    
    # Precompute distributions using dictionary
    distribution_in, distribution_out = {}, {}
    for layer in range(1, len(dims)):
        if (layer-1, layer) in dis_w_in:
            path_sub = dis_w_in[(layer-1, layer)]
            # path_sub = torch.where(path_sub > 0, path_sub, inf)
            mask = (path_sub != float('inf')) & (path_sub != 0)
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
            # weights_layer = (1./path_sub) * mask
            sum_weights = weights_layer.sum(dim=1)

            dist_prev = ((1.0 - _alpha) * weights_layer) / sum_weights[:, None, :]
            
            indices = torch.where(sum_weights <= EPSILON)[1]

            mask1 = (path_sub[:,:,indices] != float('inf'))
            dist_prev[:,:,indices] = -1
            dist_prev[:,:,indices] *= mask1
            dist_prev *= mask
            
            distribution_in[layer] = dist_prev.cpu().numpy()
            
    for layer in range(len(dims)-1):
        if (layer, layer+1) in dis_w_out:
            path_sub = dis_w_out[(layer, layer+1)]
            # path_sub = torch.where(path_sub > 0, path_sub, inf)
            mask = (path_sub != float('inf')) & (path_sub != 0)
            
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
                
            # weights_layer = (1./path_sub) * mask
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
    _nodes_value = _nodes_value.cpu().numpy()

    # Generate edges from original weights
    edges = []
    for layer in layers:
        sp_array = sp_dict[(layer, layer+1)]
        
        for b in range(batch_size):
            non_inf = torch.nonzero(~torch.isinf(sp_array[b])).cpu().numpy()
            for src, dst in non_inf:
                global_src = prefix_dims[layer] + src
                global_dst = prefix_dims[layer+1] + dst
     
                # print(f'{src} - {dst}: {sp_array[b][src][dst]} {_sp_dict[(layer, layer+1)][b][src][dst]} - {global_src}:{global_dst}')
                edges.append((b, (global_src, global_dst)))


    args = [(b, edge) for b, edge in edges]
    
    del sp_dict
    torch.cuda.empty_cache()

    # print(len(args))
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
    
    node[0,2] = 0
    node[0,12] = 0
    node[0,3] = 0
    node[0,10] = 0
    node[0,14] = 0
    node[0,16] = 0
    node[0,18] = 0

    # for i in range(4,7):
    #     print(f'Edge {i} - {7}: {w[0,(i-4)*3]}')
    #     print(f'Edge {i} - {8}: {w[0,(i-4)*3+1]}')
    #     print(f'Edge {i} - {9}: {w[0,(i-4)*3+2]}')
    
    print(node)
    # print(1./node)

    ricci_curvature = graph_curvature_main_torch(dims, weights,probability_w=node)

    print("Ricci Curvature Results:")
    for b in range(weights.shape[0]):
        print(f"\nBatch {b}:")
        for (x,y,c) in ricci_curvature[b]:
            print(x,y,c)