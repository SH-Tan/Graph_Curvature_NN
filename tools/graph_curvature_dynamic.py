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
from numba import njit, prange

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=sys.maxsize)

EPSILON = 1e-7
proc = mp.cpu_count()

_pre_n = 0
_prefix_dims = []
_alpha = 0.
_sp_curr = None
_sp_prev = None
_sp_next = None
_dist_in = None
_dist_out = None
_shorest_pc = None
_shorest_cn = None
_shorest_pn = None
_W = dict()
_GLOBAL_CHAINED_SP = {}



def get_shortest_path_between_nodes(m, n, b, i=None, j=None):
    global _sp_curr, _shorest_pc, _shorest_cn, _shorest_pn
    """
    Returns the shortest path from node m to node n across up to 3 layers:
    - If directly adjacent: use sp_curr
    - If 2 layers apart:
        - If n == j: use sp_prev + sp_curr
        - If m == i: use sp_curr + sp_next
    - If 3 layers apart: use sp_prev + sp_curr + sp_next
    """
    layer_m = np.searchsorted(_prefix_dims, m, side='right') - 1
    layer_n = np.searchsorted(_prefix_dims, n, side='right') - 1
    m_idx = m - _prefix_dims[layer_m]
    n_idx = n - _prefix_dims[layer_n]

    try:
        if layer_m == layer_n - 1:
            if _sp_curr is not None:
                return _sp_curr[b, m_idx, n_idx]

        elif layer_m == layer_n - 2:
            if j is not None and n == j and _sp_prev is not None and _sp_curr is not None:
                return _shorest_pc[b, m_idx, n_idx]
            if i is not None and m == i and _sp_curr is not None and _sp_next is not None:
                return _shorest_cn[b, m_idx, n_idx]

        elif layer_m == layer_n - 3:
            if _sp_prev is not None and _sp_curr is not None and _sp_next is not None:
                return _shorest_pn[b, m_idx, n_idx]

    except:
        pass

    return np.inf


def compose_shortest_paths(sp_a: torch.Tensor, sp_b: torch.Tensor) -> torch.Tensor:
    """
    Min-plus matrix product: computes composed shortest paths A → C via B
    sp_a: (B, N1, N2) - A → B
    sp_b: (B, N2, N3) - B → C
    returns: (B, N1, N3) - A → C
    """
    assert sp_a.device == sp_b.device, "sp_a and sp_b must be on the same device"
    B, N1, N2 = sp_a.shape
    _, N2_b, N3 = sp_b.shape
    assert N2 == N2_b, "Mismatched intermediate layer size"

    # Efficient broadcasting without unnecessary reshapes
    return torch.min(sp_a.unsqueeze(3) + sp_b.unsqueeze(1), dim=2).values
    # (B, N1, 1, N2) + (B, 1, N2, N3) → (B, N1, N2, N3) → min over N2 → (B, N1, N3)



# For CNN
def get_adjacent_shortest_path(dims, weights, layer, prefix_dims, model_dims=None, device='cuda', weight_idx=0):
    batch_size, weight_num = weights.shape
    weights_idx = weight_idx

    if model_dims is None or model_dims[layer+1]['name'] == 'fc' or model_dims[layer+1]['name'] == 'input':
        src_size = dims[layer-1]
        dst_size = dims[layer]

        block = weights[:, weights_idx:weights_idx + src_size * dst_size]
        weights_idx += src_size * dst_size
        mat = block.view(batch_size, src_size, dst_size)
        mat = torch.where(mat > 0, mat, torch.full_like(mat, float('inf')))
        return mat, weights_idx
    
    else:
        cfg = model_dims[layer+1]['dim'] # current layer
        k, s = cfg['kernel'], cfg['stride']
        out_size = cfg['out_size']
        in_cfg = model_dims[layer]['dim']
        in_size = in_cfg.get('out_size', 1)
        pre_ch = in_cfg.get('channel', 1)
        cur_ch = cfg['channel']
        padding = cfg.get('padding', 0)
        pool = cfg.get('pool', False)
        if pool:
            dst_size = dims[layer] * 2 * 2
        else:
            dst_size = dims[layer]
        src_size = dims[layer-1]
    
        # build adjacency
        adjacent = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
        # create index map
        dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()
        unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1,2).int()  # (1, patches, k*k*pre_ch)
        patches = unfolded.shape[1]

        # Create indices matrix
        n = 0
        step = k**2 * pre_ch
        end_col = weights_idx + step
        for c in range(cur_ch):
            for p in range(patches):
                cur_idx = unfolded[0,p].tolist()
                adjacent[:, cur_idx, n] = weights[:, weights_idx : end_col]
                weights_idx = end_col
                end_col = weights_idx + step
                n += 1

        mat = torch.where(adjacent > 0, adjacent, torch.full_like(adjacent, float('inf')))
        return mat, weights_idx



def compute_distribution(np_paths, alpha, mode='in'):
    """
    Compute neighbor distributions matching original Torch logic.
    np_paths: numpy array shape (B, S, D) for 'in', or (B, S, D) for 'out'.
    """
    # mask valid edges
    mask = (np_paths != float('inf'))
    weights = torch.exp(-(np_paths ** 2)) * mask

    if mode == 'in':
        # sum over source dimension
        sum_w = weights.sum(dim=1)  # (B, D)
        dist = ((1.0 - alpha) * weights) / (sum_w[:, None, :])
        # handle zero-sum columns
        indices = torch.where(sum_w <= EPSILON)[1]
        mask1 = (np_paths[:,:,indices] != float('inf'))
        dist[:,:,indices] = -1
        dist[:,:,indices] *= mask1
        dist *= mask
    else:
        # mode 'out': sum over destination dim
        sum_w = weights.sum(dim=2) # (B, S)
        dist = ((1.0 - _alpha) * weights) / (sum_w[:, :, None])
        # handle zero-sum rows
        indices = torch.where(sum_w <= EPSILON)[1]
        mask1 = (np_paths[:,indices,:] != float('inf'))
        dist[:,indices,:] = -1
        dist[:,indices,:] *= mask1
        dist *= mask
    return dist.cpu().numpy()



@njit(parallel=True)
def min_plus_mult(A, B):
    """
    Accelerated min-plus matrix multiplication using Numba with explicit loops.
    A: (B, N, K)
    B: (B, K, M)
    Output: (B, N, M)
    """
    B_sz, N, K = A.shape
    _, _, M = B.shape
    result = np.full((B_sz, N, M), np.inf, dtype=A.dtype)

    for b in prange(B_sz):
        for i in range(N):
            for j in range(M):
                min_val = np.inf
                for k in range(K):
                    val = A[b, i, k] + B[b, k, j]
                    if val < min_val:
                        min_val = val
                result[b, i, j] = min_val

    return result



def fill_shortest_paths(d_np, b, in_neigh, out_neigh, row_offset=0, col_offset=0, mode="next"):
    global _GLOBAL_CHAINED_SP
    
    if len(in_neigh) == 0 or len(out_neigh) == 0:
        return

    in_neigh = np.atleast_1d(np.array(in_neigh))
    out_neigh = np.atleast_1d(np.array(out_neigh))

    i_layer = np.searchsorted(_prefix_dims, in_neigh[0], side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, out_neigh[0], side='right') - 1

    in_idx = in_neigh - _prefix_dims[i_layer]
    out_idx = out_neigh - _prefix_dims[j_layer]

    # Two layers apart
    if i_layer + 2 == j_layer:
        key = mode
        if key in _GLOBAL_CHAINED_SP:
            # print(key)
            D = _GLOBAL_CHAINED_SP[key]
        else:
            if mode == "next" and _sp_curr is not None and _sp_next is not None:
                A = _sp_curr[b]
                B = _sp_next[b]
            elif mode == "prev" and _sp_curr is not None and _sp_prev is not None:
                A = _sp_curr[b]
                B = _sp_prev[b]
            else:
                return

            D = min_plus_mult(A, B)
            _GLOBAL_CHAINED_SP[key] = D

        d_np[row_offset:row_offset + len(in_neigh), col_offset:col_offset + len(out_neigh)] = \
            D[np.ix_(in_idx, out_idx)]

    # Three layers apart
    elif i_layer + 3 == j_layer:
        key = 'three'
        if key in _GLOBAL_CHAINED_SP:
            D = _GLOBAL_CHAINED_SP[key]
        else:
            if _sp_curr is None or _sp_next is None or _sp_prev is None:
                return
            A = _sp_curr[b]
            B = _sp_next[b]
            C = _sp_prev[b]

            D = min_plus_mult(min_plus_mult(A, B), C)
            _GLOBAL_CHAINED_SP[key] = D

        d_np[row_offset:row_offset + len(in_neigh), col_offset:col_offset + len(out_neigh)] = \
            D[np.ix_(in_idx, out_idx)]



def compute_full_path_matrix(b, in_neigh, out_neigh):
    d_np = np.full((len(in_neigh), len(out_neigh)), np.inf)

    if len(in_neigh) > 1 and len(out_neigh) > 1:
        fill_shortest_paths(d_np,  b, in_neigh[:-1], out_neigh[:-1])
    if len(in_neigh) > 0 and len(out_neigh) > 1:
        fill_shortest_paths(d_np, b, [in_neigh[-1]], out_neigh[:-1], row_offset=len(in_neigh) - 1, col_offset=0, mode = "next")
    if len(in_neigh) > 1 and len(out_neigh) > 0:
        fill_shortest_paths(d_np, b, in_neigh[:-1], [out_neigh[-1]], row_offset=0, col_offset=len(out_neigh) - 1, mode = "prev")
    return d_np



def process_edge(b, i, j):
    t1 = time.time()
    global _sp_curr, _sp_prev, _sp_next, _dist_in, _dist_out, _alpha, _prefix_dims, _pre_n, _W

    i_layer = np.searchsorted(_prefix_dims, i, side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, j, side='right') - 1

    i_idx = i - _prefix_dims[i_layer]
    j_idx = j - _prefix_dims[j_layer]
    sp = _sp_curr[b, i_idx, j_idx]
    
    if i_layer > 0 and j_layer < len(dims)-1:
        m = _W.get(i_layer)
        if m is not None:
            return (b, i, j, 1.0 - m/sp)

    if np.isinf(sp):
        return (b, i+_pre_n, j+_pre_n, 2.0)
    
    if j_layer != i_layer + 1:
        return (b, i+_pre_n, j+_pre_n, 2.0)
    
    # in-neighbors
    if _sp_prev is not None:
        mu = _dist_in[b, :, i_idx]
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
    else:
        in_neigh = [i]
        mu = np.array([1.0])

    # out-neighbors
    if _sp_next is not None:
        nu = _dist_out[b, j_idx, :]
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
    else:
        out_neigh = [j]
        nu = np.array([1.0])
        
    # cost matrix
    d_np = compute_full_path_matrix(0, in_neigh, out_neigh)
    d_np[-1, -1] = sp
    
    # d_np = np.zeros((len(in_neigh), len(out_neigh)))
    # for m_idx, m in enumerate(in_neigh):
    #     for n_idx, n in enumerate(out_neigh):
    #         if (m == n):
    #             d_np[m_idx, n_idx] = 0.
    #         else:
    #             d_np[m_idx, n_idx] = get_shortest_path_between_nodes(m, n, b, i, j)

    if d_np.size == 0 or np.isinf(d_np).all():
        return (b, i+_pre_n, j+_pre_n, 2.0)
    
    M = ot.emd2(mu, nu, d_np)
    val = 1.0 - M / sp
    _W[i_layer] = M
    
    t2 = time.time()
    # print(t2-t1)
    
    return (b, i+_pre_n, j+_pre_n, val)




def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return process_edge(*stuff)


def graph_curvature_main_torch(dims, weights, model_dims=None, alpha=0.0, device='cuda', layers_to_process=None, probability_w=None, weight_idx=0, pre_weights_idx=0, pre_n=0):
    global _sp_curr, _sp_prev, _sp_next, _dist_in, _dist_out, _alpha, _prefix_dims, _pre_n, _shorest_pc, _shorest_cn, _shorest_pn, _GLOBAL_CHAINED_SP
    _alpha = alpha
    _pre_n = pre_n

    weights = weights.to(device)
    batch_size = weights.shape[0]
    prefix_dims = np.cumsum([0] + dims).tolist()
    _prefix_dims = prefix_dims

    ricci_results = defaultdict(list)
    layers = layers_to_process or list(range(len(dims) - 1))
    if layers_to_process is None:
        layers = [l+1 for l in layers]

    layer_idx = {layers[0]-1:pre_weights_idx}
    layer_idx[layers[0]] = weight_idx

    for layer in layers:
        # compute shortest-path matrices
        weight_idx = layer_idx.get(layer, 0)
        sp_curr, weight_idx = get_adjacent_shortest_path(dims, weights, layer, prefix_dims, model_dims, device, weight_idx)
        layer_idx[layer+1] = weight_idx

        sp_prev = None; sp_next = None; dist_in = None; dist_out = None

        if layer-1 > 0:
            weight_idx = layer_idx.get(layer-1, 0)
            sp_prev, weight_idx1 = get_adjacent_shortest_path(dims, weights, layer-1, prefix_dims, model_dims, device, weight_idx)
            layer_idx[layer] = weight_idx1
            sp1 = sp_prev
            if probability_w != None:
                sp1,_ = get_adjacent_shortest_path(dims, probability_w, layer-1, prefix_dims, model_dims, device, weight_idx)
            dist_in = compute_distribution(sp1, alpha, mode='in')

        if layer + 1 < len(dims):
            weight_idx = layer_idx.get(layer+1, 0)
            sp_next, weight_idx2 = get_adjacent_shortest_path(dims, weights, layer+1, prefix_dims, model_dims, device, weight_idx)
            layer_idx[layer+2] = weight_idx2
            sp2 = sp_next
            if probability_w != None:
                sp2,_ = get_adjacent_shortest_path(dims, probability_w, layer+1, prefix_dims, model_dims, device, weight_idx)
            dist_out = compute_distribution(sp2, alpha, mode='out')

            
        # if sp_prev is not None:
        #     shortest_pc = compose_shortest_paths(sp_prev, sp_curr)  # A → C
        #     sp_prev = sp_prev.cpu().numpy()
        # else:
        #     shortest_pc = sp_curr

        # if sp_next is not None:
        #     shortest_cn = compose_shortest_paths(sp_curr, sp_next)  # B → D
        #     shortest_pn = compose_shortest_paths(shortest_pc, sp_next)
        #     sp_next = sp_next.cpu().numpy()
        #     shortest_cn = shortest_cn.cpu().numpy()
        #     shortest_pn = shortest_pn.cpu().numpy()
        # else:
        #     shortest_cn = sp_curr
        #     shortest_pn = sp_curr
        #     shortest_cn = shortest_cn.cpu().numpy()
        #     shortest_pn = shortest_pn.cpu().numpy()


        # Always move these to NumPy
        if sp_curr != None:
            sp_curr = sp_curr.cpu().numpy()
        if sp_prev != None:
            sp_prev = sp_prev.cpu().numpy()
            D = min_plus_mult(sp_prev, sp_curr)
            _GLOBAL_CHAINED_SP['prev'] = D
        if sp_next != None:
            sp_next = sp_next.cpu().numpy()
            D = min_plus_mult(sp_curr, sp_next)
            _GLOBAL_CHAINED_SP['next'] = D
        # shortest_pc = shortest_pc.cpu().numpy()
        
        _sp_curr = sp_curr
        _sp_prev = sp_prev
        _sp_next = sp_next
        # _shorest_pc = shortest_pc
        # _shorest_cn = shortest_cn
        # _shorest_pn = shortest_pn

        _dist_in = dist_in
        _dist_out = dist_out
        
        if _sp_prev != None and _sp_next != None:
            D = min_plus_mult(min_plus_mult(sp_prev, sp_curr), sp_next)
            _GLOBAL_CHAINED_SP['three'] = D

        # build tasks
        tasks = []
        for b in range(batch_size):
            for i_local in range(dims[layer-1]):
                for j_local in range(dims[layer]):
                    if np.isinf(sp_curr[b,i_local,j_local]):
                        continue
                    
                    i = prefix_dims[layer-1] + i_local
                    j = prefix_dims[layer] + j_local
                    tasks.append((b, i, j))
                    
        print(len(tasks))

        # Process edges in parallel
        with get_context('fork').Pool(processes=proc) as pool:
            chunksize, extra = divmod(len(tasks), proc * 4)
            if extra:
                chunksize += 1
                
            # print('Start multi-processing..')
            
            results = pool.imap_unordered(_wrap_compute_single_edge, tasks, chunksize=chunksize)
            pool.close()
            pool.join()
        
        for b, i, j, val in results:
            ricci_results[b].append((i,j,val))

        # cleanup
        del sp_curr, sp_prev, sp_next, dist_in, dist_out, tasks
        torch.cuda.empty_cache()
        gc.collect()

        _sp_curr = None
        _sp_prev = None
        _sp_next = None
        _dist_in = None
        _dist_out = None
        _shorest_pc = None
        _shorest_cn = None
        _shorest_pn = None
        _GLOBAL_CHAINED_SP = {}

    return ricci_results


if __name__ == '__main__':
    # dims = [2, 3, 1]
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
    
    dims = [9, 8, 2, 1]
    
    edge_num = 32 + 8*2 + 2
    weights = torch.rand(1, edge_num)
    
    print(weights)

    ricci_curvature = graph_curvature_main_torch(dims, weights, model_dims=model_dims, layers_to_process=[2,3], weight_idx=32)
    print("Ricci Curvature Results:")
    for b in range(weights.shape[0]):
        print(f"\nBatch {b}:")
        print(ricci_curvature[b])