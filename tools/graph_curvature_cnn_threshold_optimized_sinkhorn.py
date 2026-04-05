import gc
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import ot
import torch
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=sys.maxsize)

EPSILON = 1e-7

_dims = []
_prefix_dims = []
_sp_dict = {}
_distribution_in = {}
_distribution_out = {}
_alpha = 0.0
_pre_n = 0
_nodes_value = None
_layers = 0
_edge_value = None
_model_dims = None
_W_dict = {}
_upper_bound = 1.0
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _format_array_for_log(x):
    if x is None:
        return "None"

    arr = _to_numpy(x)
    return np.array2string(arr, threshold=arr.size, max_line_width=sys.maxsize)


def _safe_cost_scale_torch(C: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(C)
    if not finite.any():
        return torch.tensor(1.0, device=C.device, dtype=C.dtype)
    vals = C[finite]
    scale = torch.median(vals)
    return torch.clamp(scale, min=EPSILON)


def sinkhorn_cost_torch(
    mu,
    nu,
    C,
    reg_scales=[1.0],
    numItermax=2000,
    stopThr=1e-6,
):
    mu = torch.as_tensor(mu, dtype=torch.float64, device=_device)
    nu = torch.as_tensor(nu, dtype=torch.float64, device=_device)
    C  = torch.as_tensor(C,  dtype=torch.float64, device=_device)

    for reg in reg_scales:
        cost = ot.sinkhorn2(
            mu,
            nu,
            C,
            reg=reg,
            method="sinkhorn_log",   # or sinkhorn_stabilized / sinkhorn_epsilon_scaling
            numItermax=numItermax,
            stopThr=stopThr,
            warn=True,
        )

        # row_err = torch.max(torch.abs(P.sum(dim=1) - mu)).item()
        # col_err = torch.max(torch.abs(P.sum(dim=0) - nu)).item()
        # cost = torch.sum(P * C).item()
        # forced_entry = P[7, 18].item()   # row 8, col 19 in 0-based indexing

        # print("reg: ", reg)
        # print("row error:", row_err)
        # print("col error:", col_err)
        # print("transport cost:", cost)
        # print("P[7,18]:", forced_entry)
 
    return float(cost.detach().cpu().item() if isinstance(cost, torch.Tensor) else cost)


def emd2_cost_cpu(mu, nu, C):
    """
    Exact OT on CPU for comparison.
    POT docs note that exact OT uses a C++ CPU backend, so keep this on CPU.
    """
    mu_np = _to_numpy(mu).astype(np.float64)
    nu_np = _to_numpy(nu).astype(np.float64)
    C_np = _to_numpy(C).astype(np.float64)

    finite_mask = np.isfinite(C_np)
    if not finite_mask.any():
        return float("inf")

    try:
        return float(ot.emd2(mu_np, nu_np, C_np))
    except Exception:
        return float("inf")


def out_distribution(model_dims, sp1, device='cuda', thre=0.5, dist=None):
    out_dist_matrices = {}
    inf = torch.tensor(float('inf'), device=device)

    for (i, j), cur_sp in sorted(sp1.items()):
        batch_size, src_size, dst_size = cur_sp.shape
        cur_dist = dist[(i, j)]
        out_matrix = torch.full_like(cur_sp, inf, device=device, dtype=torch.float32)

        cur_layer = model_dims[j + 1]
        cur_n = cur_layer["name"]
        dst_prefix = _prefix_dims[j]

        for src_idx in range(src_size):
            if cur_n == "fc":
                out_neighbors = torch.arange(_prefix_dims[j], _prefix_dims[j + 1], device=device)
                node_slice = _nodes_value[:, out_neighbors]

                if thre <= 0:
                    out_matrix[:, src_idx, :] = 1.0 / node_slice
                    continue

                weight_v = 1.0 / cur_dist[:, src_idx, :]
                valid_mask = node_slice > 0
                valid_weights = weight_v[valid_mask]

                if valid_weights.numel() == 0:
                    out_matrix[:, src_idx, :] = inf
                    continue

                threshold = torch.quantile(valid_weights, thre)
                top_mask = (weight_v >= threshold) & valid_mask
                inv_vals = torch.where(top_mask, 1.0 / node_slice, inf)
                out_matrix[:, src_idx, :] = inv_vals
            else:
                mask = cur_sp[:, src_idx, :] != 0
                if not torch.any(mask):
                    continue

                valid_idx = torch.nonzero(mask[0], as_tuple=True)[0]
                inv_vals = torch.full_like(cur_sp[:, src_idx, :], inf, device=device, dtype=torch.float32)
                global_idx = dst_prefix + valid_idx
                node_slice = _nodes_value[:, global_idx]

                if thre <= 0:
                    node_inv = torch.where(node_slice != 0, 1.0 / node_slice, torch.zeros_like(node_slice))
                    out_matrix[:, src_idx, valid_idx] = node_inv
                    continue

                valid_mask = node_slice > 0
                if not torch.any(valid_mask):
                    continue

                weight_v = 1.0 / cur_dist[:, src_idx, valid_idx]
                valid_weights = weight_v[valid_mask]
                threshold = torch.quantile(valid_weights, thre)
                top_mask = (weight_v >= threshold) & valid_mask
                inv_vals[:, valid_idx] = torch.where(top_mask, 1.0 / node_slice, inf)
                out_matrix[:, src_idx, :] = inv_vals

        out_dist_matrices[(i, j)] = torch.where(out_matrix > 0, out_matrix, inf)

    return out_dist_matrices


def min_plus_mult(a, b, chunk_k=256, chunk_p=256):
    batch_size, m, n = a.shape
    _, n2, p = b.shape
    assert n == n2, "Dimension mismatch"

    device = a.device
    result = torch.full((batch_size, m, p), float('inf'), device=device)

    for start_k in range(0, n, chunk_k):
        end_k = min(start_k + chunk_k, n)
        a_chunk = a[:, :, start_k:end_k]
        b_chunk = b[:, start_k:end_k, :]

        for start_p in range(0, p, chunk_p):
            end_p = min(start_p + chunk_p, p)
            b_sub = b_chunk[:, :, start_p:end_p]
            partial = (a_chunk.unsqueeze(3) + b_sub.unsqueeze(1)).min(dim=2)[0]
            result[:, :, start_p:end_p] = torch.minimum(result[:, :, start_p:end_p], partial)

    return result


def cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda'):
    batch_size, _ = weights.shape
    total_layers = len(model_dims) - 1
    shortest_paths = {}
    inf = torch.tensor(float('inf'), device=device)

    start_layer = max(0, min(_layers) - 1)
    end_layer = min(total_layers - 1, max(_layers) + 1)

    weight_idx = 0
    for i in range(start_layer, end_layer + 1):
        l = i + 1
        current_layer = model_dims[l + 1]
        src_size = prefix_dims[i + 1] - prefix_dims[i]
        dst_size = prefix_dims[i + 2] - prefix_dims[i + 1]

        if current_layer['name'] == 'fc':
            direct_dist = weights[:, weight_idx:weight_idx + src_size * dst_size]
            direct_dist = direct_dist.view(batch_size, src_size, dst_size)
            shortest_paths[(i, i + 1)] = torch.where(direct_dist > 0, direct_dist, inf)
            weight_idx += src_size * dst_size

        elif current_layer['name'] in ['cnn', 'pooling']:
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            padding = current_layer['dim'].get('padding', 0)
            pool = current_layer['dim'].get('pool', False)
            if pool:
                dst_size = dst_size * 2 * 2

            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()
            unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1, 2).int()
            patches = unfolded.shape[1]
            step = k ** 2

            n = 0
            end_col = weight_idx + step * pre_ch
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0, p].tolist()
                    adjacent_m[:, cur_idx, n] = weights[:, weight_idx:end_col]
                    weight_idx = end_col
                    end_col = weight_idx + step * pre_ch
                    n += 1

            shortest_paths[(i, i + 1)] = torch.where(adjacent_m > 0, adjacent_m, inf)

    def adaptive_chunksize(max_chunk=512):
        if not torch.cuda.is_available():
            return 128, 128
        free_mem, _ = torch.cuda.mem_get_info()
        gb_free = free_mem / (1024 ** 3)
        if gb_free < 10:
            return 64, 64
        if gb_free < 20:
            return 128, 128
        if gb_free < 30:
            return 256, 256
        return max_chunk, max_chunk

    for d in range(2, end_layer - start_layer + 2):
        for i in range(start_layer, end_layer - d + 2):
            j = i + d
            if j > end_layer + 1:
                continue

            current_min = torch.full(
                (
                    batch_size,
                    prefix_dims[i + 1] - prefix_dims[i],
                    prefix_dims[j + 1] - prefix_dims[j],
                ),
                float('inf'),
                device=device,
            )
            chunk_k, chunk_p = adaptive_chunksize()

            for k in range(i + 1, j):
                if (i, k) not in shortest_paths or (k, j) not in shortest_paths:
                    continue
                current_min = torch.minimum(
                    current_min,
                    min_plus_mult(shortest_paths[(i, k)], shortest_paths[(k, j)], chunk_k=chunk_k, chunk_p=chunk_p),
                )

            shortest_paths[(i, j)] = current_min

    return shortest_paths


def cnn_adjacent_layer(model_dims, weights, prefix_dims, device='cuda', thre=0.5):
    batch_size, _ = weights.shape
    total_layers = len(model_dims) - 1
    shortest_paths = {}
    inf = torch.tensor(float('inf'), device=device)

    start_layer = max(0, min(_layers) - 1)
    end_layer = min(total_layers - 1, max(_layers) + 1)

    weight_idx = 0
    for i in range(start_layer, end_layer + 1):
        l = i + 1
        current_layer = model_dims[l + 1]
        src_size = prefix_dims[i + 1] - prefix_dims[i]
        dst_size = prefix_dims[i + 2] - prefix_dims[i + 1]

        if current_layer['name'] == 'fc':
            direct_dist = weights[:, weight_idx:weight_idx + src_size * dst_size].view(batch_size, src_size, dst_size)
            edge_slice = _edge_value[:, weight_idx:weight_idx + src_size * dst_size].view(batch_size, src_size, dst_size)

            if thre <= 0:
                shortest_paths[(i, i + 1)] = torch.where(direct_dist > 0, direct_dist, inf)
                weight_idx += src_size * dst_size
                continue

            adjacent_m = torch.full_like(direct_dist, inf, dtype=torch.float32, device=device)
            for col in range(dst_size):
                valid_vals = edge_slice[:, :, col]
                nonzero_mask = valid_vals != 0
                if not torch.any(nonzero_mask):
                    continue
                valid_vals = valid_vals[nonzero_mask]
                threshold_val = torch.quantile(valid_vals, thre)
                top_mask = (edge_slice[:, :, col] >= threshold_val) & nonzero_mask
                top_weights = direct_dist[:, :, col].clone()
                top_weights[~top_mask] = inf
                adjacent_m[:, :, col] = top_weights

            shortest_paths[(i, i + 1)] = torch.where(adjacent_m > 0, adjacent_m, inf)
            weight_idx += src_size * dst_size

        elif current_layer['name'] in ['cnn', 'pooling']:
            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            padding = current_layer['dim'].get('padding', 0)
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()
            unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1, 2).int()
            patches = unfolded.shape[1]
            step = k ** 2

            n = 0
            end_col = weight_idx + step * pre_ch
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0, p].tolist()
                    edge_slice = _edge_value[:, weight_idx:end_col]
                    weight_slice = weights[:, weight_idx:end_col]
                    nonzero_mask = edge_slice != 0

                    if thre <= 0:
                        adjacent_m[:, cur_idx, n] = weight_slice
                    else:
                        valid_vals = edge_slice[nonzero_mask]
                        threshold_val = torch.quantile(valid_vals, thre)
                        top_mask = (edge_slice >= threshold_val) & nonzero_mask
                        weight_slice = weight_slice.clone()
                        weight_slice[~top_mask] = inf
                        weight_slice[~nonzero_mask] = 0
                        adjacent_m[:, cur_idx, n] = weight_slice

                    weight_idx = end_col
                    end_col = weight_idx + step * pre_ch
                    n += 1

            shortest_paths[(i, i + 1)] = adjacent_m

    return shortest_paths


def _build_cost_matrix_torch(b, in_neigh, out_neigh):
    d = torch.full((len(in_neigh), len(out_neigh)), float('inf'), dtype=torch.float64, device=_device)

    if len(in_neigh) > 1 and len(out_neigh) > 1:
        _fill_shortest_paths_torch(d, b, in_neigh[:-1], out_neigh[:-1])
    if len(in_neigh) > 0 and len(out_neigh) > 1:
        _fill_shortest_paths_torch(d, b, [in_neigh[-1]], out_neigh[:-1], row_offset=len(in_neigh) - 1, col_offset=0)
    if len(in_neigh) > 1 and len(out_neigh) > 0:
        _fill_shortest_paths_torch(d, b, in_neigh[:-1], [out_neigh[-1]], row_offset=0, col_offset=len(out_neigh) - 1)

    return d


def _fill_shortest_paths_torch(d, b, in_neigh, out_neigh, row_offset=0, col_offset=0):
    if len(in_neigh) == 0 or len(out_neigh) == 0:
        return

    in_neigh = np.atleast_1d(np.array(in_neigh))
    out_neigh = np.atleast_1d(np.array(out_neigh))

    i_layer = np.searchsorted(_prefix_dims, in_neigh[0], side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, out_neigh[0], side='right') - 1

    device = d.device
    in_idx = torch.as_tensor(in_neigh - _prefix_dims[i_layer], dtype=torch.long, device=device)
    out_idx = torch.as_tensor(out_neigh - _prefix_dims[j_layer], dtype=torch.long, device=device)

    # Keep the cached shortest paths on CPU and move only the batch slice we need.
    sp_tensor = _sp_dict[(i_layer, j_layer)][b].to(device, non_blocking=True)
    submat = sp_tensor.index_select(0, in_idx).index_select(1, out_idx)
    d[row_offset:row_offset + len(in_neigh), col_offset:col_offset + len(out_neigh)] = submat


def _edge_problem_from_graph(b, edge):
    i, j = edge
    i_layer = np.searchsorted(_prefix_dims, i, side='right') - 1
    j_layer = np.searchsorted(_prefix_dims, j, side='right') - 1

    if j_layer != i_layer + 1:
        return None
    if (i_layer, j_layer) not in _sp_dict:
        return None

    model_dim_i = _model_dims[i_layer + 1]
    model_dim_j = _model_dims[j_layer + 1]

    i_size = model_dim_i['dim']['out_size']
    j_size = model_dim_j['dim']['out_size']

    node_j = float(_nodes_value[0, j].item())
    node_i = float(_nodes_value[0, i].item())

    i_idx = i - _prefix_dims[i_layer]
    j_idx = j - _prefix_dims[j_layer]
    sp = float(_sp_dict[(i_layer, j_layer)][b, i_idx, j_idx].item())

    if ((j_layer < len(_dims) - 1) and (node_j <= 0)):
        return {"fallback_curv": 1.0, "sp": sp}
    if ((i_layer > 0) and (node_i <= 0)):
        return {"fallback_curv": 1.0, "sp": sp}

    if model_dim_i['name'] != 'fc':
        pos_i = i_idx % (i_size ** 2)
        i_x = pos_i // i_size
        i_y = pos_i % i_size
    else:
        i_x, i_y = 0, i_size

    if model_dim_j['name'] != 'fc':
        pos_j = j_idx % (j_size ** 2)
        j_x = pos_j // j_size
        j_y = pos_j % j_size
    else:
        j_x, j_y = 0, j_size

    key = (i_layer, i_x, i_y, j_x, j_y)
    if (i_layer > 0) and (j_layer < len(_dims) - 1) and (key in _W_dict):
        m = _W_dict.get(key)
        curv = (1.0 - m / sp) / (1 - _alpha)
        return {"cached_curv": curv, "sp": sp}

    if i_layer == 0:
        mu = np.array([1.0], dtype=np.float64)
        in_neigh = [i]
    else:
        mu = _distribution_in[i_layer][b, :, i_idx].copy()
        if len(np.nonzero(mu)[0]) == 0:
            mu = np.array([1.0], dtype=np.float64)
            in_neigh = [i]
        else:
            if np.any(mu == -1.0):
                tmp = (1.0 - _alpha) / len(np.nonzero(mu)[0])
                mu[mu == -1.0] = tmp
            non_zero = np.nonzero(mu)[0]
            in_neigh = np.array(range(_prefix_dims[i_layer - 1], _prefix_dims[i_layer]))
            in_neigh = list(in_neigh[non_zero]) + [i]
            mu = np.hstack((mu[non_zero], np.array(_alpha)))

    if j_layer == len(_dims) - 1:
        nu = np.array([1.0], dtype=np.float64)
        out_neigh = [j]
    else:
        nu = _distribution_out[j_layer][b, j_idx, :].copy()
        out_neigh = np.array(range(_prefix_dims[j_layer + 1], _prefix_dims[j_layer + 2]))
        if len(np.nonzero(nu)[0]) == 0:
            nu = np.array([1.0], dtype=np.float64)
            out_neigh = [j]
        else:
            if np.any(nu == -1.0):
                tmp = (1.0 - _alpha) / len(np.nonzero(nu)[0])
                nu[nu == -1.0] = tmp
            non_zero = np.nonzero(nu)[0]
            out_neigh = list(out_neigh[non_zero]) + [j]
            nu = np.hstack((nu[non_zero], np.array(_alpha)))

    assert in_neigh[-1] == i and out_neigh[-1] == j
    C = _build_cost_matrix_torch(b, in_neigh, out_neigh)
    C[-1, -1] = sp

    if C.numel() == 0 or torch.isinf(C).all():
        return None

    return {"mu": mu, "nu": nu, "C": C, "sp": sp}


def process_edge_compare_gpu(b, edge, compute_exact=True, sinkhorn_reg_scales=(1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2)):
    problem = _edge_problem_from_graph(b, edge)
    i, j = edge

    if problem is None:
        return {
            "b": b,
            "i": i,
            "j": j,
            "sinkhorn_cost": float('inf'),
            "emd2_cost": float('inf') if compute_exact else None,
            "sinkhorn_curv": 20.0,
            "emd2_curv": 20.0 if compute_exact else None,
            "abs_diff": float('inf') if compute_exact else None,
            "sp": None,
            "mu": None,
            "nu": None,
        }

    if 'fallback_curv' in problem:
        curv = float(problem['fallback_curv'])
        return {
            "b": b,
            "i": i,
            "j": j,
            "sinkhorn_cost": None,
            "emd2_cost": None,
            "sinkhorn_curv": curv,
            "emd2_curv": curv if compute_exact else None,
            "abs_diff": 0.0 if compute_exact else None,
            "sp": problem.get("sp"),
            "mu": None,
            "nu": None,
        }

    if 'cached_curv' in problem:
        curv = float(problem['cached_curv'])
        return {
            "b": b,
            "i": i,
            "j": j,
            "sinkhorn_cost": None,
            "emd2_cost": None,
            "sinkhorn_curv": curv,
            "emd2_curv": curv if compute_exact else None,
            "abs_diff": 0.0 if compute_exact else None,
            "sp": problem.get("sp"),
            "mu": None,
            "nu": None,
        }

    mu = problem['mu']
    nu = problem['nu']
    C = problem['C']
    sp = problem['sp']

    sinkhorn_cost = sinkhorn_cost_torch(mu, nu, C, reg_scales=sinkhorn_reg_scales)
    sinkhorn_curv = (1.0 - sinkhorn_cost / sp) / (1.0 - _alpha) if np.isfinite(sinkhorn_cost) and sp > 0 else 20.0

    emd2_cost = None
    emd2_curv = None
    abs_diff = None
    if compute_exact:
        emd2_cost = emd2_cost_cpu(mu, nu, C)
        emd2_curv = (1.0 - emd2_cost / sp) / (1.0 - _alpha) if np.isfinite(emd2_cost) and sp > 0 else 20.0
        abs_diff = abs(sinkhorn_curv - emd2_curv) if np.isfinite(sinkhorn_curv) and np.isfinite(emd2_curv) else float('inf')

    return {
        "b": b,
        "i": i,
        "j": j,
        "sinkhorn_cost": sinkhorn_cost,
        "emd2_cost": emd2_cost,
        "sinkhorn_curv": sinkhorn_curv,
        "emd2_curv": emd2_curv,
        "abs_diff": abs_diff,
        "sp": sp,
        # "mu": mu,
        # "nu": nu,
        # "C": C
    }


def graph_curvature_main_torch_sinkhorn(
    dims,
    weights,
    model_dims=None,
    device='cuda',
    probability_w=None,
    alpha=0.0,
    pre_n=0,
    layers_to_process=None,
    nodes=None,
    edge_value=None,
    threshold=0.,
    compare_with_emd=True,
    max_compare_edges=None,
    sp_dict = None, 
    sinkhorn_reg_scales=[0.1],
):
    global _dims, _prefix_dims, _sp_dict, _distribution_in, _distribution_out
    global _alpha, _pre_n, _nodes_value, _layers, _edge_value, _model_dims, _upper_bound, _device

    _device = device
    graph_device = torch.device('cpu')
    _alpha = alpha
    _pre_n = pre_n
    _nodes_value = _as_cpu_tensor(nodes) if nodes is not None else None
    _edge_value = _as_cpu_tensor(edge_value) if edge_value is not None else None
    _model_dims = model_dims

    weights = _as_cpu_tensor(weights)
    if probability_w is not None:
        probability_w = _as_cpu_tensor(probability_w)

    batch_size = weights.shape[0]
    prefix_dims = np.cumsum([0] + dims).tolist()
    _dims = dims
    _prefix_dims = np.array(prefix_dims)
    layers = layers_to_process or list(range(len(dims) - 1))
    _layers = layers

    t1 = time.time()
    if model_dims:
        if sp_dict is None:
            sp_dict = cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device=str(graph_device))
            sp_dict = {k: v.detach().cpu() for k, v in sp_dict.items()}
        else:
            sp_dict = {k: _as_cpu_tensor(v) for k, v in sp_dict.items()}
        if probability_w is not None:
            sp1 = cnn_adjacent_layer(model_dims, probability_w, prefix_dims, device=str(graph_device), thre=threshold)
            sp2 = out_distribution(model_dims, sp1, device=str(graph_device), thre=threshold, dist=sp_dict)
        else:
            sp1, sp2 = None, None
    else:
        raise ValueError("model_dims is required for this implementation.")

    matrix_t = time.time() - t1
    _sp_dict = {k: v.detach().cpu() for k, v in sp_dict.items()}

    if probability_w is not None:
        dis_w_in, dis_w_out = sp1, sp2
    else:
        dis_w_in, dis_w_out = sp_dict, sp_dict
        
    del weights
    del probability_w
    del sp1, sp2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    distribution_in, distribution_out = {}, {}

    for layer in range(min(_layers), max(_layers) + 3):
        if (layer - 1, layer) in dis_w_in:
            path_sub = dis_w_in[(layer - 1, layer)]
            mask = (path_sub != float('inf')) & (path_sub != 0)
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
            sum_weights = weights_layer.sum(dim=1)
            dist_prev = ((1.0 - _alpha) * weights_layer) / torch.clamp(sum_weights.unsqueeze(1), min=EPSILON)
            indices = torch.where(sum_weights <= EPSILON)[1]
            if len(indices) > 0:
                mask1 = (path_sub[:, :, indices] != float('inf')) & (path_sub[:, :, indices] != 0)
                dist_prev[:, :, indices] = -1
                dist_prev[:, :, indices] *= mask1
            dist_prev *= mask
            distribution_in[layer] = dist_prev.detach().cpu().numpy()
            
            # free memory
            del path_sub, mask, weights_layer, sum_weights, dist_prev
            
    del dis_w_in

    for layer in range(min(_layers) - 1, max(_layers) + 2):
        if (layer, layer + 1) in dis_w_out:
            path_sub = dis_w_out[(layer, layer + 1)]
            mask = (path_sub != float('inf')) & (path_sub != 0)
            weights_layer = torch.exp(-(path_sub ** 2)) * mask
            sum_weights = weights_layer.sum(dim=2)
            dist_next = ((1.0 - _alpha) * weights_layer) / torch.clamp(sum_weights.unsqueeze(-1), min=EPSILON)
            indices = torch.where(sum_weights <= EPSILON)[1]
            if len(indices) > 0:
                mask1 = (path_sub[:, indices, :] != float('inf')) & (path_sub[:, indices, :] != 0)
                dist_next[:, indices, :] = -1
                dist_next[:, indices, :] *= mask1
            dist_next *= mask
            distribution_out[layer] = dist_next.detach().cpu().numpy()
            
            # free memory
            del path_sub, mask, weights_layer, sum_weights, dist_next
            
    del dis_w_out

    _distribution_in = distribution_in
    _distribution_out = distribution_out
    
    del distribution_in, distribution_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edges = []
    
    # edges = [(0,(3072,21024))] # (0,(62976,66257)) (0,(3072,21024))
    
    for layer in layers:
        sp_array = sp_dict[(layer, layer + 1)]
        for b in range(batch_size):
            non_inf = torch.nonzero(~torch.isinf(sp_array[b]), as_tuple=False)
            for src, dst in non_inf.tolist():
                global_src = prefix_dims[layer] + src
                global_dst = prefix_dims[layer + 1] + dst
                edges.append((b, (global_src, global_dst)))

    if max_compare_edges is not None:
        if max_compare_edges <= 0:
            compare_edges = []
        elif len(edges) <= max_compare_edges:
            compare_edges = edges
        else:
            compare_edges = random.sample(edges, max_compare_edges)
    else:
        compare_edges = edges
        
    del sp_dict

    t2 = time.time()
    ricci_results = defaultdict(list)
    comparison_results = defaultdict(list)

    for b, edge in compare_edges:
        res = process_edge_compare_gpu(
            b,
            edge,
            compute_exact=compare_with_emd,
            sinkhorn_reg_scales=sinkhorn_reg_scales,
        )
        ricci_results[b].append((res['i'] + _pre_n, res['j'] + _pre_n, res['sinkhorn_curv']))
        comparison_results[b].append(res)

    curv_t = time.time() - t2

    if compare_with_emd:
        all_diffs = []
        for b in comparison_results:
            for r in comparison_results[b]:
                if r['abs_diff'] is not None and np.isfinite(r['abs_diff']):
                    all_diffs.append(r['abs_diff'])
        summary = {
            'num_edges_compared': sum(len(v) for v in comparison_results.values()),
            'mean_abs_curv_diff': float(np.mean(all_diffs)) if all_diffs else None,
            'max_abs_curv_diff': float(np.max(all_diffs)) if all_diffs else None,
        }
        with open("sinkhorn vs W_alpha05_new.txt", "a+") as ff:
            for b in comparison_results:
                for r in comparison_results[b]:
                    if r['sinkhorn_cost'] is not None:
                        ff.write(
                            f"{r['b']}\t{r['i']}\t{r['j']}\t"
                            f"{r['sinkhorn_cost']}\t{r['emd2_cost']}\t"
                            f"{r['sinkhorn_curv']}\t{r['emd2_curv']}\t"
                            f"{r['abs_diff']}\t{r['sp']}\t\n"
                            # f"{_format_array_for_log(r['mu'])}\n"
                            # f"{_format_array_for_log(r['nu'])}\n"
                            # f"{_format_array_for_log(r['C'])}\n"
                        )
    else:
        summary = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return ricci_results, matrix_t, curv_t


if __name__ == '__main__':
    dims = [9, 8, 2, 1]
    seed = 29
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_dims = {
        1: {"name": "input", "dim": {"channel": 1, "out_size": 3}},
        2: {"name": "cnn", "dim": {"channel": 2, "kernel": 2, "stride": 1, "out_size": 2}},
        3: {"name": "fc", "dim": {"out_size": 2}},
        4: {"name": "fc", "dim": {"out_size": 1}},
    }

    edge_num = 27
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights = torch.rand(1, edge_num, device=device)
    node = torch.rand(1, edge_num, device=device)
    node[0, 2] = 0
    node[0, 12] = 0
    node[0, 3] = 0
    node[0, 10] = 0
    node[0, 14] = 0
    node[0, 16] = 0
    node[0, 18] = 0

    ricci_curvature, matrix_t, curv_t = graph_curvature_main_torch_sinkhorn(
        dims,
        weights,
        model_dims=model_dims,
        device=device,
        probability_w=node,
        nodes=node,
        edge_value=node,
        compare_with_emd=True,
        max_compare_edges=20,
    )

    print('Ricci Curvature Results (Sinkhorn):')
    for b in range(weights.shape[0]):
        print(f'\\nBatch {b}:')
        for (x, y, c) in ricci_curvature[b]:
            print(x, y, c)
    print(f'\\nMatrix time: {matrix_t:.4f}s')
    print(f'Curvature time: {curv_t:.4f}s')
