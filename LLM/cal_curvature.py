import numpy as np
import torch
import ot

from graph_relation import _resolve_graph_sets
import curv_analysis_utils as analysis_utils
from curv_distribution_utils import (
    _edge_distribution,
    _min_reduce_blocks,
    _normalize_node_value_per_sequence,
)
from curv_model_utils import _operation_distance_matrix_torch
from curv_sequence_utils import (
    _build_oproj_to_att_in_value_map,
    _build_vproj_to_att_out_value_map,
    masked_oproj_value_map_for_seq,
    masked_value_map_for_seq,
)
from curv_shared_utils import _from_shared_numpy, _to_shared_numpy
from curv_tensor_utils import _all_cost_matrices, _get_matrix_torch, adaptive_chunksize
from multiprocessing import get_context
from functools import partial
import multiprocessing as mp
import ctypes
import time

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

EPSILON = 1e-7
proc = mp.cpu_count()

# Cache structure:
# SP_CACHE[layer_id][short_name] = {
#     "curr_dist": np.ndarray,
#     "prev_to_curr_out_all": dict[str, np.ndarray],
#     "curr_in_to_next_all": dict[str, np.ndarray],
#     "prev_to_next_all": dict[str, np.ndarray],
# }
SP_CACHE = {}

# ---- multiprocessing shared globals ----
_SHARED_CURR_DIST = None
_SHARED_PREV_IN = None
_SHARED_NEXT_OUT = None
_SHARED_SP = None
_SHARED_ALPHA = 0.0
_A = None

_SHARED_SHORT_NAME = None
_SHARED_MODEL_META = None
_SHARED_SEQ = None
_SHARED_SEQ_LEN = 1
_FLATTEN_ORDER = "by_out_then_seq"
_SHARED_DYNAMIC_NEXT = None
_SHARED_DYNAMIC_PREV = None
_WORKER_CURR_DIST_SHM = None
_WORKER_PREV_IN_SHM = None
_WORKER_NEXT_OUT_SHM = None


def _safe_inverse_abs(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.abs(arr)
    inv = np.full(arr.shape, np.inf, dtype=np.float32)
    np.divide(1.0, arr, out=inv, where=(arr != 0))
    return inv


def _min_plus_torch(a, b, chunk_k=256, chunk_p=256):
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=a.device)

    m, n = a.shape
    n2, p = b.shape
    assert n == n2, f"Dimension mismatch: {a.shape} vs {b.shape}"

    result = torch.full((m, p), float("inf"), dtype=torch.float32, device=a.device)

    for start_k in range(0, n, chunk_k):
        end_k = min(start_k + chunk_k, n)
        a_chunk = a[:, start_k:end_k]
        b_chunk = b[start_k:end_k, :]

        for start_p in range(0, p, chunk_p):
            end_p = min(start_p + chunk_p, p)
            b_sub = b_chunk[:, start_p:end_p]
            partial = (a_chunk.unsqueeze(2) + b_sub.unsqueeze(0)).min(dim=1).values
            result[:, start_p:end_p] = torch.minimum(result[:, start_p:end_p], partial)

    return result



def _build_node_distribution(node_tensor, node_name, alpha, eps=EPSILON):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    node_tensor = node_tensor.to(dtype=torch.float32)
    if node_tensor.dim() == 3:
        if node_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for node tensor, got shape {tuple(node_tensor.shape)}"
            )
        node_tensor = node_tensor.squeeze(0)
        
    node_tensor = _normalize_node_value_per_sequence(node_tensor, node_name)
    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    weights = torch.exp(-(node_tensor ** 2)) * valid_mask

    sum_weights = weights.sum(dim=-1, keepdim=True)

    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )

    empty_mask = (sum_weights <= eps).expand_as(valid_mask)
    dist = torch.where(empty_mask & valid_mask, torch.full_like(dist, -1.0), dist)

    dist = dist * valid_mask
    return dist.numpy().astype(np.float32, copy=False)


def _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv):
    """
    Static seq-local cost matrix.
    """
    prev_count = int(len(prev_active))
    next_count = int(len(next_active))

    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=np.float32)
    sp = _SHARED_SP

    prev_to_next_all = sp.get("prev_to_next_all", {})
    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    if prev_count > 0 and next_count > 0 and prev_to_next_all:
        block = _min_reduce_blocks([
            matrix[np.ix_(prev_active, next_active)]
            for matrix in prev_to_next_all.values()
        ])
        if block is not None:
            cost[:-1, :-1] = block

    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block

    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block

    cost[-1, -1] = sp_uv
    return cost




def _precompute_vproj_next_distributions(value_map, seq_len, repeat, node_name, alpha, flatten_order):
    out = []
    for s in range(seq_len):
        masked = masked_value_map_for_seq(
            value_map, s, seq_len, repeat, flatten_order=flatten_order
        )
        dist = _build_node_distribution(masked, node_name, alpha)
        out.append(dist)
    return out


def _build_vproj_to_att_out_cost(
    s_in,
    v_idx,
    head_dim,
    repeat,
    flatten_order="by_out_then_seq",
):
    assert _A is not None

    kv_head = v_idx // head_dim
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    # A has shape [batch, num_q_heads, seq_len, seq_len]; use the single active batch.
    block = _safe_inverse_abs(_A[0, q_start:q_end, :_SHARED_SEQ_LEN, s_in].abs())

    if flatten_order == "by_out_then_seq":
        return block.reshape(-1)
    elif flatten_order == "by_seq_then_out":
        return block.transpose(1, 0).reshape(-1)
    else:
        raise ValueError(f"Unknown flatten_order: {flatten_order}")
    



def _precompute_oproj_prev_distributions(value_map, seq_len, node_name, alpha):
    out = []
    for s in range(seq_len):
        masked = masked_oproj_value_map_for_seq(value_map, s, seq_len)
        dist = _build_node_distribution(masked, node_name, alpha)
        out.append(dist)
    return out



def _build_att_out_to_o_cost(
    s_out,
    out_idx,
    head_dim,
):
    if _A is None:
        return np.empty((0,), dtype=np.float64)

    q_head = out_idx // head_dim
    
    # A has shape [batch, num_q_heads, seq_len, seq_len]; use the single active batch.
    block = _safe_inverse_abs(_A[0, q_head, s_out, :].abs())
    return block.reshape(-1)




def _edge_cost_matrix_seq_aware(u_idx, v_idx, prev_active, next_active, sp_uv, seq = 0):
    short_name = _SHARED_SHORT_NAME
    if short_name not in {"v_proj", "o_proj"}:
        return _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv)

    prev_count = int(len(prev_active))
    next_count = int(len(next_active))
    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=np.float64)

    sp = _SHARED_SP
    A = _A
    meta = _SHARED_MODEL_META
    head_dim = meta["head_dim"]
    repeat = meta["repeat"]
    
    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    # For seq-aware ops, both attention weights and attention metadata must exist.
    assert A is not None and meta is not None

    cost[-1, -1] = sp_uv

    # Static prev -> current output-node column
    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block
            
            
    # Static current input-node row -> next
    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block


    # Dynamic attention-coupled part
    if short_name == "v_proj" and next_count > 0:
        dynamic_next = _build_vproj_to_att_out_cost(
                    s_in=seq,
                    v_idx=v_idx,
                    head_dim=head_dim,
                    repeat=repeat,
                    flatten_order=_FLATTEN_ORDER,
                )
        
        # if next_active is a subset, select aligned entries first
        if dynamic_next.shape[0] != next_count:
            dynamic_next = dynamic_next[next_active]
        if dynamic_next.shape[0] != next_count:
            raise ValueError(
                f"dynamic_next shape mismatch: got {dynamic_next.shape}, expected ({next_count},)"
            )
            
        # broadcast as outer sum
        cost[:, :-1] = cost[:, -1][:, None] + dynamic_next[None, :]

    elif short_name == "o_proj" and prev_count > 0:
        dynamic_prev = _build_att_out_to_o_cost(
                    s_out=seq,
                    out_idx=u_idx,
                    head_dim=head_dim,
                )
        
        # if prev_active is a subset, select aligned entries first
        if dynamic_prev.shape[0] != prev_count:
            dynamic_prev = dynamic_prev[prev_active]
        if dynamic_prev.shape[0] != prev_count:
            raise ValueError(
                f"dynamic_prev shape mismatch: got {dynamic_prev.shape}, expected ({prev_count},)"
            )
            
        # prev nodes -> next nodes through current endpoint
        cost[:-1, :] = dynamic_prev[:, None] + cost[-1, :][None, :]

    return cost


def build_layer_cache(model, operations, layer_id, cache=None, device="cuda"):
    """
    Build or update the layer cache with distance matrices.
    Keep this once per layer/model state, then reuse across samples.
    """
    if cache is None:
        cache = {}
        

    for name in operations.keys():
        if name.startswith("prev_"):
            name = name.replace("prev_", "")
        if name in {"layer_input", "A", "Att_out", "gate_up_out"}:
            continue
   
        dist_matrix = _operation_distance_matrix_torch(model, operations, name, layer_id, device)
        cache[f"{name}__dist"] = dist_matrix # 1/|w.T| cpu tensor

    return cache


def build_shortest_path_cache(
    operations,
    layer_cache,
    short_name,
    sp_cache=None,
    device="cuda",
    graph_data =  None
):
    """
    Cache is flat: sp_cache[short_name] = ...
    If you want to rebuild for the next layer, clear sp_cache at the layer boundary.
    """
    if sp_cache is None:
        sp_cache = SP_CACHE
        
    if graph_data is None:
        graph_data = _resolve_graph_sets(operations, short_name)

    if short_name in sp_cache:
        return sp_cache[short_name], graph_data

    curr_dist = _get_matrix_torch(layer_cache, short_name, device=device)
    if curr_dist is None:
        return None, graph_data

    prev_dists = _all_cost_matrices(layer_cache, graph_data["prev_cost_names"], device=device)
    next_dists = _all_cost_matrices(layer_cache, graph_data["next_cost_names"], device=device)

    chunk_k, chunk_p = adaptive_chunksize()

    prev_to_curr_out_all = {}
    curr_in_to_next_all = {}
    prev_to_next_all = {}

    for name, prev_matrix in prev_dists.items():
        prev_to_curr_out_all[name] = _min_plus_torch(prev_matrix, curr_dist, chunk_k=chunk_k, chunk_p=chunk_p)

    for name, next_matrix in next_dists.items():
        curr_in_to_next_all[name] = _min_plus_torch(curr_dist, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p)

    for prev_name, prev_to_curr_out in prev_to_curr_out_all.items():
        for next_name, next_matrix in next_dists.items():
            key = f"{prev_name}->{next_name}"
            prev_to_next_all[key] = (
                _min_plus_torch(prev_to_curr_out, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p)
                if (prev_to_curr_out.numel() and next_matrix.numel())
                else torch.empty((prev_to_curr_out.shape[0], next_matrix.shape[1]), dtype=torch.float32, device=device)
            )

    sp = {
        "curr_dist": curr_dist.cpu().contiguous().numpy(),
        "prev_to_curr_out_all": {k: v.cpu().contiguous().numpy() for k, v in prev_to_curr_out_all.items()},
        "curr_in_to_next_all": {k: v.cpu().contiguous().numpy() for k, v in curr_in_to_next_all.items()},
        "prev_to_next_all": {k: v.cpu().contiguous().numpy() for k, v in prev_to_next_all.items()},
    }
    del prev_dists, next_dists, prev_to_curr_out_all, curr_in_to_next_all, prev_to_next_all
    del curr_dist

    sp_cache[short_name] = sp
    return sp, graph_data


def _compute_single_edge_seq_global(edge_info, seq_info):
    u_idx, v_idx = edge_info

    sp_uv = float(_SHARED_CURR_DIST[u_idx, v_idx])
    seq_idx = _SHARED_SEQ

    # For seq-aware nodes, pick the row for this sequence
    if _SHARED_SHORT_NAME in {"v_proj"}:
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[v_idx]
    else:
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[seq_idx]
    
    if _SHARED_SHORT_NAME in {"o_proj"}:
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[u_idx]
    else:
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[seq_idx]
    
    mu, prev_active = _edge_distribution(prev_row, _SHARED_ALPHA)
    nu, next_active = _edge_distribution(next_row, _SHARED_ALPHA)

    cost = _edge_cost_matrix_seq_aware(
        u_idx=u_idx,
        v_idx=v_idx,
        prev_active=prev_active,
        next_active=next_active,
        sp_uv=sp_uv,
        seq = seq_info
    )
    if np.isinf(cost).any():
        print(f"cost has inf value for u_idx={u_idx}, v_idx={v_idx}, seq={seq_info}")

    try:
        w_dist = float(ot.emd2(mu, nu, cost))
    except Exception:
        return (v_idx, u_idx, float('inf'))

    curv = 1.0 - (w_dist / sp_uv)
    curv = curv / (1-_SHARED_ALPHA)

    return (v_idx, u_idx, np.float32(curv))


def _init_worker(curr_dist_meta, prev_meta, next_meta):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _WORKER_CURR_DIST_SHM, _WORKER_PREV_IN_SHM, _WORKER_NEXT_OUT_SHM

    _WORKER_CURR_DIST_SHM, _SHARED_CURR_DIST = _from_shared_numpy(curr_dist_meta)

    _WORKER_PREV_IN_SHM = None
    _SHARED_PREV_IN = None
    if prev_meta is not None:
        _WORKER_PREV_IN_SHM, _SHARED_PREV_IN = _from_shared_numpy(prev_meta)

    _WORKER_NEXT_OUT_SHM = None
    _SHARED_NEXT_OUT = None
    if next_meta is not None:
        _WORKER_NEXT_OUT_SHM, _SHARED_NEXT_OUT = _from_shared_numpy(next_meta)
        

def _compute_edge_chunk_with_seq(task):
    seq_idx, edge_chunk, prev_meta, next_meta = task

    global _SHARED_SEQ, _SHARED_PREV_IN, _SHARED_NEXT_OUT

    _SHARED_SEQ = seq_idx

    prev_in_distribution = _SHARED_PREV_IN
    next_out_distribution = _SHARED_NEXT_OUT

    local_prev_shm = None
    local_next_shm = None

    if prev_meta is not None:
        local_prev_shm, prev_in_distribution = _from_shared_numpy(prev_meta)
    if next_meta is not None:
        local_next_shm, next_out_distribution = _from_shared_numpy(next_meta)

    old_prev = _SHARED_PREV_IN
    old_next = _SHARED_NEXT_OUT
    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution

    try:
        out = []
        append = out.append
        for edge in edge_chunk:
            append(_compute_single_edge_seq_global(edge, seq_idx))
        return out
    finally:
        _SHARED_PREV_IN = old_prev
        _SHARED_NEXT_OUT = old_next

        if local_prev_shm is not None:
            local_prev_shm.close()
        if local_next_shm is not None:
            local_next_shm.close()



def compute_op_curvature(
    operations,
    short_name,
    layer_id,
    layer_cache,
    sp_cache=None,
    alpha=0.0,
    device="cpu",
    seq_len = 1,
    num_q_heads=0, num_kv_heads=0, head_dim=0, repeat=0,
    flatten_order="by_seq_then_out"
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _FLATTEN_ORDER
    global _SHARED_DYNAMIC_NEXT, _SHARED_DYNAMIC_PREV
    global _SHARED_SHORT_NAME, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    
    _FLATTEN_ORDER = flatten_order

    if operations is None or layer_cache is None:
        return None

    if sp_cache is None:
        sp_cache = SP_CACHE

    sp, graph_data = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name=short_name,
        sp_cache=sp_cache,
        device=device,
    )
    if sp is None:
        return None
    
    curr_dist = sp["curr_dist"]
    in_dim, out_dim = curr_dist.shape
    
    prev_in_distribution = None
    next_out_distribution = None
     
    if ("A" not in graph_data["prev_cost_names"]):
        if graph_data["prev_in"] is not None:
            prev_in_distribution = _build_node_distribution(
                graph_data["prev_in"],
                graph_data["prev_in_name"],
                alpha,
            )
            
    if ("A" not in graph_data["next_cost_names"]):
        if graph_data["next_out"] is not None:
            next_out_distribution = _build_node_distribution(
                graph_data["next_out"],
                graph_data["next_out_name"],
                alpha,
            ) # [batch, seq, hidden size]
    

    _SHARED_CURR_DIST = curr_dist
    _SHARED_SP = sp
    _SHARED_ALPHA = alpha
    _SHARED_SHORT_NAME = short_name
    _SHARED_SEQ_LEN = seq_len
    
    _SHARED_MODEL_META = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "repeat": repeat,
    }

    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    
    curvature = torch.full((out_dim, in_dim), float("inf"), dtype=torch.float32)

    # compute curvature for all parameters oer seq
    if torch.is_tensor(curr_dist):
        curr_dist_np = curr_dist.detach().cpu().numpy().astype(np.float32, copy=False)
        finite_edges = np.argwhere(np.isfinite(curr_dist_np) & (curr_dist_np > 0))
    else:
        curr_dist_np = np.asarray(curr_dist, dtype=np.float32)
        finite_edges = np.argwhere(np.isfinite(curr_dist_np) & (curr_dist_np > 0))
        
    finite_edges = finite_edges[:5]

    edge_chunk_size = 4096
    edge_chunks = [finite_edges[i:i + edge_chunk_size] for i in range(0, len(finite_edges), edge_chunk_size)]
    
    print(f'op = {short_name}, seq = {seq_len}, total edges = {len(finite_edges)} per seq, cur dist shape = {curr_dist.shape}')
    
    analysis_path = analysis_utils.start_curvature_analysis(
        layer_id=layer_id,
        short_name=short_name,
        curvature_shape=curvature.shape,
        total_edges=len(finite_edges),
        seq_len=seq_len,
    )
    analysis_utils.append_cur_dist_analysis(
        analysis_path=analysis_path,
        curr_dist=curr_dist,
    )

    precomputed_prev_dists = None
    precomputed_next_dists = None
    
    
    if ("A" in graph_data["next_cost_names"]) or ("A" in graph_data["prev_cost_names"]):
        _A = operations.get("A", None)
        
        if short_name in {"v_proj"}:
            value_map = _build_vproj_to_att_out_value_map(
                graph_data["next_out"], out_dim, seq_len, head_dim, repeat,
                flatten_order=flatten_order
            )
            precomputed_next_dists = _precompute_vproj_next_distributions(
                value_map, seq_len, repeat, graph_data["next_out_name"], alpha, flatten_order=flatten_order
            )
            
        elif short_name in {"o_proj"}:
            value_map = _build_oproj_to_att_in_value_map(
                graph_data["prev_in"], in_dim, seq_len, head_dim, repeat
            )
            precomputed_prev_dists = _precompute_oproj_prev_distributions(
                value_map, seq_len, graph_data["prev_in_name"], alpha
            )
    
    ctx = get_context("fork")

    base_owned_shms = []
    curr_shm, curr_meta = _to_shared_numpy(curr_dist_np)
    base_owned_shms.append(curr_shm)

    base_prev_meta = None
    base_next_meta = None

    if _SHARED_PREV_IN is not None:
        shm, base_prev_meta = _to_shared_numpy(np.asarray(_SHARED_PREV_IN, dtype=np.float32))
        base_owned_shms.append(shm)

    if _SHARED_NEXT_OUT is not None:
        shm, base_next_meta = _to_shared_numpy(np.asarray(_SHARED_NEXT_OUT, dtype=np.float32))
        base_owned_shms.append(shm)
        
    print(f'Start creating Pool....')
    
    with ctx.Pool(processes=proc, initializer=_init_worker,
        initargs=(curr_meta, base_prev_meta, base_next_meta),) as pool:
        for s in range(seq_len):
            t1 = time.time()

            prev_meta = None
            next_meta = None
            seq_owned_shms = []

            if short_name == "v_proj":
                shm, next_meta = _to_shared_numpy(
                    np.asarray(precomputed_next_dists[s], dtype=np.float32)
                )
                seq_owned_shms.append(shm)

            elif short_name == "o_proj":
                shm, prev_meta = _to_shared_numpy(
                    np.asarray(precomputed_prev_dists[s], dtype=np.float32)
                )
                seq_owned_shms.append(shm)

            task_iter = (
                (s, edge_chunk, prev_meta, next_meta)
                for edge_chunk in edge_chunks
            )
            
            seq_v_parts = []
            seq_u_parts = []
            seq_curv_parts = []

            for chunk_res in pool.imap_unordered(_compute_edge_chunk_with_seq, task_iter, chunksize=8):
                if not chunk_res:
                    continue

                for v, u, c in chunk_res:
                    seq_v_parts.append(v)
                    seq_u_parts.append(u)
                    seq_curv_parts.append(c)

            for shm in seq_owned_shms:
                shm.close()
                shm.unlink()

            print(f"seq = {s}, time = {time.time() - t1} s, ")
            input()

            if seq_v_parts:
                seq_v_idx = torch.tensor(seq_v_parts, dtype=torch.long)
                seq_u_idx = torch.tensor(seq_u_parts, dtype=torch.long)
                seq_curv_vals = torch.tensor(seq_curv_parts, dtype=torch.float32)

                prev_vals = curvature[seq_v_idx, seq_u_idx].clone()
                new_vals = torch.minimum(prev_vals, seq_curv_vals)
                curvature[seq_v_idx, seq_u_idx] = new_vals

                analysis_utils.append_seq_curvature_analysis(
                    analysis_path=analysis_path,
                    seq_idx=s,
                    prev_vals=prev_vals,
                    new_vals=new_vals,
                    v_idx=seq_v_idx,
                    u_idx=seq_u_idx,
                )

    _SHARED_CURR_DIST = None
    _SHARED_PREV_IN = None
    _SHARED_NEXT_OUT = None
    _SHARED_SP = None
    _SHARED_ALPHA = 0.0
    _A = None
    _SHARED_SHORT_NAME = None
    _SHARED_MODEL_META = None

    # curvature = _merge_gqa_curvature(curvature, model, layer_id, short_name)
    return curvature
