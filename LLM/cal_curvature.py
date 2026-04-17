import numpy as np
import torch
import ot

import curv_analysis_utils as analysis_utils
from curv_distribution_utils import (
    _build_node_distribution,
    _edge_distribution,
    _min_reduce_blocks,
)

from curv_sequence_utils import (
    _build_att_out_to_o_cost,
    _build_oproj_to_att_in_value_map,
    _build_vproj_to_att_out_cost,
    _build_vproj_to_att_out_value_map,
    _precompute_oproj_prev_distributions,
    _precompute_vproj_next_distributions,
)
from curv_shortest_path_utils import build_shortest_path_cache
from curv_shared_utils import _from_shared_numpy, _to_shared_numpy, _load_worker_seq_distribution, _to_shared_seq_metas
from curv_tensor_utils import _build_v_to_att_out_template, _build_x_to_out_cost

from multiprocessing import get_context
import multiprocessing as mp
import time

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

proc = mp.cpu_count()

_SHARED_CURR_DIST = None
_SHARED_PREV_IN = None
_SHARED_NEXT_OUT = None
_SHARED_SP = None
_SHARED_ALPHA = 0.0
_A = None
_Q_to_A = None
_V_COST = None
_SPK = None

_SHARED_SHORT_NAME = None
_SHARED_MODEL_META = None
_SHARED_SEQ_LEN = 1
_FLATTEN_ORDER = "by_out_then_seq"
_WORKER_CURR_DIST_SHM = None
_WORKER_PREV_IN_SHM = None
_WORKER_NEXT_OUT_SHM = None
_WORKER_PREV_SEQ_METAS = None
_WORKER_NEXT_SEQ_METAS = None
_WORKER_PREV_SEQ_SHMS = {}
_WORKER_NEXT_SEQ_SHMS = {}
_WORKER_PREV_SEQ_CACHE = {}
_WORKER_NEXT_SEQ_CACHE = {}


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


def _get_min_QK_A_cost(v_idx, s, prev_active):
    head_dim = _SHARED_MODEL_META["head_dim"]
    repeat = _SHARED_MODEL_META["repeat"]

    prev_active = torch.as_tensor(prev_active, device=_Q_to_A.device, dtype=torch.long)

    d = v_idx % head_dim
    kv_head = v_idx // head_dim
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    shared_q_heads = torch.arange(q_start, q_end, device=_Q_to_A.device)
    shared_out_idx = shared_q_heads * head_dim + d

    # Q cost only exists on shared_out_idx, only for current seq
    q_cost = _Q_to_A[prev_active][:, shared_out_idx]

    # K cost
    k_down = _V_COST[q_start:q_end, s, d]
    if _SPK.ndim == 2:
        seq_len = _SHARED_SEQ_LEN
        base = kv_head * seq_len
        r_idx = torch.arange(repeat - 1, device=_Q_to_A.device)
        k_block_idx = base + r_idx * seq_len
        k_prefix = _SPK[prev_active][k_block_idx]

    k_cost = k_prefix + k_down[None, :]

    # Q only overlaps K on out_seq = s
    shared_start = s * repeat
    shared_end = (s + 1) * repeat

    merged_cost = k_cost.clone()
    merged_cost[:, shared_start:shared_end] = torch.minimum(
        q_cost,
        merged_cost[:, shared_start:shared_end]
    )

    return merged_cost

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
            a=A,
            seq_len=_SHARED_SEQ_LEN,
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
            a=A,
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



def _compute_single_edge_seq_global(edge_info, seq_info):
    u_idx, v_idx = edge_info

    sp_uv = float(_SHARED_CURR_DIST[u_idx, v_idx])

    # For seq-aware nodes, pick the row for this sequence
    if _SHARED_SHORT_NAME == "v_proj":
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[v_idx]
    else:
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[seq_info]
    
    if _SHARED_SHORT_NAME == "o_proj":
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[u_idx]
    else:
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[seq_info]
    
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





def _init_worker(curr_dist_meta, prev_meta, next_meta, prev_seq_metas=None, next_seq_metas=None):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _WORKER_CURR_DIST_SHM, _WORKER_PREV_IN_SHM, _WORKER_NEXT_OUT_SHM
    global _WORKER_PREV_SEQ_METAS, _WORKER_NEXT_SEQ_METAS
    global _WORKER_PREV_SEQ_SHMS, _WORKER_NEXT_SEQ_SHMS
    global _WORKER_PREV_SEQ_CACHE, _WORKER_NEXT_SEQ_CACHE

    _WORKER_CURR_DIST_SHM, _SHARED_CURR_DIST = _from_shared_numpy(curr_dist_meta)

    _WORKER_PREV_IN_SHM = None
    _SHARED_PREV_IN = None
    if prev_meta is not None:
        _WORKER_PREV_IN_SHM, _SHARED_PREV_IN = _from_shared_numpy(prev_meta)

    _WORKER_NEXT_OUT_SHM = None
    _SHARED_NEXT_OUT = None
    if next_meta is not None:
        _WORKER_NEXT_OUT_SHM, _SHARED_NEXT_OUT = _from_shared_numpy(next_meta)

    _WORKER_PREV_SEQ_METAS = prev_seq_metas
    _WORKER_NEXT_SEQ_METAS = next_seq_metas
    _WORKER_PREV_SEQ_SHMS = {}
    _WORKER_NEXT_SEQ_SHMS = {}
    _WORKER_PREV_SEQ_CACHE = {}
    _WORKER_NEXT_SEQ_CACHE = {}


def _compute_edge_with_seq(task):
    seq_idx, edge = task

    global _SHARED_PREV_IN, _SHARED_NEXT_OUT

    prev_in_distribution = _SHARED_PREV_IN
    next_out_distribution = _SHARED_NEXT_OUT

    if _SHARED_SHORT_NAME == "o_proj":
        prev_in_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_PREV_SEQ_METAS,
            _WORKER_PREV_SEQ_SHMS,
            _WORKER_PREV_SEQ_CACHE,
        )
    elif _SHARED_SHORT_NAME == "v_proj":
        next_out_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_NEXT_SEQ_METAS,
            _WORKER_NEXT_SEQ_SHMS,
            _WORKER_NEXT_SEQ_CACHE,
        )

    old_prev = _SHARED_PREV_IN
    old_next = _SHARED_NEXT_OUT
    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution

    try:
        return _compute_single_edge_seq_global(edge, seq_idx)
    finally:
        _SHARED_PREV_IN = old_prev
        _SHARED_NEXT_OUT = old_next


def _reset_shared_state():
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SHORT_NAME, _SHARED_MODEL_META, _SHARED_SEQ_LEN

    _SHARED_CURR_DIST = None
    _SHARED_PREV_IN = None
    _SHARED_NEXT_OUT = None
    _SHARED_SP = None
    _SHARED_ALPHA = 0.0
    _A = None
    _Q_to_A = None
    _V_COST = None
    _SPK = None
    _SHARED_SHORT_NAME = None
    _SHARED_MODEL_META = None
    _SHARED_SEQ_LEN = 1


def _get_vproj_aux_shortest_paths(operations, layer_cache, sp_cache, device):
    sp_q, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="q_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    sp_k, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="k_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    return sp_q, sp_k

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
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _FLATTEN_ORDER, _SPK
    global _SHARED_SHORT_NAME, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    
    _FLATTEN_ORDER = flatten_order

    if operations is None or layer_cache is None:
        return None

    _reset_shared_state()
    _SHARED_MODEL_META = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "repeat": repeat,
    }

    sp, graph_data = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name=short_name,
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    if sp is None:
        return None

    if short_name == "v_proj":
        sp_q, sp_k = _get_vproj_aux_shortest_paths(
            operations=operations,
            layer_cache=layer_cache,
            sp_cache=sp_cache,
            device=device,
        )
        if sp_q is None or sp_k is None:
            raise ValueError("q_proj and k_proj shortest-path caches are required for v_proj")
        
        _SPK = (
            next(iter(sp_k["prev_to_next_all"].values()))
            if sp_k["prev_to_next_all"]
            else sp_k["curr_in_to_next_all"]["q_proj"]
        ) # [input, seq * q_head]

    curr_dist = sp["curr_dist"]
    in_dim, out_dim = curr_dist.shape
    
    prev_in_distribution = None
    next_out_distribution = None
    
    # if is o_proj, prev is A
    if (short_name not in ["o_proj"]) and graph_data["prev_in"] is not None:
        prev_in_distribution = _build_node_distribution(
            graph_data["prev_in"],
            graph_data["prev_in_name"],
            alpha,
        )
    
    # if is q, k, v, the next is A or attention out     
    if (short_name not in ["q_proj", "k_proj", "v_proj"]) and graph_data["next_out"] is not None:
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

    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    
    curvature = torch.full((out_dim, in_dim), float("inf"), dtype=torch.float32)

    curr_dist_np = np.asarray(curr_dist, dtype=np.float32)
    finite_edges = np.argwhere(np.isfinite(curr_dist_np) & (curr_dist_np > 0))
        
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
    
    
    if short_name in {"v_proj", "o_proj"}:
        _A = operations.get("A", None)
        
        if short_name == "v_proj":
            value_map = _build_vproj_to_att_out_value_map(
                graph_data["next_out"], out_dim, seq_len, head_dim, repeat,
                flatten_order=flatten_order
            )
            
            # node distribution for all seq
            precomputed_next_dists = _precompute_vproj_next_distributions(
                value_map, seq_len, repeat, graph_data["next_out_name"], alpha, flatten_order=flatten_order
            )

            # x -> Q -> A -> out
            v_cost = operations.get("v_proj", None)
            if v_cost is not None:
                _V_COST = _build_v_to_att_out_template(v_cost, _SHARED_MODEL_META)
                _Q_to_A = _build_x_to_out_cost(
                    _V_COST,
                    sp_q,
                    _SHARED_MODEL_META,
                    device,
                )
                
        elif short_name == "o_proj":
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

    seq_prev_metas, prev_seq_shms = _to_shared_seq_metas(precomputed_prev_dists)
    seq_next_metas, next_seq_shms = _to_shared_seq_metas(precomputed_next_dists)
    seq_owned_shms = prev_seq_shms + next_seq_shms

    print(f'Start creating Pool....')
    
    try:
        with ctx.Pool(processes=proc, initializer=_init_worker,
            initargs=(curr_meta, base_prev_meta, base_next_meta, seq_prev_metas, seq_next_metas),) as pool:
            for s in range(seq_len):
                t1 = time.time()

                task_iter = (
                    (s, edge)
                    for edge in finite_edges
                )
                
                seq_v_parts = []
                seq_u_parts = []
                seq_curv_parts = []

                for edge_res in pool.imap_unordered(_compute_edge_with_seq, task_iter, chunksize=1):
                    if not edge_res:
                        continue

                    v_idx, u_idx, curv = edge_res
                    seq_v_parts.append(v_idx)
                    seq_u_parts.append(u_idx)
                    seq_curv_parts.append(curv)

                print(f"seq = {s}, time = {time.time() - t1} s, ")

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
    finally:
        for shm in seq_owned_shms:
            shm.close()
            shm.unlink()

        for shm in base_owned_shms:
            shm.close()
            shm.unlink()

    _reset_shared_state()

    # curvature = _merge_gqa_curvature(curvature, model, layer_id, short_name)
    return curvature
