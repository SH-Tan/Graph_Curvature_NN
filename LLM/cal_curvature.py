import numpy as np
import os
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
_SHARED_SAMPLE_IDX = None
_SHARED_MODEL_META = None
_SHARED_SEQ_LEN = 1
_WORKER_CURR_DIST_SHM = None
_WORKER_PREV_IN_SHM = None
_WORKER_NEXT_OUT_SHM = None
_WORKER_PREV_SEQ_METAS = None
_WORKER_NEXT_SEQ_METAS = None
_WORKER_PREV_SEQ_SHMS = {}
_WORKER_NEXT_SEQ_SHMS = {}
_WORKER_PREV_SEQ_CACHE = {}
_WORKER_NEXT_SEQ_CACHE = {}
_INF_COST_LOG_DIR = os.path.join(os.path.dirname(__file__), "cost_inf_debug")


def _as_index_array(active):
    if active is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(active, dtype=np.int64).reshape(-1)


def _append_inf_cost_log(lines, u_idx, v_idx):
    os.makedirs(_INF_COST_LOG_DIR, exist_ok=True)
    log_path = os.path.join(_INF_COST_LOG_DIR, f"edge_u{u_idx}_v{v_idx}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n" + "=" * 80 + "\n")


def _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv):
    """
    Static seq-local cost matrix.
    """
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

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


def _get_min_QK_A_cost(v_idx, u_idx, s, prev_active):
    prev_active = _as_index_array(prev_active)

    head_dim = _SHARED_MODEL_META["head_dim"]
    repeat = _SHARED_MODEL_META["repeat"]
    seq_len = _SHARED_SEQ_LEN

    d = v_idx % head_dim
    kv_head = v_idx // head_dim
    
    # repeated q-heads that share this kv head
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    shared_q_heads = np.arange(q_start, q_end)
    # seq @ repeat
    shared_out_idx = shared_q_heads * head_dim + d   # shape [repeat]
    
    # k cost, _SPK["prev_to_next_all"] = [input, seq * q head]
    k_start = kv_head * seq_len * repeat
    k_end = k_start + repeat * seq_len
    shared_k_heads = np.arange(k_start, k_end)    # [repeat * seq_len]
    
    # all seq @ repeat
    k_down = _V_COST[q_start:q_end, s, d]
    
    # Q only overlaps K on out_seq = s
    shared_start = s * repeat
    shared_end = (s + 1) * repeat
    merged_width = seq_len * repeat

    prev_merged = np.empty((0, merged_width), dtype=np.float32)

    if len(prev_active) > 0:
        # previous input -> Q-local-out
        q_prev = _Q_to_A["prev_to_next_all"][np.ix_(prev_active, shared_out_idx)]     # [P, repeat]

        # previous input -> K-local-node over all seq/repeated heads
        k_prev_prefix = _SPK["prev_to_next_all"][np.ix_(prev_active, shared_k_heads)]   # [P, seq*repeat]
        # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
        k_prev_prefix = k_prev_prefix.reshape(len(prev_active), repeat, seq_len).transpose(0, 2, 1)

        # previous input -> K -> V-out, broadcast repeated-head cost across all seq
        k_prev = k_prev_prefix + k_down[None, None, :]                             # [P, seq, repeat]
        prev_merged = k_prev.reshape(len(prev_active), merged_width)
        
        prev_merged[:, shared_start:shared_end] = np.minimum(
            q_prev,
            prev_merged[:, shared_start:shared_end]
        )

        
    # current input -> Q-local-out
    q_curr = _Q_to_A["curr_in_to_next_all"][u_idx, shared_out_idx]             # [repeat]

    # current input -> K-local-node over all seq/repeated heads
    k_curr_prefix = _SPK["curr_in_to_next_all"][u_idx, shared_k_heads]         # [seq*repeat]
    # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
    k_curr_prefix = k_curr_prefix.reshape(repeat, seq_len).transpose(1, 0)     # [seq, repeat]
    
    # current input -> K -> V-out, broadcast repeated-head cost across all seq
    k_curr = k_curr_prefix + k_down[None, :]                                     # [seq, repeat]
    
    curr_merged = k_curr.reshape(merged_width)
    curr_merged[shared_start:shared_end] = np.minimum(
        q_curr,
        curr_merged[shared_start:shared_end]
    )

    return {
        "prev_to_out": prev_merged,
        "curr_to_out": curr_merged,
    }



def _edge_cost_matrix_seq_aware(u_idx, v_idx, prev_active, next_active, sp_uv, seq = 0):
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

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
        merge_cost = _get_min_QK_A_cost(v_idx, u_idx, seq, prev_active)
        
        dynamic_next = _build_vproj_to_att_out_cost(
            a=A,
            seq_len=_SHARED_SEQ_LEN,
            s_in=seq,
            v_idx=v_idx,
            head_dim=head_dim,
            repeat=repeat,
        )
        
        # if next_active is a subset, select aligned entries first
        if dynamic_next.shape[0] != next_count:
            dynamic_next = dynamic_next[next_active]
        if dynamic_next.shape[0] != next_count:
            raise ValueError(
                f"dynamic_next shape mismatch: got {dynamic_next.shape}, expected ({next_count},)"
            )

        merge_prev = merge_cost["prev_to_out"]
        merge_curr = merge_cost["curr_to_out"]
        
        if merge_prev.shape[1] != next_count and merge_prev.shape[1] > 0:
            merge_prev = merge_prev[:, next_active]
        if merge_curr.shape[0] != next_count:
            merge_curr = merge_curr[next_active]

        # Base dynamic path uses current v_proj edge then attention-to-output.
        base_prev = cost[:-1, -1][:, None] + dynamic_next[None, :]
        base_curr = cost[-1, -1] + dynamic_next

        if prev_count > 0 and merge_prev.shape[1] == next_count:
            cost[:-1, :-1] = np.minimum(base_prev, merge_prev)
        else:
            cost[:-1, :-1] = base_prev

        # Current input row uses the better of direct v->A->out and merged Q/K->A->out.
        cost[-1, :-1] = np.minimum(base_curr, merge_curr)

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
    edge_t0 = time.time()
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
    cost_inf_count = int(np.isinf(cost).sum())
    if cost_inf_count > 0:
        log_lines = [
            f"cost has inf value for u_idx={u_idx}, v_idx={v_idx}, seq={seq_info}",
            f"cost shape={cost.shape}, sp_uv={sp_uv}, short_name={_SHARED_SHORT_NAME}, alpha={_SHARED_ALPHA}",
            f"mu shape={mu.shape}, nu shape={nu.shape}, prev_active_len={len(prev_active)}, next_active_len={len(next_active)}",
        ]
        _append_inf_cost_log(log_lines, u_idx=u_idx, v_idx=v_idx)

    try:
        w_dist = float(ot.emd2(mu, nu, cost))
    except Exception:
        return {
            "seq_idx": seq_info,
            "v_idx": v_idx,
            "u_idx": u_idx,
            "curv": float("inf"),
            "mu_len": int(len(mu)),
            "nu_len": int(len(nu)),
            "w_dist": float("inf"),
            "sp_uv": sp_uv,
            "cost_has_inf": bool(cost_inf_count > 0),
            "cost_inf_count": cost_inf_count,
            "edge_time_sec": float(time.time() - edge_t0),
        }

    curv = 1.0 - (w_dist / sp_uv)
    curv = curv / (1-_SHARED_ALPHA)

    return {
        "seq_idx": seq_info,
        "v_idx": v_idx,
        "u_idx": u_idx,
        "curv": np.float32(curv),
        "mu_len": int(len(mu)),
        "nu_len": int(len(nu)),
        "w_dist": w_dist,
        "sp_uv": sp_uv,
        "cost_has_inf": bool(cost_inf_count > 0),
        "cost_inf_count": cost_inf_count,
        "edge_time_sec": float(time.time() - edge_t0),
    }





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
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN

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
    _SHARED_SAMPLE_IDX = None
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
    sample_idx,
    layer_cache,
    sp_cache=None,
    alpha=0.0,
    device="cpu",
    seq_len = 1,
    num_q_heads=0, num_kv_heads=0, head_dim=0, repeat=0,
    sample_edge_num = -1,
    dataset_name="unknown_dataset",
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN

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
        
        _SPK = {}
        
        if sp_k["prev_to_next_all"]:
            _SPK["prev_to_next_all"] = next(iter(sp_k["prev_to_next_all"].values()))
        _SPK["curr_in_to_next_all"] = sp_k["curr_in_to_next_all"]["q_proj"]  # [input, seq * q_head]

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
    _SHARED_SAMPLE_IDX = sample_idx
    _SHARED_SEQ_LEN = seq_len

    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    
    curvature = torch.full((out_dim, in_dim), float("inf"), dtype=torch.float32)

    curr_dist_np = np.asarray(curr_dist, dtype=np.float64)
    curr_dist_finite_edges = int(np.isfinite(curr_dist_np).sum())
    curr_dist_infinite_edges = int(np.isinf(curr_dist_np).sum())
    finite_edges = np.argwhere(np.isfinite(curr_dist_np) & (curr_dist_np > 0))

    analysis_path = analysis_utils.start_curvature_analysis(
        layer_id=layer_id,
        short_name=short_name,
        sample_idx=sample_idx,
        curvature_shape=curvature.shape,
        seq_len=seq_len,
        dataset_name=dataset_name,
    )
    analysis_utils.start_sample_section(
        analysis_path=analysis_path,
        sample_idx=sample_idx,
    )

    precomputed_prev_dists = None
    precomputed_next_dists = None
    
    
    # build neighbor distribution for v_proj or o_proj with A
    if short_name in {"v_proj", "o_proj"}:
        _A = operations.get("A", None)
        
        if short_name == "v_proj":
            value_map = _build_vproj_to_att_out_value_map(
                graph_data["next_out"], out_dim, seq_len, head_dim, repeat
            )
            
            # # node distribution for all seq w/ mask
            # precomputed_next_dists = _precompute_vproj_next_distributions(
            #     value_map, seq_len, repeat, graph_data["next_out_name"], alpha
            # )
            
            # node distribution for all seq w/o mask
            precomputed_next_dists = _precompute_vproj_next_distributions(value_map, graph_data["next_out_name"], alpha)
      
            # x -> Q -> A -> out
            v_cost = operations.get("v_proj", None)
            if v_cost is not None:                
                # [q_heads, seq, head_dim]
                _V_COST = _build_v_to_att_out_template(v_cost, _SHARED_MODEL_META) 
                
                # cost matrix from inout to attention out via q_proj
                _Q_to_A = _build_x_to_out_cost(
                    _V_COST,
                    sp_q,
                    _SHARED_MODEL_META,
                    device,
                ) # [input, head_dim * q_head]
                
                _V_COST = _V_COST.cpu().contiguous().numpy()
                
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
    
    # select edge if sample_edge_num > 0
    if sample_edge_num > 0 and len(finite_edges) > 0:
        seed = 13
        rng = np.random.default_rng(seed)
        selected_idx = rng.choice(len(finite_edges), size=sample_edge_num, replace=False)
        finite_edges = finite_edges[np.sort(selected_idx)]
    
    print(
        f'op = {short_name}, sample = {sample_idx}, seq = {seq_len}, '
        f'total edges = {len(finite_edges)} per seq, cur dist shape = {curr_dist.shape}'
    )

    print(f'Start creating Pool....')
    mu_len_total = 0.0
    nu_len_total = 0.0
    mu_nu_count = 0
    
    t1 = time.time()
    
    try:
        with ctx.Pool(processes=proc, initializer=_init_worker,
            initargs=(curr_meta, base_prev_meta, base_next_meta, seq_prev_metas, seq_next_metas),) as pool:
            for s in range(seq_len):
                task_iter = (
                    (s, edge)
                    for edge in finite_edges
                )
                
                seq_v_parts = []
                seq_u_parts = []
                seq_curv_parts = []
                seq_edge_results = []

                for edge_res in pool.imap_unordered(_compute_edge_with_seq, task_iter, chunksize=1):
                    if not edge_res:
                        continue

                    v_idx = edge_res["v_idx"]
                    u_idx = edge_res["u_idx"]
                    curv = edge_res["curv"]
                    seq_v_parts.append(v_idx)
                    seq_u_parts.append(u_idx)
                    seq_curv_parts.append(curv)
                    mu_len_total += float(edge_res["mu_len"])
                    nu_len_total += float(edge_res["nu_len"])
                    mu_nu_count += 1
                    seq_edge_results.append(edge_res)

                if seq_v_parts:
                    seq_v_idx = torch.tensor(seq_v_parts, dtype=torch.long)
                    seq_u_idx = torch.tensor(seq_u_parts, dtype=torch.long)
                    seq_curv_vals = torch.tensor(seq_curv_parts, dtype=torch.float32)

                    curvature[seq_v_idx, seq_u_idx] = torch.minimum(
                        curvature[seq_v_idx, seq_u_idx],
                        seq_curv_vals,
                    )

                analysis_utils.append_edge_curvature_details(
                    analysis_path=analysis_path,
                    sample_idx=sample_idx,
                    edge_results=seq_edge_results,
                )
    finally:
        for shm in seq_owned_shms:
            shm.close()
            shm.unlink()

        for shm in base_owned_shms:
            shm.close()
            shm.unlink()
            
    print(f"sample_idx = {sample_idx}, time = {time.time() - t1} s, ")

    _reset_shared_state()

    avg_mu_len = mu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")
    avg_nu_len = nu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")

    analysis_utils.append_final_min_curvature_summary(
        analysis_path=analysis_path,
        sample_idx=sample_idx,
        curvature=curvature,
        avg_mu_len=avg_mu_len,
        avg_nu_len=avg_nu_len,
        curr_dist_finite_edges=curr_dist_finite_edges,
        curr_dist_infinite_edges=curr_dist_infinite_edges,
    )

    # curvature = _merge_gqa_curvature(curvature, model, layer_id, short_name)
    return curvature
