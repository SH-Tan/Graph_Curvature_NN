import numpy as np
import torch
import ot

from graph_relation import GRAPH_RELATION
import curv_prune_utils as utils
from multiprocessing import get_context, cpu_count


EPSILON = 1e-7
W_EPSILON = 1e-30

# Cache structure:
# SP_CACHE[layer_id][short_name] = {
#     "curr_dist": np.ndarray,
#     "prev_dist": np.ndarray,
#     "next_dist": np.ndarray,
#     "prev_to_curr_out": np.ndarray,
#     "curr_in_to_next": np.ndarray,
#     "prev_to_next": np.ndarray,
# }
SP_CACHE = {}

# ---- multiprocessing shared globals ----
_SHARED_CURR_DIST = None
_SHARED_PREV_IN = None
_SHARED_NEXT_OUT = None
_SHARED_SP = None
_SHARED_ALPHA = 0.0



def _flatten_node(node_value):
    if node_value is None:
        return None
    if isinstance(node_value, np.ndarray):
        arr = node_value
    elif torch.is_tensor(node_value):
        arr = node_value.detach().float().cpu().numpy()
    else:
        return None

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)



def _resolve_prev_in_names(cur_in_names):
    prev_in = []
    for name in utils._as_name_list(cur_in_names):
        rel = GRAPH_RELATION.get(name, None)
        if rel is None:
            continue
        prev_in.extend(utils._as_name_list(rel.get("prev", [])))
    return prev_in


def _resolve_graph_sets(operations, short_name):
    rel = GRAPH_RELATION.get(short_name, {})

    base_prev_names = rel.get("prev", [])
    prev_in_names = _resolve_prev_in_names(base_prev_names)

    cur_in_names = list(base_prev_names)
    if short_name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        cur_in_names = ["layer_input"]

    next_out_names = rel.get("next", [])

    prev_cost_names = rel.get("prev_cost", base_prev_names)
    next_cost_names = rel.get("next_cost", next_out_names)

    return {
        "prev_in": _flatten_node(operations.get(prev_in_names[0], {}).get("node")),
        "cur_in": _flatten_node(operations.get(cur_in_names[0], {}).get("node")),
        "cur_out": _flatten_node(operations.get(short_name, {}).get("node")),
        "next_out": _flatten_node(operations.get(next_out_names[0], {}).get("node")),
        "prev_cost_names": prev_cost_names,
        "next_cost_names": next_cost_names,
    }


def _get_matrix_torch(layer_cache, name, device):
    matrix = layer_cache.get(f"{name}__dist")
    if matrix is None:
        return None
    if torch.is_tensor(matrix):
        return matrix.to(device)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return torch.as_tensor(arr, dtype=torch.float32, device=device)



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




def _build_node_distribution(node_values, path_sub, alpha, normalize_dim, device):
    if node_values is None or path_sub is None or path_sub.size == 0:
        return None

    node_values = np.asarray(node_values, dtype=np.float32).reshape(-1)
    path_sub = np.asarray(path_sub, dtype=np.float32)

    if path_sub.ndim != 2:
        return None

    if normalize_dim == 0:
        if node_values.size != path_sub.shape[0]:
            return None
        node_tensor = torch.as_tensor(node_values, dtype=torch.float32, device=device).view(-1, 1)
        node_matrix = node_tensor.expand(-1, path_sub.shape[1]).contiguous()
    elif normalize_dim == 1:
        if node_values.size != path_sub.shape[1]:
            return None
        node_tensor = torch.as_tensor(node_values, dtype=torch.float32, device=device).view(1, -1)
        node_matrix = node_tensor.expand(path_sub.shape[0], -1).contiguous()
    else:
        raise ValueError(f"Unsupported normalize_dim: {normalize_dim}")

    valid_mask = torch.isfinite(node_matrix) & (node_matrix > 0)

    weights = torch.exp(-(node_matrix ** 2)) * valid_mask
    sum_weights = weights.sum(dim=normalize_dim, keepdim=True)

    dist = torch.zeros_like(weights, dtype=torch.float32)
    positive = sum_weights > EPSILON
    dist = torch.where(positive, ((1.0 - alpha) * weights) / sum_weights.clamp_min(EPSILON), dist)

    # Keep the CNN-style fallback marker for valid edges when a whole row/column has zero mass.
    if normalize_dim == 0:
        empty = (sum_weights <= EPSILON).squeeze(0)
        if empty.any():
            dist[:, empty] = torch.where(valid_mask[:, empty], torch.full_like(dist[:, empty], -1.0), dist[:, empty])
    else:
        empty = (sum_weights <= EPSILON).squeeze(1)
        if empty.any():
            dist[empty, :] = torch.where(valid_mask[empty, :], torch.full_like(dist[empty, :], -1.0), dist[empty, :])

    dist = dist * valid_mask
    return dist.detach().cpu().numpy().astype(np.float64, copy=False)


def _build_in_distribution(node_values, path_sub, alpha, device):
    return _build_node_distribution(node_values, path_sub, alpha, normalize_dim=0, device=device)


def _build_out_distribution(node_values, path_sub, alpha, device):
    return _build_node_distribution(node_values, path_sub, alpha, normalize_dim=1, device=device)


def _edge_distribution(dist_row_or_col, alpha):
    if dist_row_or_col is None:
        return np.array([1.0], dtype=np.float64), np.empty((0,), dtype=np.int64)

    probs = np.asarray(dist_row_or_col, dtype=np.float64).reshape(-1)
    active = np.flatnonzero(np.isfinite(probs) & (probs > 0))
    if active.size == 0:
        return np.array([1.0], dtype=np.float64), active

    return np.hstack([probs[active], np.array([alpha], dtype=np.float64)]), active


def _edge_cost_matrix(sp, u_idx, v_idx, prev_active, next_active, sp_uv):
    prev_count = int(len(prev_active))
    next_count = int(len(next_active))

    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=np.float64)

    if prev_count > 0 and next_count > 0 and sp["prev_to_next"] is not None:
        cost[:-1, :-1] = sp["prev_to_next"][np.ix_(prev_active, next_active)]
    if prev_count > 0 and sp["prev_to_curr_out"] is not None:
        cost[:-1, -1] = sp["prev_to_curr_out"][prev_active, v_idx]
    if next_count > 0 and sp["curr_in_to_next"] is not None:
        cost[-1, :-1] = sp["curr_in_to_next"][u_idx, next_active]

    cost[-1, -1] = sp_uv
    return np.nan_to_num(cost, nan=1e12, posinf=1e12, neginf=1e12)


def build_layer_cache(model, operations, layer_id, cache=None, device="cuda"):
    """
    Build or update the layer cache with distance matrices.
    Keep this once per layer/model state, then reuse across samples.
    """
    if cache is None:
        cache = {}

    for name in operations.keys():
        if name in ["layer_input", "A_q", "A_k"]:
            continue
        dist_matrix = utils._operation_distance_matrix_torch(model, operations, name, layer_id, device)
        if dist_matrix is not None:
            cache[f"{name}__dist"] = dist_matrix

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

    if (short_name not in ["q_proj", "k_proj"]) and (short_name in sp_cache):
        return sp_cache[short_name], graph_data

    curr_dist = _get_matrix_torch(layer_cache, short_name, device=device)
    if curr_dist is None:
        return None

    prev_dist = _get_matrix_torch(layer_cache, graph_data["prev_cost_names"][0], device)
    next_dist = _get_matrix_torch(layer_cache, graph_data["next_cost_names"][0], device)

    chunk_k, chunk_p = utils.adaptive_chunksize()

    prev_to_curr_out = None
    curr_in_to_next = None
    prev_to_next = None

    if prev_dist is not None:
        prev_to_curr_out = _min_plus_torch(prev_dist, curr_dist, chunk_k=chunk_k, chunk_p=chunk_p)
    if next_dist is not None:
        curr_in_to_next = _min_plus_torch(curr_dist, next_dist, chunk_k=chunk_k, chunk_p=chunk_p)
    if prev_dist is not None and next_dist is not None:
        prev_to_next = (
            _min_plus_torch(prev_to_curr_out, next_dist, chunk_k=chunk_k, chunk_p=chunk_p)
            if (prev_to_curr_out.numel() and next_dist.numel())
            else torch.empty((prev_dist.shape[0], next_dist.shape[1]), dtype=torch.float32, device=device)
        )

    sp = {
        "curr_dist": utils._to_cpu_numpy(curr_dist),
        "prev_dist": None if prev_dist is None else utils._to_cpu_numpy(prev_dist),
        "next_dist": None if next_dist is None else utils._to_cpu_numpy(next_dist),
        "prev_to_curr_out": utils._to_cpu_numpy(prev_to_curr_out),
        "curr_in_to_next": utils._to_cpu_numpy(curr_in_to_next),
        "prev_to_next": utils._to_cpu_numpy(prev_to_next),
    }

    del curr_dist, prev_dist, next_dist, prev_to_curr_out, curr_in_to_next, prev_to_next
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    sp_cache[short_name] = sp
    return sp, graph_data


def _merge_gqa_curvature(curvature, model, layer_id, short_name):
    if short_name not in {"k_proj", "v_proj"}:
        return curvature

    num_q_heads, num_kv_heads, _, head_dim = utils._resolve_attention_dims_for_layer(model, layer_id)
    if num_q_heads == num_kv_heads:
        return curvature

    repeat = num_q_heads // num_kv_heads
    if repeat <= 1:
        return curvature

    # curvature: [out_features, in_features]
    out_dim, in_dim = curvature.shape
    expected = num_q_heads * head_dim
    if out_dim != expected:
        return curvature

    curvature = curvature.view(num_q_heads, head_dim, in_dim)
    curvature = curvature.view(num_kv_heads, repeat, head_dim, in_dim).min(dim=1).values
    return curvature.reshape(num_kv_heads * head_dim, in_dim)



def _compute_single_edge_global(edge):
    u_idx, v_idx = edge

    curr_dist = _SHARED_CURR_DIST
    sp = _SHARED_SP
    alpha = _SHARED_ALPHA

    prev_in_distribution = _SHARED_PREV_IN
    next_out_distribution = _SHARED_NEXT_OUT

    sp_uv = float(curr_dist[u_idx, v_idx])
    if not np.isfinite(sp_uv) or sp_uv <= EPSILON:
        return (v_idx, u_idx, 1.0)

    mu, prev_active = _edge_distribution(
        None if prev_in_distribution is None else prev_in_distribution[:, u_idx],
        alpha,
    )

    nu, next_active = _edge_distribution(
        None if next_out_distribution is None else next_out_distribution[v_idx, :],
        alpha,
    )

    cost = _edge_cost_matrix(sp, u_idx, v_idx, prev_active, next_active, sp_uv)

    try:
        w_dist = float(ot.emd2(mu, nu, cost))
    except Exception:
        return (v_idx, u_idx, 1.0)

    curv = 1.0 - (w_dist / max(sp_uv, EPSILON))
    if (1.0 - alpha) > EPSILON:
        curv = curv / (1.0 - alpha)

    if not np.isfinite(curv):
        curv = 1.0

    return (v_idx, u_idx, np.float32(curv))


def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_single_edge_global(*stuff)


def compute_op_curvature(
    model,
    operations,
    short_name,
    layer_id,
    layer_cache,
    sp_cache=None,
    alpha=0.0,
    device="cpu",
):
    global _SHARED_CURR_DIST
    global _SHARED_PREV_IN
    global _SHARED_NEXT_OUT
    global _SHARED_SP
    global _SHARED_ALPHA


    if operations is None or layer_cache is None:
        return None

    if sp_cache is None:
        sp_cache = SP_CACHE
        
    print(SP_CACHE.keys())

    # dict_keys(['curr_dist', 'prev_dist', 'next_dist', \
    # 'prev_to_curr_out', 'curr_in_to_next', 'prev_to_next'])
    sp, graph_data = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name=short_name,
        sp_cache=sp_cache,
        device=device,
    )
    
    if sp is None:
        return None

    # graph_data = _resolve_graph_sets(operations, short_name)

    curr_dist = sp["curr_dist"]
    prev_dist = sp["prev_dist"]
    next_dist = sp["next_dist"]

    cur_in_dim, cur_out_dim = curr_dist.shape
    prev_in_distribution = _build_in_distribution(graph_data["prev_in"], prev_dist, alpha, device)
    next_out_distribution = _build_out_distribution(graph_data["next_out"], next_dist, alpha, device)
    
    _SHARED_CURR_DIST = curr_dist
    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    _SHARED_SP = sp
    _SHARED_ALPHA = alpha

    finite_edges = np.argwhere(np.isfinite(curr_dist) & (curr_dist > 0))
    if finite_edges.size == 0:
        curvature = torch.from_numpy(output)
        return _merge_gqa_curvature(curvature, model, layer_id, short_name)


    args = [(int(u), int(v)) for u, v in finite_edges]

    print(f'Total {len(args)} edges for op = {short_name}')
    
    output = np.full((cur_out_dim, cur_in_dim), 100.0, dtype=np.float32)

    ctx = get_context("fork")  # important for speed
    proc = cpu_count()

    chunksize = max(500, len(args) // (proc * 2))

    with ctx.Pool(proc) as pool:
        results = pool.map(_wrap_compute_single_edge, args, chunksize=chunksize)

    for v_idx, u_idx, curv in results:
        output[v_idx, u_idx] = curv

    curvature = torch.from_numpy(output)
    curvature = _merge_gqa_curvature(curvature, model, layer_id, short_name)
    return curvature
