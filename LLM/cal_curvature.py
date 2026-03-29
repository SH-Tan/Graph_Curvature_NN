import multiprocessing as mp

import numpy as np
import torch
import ot

from graph_relation import GRAPH_RELATION


EPSILON = 1e-7
W_EPSILON = 1e-30

_MP_EDGE_STATE = {}


def _weight_from_model(model, short_name, layer_id):
    if short_name == "lm_head":
        return model.lm_head.weight.detach().cpu()

    layer = model.model.layers[layer_id]
    if short_name == "q_proj":
        return layer.self_attn.q_proj.weight.detach().cpu()
    if short_name == "k_proj":
        return layer.self_attn.k_proj.weight.detach().cpu()
    if short_name == "v_proj":
        return layer.self_attn.v_proj.weight.detach().cpu()
    if short_name == "o_proj":
        return layer.self_attn.o_proj.weight.detach().cpu()
    if short_name == "gate_proj":
        return layer.mlp.gate_proj.weight.detach().cpu()
    if short_name == "up_proj":
        return layer.mlp.up_proj.weight.detach().cpu()
    if short_name == "down_proj":
        return layer.mlp.down_proj.weight.detach().cpu()
    raise KeyError(f"Unknown op short_name: {short_name}")


def _scalar_distance(weight):
    weight = weight.detach().float().cpu()
    valid_mask = weight.isfinite() & (weight != 0)
    if not valid_mask.any():
        return None
    dist = (1.0 / weight.abs()[valid_mask].clamp_min(W_EPSILON)).mean().item()
    return max(dist, W_EPSILON)


def _distance_matrix(weight):
    weight = weight.detach().float().cpu()
    dist = torch.full_like(weight, float("inf"), dtype=torch.float32)
    valid = weight.isfinite() & (weight != 0)
    if valid.any():
        dist[valid] = 1.0 / weight[valid].abs().clamp_min(W_EPSILON)
    # Linear weights are [out_features, in_features].
    # We use [in_nodes, out_nodes] to match graph edge direction.
    return dist.T.contiguous().numpy().astype(np.float64, copy=False)


def _extra_distance_matrix(extra):
    if extra is None:
        return None

    if torch.is_tensor(extra):
        extra = extra.detach().float().cpu()
        if extra.numel() == 0:
            return None
        if extra.dim() == 4:
            extra = extra.mean(dim=(0, 1))
        elif extra.dim() == 3:
            extra = extra.mean(dim=0)
        elif extra.dim() > 2:
            extra = extra.reshape(extra.shape[-2], extra.shape[-1])

        if extra.dim() != 2:
            return None

        dist = torch.full_like(extra, float("inf"), dtype=torch.float32)
        valid = extra.isfinite() & (extra != 0)
        if valid.any():
            dist[valid] = 1.0 / extra[valid].abs().clamp_min(W_EPSILON)
        return dist.contiguous().numpy().astype(np.float64, copy=False)

    arr = np.asarray(extra, dtype=np.float64)
    if arr.ndim < 2 or arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    dist = np.full(arr.shape, np.inf, dtype=np.float64)
    valid = np.isfinite(arr) & (arr != 0)
    if np.any(valid):
        dist[valid] = 1.0 / np.abs(arr[valid]).clip(min=W_EPSILON)
    return dist


def _operation_distance_matrix(model, operations, short_name, layer_id):
    try:
        return _distance_matrix(_weight_from_model(model, short_name, layer_id))
    except KeyError:
        op = operations.get(short_name, {})
        return _extra_distance_matrix(op.get("extra"))


def _as_name_list(names):
    if names is None:
        return []
    if isinstance(names, str):
        return [names]
    return list(names)


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


def _concat_node_values(operations, names):
    vectors = []
    for name in _as_name_list(names):
        op = operations.get(name)
        if op is None:
            continue
        vec = _flatten_node(op.get("node"))
        if vec is not None and vec.size > 0:
            vectors.append(vec)
    if len(vectors) == 0:
        return None
    return np.concatenate(vectors, axis=0)


def _normalize_neighbor_probabilities(probabilities, alpha):
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs[np.isfinite(probs) & (probs > 0)]

    if probs.size == 0:
        return np.array([1.0], dtype=np.float64)

    total = probs.sum()
    if total <= EPSILON:
        return np.array([1.0], dtype=np.float64)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    probs = ((1.0 - alpha) * probs) / total
    return np.hstack((probs, np.array([alpha], dtype=np.float64)))


def _build_cost(prev_d, curr_d, next_d):
    prev_d = np.asarray(prev_d, dtype=np.float64)
    next_d = np.asarray(next_d, dtype=np.float64)

    rows = len(prev_d) + 1
    cols = len(next_d) + 1

    cost = np.empty((rows, cols), dtype=np.float64)

    if len(prev_d) and len(next_d):
        cost[:-1, :-1] = prev_d[:, None] + curr_d + next_d[None, :]
    if len(prev_d):
        cost[:-1, -1] = prev_d + curr_d
    if len(next_d):
        cost[-1, :-1] = curr_d + next_d

    cost[-1, -1] = curr_d
    return cost


def _edge_distribution_from_dist(dist_vec, alpha):
    dist_vec = np.asarray(dist_vec, dtype=np.float64)
    valid = np.isfinite(dist_vec) & (dist_vec > 0)
    dist_valid = dist_vec[valid]

    if dist_valid.size == 0:
        return np.array([1.0], dtype=np.float64), np.array([], dtype=np.float64)

    scores = np.exp(-(dist_valid ** 2))
    prob = _normalize_neighbor_probabilities(scores, alpha)
    return prob, dist_valid


def _resolve_prev_in_names(cur_in_names):
    prev_in = []
    for name in _as_name_list(cur_in_names):
        rel = GRAPH_RELATION.get(name)
        if rel is None:
            continue
        prev_in.extend(_as_name_list(rel.get("prev", [])))
    return prev_in


def _get_matrix(layer_cache, name):
    matrix = layer_cache.get(f"{name}__dist")
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        return None
    return arr


def _concat_prev_matrices(layer_cache, names, cur_in_dim):
    matrices = []
    for name in _as_name_list(names):
        mat = _get_matrix(layer_cache, name)
        if mat is not None and mat.shape[1] == cur_in_dim:
            matrices.append(mat)
            continue

        scalar = layer_cache.get(name)
        if scalar is not None:
            matrices.append(np.full((1, cur_in_dim), float(scalar), dtype=np.float64))
    if len(matrices) == 0:
        return np.empty((0, cur_in_dim), dtype=np.float64)
    return np.concatenate(matrices, axis=0)


def _concat_next_matrices(layer_cache, names, cur_out_dim):
    matrices = []
    for name in _as_name_list(names):
        mat = _get_matrix(layer_cache, name)
        if mat is not None and mat.shape[0] == cur_out_dim:
            matrices.append(mat)
            continue

        scalar = layer_cache.get(name)
        if scalar is not None:
            matrices.append(np.full((cur_out_dim, 1), float(scalar), dtype=np.float64))
    if len(matrices) == 0:
        return np.empty((cur_out_dim, 0), dtype=np.float64)
    return np.concatenate(matrices, axis=1)


def _init_mp_edge_state(curr_dist, prev_dist, next_dist, alpha):
    global _MP_EDGE_STATE
    _MP_EDGE_STATE = {
        "curr": curr_dist,
        "prev": prev_dist,
        "next": next_dist,
        "alpha": float(alpha),
    }


def _edge_curvature_worker(edge):
    u_idx, v_idx = edge

    curr = _MP_EDGE_STATE["curr"]
    prev = _MP_EDGE_STATE["prev"]
    next_m = _MP_EDGE_STATE["next"]
    alpha = _MP_EDGE_STATE["alpha"]

    sp = float(curr[u_idx, v_idx])
    if not np.isfinite(sp) or sp <= EPSILON:
        return u_idx, v_idx, 1.0

    prev_vec = prev[:, u_idx] if prev.size else np.empty((0,), dtype=np.float64)
    next_vec = next_m[v_idx, :] if next_m.size else np.empty((0,), dtype=np.float64)

    mu, prev_valid = _edge_distribution_from_dist(prev_vec, alpha)
    nu, next_valid = _edge_distribution_from_dist(next_vec, alpha)

    cost = _build_cost(prev_valid, sp, next_valid)

    if cost.size == 0 or np.isinf(cost).all():
        return u_idx, v_idx, 1.0

    try:
        w_dist = float(ot.emd2(mu, nu, cost))
    except Exception:
        return u_idx, v_idx, 1.0

    curv = 1.0 - (w_dist / max(sp, EPSILON))
    if (1.0 - alpha) > EPSILON:
        curv = curv / (1.0 - alpha)

    if not np.isfinite(curv):
        curv = 1.0
    return u_idx, v_idx, curv


def build_layer_cache(model, operations, layer_id, cache=None):
    if cache is None:
        cache = {}

    if cache:
        for name in ["gate_proj", "up_proj", "down_proj"]:
            if name in cache:
                cache[f"prev_{name}"] = cache[name]
            if f"{name}__dist" in cache:
                cache[f"prev_{name}__dist"] = cache[f"{name}__dist"]

    for name in operations.keys():
        try:
            weight = _weight_from_model(model, name, layer_id)
        except KeyError:
            weight = None

        if weight is not None:
            cache[name] = _scalar_distance(weight)

        dist_matrix = _operation_distance_matrix(model, operations, name, layer_id)
        if dist_matrix is not None:
            cache[f"{name}__dist"] = dist_matrix
            if name not in cache:
                finite = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
                if finite.size > 0:
                    cache[name] = float(finite.mean())

    return cache


def build_in_distribution(x, y):
    x = x.squeeze(0)
    return x.unsqueeze(1).expand(-1, y.shape[-1])


def build_out_distribution(y, x):
    y = y.squeeze(0)
    return y.unsqueeze(0).expand(x.shape[-1], -1)


def get_nodes_from_names(operations, names):
    nodes = []
    for name in _as_name_list(names):
        op = operations.get(name)
        if op is None:
            continue
        node = op.get("node")
        if node is not None:
            nodes.append(node)
    return nodes


def _resolve_graph_sets(operations, short_name):
    rel = GRAPH_RELATION.get(short_name, {})

    base_prev_names = _as_name_list(rel.get("prev", []))
    cur_in_names = list(base_prev_names)
    if short_name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        cur_in_names = ["layer_input"]

    next_out_names = _as_name_list(rel.get("next", []))
    prev_in_names = _resolve_prev_in_names(base_prev_names)

    prev_cost_names = _as_name_list(rel.get("prev_cost", base_prev_names))
    next_cost_names = _as_name_list(rel.get("next_cost", next_out_names))

    return {
        "prev_in": _concat_node_values(operations, prev_in_names),
        "cur_in": _concat_node_values(operations, cur_in_names),
        "cur_out": _flatten_node(operations.get(short_name, {}).get("node")),
        "prev_out": _concat_node_values(operations, next_out_names),
        "prev_cost_names": prev_cost_names,
        "next_cost_names": next_cost_names,
    }


def _compute_curvature_matrix(curr_dist, prev_dist, next_dist, alpha):
    in_dim, out_dim = curr_dist.shape
    output = np.full((out_dim, in_dim), 1.0, dtype=np.float32)

    finite_edges = np.argwhere(np.isfinite(curr_dist) & (curr_dist > 0))
    if finite_edges.size == 0:
        return torch.from_numpy(output)

    edges = [tuple(map(int, e)) for e in finite_edges]
    _init_mp_edge_state(curr_dist, prev_dist, next_dist, alpha)

    if len(edges) < 64:
        results = [_edge_curvature_worker(e) for e in edges]
    else:
        workers = min(mp.cpu_count(), len(edges))
        if workers <= 1:
            results = [_edge_curvature_worker(e) for e in edges]
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context()
            with ctx.Pool(
                processes=workers,
                initializer=_init_mp_edge_state,
                initargs=(curr_dist, prev_dist, next_dist, alpha),
            ) as pool:
                chunk = max(64, len(edges) // (workers * 4))
                results = pool.map(_edge_curvature_worker, edges, chunksize=chunk)

    for u_idx, v_idx, curv in results:
        output[v_idx, u_idx] = np.float32(curv)

    return torch.from_numpy(output)


def compute_op_curvature(operations, short_name, layer_cache, alpha=0.0, last_layer=False, device="cpu"):
    del device

    if operations is None or layer_cache is None:
        return None

    graph_data = _resolve_graph_sets(operations, short_name)

    curr_dist = _get_matrix(layer_cache, short_name)
    if curr_dist is None:
        return None

    cur_in_dim, cur_out_dim = curr_dist.shape

    prev_dist = _concat_prev_matrices(layer_cache, graph_data["prev_cost_names"], cur_in_dim)
    next_dist = _concat_next_matrices(layer_cache, graph_data["next_cost_names"], cur_out_dim)

    curvature = _compute_curvature_matrix(
        curr_dist=curr_dist,
        prev_dist=prev_dist,
        next_dist=next_dist,
        alpha=alpha,
    )
    if short_name == "lm_head":
        return {
            "curvature_score": float(curvature.float().mean().item()),
            "input_value": None if graph_data["cur_in"] is None else torch.from_numpy(graph_data["cur_in"]).float(),
            "output_value": None if graph_data["cur_out"] is None else torch.from_numpy(graph_data["cur_out"]).float(),
            "prev_in": graph_data["prev_in"],
            "cur_in": graph_data["cur_in"],
            "cur_out": graph_data["cur_out"],
            "prev_out": graph_data["prev_out"],
            "last_layer": bool(last_layer),
        }

    return curvature
