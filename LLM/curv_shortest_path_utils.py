import torch

from graph_relation import _resolve_graph_sets
from curv_tensor_utils import (
    _all_cost_matrices,
    _get_matrix_torch,
    _get_qk_next_cost,
    adaptive_chunksize,
)

SP_CACHE = {}


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


def build_shortest_path_cache(
    operations,
    layer_cache,
    short_name,
    sp_cache=None,
    device="cuda",
    graph_data=None,
    model_meta=None,
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

    # 1/|w| and transpose shape = (int, out)
    curr_dist = _get_matrix_torch(layer_cache, short_name, device=device)
    if curr_dist is None:
        return None, graph_data

    prev_dists = _all_cost_matrices(layer_cache, graph_data["prev_cost_names"], device=device)

    if short_name in {"q_proj", "k_proj"}:
        cost_n = graph_data["next_cost_names"][0]
        cost = operations.get(cost_n)
        if model_meta is None:
            raise ValueError(f"model_meta is required for {short_name}")
        next_dists = {
            name: _get_qk_next_cost(cost, short_name, model_meta, device=device)
            for name in graph_data["next_cost_names"]
        }
    else:
        next_dists = _all_cost_matrices(layer_cache, graph_data["next_cost_names"], device=device)

    chunk_k, chunk_p = adaptive_chunksize()

    prev_to_curr_out_all = {}
    curr_in_to_next_all = {}
    prev_to_next_all = {}

    for name, prev_matrix in prev_dists.items():
        prev_to_curr_out_all[name] = _min_plus_torch(
            prev_matrix, curr_dist, chunk_k=chunk_k, chunk_p=chunk_p
        )

    for name, next_matrix in next_dists.items():
        curr_in_to_next_all[name] = _min_plus_torch(
            curr_dist, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p
        )

    for prev_name, prev_to_curr_out in prev_to_curr_out_all.items():
        for next_name, next_matrix in next_dists.items():
            key = f"{prev_name}->{next_name}"
            prev_to_next_all[key] = (
                _min_plus_torch(prev_to_curr_out, next_matrix, chunk_k=chunk_k, chunk_p=chunk_p)
                if (prev_to_curr_out.numel() and next_matrix.numel())
                else torch.empty(
                    (prev_to_curr_out.shape[0], next_matrix.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
            )

    sp = {
        "curr_dist": curr_dist.cpu().contiguous().numpy(),
        "prev_to_curr_out_all": {
            k: v.cpu().contiguous().numpy() for k, v in prev_to_curr_out_all.items()
        },
        "curr_in_to_next_all": {
            k: v.cpu().contiguous().numpy() for k, v in curr_in_to_next_all.items()
        },
        "prev_to_next_all": {
            k: v.cpu().contiguous().numpy() for k, v in prev_to_next_all.items()
        },
    }
    del prev_dists, next_dists, prev_to_curr_out_all, curr_in_to_next_all, prev_to_next_all
    del curr_dist

    sp_cache[short_name] = sp
    return sp, graph_data
