import numpy as np
import torch


def adaptive_chunksize(max_chunk=512):
    if not torch.cuda.is_available():
        return max_chunk, max_chunk

    free_mem, total_mem = torch.cuda.mem_get_info()
    gb_free = free_mem / (1024**3)
    if gb_free < 10:
        return 64, 64
    elif gb_free < 20:
        return 128, 128
    elif gb_free < 30:
        return 256, 256
    else:
        return max_chunk, max_chunk


def _to_cpu_numpy(x, dtype=np.float32):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=False)
    if torch.is_tensor(x):
        return x.detach().to("cpu", copy=False).numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def _to_numpy_node(node_value):
    return _to_cpu_numpy(node_value, dtype=np.float32)


def _get_matrix_torch(layer_cache, name, device):
    matrix = layer_cache.get(f"{name}__dist")
    if matrix is None:
        return None
    if torch.is_tensor(matrix):
        return matrix if matrix.device.type == device.split(":")[0] else matrix.to(device, non_blocking=True)
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _all_cost_matrices(layer_cache, names, device):
    matrices = {}
    for name in names:
        if not name:
            continue
        matrix = _get_matrix_torch(layer_cache, name, device)
        if matrix is not None:
            matrices[name] = matrix
    return matrices


def _to_numpy_dict(matrix_dict):
    return {name: _to_cpu_numpy(matrix) for name, matrix in matrix_dict.items()}
