from multiprocessing import shared_memory

import numpy as np


def _to_shared_numpy(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr
    meta = (shm.name, arr.shape, str(arr.dtype))
    return shm, meta


def _from_shared_numpy(meta):
    if meta is None:
        return None, None
    name, shape, dtype = meta
    shm = shared_memory.SharedMemory(name=name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)
    return shm, arr



def _load_worker_seq_distribution(seq_idx, seq_metas, shm_map, cache_map):
    if seq_metas is None:
        return None
    if type(seq_metas) is not list:
        return seq_metas
    cached = cache_map.get(seq_idx)
    if cached is not None:
        return cached

    if (type(seq_metas) is list):
        shm, arr = _from_shared_numpy(seq_metas[seq_idx])
        shm_map[seq_idx] = shm
        cache_map[seq_idx] = arr
    return arr


def _to_shared_seq_metas(seq_distributions):
    if seq_distributions is None:
        return None, []
    if type(seq_distributions) is not list:
        return seq_distributions, []

    metas = []
    owned_shms = []
    for seq_dist in seq_distributions:
        shm, meta = _to_shared_numpy(np.asarray(seq_dist, dtype=np.float32))
        owned_shms.append(shm)
        metas.append(meta)
    return metas, owned_shms
