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
