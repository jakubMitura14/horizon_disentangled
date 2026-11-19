import os
import h5py
import hdf5plugin
import numpy as np
import orjson
import xxhash
from typing import Dict, Any, Optional, Tuple, List

def get_blosc2_filter(dtype: np.dtype, clevel: int = 5):
    """Returns Blosc2 Zstd filter with SHUFFLE (float32) or BITSHUFFLE (uint8)."""
    if np.issubdtype(dtype, np.floating):
        return hdf5plugin.Blosc2(cname='zstd', clevel=clevel, filters=hdf5plugin.Blosc2.SHUFFLE)
    elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return hdf5plugin.Blosc2(cname='zstd', clevel=clevel, filters=hdf5plugin.Blosc2.BITSHUFFLE)
    else:
        return hdf5plugin.Blosc2(cname='zstd', clevel=clevel, filters=hdf5plugin.Blosc2.SHUFFLE)

def compute_xxhash64(arr: np.ndarray) -> str:
    """Computes xxHash64 hex digest of array bytes."""
    return xxhash.xxh64(arr.tobytes()).hexdigest()

class HDF5ShardWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        exists = os.path.exists(self.filepath)
        if exists:
            self.h5 = h5py.File(self.filepath, 'a', libver='latest')
        else:
            self.h5 = h5py.File(self.filepath, 'w', libver='latest', fs_strategy='fsm')

    def close(self):
        if self.h5:
            self.h5.flush()
            try:
                handle = self.h5.id.get_vfd_handle()
                os.fsync(handle)
            except Exception:
                pass
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def flush_and_sync(self):
        self.h5.flush()
        try:
            handle = self.h5.id.get_vfd_handle()
            os.fsync(handle)
        except Exception:
            pass

    def write_array(self, group: h5py.Group, name: str, data: np.ndarray,
                    chunks: Optional[Tuple[int, ...]] = None,
                    clevel: int = 5,
                    attrs: Optional[Dict[str, Any]] = None) -> Tuple[h5py.Dataset, str]:
        """
        Writes a NumPy array with Blosc2 compression and verified checksum.
        Returns the created dataset and its xxHash64 digest.
        """
        if name in group:
            del group[name]

        digest = compute_xxhash64(data)
        compress_kwargs = get_blosc2_filter(data.dtype, clevel=clevel)

        if chunks is None:
            # Automatic chunking or contiguous for small 1D data
            if data.size < 1024:
                dset = group.create_dataset(name, data=data)
            else:
                dset = group.create_dataset(name, data=data, **compress_kwargs)
        else:
            dset = group.create_dataset(name, data=data, chunks=chunks, **compress_kwargs)

        if attrs:
            for k, v in attrs.items():
                if isinstance(v, str):
                    dset.attrs[k] = np.bytes_(v.encode('ascii', errors='ignore'))
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    dset.attrs[k] = v
                elif isinstance(v, (list, tuple, np.ndarray)):
                    arr_v = np.array(v)
                    if arr_v.dtype.kind in ('U', 'O'):
                        # Fixed-width ASCII strings
                        dset.attrs[k] = np.array([str(x).encode('ascii', errors='ignore') for x in v], dtype='S32')
                    else:
                        dset.attrs[k] = arr_v

        dset.attrs['xxhash64'] = np.bytes_(digest.encode('ascii'))
        return dset, digest

    def write_json_blob(self, group: h5py.Group, name: str, obj: Any) -> Tuple[h5py.Dataset, str]:
        """Serializes dictionary/object via orjson into a compressed uint8 dataset."""
        if name in group:
            del group[name]
        raw_bytes = orjson.dumps(obj)
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        digest = compute_xxhash64(data)
        compress_kwargs = hdf5plugin.Blosc2(cname='zstd', clevel=7, filters=hdf5plugin.Blosc2.SHUFFLE)
        dset = group.create_dataset(name, data=data, chunks=(len(data),), **compress_kwargs)
        dset.attrs['xxhash64'] = np.bytes_(digest.encode('ascii'))
        return dset, digest

    def verify_dataset(self, dset_path: str, expected_digest: str, expected_shape: Tuple[int, ...], expected_dtype: np.dtype) -> bool:
        """
        Re-reads dataset from disk, verifies shape, dtype, finite values, and xxHash64 digest.
        """
        if dset_path not in self.h5:
            return False
        dset = self.h5[dset_path]
        if dset.shape != expected_shape:
            return False
        if dset.dtype != expected_dtype:
            return False
        arr = dset[:]
        if np.issubdtype(arr.dtype, np.floating):
            if not np.isfinite(arr).all():
                return False
        actual_digest = compute_xxhash64(arr)
        return actual_digest == expected_digest
