"""Fast loading of the patch dataset.

data/patches_31.npz is compressed (201 MB on disk, 1.6 GB in memory) so that it
is cheap to copy between machines -- but zlib decompression of the patch array
costs ~160 s single-threaded, every time, in every notebook.

load_patches() pays that once and then keeps an uncompressed, memory-mappable
cache beside it.  Subsequent loads are effectively instant, and because the
patch array is memory-mapped, only the pages actually touched are read.

    from qmllar.data import load_patches
    d = load_patches()            # dict-like: d["patches"], d["cls"], ...

Cache location: <npz dir>/cache/ by default; override with $QML_CACHE_DIR
(point it at local disk if the data lives on a slow network filesystem).
"""
import os
import pathlib

import numpy as np

_SMALL = ("cls", "pid", "efrac", "ke_mev", "anode", "flavour", "archive", "split")


def _cache_dir(npz_path: pathlib.Path) -> pathlib.Path:
    env = os.environ.get("QML_CACHE_DIR")
    return pathlib.Path(env) if env else npz_path.parent / "cache"


def load_patches(npz_path=None, verbose=True):
    """Return a dict of the dataset arrays, building a fast cache on first use.

    patches is returned as a read-only memmap; the small arrays are in RAM.
    """
    if npz_path is None:
        here = pathlib.Path(__file__).resolve().parent.parent
        npz_path = here / "data" / "patches_31.npz"
    npz_path = pathlib.Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Build it with scripts/build_patches.py, or copy "
            "it from a machine that has it (it is the only data file the notebooks need)."
        )

    cdir = _cache_dir(npz_path)
    patches_npy, meta_npz = cdir / "patches.npy", cdir / "meta.npz"
    stamp = cdir / "source.txt"
    fresh = (patches_npy.exists() and meta_npz.exists() and stamp.exists()
             and stamp.read_text().strip() == f"{npz_path}:{npz_path.stat().st_mtime_ns}")

    if not fresh:
        if verbose:
            print(f"building one-time fast cache in {cdir} (~3 min; later loads are instant)…",
                  flush=True)
        cdir.mkdir(parents=True, exist_ok=True)
        with np.load(npz_path) as z:
            np.save(patches_npy, z["patches"])                 # uncompressed -> memmap-able
            np.savez(meta_npz, **{k: z[k] for k in _SMALL if k in z.files})
        stamp.write_text(f"{npz_path}:{npz_path.stat().st_mtime_ns}")
        if verbose:
            print("cache built.", flush=True)

    out = {"patches": np.load(patches_npy, mmap_mode="r")}
    with np.load(meta_npz) as m:
        out.update({k: m[k] for k in m.files})
    return out
