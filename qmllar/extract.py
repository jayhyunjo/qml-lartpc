"""Extract labelled patches from the Wire-Cell AI/ML campaign samples.

READ-ONLY with respect to the campaign data.  Nothing is written back into
the campaign files; labels are derived in memory and only our own consolidated
output file is written.

Label derivation (see docs/wc-aiml-dataset-inventory.md section 4a):
    frame_trackid_1st -> abs() -> mcpart/track_ids -> mcpart/pids -> class

abs() is mandatory: ~90% of pixel track IDs are negative (Geant4 EM-shower
daughter convention) and a raw join silently fails on them.  This matches the
campaign's own classify_pixels.py:359.

We store the raw PDG per patch, so finer label schemes (the campaign's 7-class
Track/Shower/Michel/DeltaRay/Blip) can be derived downstream without
re-extraction.
"""

import glob
import os
import tarfile
import tempfile

import h5py
import numpy as np

# The campaign data has moved once already (/srv/data/1 -> /data/0), so resolve it
# rather than hardcoding.  Override with:  export WC_AIML_BASE=/path/to/training_samples
_CANDIDATES = (
    "/data/0/jjo/WC_AIML/training_samples",
    "/srv/data/1/jjo/WC_AIML/training_samples",
)


def _find_base() -> str:
    env = os.environ.get("WC_AIML_BASE")
    if env:
        if not os.path.isdir(env):
            raise FileNotFoundError(f"WC_AIML_BASE={env} is not a directory")
        return env
    for c in _CANDIDATES:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "Campaign data not found. Tried:\n  " + "\n  ".join(_CANDIDATES)
        + "\nSet WC_AIML_BASE to the training_samples directory."
    )


# Resolved lazily (PEP 562): machines without the raw campaign data (e.g. FNAL
# dunegpvm) can still import this module and run everything that only needs the
# extracted patch file.  Accessing extract.BASE is what raises if data is absent.
def __getattr__(name):
    if name == "BASE":
        return _find_base()
    raise AttributeError(name)

# Coarse classes, matching the PRD's binary track/shower task.
TRACK_PDGS  = {13, 211, 2212, 321}
SHOWER_PDGS = {11, 22, 111}
CLS_TRACK, CLS_SHOWER, CLS_OTHER = 0, 1, -1


def pdg_to_class(pid: int) -> int:
    a = abs(int(pid))
    if a in TRACK_PDGS:
        return CLS_TRACK
    if a in SHOWER_PDGS:
        return CLS_SHOWER
    return CLS_OTHER


def _ke_mev(start_mom):
    """Kinetic energy in MeV from a [px, py, pz, E] 4-vector in GeV."""
    px, py, pz, E = start_mom
    m2 = max(0.0, E * E - px * px - py * py - pz * pz)
    return (E - np.sqrt(m2)) * 1000.0


def extract_event(px_path, tm_path, ev, half, rng, per_anode, anode_idx):
    """Yield (patch, cls, pid, efrac, ke, anode) for one event/anode."""
    out = []
    with h5py.File(tm_path, "r") as ft:
        if ev not in ft:
            return out
        tids = ft[f"/{ev}/mcpart/track_ids"][:].astype(np.int64)
        pids = ft[f"/{ev}/mcpart/pids"][:].astype(np.int64)
        moms = ft[f"/{ev}/mcpart/start_moms"][:]
    t2i = {int(t): i for i, t in enumerate(tids)}

    with h5py.File(px_path, "r") as fp:
        if ev not in fp:
            return out
        g = fp[ev]
        rc = g["frame_rebinned_reco/coords"][:].astype(np.int64)
        rf = g["frame_rebinned_reco/features"][:].astype(np.float32)
        tc = g["frame_trackid_1st/coords"][:].astype(np.int64)
        tf = g["frame_trackid_1st/features"][:]
        ec = g["frame_energyfrac_1st/coords"][:].astype(np.int64)
        ef = g["frame_energyfrac_1st/features"][:].astype(np.float32)

    if len(rc) == 0 or len(tc) == 0:
        return out

    # Dense charge plane for this anode; cheap and transient (~15 MB worst case).
    tmin = int(min(rc[:, 1].min(), tc[:, 1].min()))
    tmax = int(max(rc[:, 1].max(), tc[:, 1].max()))
    nch, ntk = 2560, tmax - tmin + 1
    plane = np.zeros((nch, ntk), dtype=np.float32)
    plane[rc[:, 0], rc[:, 1] - tmin] = rf

    # energyfrac lookup keyed on coords (each frame has its own coord list)
    efrac = {(int(c), int(t)): v for (c, t), v in zip(ec, ef)}

    # candidate centres: labelled pixels far enough from the plane edges
    ch, tk = tc[:, 0], tc[:, 1] - tmin
    ok = (ch >= half) & (ch < nch - half) & (tk >= half) & (tk < ntk - half)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    if idx.size > per_anode:
        idx = rng.choice(idx, per_anode, replace=False)

    for i in idx:
        raw_tid = int(tf[i])
        j = t2i.get(abs(raw_tid))
        if j is None:
            continue
        cls = pdg_to_class(pids[j])
        if cls == CLS_OTHER:
            continue
        c, t = int(ch[i]), int(tk[i])
        patch = plane[c - half:c + half + 1, t - half:t + half + 1].copy()
        out.append((
            patch,
            cls,
            int(pids[j]),
            float(efrac.get((int(tc[i, 0]), int(tc[i, 1])), np.nan)),
            float(_ke_mev(moms[j])),
            anode_idx,
        ))
    return out


def extract_archive(tgz, half=15, per_anode=8, seed=0, tmpdir=None):
    """Extract patches from one .tgz (10 events x 12 anodes). Read-only on source."""
    rng = np.random.default_rng(seed)
    rows = []
    d = tempfile.mkdtemp(dir=tmpdir)
    try:
        with tarfile.open(tgz) as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(d, filter="data")
            else:
                tf.extractall(d)
        tm = glob.glob(f"{d}/sparse/*trackid_pid_map.h5")
        if not tm:
            return rows
        tm = tm[0]
        with h5py.File(tm, "r") as ft:
            events = sorted(ft.keys(), key=lambda k: int(k))
        for px in sorted(glob.glob(f"{d}/sparse/*pixeldata-anode*.h5")):
            ai = int(px.split("anode")[-1].split(".")[0])
            for ev in events:
                rows += extract_event(px, tm, ev, half, rng, per_anode, ai)
    finally:
        os.system(f"rm -rf {d}")
    return rows
