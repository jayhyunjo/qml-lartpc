#!/usr/bin/env python3
"""Consolidate campaign samples into one training-ready patch file.

Read-only on the campaign data (see qmllar.extract.BASE).  Output goes to data/.

IMPORTANT: train/val/test are split at ARCHIVE level, never at patch level.
Patches from one event share particles, so a patch-level split leaks.
"""
import argparse, glob, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qmllar.extract import BASE, extract_archive

SCRATCH = "/dev/shm/qml_extract"   # RAM-backed; tar extraction is transient


def worker(args):
    tgz, half, per_anode, seed = args
    try:
        return extract_archive(tgz, half=half, per_anode=per_anode,
                               seed=seed, tmpdir=SCRATCH)
    except Exception as e:                      # keep one bad archive from killing the run
        return {"error": f"{tgz}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-archives", type=int, default=800, help="per flavour")
    ap.add_argument("--half", type=int, default=15, help="patch half-width (31x31 default)")
    ap.add_argument("--per-anode", type=int, default=8)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default="data/patches_31.npz")
    a = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)
    samples = {
        0: f"{BASE}/fdhd_sparse_nue_truthlabel/001",    # flavour 0 = nue
        1: f"{BASE}/fdhd_sparse_numu_truthlabel/001",   # flavour 1 = numu
    }

    jobs, meta = [], []
    for flav, d in samples.items():
        files = sorted(glob.glob(f"{d}/*.tgz"))[: a.n_archives]
        print(f"flavour {flav}: {len(files)} archives from {os.path.basename(d)}")
        for k, f in enumerate(files):
            jobs.append((f, a.half, a.per_anode, k))
            meta.append((flav, k))

    # archive-level split, deterministic
    rng = np.random.default_rng(12345)
    order = rng.permutation(len(jobs))
    n = len(order)
    split_of = np.empty(n, dtype=np.int8)
    split_of[order[: int(0.70 * n)]] = 0                      # train
    split_of[order[int(0.70 * n): int(0.85 * n)]] = 1         # val
    split_of[order[int(0.85 * n):]] = 2                       # test

    t0 = time.time()
    P, C, PID, EF, KE, AN, FL, AR, SP = [], [], [], [], [], [], [], [], []
    errs = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, rows in enumerate(ex.map(worker, jobs, chunksize=4)):
            if isinstance(rows, dict):
                errs += 1
                if errs <= 3:
                    print("  ERROR", rows["error"], flush=True)
                continue
            flav, arch = meta[i]
            for patch, cls, pid, ef, ke, an in rows:
                P.append(patch); C.append(cls); PID.append(pid); EF.append(ef)
                KE.append(ke); AN.append(an); FL.append(flav)
                AR.append(i); SP.append(split_of[i])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)} archives, {len(P)} patches, "
                      f"{time.time()-t0:.0f}s", flush=True)

    P = np.stack(P).astype(np.float32)
    out = dict(patches=P, cls=np.array(C, np.int8), pid=np.array(PID, np.int32),
               efrac=np.array(EF, np.float32), ke_mev=np.array(KE, np.float32),
               anode=np.array(AN, np.int8), flavour=np.array(FL, np.int8),
               archive=np.array(AR, np.int32), split=np.array(SP, np.int8))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, **out)

    c = out["cls"]; s = out["split"]
    print(f"\ndone in {time.time()-t0:.0f}s; {errs} archive errors")
    print(f"  {len(c)} patches, shape {P.shape[1:]}, {os.path.getsize(a.out)/1e6:.0f} MB -> {a.out}")
    for name, v in (("train", 0), ("val", 1), ("test", 2)):
        m = s == v
        print(f"  {name:5s}: {m.sum():7d} patches  track={np.mean(c[m]==0):.3f} "
              f"shower={np.mean(c[m]==1):.3f}  archives={len(np.unique(out['archive'][m]))}")


if __name__ == "__main__":
    main()
