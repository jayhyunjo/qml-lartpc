# Runbook — QML for LArTPC

> **Students: start with the notebooks instead** — [`../notebooks/README.md`](../notebooks/README.md).
> They cover everything here except §1 (installs) and §6 (troubleshooting), with figures and
> explanations inline. This file remains the reference for shell-level setup, which cannot run in a
> notebook.

Commands in the order you run them. Concepts are in **[`PRIMER.md`](PRIMER.md)** — this file
assumes you either know them or will look them up.

Every command below has been run on this machine; the outputs shown are real. Project root is
`/home/jjo/Work/QML`. Nothing here writes to the campaign data — it is read-only throughout.
It currently lives at `/data/0/jjo/WC_AIML/training_samples`; the scripts resolve this
automatically, or set `WC_AIML_BASE` to override.

**Endpoint:** a verified environment, a 415k-patch dataset, and a validated quantum layer.
**Not** a trained model — that code does not exist yet (§5).

---

## 1. Install

Already done on this machine — run §3.1 to confirm, and skip ahead if it passes.

```
mkdir -p /home/jjo/.pip-tmp

TMPDIR=/home/jjo/.pip-tmp python3 -m pip install \
    --cache-dir=/home/jjo/.pip-tmp/cache h5py scikit-learn pennylane

TMPDIR=/home/jjo/.pip-tmp python3 -m pip install \
    --cache-dir=/home/jjo/.pip-tmp/cache --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126 torch
```

⚠️ Two non-obvious bits, both required: `TMPDIR=...` because `/tmp` is a 1.8 GB partition that
fills; `--index-url .../cu126` plus `--force-reinstall` because the default torch is built for
CUDA 13 and this driver supports 12.5. Details in §6.

Torch is several GB onto NFS. To background it:

```
cd /home/jjo && TMPDIR=/home/jjo/.pip-tmp nohup python3 -m pip install \
    --cache-dir=/home/jjo/.pip-tmp/cache --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu126 torch \
    > /home/jjo/.pip-tmp/torch.log 2>&1 &
```

---

## 2. Build the samples

The one command that does real work: reads the campaign archives, extracts 31×31 windows centred on
labelled pixels, and writes one consolidated file.

```
cd /home/jjo/Work/QML
python3 scripts/build_patches.py --n-archives 800 --workers 48
```

```
flavour 0: 800 archives from 001
flavour 1: 800 archives from 001
  1600/1600 archives, 415509 patches, ...

done in ...s; 0 archive errors
  415509 patches, shape (31, 31), 201 MB -> data/patches_31.npz
  train:  292603 patches  track=0.425 shower=0.575  archives=1120
  val  :   60677 patches  track=0.395 shower=0.605  archives=240
  test :   62229 patches  track=0.433 shower=0.567  archives=240
```

**Expect roughly 4 minutes.** The campaign data now sits on `/data/0` (`/dev/sdb`, a **spinning
disk**); it was previously on NVMe, where the same build took 55 s. Measured throughput from the new
location is ~6.7 archives/s against ~29/s before.

`data/patches_31.npz` already exists, so this is a re-run — harmless, and worth watching once.

> On a spinning disk, more workers is not always faster: 48 parallel readers cause seek contention.
> If the build is slower than expected, try `--workers 16`.

| Flag | Default | |
|---|---|---|
| `--n-archives` | 800 | per flavour; ~10,000 available |
| `--half` | 15 | half-width → 31×31 |
| `--per-anode` | 8 | patches per anode per event |
| `--workers` | 48 | 64 cores available |
| `--out` | `data/patches_31.npz` | |

*Why 31×31, why both flavours, why these splits → [PRIMER §3](PRIMER.md#3-preparing-samples).*

---

## 3. Verify

### 3.1 Environment

```
python3 -c "
import pennylane, h5py, numpy, sklearn, torch
print('pennylane', pennylane.__version__, '| h5py', h5py.__version__)
print('numpy    ', numpy.__version__, '| sklearn', sklearn.__version__)
print('torch    ', torch.__version__, '| cuda available:', torch.cuda.is_available())
"
```

```
pennylane 0.45.1 | h5py 3.16.0
numpy     2.4.3 | sklearn 1.9.0
torch     2.13.0+cu126 | cuda available: True
```

**`cuda available: True` is the thing to check.** `False` with 2 devices → §6.

### 3.2 The dataset — leakage and label integrity

```
python3 -c "
import numpy as np
d = np.load('data/patches_31.npz'); s, c, a = d['split'], d['cls'], d['archive']
for name, v in (('train',0), ('val',1), ('test',2)):
    m = s == v
    print(f'{name:5s} {m.sum():7d} patches  track={np.mean(c[m]==0):.3f}  archives={len(np.unique(a[m]))}')
byarch = {int(x): set(s[a == x].tolist()) for x in np.unique(a)}
print('archives spanning >1 split:', sum(len(v) > 1 for v in byarch.values()), '(must be 0)')
print('efrac NaN fraction:', np.isnan(d['efrac']).mean(), '(must be 0.0)')
"
```

```
train  292603 patches  track=0.425  archives=1120
val     60677 patches  track=0.395  archives=240
test    62229 patches  track=0.433  archives=240
archives spanning >1 split: 0 (must be 0)
efrac NaN fraction: 0.0 (must be 0.0)
```

**Both must pass before any training.** A suspiciously high AUC later is almost always leakage.
*Why archives and not patches → [PRIMER §3.3](PRIMER.md#33-three-splits-and-the-leakage-trap).*

### 3.3 The quantum layer

Two independent implementations of the same circuits must agree. Run after any change to
`qmllar/circuits.py` or `qmllar/fast.py`.

```
python3 tests/test_fast_matches_pennylane.py
```

```
  [OK  ] chen2020_conv1_3x3           max|err| = 4.44e-16
  [OK  ] chen2020_conv2_2x2           max|err| = 4.16e-16
  [OK  ] tutorial2020_fig3_2x2        max|err| = 7.77e-16
  [OK  ] prd2025_quanv_2x2            max|err| = 7.77e-16
ALL PASS
```

This test has already caught two silently-wrong bugs. *Which ones, and why it matters →
[PRIMER §4.5](PRIMER.md#45-why-two-implementations).*

---

## 4. Inspect (optional, but do §4.1 at least once)

### 4.1 Draw a circuit

The fastest way to understand what the quantum layer actually is.

```
python3 -c "
import sys; sys.path.insert(0,'.')
import numpy as np, pennylane as qml
from qmllar import configs, build_circuit, init_params
cfg = configs.CHEN_CONV2
params = init_params(cfg, np.random.default_rng(0))
print(qml.draw(build_circuit(cfg).channel_qnode)(np.array([0.1,0.2,0.3,0.4]), params[0]))
"
```

```
0: ──RY(0.10)──RZ(0.01)─╭●──Rot(0.86,-1.45,-2.88)───────────────────────╭●─── ···
1: ──RY(0.20)──RZ(0.04)─╰X─╭●──────────────────────Rot(-3.04,1.97,2.59)─╰X─── ···
2: ──RY(0.29)──RZ(0.09)────╰X─────────────────────╭●──────────────────────── ···
3: ──RY(0.38)──RZ(0.16)───────────────────────────╰X──────────────────────── ···

0: ··· ──Rot(2.25,-2.93,1.44)──────────────────────────┤  <Z>
1: ··· ─╭●─────────────────────Rot(-2.04,2.28,0.26)────┤
2: ··· ─╰X────────────────────╭●───────────────────────┤
3: ··· ───────────────────────╰X───────────────────────┤
```

Four pixels in, one number out. The `RY/RZ` on the left are the data encoding; the CNOTs and `Rot`
gates are the trainable part. *Full annotation →
[PRIMER §4.4](PRIMER.md#44-reading-an-actual-circuit).*

Try `configs.PRD_QUANV`, `configs.TUTORIAL_FIG3`, `configs.CHEN_CONV1` to compare.

### 4.2 Benchmark

```
python3 -c "
import sys, time; sys.path.insert(0,'.')
import numpy as np
from qmllar import configs, init_params
from qmllar.fast import quanv_forward
cfg = configs.PRD_QUANV
params = init_params(cfg, np.random.default_rng(0))
B = 200_000
angles = np.random.default_rng(0).normal(size=(B, cfg.n_qubits, len(cfg.encode_gates)))
quanv_forward(cfg, angles[:1000], params)
t = time.time(); quanv_forward(cfg, angles, params); dt = time.time() - t
print(f'{dt/B*1e6:.2f} us/window | forward pass over 292,603 patches: {dt/B*900*292603/60:.1f} min')
"
```

```
4.65 us/window | forward pass over 292,603 patches: 20.4 min
```

The naive one-call-per-window path takes **331 h** for the same work. *Why →
[PRIMER §5](PRIMER.md#5-two-ways-to-differentiate-a-quantum-circuit).*

### 4.3 Look at the source data

Optional. Unpacks one archive to see the raw format.

```
cd /dev/shm && rm -rf peek && mkdir -p peek && cd peek
D=/data/0/jjo/WC_AIML/training_samples/fdhd_sparse_nue_truthlabel/001
tar xzf $D/$(ls $D | head -1)
h5ls "$(ls sparse/*pixeldata-anode0.h5)/1"
cd /dev/shm && rm -rf peek
```

Note the quoted `$(ls ...)` — a bare glob with `/1` appended will not expand.

---

## 5. Train

**The training skeleton now exists as [`../notebooks/03_train.ipynb`](../notebooks/03_train.ipynb)**
— the differentiable quantum layer is `qmllar/torch_quanv.py` (validated by
`tests/test_torch_matches_fast.py`, forward to 1e-16 and gradients vs finite differences to 1e-9),
and the harness lives in the notebook's cells, deliberately, so it can be read and owned.

What does **not** exist yet: the production sweep scripts below (parameter-count and
training-set-size sweeps at scale, equal-budget HPO). Milestones A2–A4 in the plan.

**Note the order — the conventional baseline comes first and is not optional:**

```
# 1. classical curve (A2) -- no quantum computing involved
python3 scripts/train.py --arm cnn      --patch 21 --norm log1p --seeds 5
python3 scripts/train.py --arm cnn_deep --patch 21 --norm log1p --seeds 5

# 2. only now does a quantum number mean anything (A3)
python3 scripts/train.py --arm quanv        --config prd2025_quanv_2x2 --seeds 10
python3 scripts/train.py --arm quanv_frozen --config prd2025_quanv_2x2 --seeds 10

# 3. sweeps
python3 scripts/sweep.py --axis n-train --values 100,300,1000,3000,10000,100000
```

The claim under test is *comparable performance at far fewer parameters* — a statement about where
the quantum point sits relative to the **classical curve**. You need classical AUC at ~10², ~10³ and
~10⁵ parameters before a quantum AUC means anything.

*What a training harness must contain → [PRIMER §6](PRIMER.md#6-training).*

**One thing to settle first:** charge normalisation. Raw ADC saturates the encoding — 84% of pixels
map to the same angle. See [PRIMER §3.4](PRIMER.md#34-charge-normalisation).

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: [Errno 28] No space left on device` during `pip` | `/tmp` is 1.8 GB and fills | prefix `TMPDIR=/home/jjo/.pip-tmp` |
| `torch.cuda.is_available()` is `False` but `device_count()` is `2` | torch built for CUDA 13; driver 555.42.06 supports 12.5 | reinstall from `.../cu126` **with `--force-reinstall`** |
| `pip install --index-url .../cu126 torch` says "Requirement already satisfied", changes nothing | a torch build is already present | add `--force-reinstall` |
| `h5ls sparse/*anode0.h5/1` → "unable to open file" | shell will not expand a glob with `/1` appended | `h5ls "$(ls sparse/*anode0.h5)/1"` |
| Label composition shows a big "other"/"unknown" class | joined pixel track IDs without `abs()` | [PRIMER §2.2](PRIMER.md#22-how-a-pixel-gets-a-label) |
| Every patch looks identical to the circuit | raw charge; encoding saturated | [PRIMER §3.4](PRIMER.md#34-charge-normalisation) |
| Test AUC suspiciously high | patch-level rather than archive-level split | §3.2 |
| `test_fast_matches_pennylane.py` → MISMATCH | the two implementations diverged | check `n_channels` and per-reupload parameters |
| `FileNotFoundError: Campaign data not found` | the dataset moved | set `export WC_AIML_BASE=/path/to/training_samples` |
| Build reports many archive errors | corrupt tar, or `/dev/shm` full | `df -h /dev/shm`; script prints first 3 errors |
| Identical runs give different results | different random seeds | expected — [PRIMER §6.6](PRIMER.md#66-seeds) |

**Which CUDA index?** `cu124` → torch 2.6.0, `cu126` → torch 2.14.0, `cu128` → torch 2.11.0. Any
`cu12x` works on a 12.x driver; `cu13x` does not. Use `cu126`.
