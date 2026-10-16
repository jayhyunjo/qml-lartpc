# Notebooks — the student's primary path

Run in order, in JupyterLab, from the project root (`jupyter lab` in `/home/jjo/Work/QML`).
Each ships with its outputs baked in, so you can read before running. Before trusting any result:
**Kernel → Restart & Run All.**

| Notebook | What it does | Runtime |
|---|---|---|
| `00_start_here.ipynb` | orientation, environment check, build + verify the dataset | ~1 min (≈5 min if the dataset needs building) |
| `01_explore_data.ipynb` | the data: tracks vs showers, label purity, the negative-ID trap, normalisation | ~2 min |
| `02_quantum_layer.ipynb` | the circuit: reading it, validating it, timing it, running it on a real patch | ~2 min |
| `03_train.ipynb` | **train**: the four-arm comparison (CNN, deep CNN, quanv, frozen quanv), harness in-cell | ~5 min on GPU |
| `04_study.ipynb` | **the study, small**: AUC vs training size (10²→10⁵), vs parameter count, vs label purity — 63 cached runs | ~45 min cold, seconds cached |
| `05_confusion.ipynb` | **confusion matrices**: 2×2 and species-resolved — which particles leak across track/shower | ~6 min |

## Running somewhere other than wcgpu1 (e.g. FNAL dunegpvm)

The **raw** campaign data (492 GB) lives on BNL `wcgpu1` and is only needed to *build* the patch
file. Everything else needs just **`data/patches_31.npz` (201 MB)** — copy it privately (scp) into
your area, e.g. `/exp/dune/data/users/<user>/` or `/pnfs/dune/persistent/users/<user>/`; it is
derived from internal DUNE MC, so do **not** post it publicly. With it in place, notebooks 00–05
run; the two cells that read raw archives detect its absence and skip with a message (their baked
outputs remain readable here).

Compute: **no GPU is required for this series** — measured on 4 CPU threads (be polite on shared
nodes: `torch.set_num_threads(4)` before training): notebook 03's quantum arm ≈ 1.4 min/seed,
one notebook-04 run ≈ 6 min, the full 63-run 04 sweep ≈ 3 h — and the 04 cache
(`results/study04.jsonl`) makes that sweep resumable after any interruption, so it suits a shared
interactive node. GPU becomes worthwhile at the tuned-study stage (per-point hyperparameter
search), which is a later phase. Longer term: the campaign archives are justIN production
outputs, so they likely exist in DUNE Rucio/MetaCat storage reachable from FNAL — getting the
dataset name removes the wcgpu1 dependency entirely.

Written references the notebooks link into:
- [`../docs/PRIMER.md`](../docs/PRIMER.md) — concepts, textbook-style
- [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) — shell-level setup (installs, GPU driver), which cannot run in a notebook
- the research plan the notebooks cite ("plan §…") is an internal document, not in this repository
