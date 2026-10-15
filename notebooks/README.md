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

Written references the notebooks link into:
- [`../docs/PRIMER.md`](../docs/PRIMER.md) — concepts, textbook-style
- [`../docs/RUNBOOK.md`](../docs/RUNBOOK.md) — shell-level setup (installs, GPU driver), which cannot run in a notebook
- [`../docs/qml-lartpc-research-plan.md`](../docs/qml-lartpc-research-plan.md) — the plan these notebooks serve
