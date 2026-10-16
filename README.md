# QML for LArTPC

Can a classifier built partly from a small **quantum circuit** match a conventional neural network
on liquid-argon TPC images **using far fewer adjustable parameters** — and if it can, is the cause
*quantum*, or merely *small*? Two published studies claim an advantage, both on simulation only,
both in the 100–1,000-training-sample regime. This project reproduces that claim under controlled
conditions on DUNE FD simulation (**Phase A**), then tests whether it survives contact with *real
detector data* (**Phase B**).

A null result is expected and publishable. The study measures a comparison; it does not set out to
make QML win.

## Start here

```bash
cd notebooks/ && jupyter lab        # then open 00_start_here.ipynb
```

The executed notebook series **is** the primary documentation — figures and analysis baked in:

| | | |
|---|---|---|
| `notebooks/00–02` | setup · the data · the quantum layer | the tutorial half |
| `notebooks/03` | first training, 4 arms, harness in-cell | milestone A1 |
| `notebooks/04` | the study, small: AUC vs data size / parameters / label purity | first pass of A4 |

## Current headline (notebook 04, 63 runs, untuned — see caveats there)

At n=100 training patches, a **248-parameter** quantum model ties a **30,000-parameter** CNN.
By n=100,000 the advantage is gone: the deep CNN reaches 0.969 AUC while the quantum model
saturates at ~0.911 — as does a same-size classical CNN. The quantum arms are ~10× more
seed-stable than small CNNs. Next step: the equal-budget tuned study.

## Repo map

| path | contents |
|---|---|
| `docs/PRIMER.md` / `docs/RUNBOOK.md` | concepts textbook · shell-level setup |
| `qmllar/` | the quanvolution in three cross-validated implementations (PennyLane / numpy / torch, agreement ~1e-15) + patch extraction |
| `tests/` | the cross-checks — they have caught three real, silent bugs; run after any change |
| `results/study04.jsonl` | experiment cache: every number in notebook 04 traces to a line here |

## Data

Training data is **not** in the repo: `data/patches_31.npz` (201 MB) rebuilds in ~4 min via
`python3 scripts/build_patches.py`, which reads the Wire-Cell AI/ML campaign samples
(`/data/0/jjo/WC_AIML`, BNL `wcgpu1`; override location with `WC_AIML_BASE`). Inventory and the
one dangerous labelling convention: `docs/wc-aiml-dataset-inventory.md`.

## Scope of this repository

Code, executed notebooks, and technical documentation. The research plan, the primary-source
audit of the reference papers, and the reference PDFs are **internal documents not included
here** — references to "the plan" or "findings §…" inside notebooks and docs point to those.
Results shown are **preliminary and un-reviewed** (single configurations, no hyperparameter
search; see the caveats inside each notebook) and are not DUNE collaboration results.
