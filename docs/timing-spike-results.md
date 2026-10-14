# Timing Spike — Measured Results

**Date:** 2026-09-01
**Machine:** 64 cores, 251 GB RAM, 2 × RTX 4090 (24 GB)
**Verdict:** **GO.** The published compute wall is a framework artifact, not a physical cost.

This is prep-plan Phase 3, the go/no-go gate. Research plan §4 and
`primary-source-findings.md` §J argued the published costs all assume per-window
parameter-shift gradients and that a vectorised statevector path would be far cheaper.
That argument is now **measured**, not asserted.

---

## 1. The dataset it was measured on

Built by `scripts/build_patches.py` from the campaign samples, read-only:

| | |
|---|---|
| source | `fdhd_sparse_{nue,numu}_truthlabel/001` (smeared — see inventory §4c) |
| archives | 800 per flavour (1,600 total) |
| **patches** | **415,509** at 31×31, float32, 201 MB |
| train / val / test | 292,603 / 60,677 / 62,229 |
| class balance (train) | track 0.425 / shower 0.575 |
| build time | **55 s** on 48 workers |

**Splits are at archive level, never patch level** — patches from one event share
particles, so a patch-level split leaks. Each split draws on disjoint archives
(1,120 / 240 / 240).

Stored per patch, so finer label schemes need no re-extraction: `cls` (binary
track/shower), `pid` (raw PDG), `efrac` (label purity grade), `ke_mev` (dominant
particle kinetic energy), `anode`, `flavour`, `archive`, `split`. Charge is stored **raw**;
normalisation is a swept choice applied at load time (§4a).

`efrac` join integrity: **0 of 415,509** patches have a NaN energy fraction, so the
label-purity axis is complete. (The four frame groups carry independent coord lists, so this
was worth checking rather than assuming.)

For scale: the PRD trained on **100–1000** patches. This is ~300× their training set,
from one tenth of one flavour's available archives.

---

## 2. The measurement

⚠️ **These are FORWARD-PASS costs.** Both rows measure a forward evaluation over all
292,603 training patches at 31×31 (900 windows per patch for a 2×2 filter, 841 for 3×3).
Training adds a backward pass — see the caveat below. The **ratio** is apples-to-apples;
the absolute figures are lower bounds.

| Path | µs / window | ms / patch | **forward pass over 292,603 patches** |
|---|---|---|---|
| PennyLane QNode, `parameter-shift` | 4774 | 4300 | **349 h** |
| PennyLane QNode, `backprop` | 4518 | 4070 | **331 h** |
| **Vectorised statevector (numpy, complex128)** | **4.65** | **4.18** | **20.4 min** |

**≈ 970× faster**, for identical output (validated to ~1e-15, §3).

Chen's 9-qubit 3×3 first layer, same vectorised path: 17.0 µs/window → 14.3 ms/patch →
**70 min** per forward pass. Cost grows with qubit count as expected (2⁹ = 512-dim state
vs 2⁴ = 16).

### Caveat: what a real epoch costs

`fast.py` is forward-only; the gradient path is not implemented yet (that is what `torch`
is for). So:

- **Vectorised:** a backward pass through these matmuls is the same order as the forward
  one, so an epoch is plausibly ~2–3× the 20.4 min figure — still under an hour, still
  hours per full run. The verdict does not move.
- **Naive parameter-shift:** the 349 h figure is a **floor, not an estimate**. Parameter
  shift needs two extra circuit evaluations *per parameter per window*; at 96 parameters
  that is ~192× the forward cost. A genuine parameter-shift epoch is in the thousands of
  hours.

So the real training-time gap is *wider* than 970×, not narrower.

**Do not read anything into `parameter-shift ≈ backprop` above.** On a forward-only call
`diff_method` is nearly inert because no gradient is computed, so this measurement does not
establish where the naive path's time goes. The likely explanation is per-window Python
QNode overhead — 900 tape constructions per patch, against a circuit that is a handful of
16×16 matmuls — but that remains a hypothesis here, not a result.

**The load-bearing conclusion is unaffected:** anyone gating a go/no-go decision on the
naive path would kill this project for no reason. Chen et al.'s own O(N²S) warning and the
PRD's "multiple days per patch size" are both consistent with this regime.

---

## 3. Correctness — and two bugs the validation caught

`tests/test_fast_matches_pennylane.py` checks the vectorised path against PennyLane QNodes
for all four paper configurations:

```
[OK] chen2020_conv1_3x3      max|err| = 4.44e-16
[OK] chen2020_conv2_2x2      max|err| = 4.16e-16
[OK] tutorial2020_fig3_2x2   max|err| = 7.77e-16
[OK] prd2025_quanv_2x2       max|err| = 7.77e-16
```

Parameter counts also reproduce Chen's published values exactly: **54** for the 3×3 layer
(9 qubits × 3 params × 2 blocks) and **24** for the 2×2 layer. That is independent
confirmation that Fig. 5 was read correctly.

The cross-check earned its keep by finding two real bugs:

1. **Channels were not independent.** `build_circuit` looped over channels applying gates
   cumulatively to the *same* wires and returned a single measurement. A conv layer's
   filters are independent kernels; each channel must be its own circuit. Chen's configs
   hid this because they use `n_channels = 1`.
2. **Reupload parameters were being reused.** The vectorised path applied the first
   reupload's `U_θ` at every reupload. Only the PRD config exposed it (`n_reuploads = 2`).

Neither would have thrown an error. Both would have produced plausible, wrong numbers.

**Open ambiguity, recorded not resolved:** the PRD's Eq. (3) writes `(U_θ U_F)^R`, whose
literal reading is *shared* weights across reuploads. Both implementations here use
*independent* parameters per reupload. Both are defensible; it changes the parameter count
at fixed R. Worth pinning down with the authors, or sweeping both.

---

## 4. What this means for scope

At ~20 min per forward pass — call it under an hour per epoch once backward is included — a
50-epoch run is tens of hours and a 12-config sweep a couple of weeks. Viable but not
comfortable. Headroom not yet used, in rough order of payoff:

- **complex64 instead of complex128** — ~2×, no accuracy concern at this depth.
- **GPU.** The workload is dense small matmuls on a 4090; expect a further large factor.
  (Measurement pending — `torch` was still installing.)
- **Smaller patches.** The PRD's optimum was 21×21 → 400 windows instead of 900, a 2.25×
  saving that is a *scientific* choice, not a compromise.
- **Batch over channels** rather than looping.

The go/no-go table in the research plan asked: hours, days, or weeks? **Answer: hours per
run on the optimised path, with the sweep affordable after the easy optimisations** — versus
weeks per *epoch* on the naive path. GO.

---

## 4a. ⚠️ Charge must be normalised — none of the three paper encodings works on raw ADC

Raw campaign charge spans 0 to 2.4e5 (nonzero median ~720, p99 ~1.6e4). The paper encodings
assume O(1) inputs; Chen's were pre-scaled 30×30 images, not raw ADC. Measured effect on
Chen's `arctan` encoding over 4,000 patches:

| normalisation | charge p50 | charge p99 | angle p50 | angle p99 | **fraction within 1% of π/2** |
|---|---|---|---|---|---|
| `per_patch_max` | 0.071 | 0.913 | 0.070 | 0.740 | 0.0000 |
| `global_p99` | 0.058 | 1.000 | 0.058 | 0.785 | 0.0000 |
| `log1p` | 0.705 | 1.025 | 0.614 | 0.798 | 0.0000 |
| `arcsinh` | 0.058 | 1.063 | 0.058 | 0.816 | 0.0000 |
| **`none` (raw)** | 907 | 19903 | 1.570 | 1.571 | **0.8417** |

On raw charge **84% of nonzero pixels encode to within 1% of π/2** — the encoding is
saturated and carries almost no information. The PRD's `identity` and the tutorial's
`pi_scaled` are worse: `RX(720)` wraps the Bloch sphere ~115 times.

**This fails silently.** A model trained on raw charge returns a plausible AUC that measures
normalisation, not architecture.

Handled in `qmllar/preprocess.py`, which keeps raw charge in the patch file and applies
normalisation as a **swept config field**. It must be swept jointly with the encoding —
`arctan` on `log1p` charge is a different model from `arctan` on max-scaled charge — and
both arms must use the same normalisation.

---

## 5. Reproducing

```bash
python3 scripts/build_patches.py --n-archives 800 --workers 48   # 55 s -> data/patches_31.npz
python3 tests/test_fast_matches_pennylane.py                     # correctness
```

Nothing writes to the campaign data. Labels are derived in memory
(`frame_trackid_1st` → `abs()` → `mcpart/pids`); tar extraction uses `/dev/shm`.
