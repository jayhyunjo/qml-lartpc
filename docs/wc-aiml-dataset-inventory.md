# Wire-Cell AI/ML Campaign Dataset — Inventory and What It Enables

**Location:** `/data/0/jjo/WC_AIML/training_samples` — 491 GB
(*moved from `/srv/data/1/jjo/WC_AIML` between sessions; code resolves it automatically or via
`WC_AIML_BASE`*)
**Source:** Wire-Cell Foundation Model development campaign (separate project, shared assets)
**Authoritative documentation:** <https://github.com/HaiwangYu/cffm-if> —
`dune10kt-1x2x6/exa_10evts_for_tag/MC_TRUTH_AND_LABELLING.md` (the spec) and
`classify_pixels.py` (the labelling implementation). Generation configs: `gen_genie.fcl`, `g4.fcl`,
`wcls-labelling2d_sep.fcl`. The repo is **160 MB** (plots, logs, zips) — sparse-checkout
`dune10kt-1x2x6/exa_10evts_for_tag/` rather than cloning it whole; a full clone times out.
**Inspected:** 2026-09-01, first by reverse-engineering with `h5ls`/`h5dump`, then checked against
the repo. Everything below that was reverse-engineered is now confirmed against the spec.

---

## 1. Inventory

`fdhd` = **F**ar **D**etector **H**orizontal **D**rift — i.e. DUNE FD1, the same technology family as
ProtoDUNE-HD. Every archive holds **10 events**; counts below are files × 10.

| Sample | Files | Events | Per-pixel truth | Run |
|---|---|---|---|---|
| `fdhd_sparse_nue_training` | 99,930 | **~1.0 M** | no | 016898 |
| `fdhd_sparse_numu_training` (10 subdirs) | ~100,957 | **~1.0 M** | no | 13717–13826 |
| `fdhd_sparse_nue_truthlabel/001…010` | 10,000 | **~100 k** | **yes**, smeared | 018023 |
| `fdhd_sparse_nue_truthlabel/no_smearing` | 9,992 | **~100 k** | **yes**, unsmeared | 016895 |
| `fdhd_sparse_numu_truthlabel/001…010` | 9,974 | **~100 k** | **yes**, smeared | 018020 |
| `fdhd_sparse_numu_truthlabel/no_smearing` | 9,995 | **~100 k** | **yes**, unsmeared | 016908 |

So roughly **2 M unlabelled and 400 k truth-labelled events**, νe and νμ, smeared and unsmeared.

`001`…`010` are **not** ten systematic variants — all ten share one run number and sequential file
indices. Confirmed with Josh: **job batch numbers only.** The one meaningful production split is
smeared vs `no_smearing`, and per §4(c) that is a label-convention difference, not a physics one.

---

## 2. Contents of one event

Each `.tgz` unpacks to `sparse/` containing 12 anode files + metadata + truth:

**`*_pixeldata-anodeN.h5`, N = 0…11** — one per anode, groups `/1`…`/10` per event:
- `frame_rebinned_reco/coords` {N_hit, 2} — sparse **(channel, tick)**; channel ∈ [0, 2560) within
  the anode (DUNE APA: 2×800 induction + 960 collection)
- `frame_rebinned_reco/features` {N_hit} — charge
- `channels_rebinned_reco` {2560}, `tickinfo_rebinned_reco` {3} — `[t0, tick_period, ?] = [0, 500, 0]`

Typical occupancy ~7k–15k hits per anode per event, i.e. genuinely sparse.

**`*_metadata.h5`** — per event: `nu_pdg`, `nu_ccnc`, `nu_intType`, `nu_energy`,
`nu_vertex_{x,y,z}`. (Example: `12, 0, 1003, 4.795 GeV, (-171.2, -320.1, 168.0)` = νe CC.)

**`*_trackid_pid_map.h5`** — per event, full Geant4 record:
- `mcpart/`: `track_ids`, `pids`, `masses`, `start_moms`/`end_moms` {N,4},
  `start_xyzts`/`end_xyzts` {N,4}, `processes`, `end_processes`, `mother_ids`, `mother_pids`,
  `ndaughters`, `ntrajpts`, `statuses`. (819 particles in the example event.)
- `simchnl/`: `track_ids`, `pids`, `mother_ids`, `mother_pids`, `processes`, `energies` — a
  **per-particle** summary, not a per-pixel map.

Note `track_ids` are **not contiguous** (1…13, then 18, 19, …) — a lookup table is required, not
array indexing.

---

## 3. What `truthlabel` adds — the important part

The truth-labelled samples carry **per-pixel** truth alongside the reco image:

| Dataset | Meaning |
|---|---|
| `frame_trackid_1st` | track ID of the **dominant** depositing particle at each pixel |
| `frame_trackid_2nd` | track ID of the **second** contributor |
| `frame_energyfrac_1st` | energy fraction from the dominant particle |
| `frame_energyfrac_2nd` | energy fraction from the second |
| `frame_total_numelectrons` | true electron count per pixel |

Each has its **own** `coords`, so joins must be on coordinates, not array position (the four groups
had 5487 / 4807 / 4807 / 2061 entries on the same anode).

**Verified end-to-end:** pixels with `trackid_1st = 2` → `mcpart/track_ids` index of 2 →
`pids = 2212` = proton, with `energyfrac_1st = 1.0` (unambiguous). Some pixels have
`energyfrac_2nd ≈ 0.42`, i.e. genuinely shared deposits.

### This is the PRD's labelling scheme, and strictly better

Duffy et al. (PRD 112, 092006) §III:

> "each patch is assigned a label based on the particle that contributes most to the intensity of the
> central pixel. While multiple charged particles can deposit charge at the same location, we adopt a
> **single-particle labeling scheme**"

That is exactly `frame_trackid_1st`. We additionally have the **second** contributor and **both
energy fractions** — so label ambiguity is measurable and cuttable. The PRD names this as an unmet
need in its own conclusions:

> "One could devise more precise metrics by quantifying the amount of 'shower' pixels in a 'track'
> patch, for example."

---

## 3a. ⚑ Haiwang Yu is a co-author of Chen et al. 2012.12177

The repo owner — and the author of this labelling pipeline — is **Haiwang Yu**, the fourth author of
`arXiv:2012.12177`, our primary reference. He is a current colleague on the Wire-Cell FM campaign.

**This is a far better route to the Phase 0 questions than a cold approach to Chao Zhang**, and it
changes the prep sequencing (research plan §6, Phase 0). Ask Haiwang:
- Does the original QCNN single-particle DUNE dataset still exist? What were `n_train` / `n_test`,
  the class balance, and was there a validation set? (Our 160/40 inference — findings §C.)
- Was the QCNN code released or kept internally?
- Why did the line stop — priorities, or something that did not hold up?

The lineage question is already settled from the paper itself (findings §A), so this is about assets
and history, not architecture.

---

## 3b. There is an authoritative 7-class labelling scheme — and it is richer than the PRD's

From `MC_TRUTH_AND_LABELLING.md` §3 and `classify_pixels.py`:

| Label | Name | Definition |
|---|---|---|
| 0 | Background | `track_id == 0`, no MC hit |
| 1 | **Track** | μ±, π±, p/p̄ (`_TRACK_PDGS = {13,-13,211,-211,2212,-2212}`) |
| 2 | **Shower** | e±, γ, π⁰ that is EM-shower-seeded |
| 3 | **Michel** | e± from muon Decay (`pid==11 & proc==1 & |mother_pid|==13`) |
| 4 | **DeltaRay** | ionization e± (eIoni/muIoni/hIoni) with a Track-type parent |
| 5 | **Blip** | isolated EM deposit — nCapture γ, hadronic e±: EM but *not* shower-seeded |
| 6 | Other | neutrinos, nuclei, neutrons |

Resolution order is **Michel > DeltaRay > Track > Shower > Blip > Other**. "Shower-seeded" means the
particle or an ancestor was created by an EM process (`conv`, `compt`, `phot`, `Brem`, `annihil`,
`LowEnConversion`, `Pair`) — or is any π⁰, since π⁰ → γγ is prompt.

**This is essentially PILArNet's class set** (HIP, MIP, EM shower, delta ray, Michel), which the prep
plan already named as the public fallback — so results here are directly comparable to the
PILArNet-based literature.

**And it is strictly richer than the PRD's binary track/shower.** That is a differentiation
opportunity: the 5-way semantic task (Track / Shower / Michel / DeltaRay / Blip) is what LArTPC
reconstruction actually needs, is harder, and is *not* what Duffy et al. did. Doing binary
track/shower reproduces them; doing the 5-way extends beyond them on a non-symmetry axis.

---

## 3c. ⚠️ The labels are NOT in the production files — you must generate them

Verified on a production file: `pixeldata-anode0.h5` contains only

    frame_rebinned_reco, frame_total_numelectrons,
    frame_trackid_1st/2nd, frame_energyfrac_1st/2nd

`frame_label_1st`, `frame_label_2nd` are **absent**, and `mcpart/labels` is **absent** too. Per the
spec these are written by `classify_pixels.py`, which has not been run on this production set.

**So running (or reimplementing) `classify_pixels.py` is a required preprocessing step.** Two
complications:

1. **Format mismatch.** The spec and the script describe **dense** frames — *"`frame_rebinned_reco`
   (Nch, Ntick) … Typical dimensions: Nch=2560, Ntick=1500"*. The production files are **sparse**
   (`coords` (N,2) + `features` (N)) — hence `fdhd_sparse_*` in the directory names. The script needs
   adapting to the sparse layout.
2. **It is explicitly a draft.** The spec says: *"This is the main script collaborators should review
   and improve"*, and §7 lists five open issues — the Shower/Blip boundary, the DeltaRay definition,
   Michel vs muon capture, whether to weight labels by energy fraction, and negative-track-ID edge
   cases.

Item (1) is work we must do; item (2) is an **opening**: label-definition choices are unsettled, we
have `energyfrac_1st`/`_2nd` to quantify them, and §7.4 explicitly floats energy-fraction weighting as
future work. That is a contribution available for the asking.

---

## 3d. Kinematics are available — energy-binned reporting is possible

`mcpart` carries `start_moms`/`end_moms` (4-momentum in GeV) and `start_xyzts`/`end_xyzts` (cm, ns),
so per-particle kinetic energy is computable (the spec gives the recipe). `simchnl/energies` gives
deposited energy in MeV.

This **partially recovers something I thought was lost**: momentum-binned Δ reporting was the
mitigation for per-class distribution mismatch on beam data (research plan §3). The same binning is
available here in simulation, so any label-quality or domain-shift study can be reported
energy-differentially rather than integrated.

---

## 4. What this enables

**(a) Both task formulations, on one dataset.**
- *Pixel-level* track/shower (the PRD's task): `trackid_1st` → `pid` → track-like (μ, π, p, K) vs
  shower-like (e, γ). A **direct** reproduction target rather than an analogue.
- *Particle-level* (Chen's task, and ours): group pixels by `trackid_1st` to get per-particle
  clusters, label by `pid`. Not identical to Chen — his particles were isolated in an otherwise empty
  detector, these sit in busy neutrino events — but closer to real analysis conditions.

**(b) A label-quality axis nobody in this literature has used.** `energyfrac_1st` grades every label
from clean (1.0) to ambiguous (~0.5). Performance versus label purity is a **robustness** measurement
of the same kind as sim-to-data — does the low-parameter model degrade more gracefully under label
noise? Cheap, entirely simulation-based, and *not* on the PRD's symmetry axis.

**(c) ~~A detector-effects domain shift~~ — RETRACTED. It is a label-convention difference.**

I initially proposed training on `no_smearing` and testing on smeared as a simulation-only dress
rehearsal for the sim-to-data measurement. **That was wrong**, on two counts, both now checked:

- **The smearing is applied to the truth label**, not as a physics variation of the events
  (confirmed with Josh). So the two samples do not differ in detector physics.
- They are also **different physics events** — event 1 of each has different `nu_energy`,
  `nu_intType` and vertex — so it is not a paired ablation either.

Measured difference over 50 events each (busiest anode per event):

| | smeared (`001`) | `no_smearing` |
|---|---|---|
| fraction of reco pixels carrying a label | **0.880 ± 0.086** | 0.690 ± 0.145 |
| fraction of labels sitting on reco pixels | **1.000** | **1.000** |
| `energyfrac_1st` mean | 0.923 | 0.926 |
| `energyfrac_1st == 1.0` | 0.505 | 0.566 |

So neither sample is *misaligned* — in both, every truth label lands on a reco pixel. What smearing
buys is **coverage**: labels follow the detector response the way the charge does, so 88% of reco
pixels get a label instead of 69%. Purity is essentially unchanged.

**Consequence: use the smeared samples for all pixel-level work.** `no_smearing` would leave ~31% of
charge unlabelled, forcing either a biased discard or an extra "unlabelled" class. Training on one and
testing on the other measures sensitivity to label convention — a legitimate but much weaker
experiment than the one I claimed.

**Combined with the campaign having FD samples only (no ProtoDUNE-HD), both of my proposed
simulation-only substitutes for the sim-to-data test are gone.** That measurement genuinely requires
ProtoDUNE-HD beam data and its own MC campaign. This dataset makes everything upstream cheaper; it
does not shorten the critical path to the novel result.

**(d) The strongest classical baseline in the literature.** ~2 M unlabelled events is exactly what a
pretrained encoder needs. The QML parameter-efficiency claim is pitched at the low-labelled-data
regime (the PRD trained on 100–1000 patches); the modern classical answer to few labels is
pretraining, and nobody has run that comparison. See research plan §3 arm 7.

**(e) It removes most of prep Phase 4's simulation half.** No LArSoft, no grid, no art files — this is
frozen, on local disk, and readable with `h5py`. What remains is plain data engineering
(§5 below), not DUNE computing.

**(f) Same technology as the eventual data target.** `fdhd` → ProtoDUNE-HD is a geometry/size shift,
not a technology shift, so models trained here transfer meaningfully.

---

## 4a. ⚠️ `abs(trackid)` is mandatory — the one gotcha that would silently ruin everything

**Pixel track IDs are negative for ~90% of pixels.** In the first event surveyed, `frame_trackid_1st`
ranges over **−970 … 1004**, and the single most common value is **−125, carrying 3535 of 4807
pixels**. Meanwhile `mcpart/track_ids` contains only positive IDs.

This is the standard LArSoft/Geant4 convention: EM-cascade secondaries that were not tracked
individually carry the **negated ancestor track ID**. Consequence, measured:

| Lookup | Pixel track IDs not found in `mcpart/track_ids` |
|---|---|
| raw `trackid` | **89.8 %** |
| `abs(trackid)` | **0.0 %** |

A naive join mislabels ~90% of pixels — and fails *silently*, since the misses look like an "unknown"
class rather than an error. With the raw IDs the composition came out "75% other"; with `abs()` it
resolves completely.

**Confirmed as the official convention.** `classify_pixels.py:359` does exactly this:
`label_map.get(int(abs(t)), LABEL_OTHER)` — a pixel inherits its shower ancestor's label. The spec
§7.5 flags it as *"handled but worth verifying edge cases."*

**Alternative route, also valid:** `simchnl` retains the negative IDs directly (8.6% of its entries),
and for the cases checked gives the same PDG as `abs()` + `mcpart` (trackid −125 → pid 22 in both).
Prefer `abs()` + `mcpart`, because `mcpart` also carries the kinematics (§3d); `simchnl` uniquely
carries deposited energy.

Composition with `abs()`:

| | νe smeared | νμ smeared |
|---|---|---|
| shower-like (e, γ) | **0.802** | 0.359 |
| track-like (μ, π, p, K) | 0.196 | **0.640** |
| nucleus | 0.002 | 0.001 |
| unmapped | **0.000** | 0.000 |

Class balance is worth noting: **νe events are ~80/20 shower/track and νμ are ~36/64** — pooling the
two flavours gives a roughly balanced pixel-level track/shower problem, which is the PRD's task.

## 4b. Label ambiguity is real, graded, and the best new axis this data offers

Over 150 surveyed events, `energyfrac_1st` (the dominant particle's share of a pixel's energy):

- mean **0.923**
- exactly 1.0 (unambiguous): **~51–55 %** of pixels
- above 0.9: ~75 %
- below 0.7 (genuinely shared deposits): **~9–10 %**

So roughly half of all pixels carry a label that is *not* clean, and one in ten is substantially
contaminated. This is a continuous, per-pixel quality grade — you can cut on it, stratify by it, or
sweep it. The PRD had no equivalent and named the lack as future work.

**Performance versus label purity is a robustness measurement of the same family as sim-to-data**, it
runs entirely in simulation, it needs nothing from Phase 0, and it is not on the PRD's symmetry axis.

## 4c. Particle-level extraction is viable but selection-biased — state the bias

Per busiest anode per event, using `abs(trackid)` so EM showers merge into their ancestor:

| | νe smeared | νμ smeared |
|---|---|---|
| particles depositing | 104 (median 96) | 68 (median 52) |
| pixels per particle, median | 9 | 11 |
| pixels per particle, p90 | 82 | 93 |
| particles with ≥100 px | **8.1 %** | **9.3 %** |
| particles with ≥400 px | 4.3 % | 5.1 % |
| largest single particle | 36,801 px | 8,136 px |

**Most particles deposit almost nothing.** A Chen-style single-particle image needs a few hundred
pixels, so a size cut keeps only ~5–9 % of particles — and that cut selects the energetic, long ones.

Statistics are still ample (100 k events × 12 anodes × ~100 particles × 8 % ≈ 10⁷ candidates), but the
**selection bias must be stated and, ideally, swept**: report performance as a function of the minimum
pixel count, rather than picking one threshold. This is also a fair-comparison issue — quantum and
classical arms must see exactly the same selection.

---

## 5. Open questions and costs

**Questions — all four now answered (2026-09-01):**
1. *What is "smearing"?* → **Detector response and noise smearing applied to the truth label.** Hence
   the retraction in §4(c).
2. *Are `001`…`010` variants?* → **No, job batch numbers only.**
3. *Any ProtoDUNE-HD samples?* → **No. FD only.** So no sim-to-sim rehearsal from this campaign.
4. *Environment?* → No preference. `h5py 3.16.0` now installed (see machine note below).

**Costs to budget:**
- `h5py` is **not installed**; neither are `torch`, `jax`, `sklearn`. At minimum `h5py` + one autodiff
  backend is needed.
- The layout is ~200 k `.tgz` archives × 12 anodes. A **consolidation pass** into a few large
  contiguous arrays (single-particle crops or patch tensors) is required before training — untarring
  per epoch would dominate runtime. This is the real preprocessing cost, and it is ordinary data
  engineering rather than LArSoft work.
- Neutrino events are multi-particle, so a **truth-matched clustering step** is needed for the
  particle-level formulation. Decide the isolation criterion explicitly and write it down.

**Hardware available:** 64 cores, 251 GB RAM, 2 × RTX 4090 (24 GB). Ample; the RAM in particular means
a consolidated patch tensor can likely be held in memory.

**⚠️ Machine housekeeping (found while installing):**
- **`/tmp` is 100 % full** — a 1.8 GB partition with ~5 MB free. This breaks `pip` and anything else
  using the default temp dir.
- **`/` is at 95 %**, 1.2 GB free.
- Workaround used: `TMPDIR=/home/jjo/.pip-tmp` (that path is on the NFS home, 392 GB free). `h5py`
  installed this way.
- Still needed for training: an autodiff backend (`jax` or `torch`) and `scikit-learn` for metrics.
  Install with the same `TMPDIR` redirect.
