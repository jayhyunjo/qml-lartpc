# Primer — QML for LArTPC

The concepts behind the project. No commands — those are in **[`RUNBOOK.md`](RUNBOOK.md)**.

Written for a physics graduate student who knows quantum mechanics and HEP data analysis but has not
done machine learning. Readable in order, or dip in via the [glossary](#7-glossary).

---

## 1. What this project computes

### 1.1 The physics question

A liquid argon TPC records charged particles as 2D images: one axis is the sense wire, the other is
drift time, intensity is collected ionization charge. To reconstruct a neutrino interaction you must
first decide, for each blob of charge, *what kind of thing made it* — a narrow ionization trail from
a muon, pion or proton (**track-like**), or a spreading electromagnetic cascade from an electron or
photon (**shower-like**).

We are testing whether a classifier built partly from a quantum circuit does this as well as a
conventional one **using far fewer adjustable parameters** — and, if so, whether the cause is
*quantum* or merely *small*.

### 1.2 The pipeline

```
   DUNE FD simulation (Wire-Cell campaign, 492 GB on disk)
              |
              |  extract 31x31 windows centred on labelled pixels
              v
   data/patches_31.npz          415,509 patches + labels
              |
              |  normalise charge -> rotation angles
              v
   +---------------------------------------------------+
   |  MODEL                                            |
   |    quanvolution layer (quantum)                   |
   |      slides a 2x2 window over the patch;          |
   |      each window -> 4-qubit circuit -> one number |
   |    classical head (ordinary neural net)           |
   |      feature map -> track / shower                |
   +---------------------------------------------------+
              |
              |  training: adjust the knobs to fit the labels
              v
   ROC AUC, compared against a conventional CNN
```

### 1.3 The one idea to hold onto: a model is a function with knobs

A classifier is a function `f(x; θ)` — `x` is the image, `θ` is a list of adjustable numbers called
**parameters** (ML also says *weights*). It returns a score; high means shower, low means track.

**Training is a fit.** The differences from a χ² fit you have done before are scale and method, not
concept:

| χ² fit you know | Training a network |
|---|---|
| 2–5 parameters | 10² to 10⁵ parameters |
| minimise χ² | minimise a **loss function** ([§6.1](#61-the-loss-function)) |
| MINUIT / `curve_fit` | **gradient descent** ([§6.2](#62-gradient-descent)) |
| analytic or numerical derivatives | **automatic differentiation** ([§6.3](#63-automatic-differentiation-the-gradient-path)) |
| judge by χ²/dof | judge on **held-out data** ([§3.3](#33-three-splits-and-the-leakage-trap)) |

So "the gradient path is not implemented" means: *we can evaluate `f(x; θ)` but not `∂f/∂θ`, so we
cannot do the fit.* That is all.

### 1.4 What is quantum about it

One layer is a small quantum circuit. For each 2×2 window of pixels:

1. Four pixel values become four rotation angles.
2. Those rotations prepare a 4-qubit state — the **encoding**, `U_F`.
3. A parameterised unitary acts on it — the **ansatz**, `U_θ`. Its angles *are* the parameters.
4. Measure `⟨Z⟩` on one qubit. **Four pixels in, one number out.**

Slide that over the image and you get a new image, exactly as a convolutional filter would. Because
only one window is processed at a time, four qubits suffice for an image of any size — the entire
reason this is feasible on near-term hardware.

We simulate the circuit classically. **No quantum hardware is involved anywhere in this project.**

---

## 2. The data

### 2.1 A LArTPC event as numbers

An **anode** (an APA in DUNE) reads 2560 wire channels, digitised in time bins called **ticks**. So
one anode's readout is an array `charge[channel, tick]` — an image. DUNE FD's 1×2×6 workspace
geometry has **12 anodes**, so one event is 12 images.

Almost all pixels are empty: ~7,000–15,000 of 2560×1500 ≈ 3.8 M carry charge. The campaign therefore
uses a **sparse representation** — a list of coordinates and a matching list of values, like a COO
sparse matrix:

```
frame_rebinned_reco/coords     (N, 2)   integer (channel, tick)
frame_rebinned_reco/features   (N,)     float   charge
```

**Consequence that matters:** different quantities have *different* coordinate lists, so combining
them means matching on coordinates, never on array position.

### 2.2 How a pixel gets a label

The simulation knows what it did. Geant4 gives every particle a **track ID** and records its PDG
code, momentum, production process and parent. The campaign also stores, per pixel, which particle
deposited the charge:

| Dataset | Meaning |
|---|---|
| `frame_trackid_1st` | track ID of the largest energy contributor at this pixel |
| `frame_trackid_2nd` | the second largest |
| `frame_energyfrac_1st` | the dominant particle's share of the energy, 0–1 |
| `frame_energyfrac_2nd` | the second's share |

A pixel's label is then: pixel → dominant track ID → PDG code → track-like or shower-like.

`energyfrac_1st` is a **label quality grade**. 1.0 means one particle deposited everything and the
label is unambiguous; 0.55 means two contributed almost equally and the label is near-arbitrary.
Measured over our patches: mean 0.92, but only ~51% are exactly 1.0 and ~9% are below 0.7.
Published work in this area has no equivalent handle, which makes "performance versus label purity"
an axis available to us and nobody else.

#### ⚠️ The negative track ID trap

Geant4 can produce enormous numbers of low-energy electrons and photons in an EM cascade. Tracking
each is wasteful, so LArSoft runs with `keepEMShowerDaughters = false`: cascade secondaries are not
stored as separate particles, and their deposits are recorded against the **negated track ID of the
shower ancestor**.

So **~90% of pixel track IDs are negative**, and none appears in the particle list, which holds only
positive IDs:

| Lookup | Pixel track IDs not found |
|---|---|
| raw `trackid` | **89.8%** |
| `abs(trackid)` | **0.0%** |

Taking the absolute value fixes it completely — the pixel inherits its shower ancestor's identity.
This is what the campaign's own `classify_pixels.py` does.

**Why it is dangerous rather than annoying:** the failed lookups raise no error. They look like an
extra "unknown" class. Get it wrong and the composition comes out ~75% "other" instead of ~80%
shower / 20% track — a plausible-looking, completely wrong dataset.

### 2.3 Which samples to use

`fdhd` = Far Detector Horizontal Drift. The `*_truthlabel` samples have per-pixel truth; the
`*_training` samples (~2 M events) do not, and exist for pretraining a foundation model on unlabelled
data.

Within `*_truthlabel`, directories `001`–`010` are **job batch numbers, not variants** — all one
production run. The meaningful split is smeared versus `no_smearing`, where "smearing" means
detector response and noise applied to the truth label. **Use the smeared ones:** their labels cover
88% of reconstructed pixels versus 69%, at the same purity. Unsmeared labels sit on un-blurred
deposit positions and miss a third of the reconstructed charge.

---

## 3. Preparing samples

### 3.1 Why patches

We could feed a whole 2560×1500 anode image to a classifier and ask it to label every pixel at once
(**semantic segmentation**). Instead we cut a small window around one pixel and ask a single
question — *is the pixel at the centre track-like or shower-like?*

```
        31 x 31 window                 one label
   +---------------------+
   |        . . .        |
   |      . . X . .      |    X = the pixel being classified
   |        . . .        |
   +---------------------+
```

Three reasons:

- **It makes the problem small enough for a quantum circuit.** A 4-qubit circuit processing 2×2
  windows can sweep a 31×31 patch. It cannot sweep a full event in any useful sense.
- **The surroundings carry the information.** A single pixel's charge says little; whether it sits on
  a straight thin trail or inside a spreading blob says everything. Hence patch size matters and is
  worth sweeping — too small and there is no context, too large and unrelated activity creeps in. The
  reference paper found ~21×21 optimal, which is why we extract 31×31 and crop down rather than
  re-extracting.
- Repeat over every pixel and you have segmented the event. We just train one pixel at a time.

### 3.2 The two classes, and imbalance

**Track-like** (label 0): μ±, π±, p, K±. **Shower-like** (label 1): e±, γ, π⁰.

**Class imbalance** matters: if 95% of examples are one class, a model that always guesses that class
is 95% "accurate" and useless. Measured here:

| Sample | shower | track |
|---|---|---|
| νe events | 0.80 | 0.20 |
| νμ events | 0.36 | 0.64 |
| **both pooled** | **0.57** | **0.43** |

Pooling the two flavours gives a nearly balanced problem for free. That is why the extractor reads
both.

### 3.3 Three splits, and the leakage trap

**Why three.** You need data the model has never seen, to know whether it learned the physics or
memorised the examples — **overfitting**, familiar to you as fitting noise with too many free
parameters. But you also make choices along the way (layer count, learning rate, when to stop). Make
those by looking at your final test set and you have tuned on it; it is no longer unseen.

| Split | Used for | Analogy |
|---|---|---|
| **train** | adjusting parameters | the data in your fit |
| **validation** | choosing settings, deciding when to stop | your control region |
| **test** | quoted once, at the end | your signal region, kept blind |

This is **blind analysis**. Look at the test set once.

**The leakage trap.** Patches from the same event are *not independent* — they may lie on the same
particle, metres of the same muon track, the same shower. Put some in train and others in test and
the model can score well by recognising that specific event. Your test AUC then measures
memorisation and looks wonderful.

So whole **archives** (10 events each) are assigned to a split, never individual patches. An
anomalously high test AUC is almost always leakage, not brilliance.

### 3.4 Charge normalisation

The circuit turns a pixel value into a rotation angle. Rotations are periodic and bounded: `RX(θ)`
for θ = 720 radians is indistinguishable from some small angle, and `arctan(x)` approaches π/2 for
any x beyond ~10, so all large values collapse to the *same* angle. **The encoding only carries
information if its input is O(1).**

Raw campaign charge is not O(1) — it spans 0 to 2.4×10⁵, nonzero median ~720. The original paper's
inputs were pre-scaled 30×30 images, not raw ADC counts.

Measured effect on the `arctan` encoding:

| Mode | formula | angle p50 | **fraction saturated** |
|---|---|---|---|
| `per_patch_max` | `x / max(patch)` | 0.070 | 0.0000 |
| `global_p99` | `clip(x / p99, 0, 1)` | 0.058 | 0.0000 |
| `log1p` | `log1p(x) / log1p(p99)` | 0.614 | 0.0000 |
| `arcsinh` | `arcsinh(x / p99)` | 0.058 | 0.0000 |
| **`none`** | raw | 1.570 | **0.8417** |

With raw charge, **84% of nonzero pixels encode to within 1% of π/2.** Every one looks identical to
the circuit. The model would still train and still report an AUC — it would simply be measuring your
normalisation choice, not the architecture.

What each mode preserves: `per_patch_max` keeps shape within a patch but loses absolute dE/dx;
`global_p99` keeps absolute scale but clips tails; `log1p` and `arcsinh` keep the full four-decade
range, compressed, at the cost of linearity.

Normalisation must be **swept jointly with the encoding** — `arctan` on `log1p` charge is a different
model from `arctan` on max-scaled charge — and must be **identical across all arms** being compared.

---

## 4. The quantum layer

### 4.1 A variational circuit is a parameterised unitary

You know the ingredients. A 4-qubit state is a complex vector of dimension 2⁴ = 16. Gates are
unitaries. Measuring `⟨Z⟩` gives a real number in [−1, 1].

A **variational quantum circuit** is a unitary built from two kinds of block:

**`U_F(x)` — the encoding** ("F" for feature map). Single-qubit rotations whose angles come from the
*data*. **Not adjustable** — this is how classical numbers get into the Hilbert space. In our
reference circuit: `Ry(arctan xᵢ)` then `Rz(arctan xᵢ²)` on qubit *i*.

**`U_θ` — the ansatz.** Entangling gates (CNOTs) plus rotations whose angles **are the adjustable
parameters**. In our reference circuit: a CNOT chain plus a general `Rot(α, β, γ)` on each qubit —
3 parameters per qubit.

```
   |0000>  --U_F(x)-->  --U_theta-->  measure <Z> on qubit 0  -->  one real number
```

That number is a differentiable function of both the pixels and the parameters. At this level, a
"quantum machine learning model" is simply **a parameterised function you fit**, whose functional
form happens to be computed by a quantum circuit.

### 4.2 Quanvolution: why 4 qubits suffice for any image

A classical **convolution** slides a small filter across an image, computing a weighted sum of the
pixels under it at each position, writing the result into a new image (a **feature map**). Stacking
such layers is a **CNN** — the standard tool for image tasks, and what LArTPC experiments already use.

A **quanvolution** replaces "weighted sum" with "run this quantum circuit".

```
   patch 31x31              slide a 2x2 window              feature map 30x30
   +-----------+            at each position:               +----------+
   | . . . . . |            4 pixels -> 4 angles            | o o o o  |
   | . [##] . .|  ------>   -> 4-qubit circuit    ----->    | o o o o  |
   | . [##] . .|            -> <Z> -> one number            | o o o o  |
   +-----------+                                            +----------+
```

**The key structural point:** the circuit only ever sees one window, so its size is set by the
*filter*, not the image. A 2×2 filter needs 4 qubits whether the image is 31×31 or 3000×3000. This
is why small quantum devices can touch real images at all.

A **channel** is one filter. A convolutional layer usually has several, each learning a different
feature and producing its own output feature map.

**This also tells you what the classical baseline must be.** A quanvolutional layer *is* a
convolutional layer with the filter replaced by a circuit — same window, same stride, same output.
So the matched comparison is a **CNN**: swap the circuit back for an ordinary weighted sum and
everything else is identical. That isolates the quantum part instead of comparing two unrelated
designs.

### 4.3 Layers versus reuploads

Two different things can be repeated. Conflating them caused one of the two real bugs in this
codebase.

**Repeating `U_θ`** — more variational layers, data encoded once:
```
   U_F(x) -> U_theta -> U_theta -> measure            n_layers = 2
```
This adds parameters and expressive power. Our reference paper does exactly this: *"the grouped box
may repeat several times to increase the number of parameters."*

**Repeating `U_F` too** — re-injecting the data between variational blocks:
```
   U_F(x) -> U_theta -> U_F(x) -> U_theta -> measure  n_reuploads = 2
```
This is **data re-uploading**. A single encoding restricts which functions of `x` the circuit can
represent — loosely, it fixes the available "frequencies". Re-injecting enriches that set. The other
reference paper does this.

**An unresolved question in our project:** when data is re-uploaded, does the second `U_θ` reuse the
first's parameters or get its own? The paper writes the block as `(U_θ U_F)^R`, whose literal reading
is *shared*. Our code uses *independent*. At R = 2 that is 16 versus 32 parameters — and since the
whole study compares **at matched parameter count**, a factor-of-2 error on the x-axis would matter.
We sweep both rather than guess.

### 4.4 Reading an actual circuit

`RUNBOOK §4.1` prints this — a 2×2 kernel, 4 qubits, pixel values `(0.1, 0.2, 0.3, 0.4)`:

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

Everything in §4.1–4.3 is visible here. Left to right:

- `RY(0.10) RZ(0.01)` on qubit 0 is **`U_F`**. The angles come from the pixel value:
  `arctan(0.1) = 0.0997`, `arctan(0.1²) = 0.00999`. Fixed by the data, **never adjusted by the fit**.
- The `╭●─╰X` pairs and every `Rot(α,β,γ)` are **`U_θ`**. Those numbers are the adjustable
  parameters — currently random, because nothing has been fitted.
- The CNOT-chain-plus-`Rot` block appears **twice** → `n_layers = 2`.
- `U_F` appears **once**, far left only → **no re-uploading**, `n_reuploads = 1`. Draw `PRD_QUANV`
  instead and `RX` encoding gates reappear mid-circuit — that is `n_reuploads = 2`.
- `⟨Z⟩` on qubit 0 only → 4 pixels in, **one number out**.
- Parameter count: 4 qubits × 3 (`Rot`) × 2 layers = **24**, matching the published value.

### 4.5 Why two implementations

We have no quantum computer, so we track the state vector explicitly: 16 complex numbers for 4
qubits, each gate a matrix multiplication. Exact — no sampling noise. (It is also why qubit count is
the cost driver: 9 qubits is a 512-dimensional state; 30 would be 10⁹ and infeasible.)

Two implementations of the *same* circuits exist deliberately:

| File | Approach | Speed | Purpose |
|---|---|---|---|
| `qmllar/circuits.py` | PennyLane, one window at a time | slow | **readable** — print the gates, draw the circuit, reason about it |
| `qmllar/fast.py` | explicit batched state vectors | ~970× faster | production runs |

They agree to ~1×10⁻¹⁵. Keep both: the slow one is how you understand what is happening, the fast
one is how you get results, and agreement is how you know your understanding is right.

**This cross-check has already caught two silently-wrong bugs**, neither of which raised an
exception:

1. **Channels were not independent** — the PennyLane path applied all channels' gates cumulatively to
   the *same* qubits and returned one measurement, instead of one circuit per channel. Invisible when
   `n_channels = 1`.
2. **Reupload parameters were reused** — the fast path applied the first reupload's `U_θ` at every
   reupload. Only visible when `n_reuploads = 2`.

Both produced plausible, wrong numbers. The general lesson: a quantum layer has no natural "does this
look right" check, so validating against an independent implementation is not optional.

---

## 5. Two ways to differentiate a quantum circuit

This explains the phrase **"gradient path"** — the thing currently missing.

### 5.1 Forward and backward

Evaluating `f(x; θ)` is the **forward pass**. Fitting also needs `∂f/∂θ` for every parameter, so you
know which way to move each one — the **backward pass**. For an ordinary network this is the chain
rule applied mechanically, called **backpropagation**, and it costs about the same as the forward
pass.

### 5.2 Parameter-shift versus backpropagation

**Parameter-shift rule.** For a circuit expectation value with rotation parameter θᵢ there is an
exact identity:

```
   df/dtheta_i  =  [ f(theta_i + pi/2) - f(theta_i - pi/2) ] / 2
```

Exact — no finite-difference error — and it needs **only circuit evaluations**, which is why it is
the *only* option on hardware: you cannot reach inside a physical device and apply the chain rule.
But it costs **two extra circuit evaluations per parameter**. At 96 parameters, ~192 extra runs per
data point.

**Backpropagation through the state vector.** On a *simulator* the situation is completely different.
We hold all 16 amplitudes in memory and every gate is a matrix multiplication — an ordinary
differentiable operation. So the chain rule applies directly, at the cost of roughly one extra
forward pass *regardless of parameter count*.

**This is the most important practical fact in the project.** The published compute costs in this
field — "multiple days per configuration", warnings about circuit-evaluation scaling — all come from
the parameter-shift regime. We never touch hardware, so we are not obliged to pay it.

Measured on this machine, for one forward pass over 292,603 patches:

| Path | µs / window | total |
|---|---|---|
| PennyLane, one call per window | ~4,500 | **331 h** |
| Vectorised state vector | **4.65** | **20.4 min** |

≈970×, for output identical to 1e-15. A 31×31 patch has 900 window positions, so the naive path makes
900 separate Python-level circuit calls per patch, each rebuilding the circuit — against a
computation that is a handful of 16×16 matrix multiplications.

**Never benchmark or make a go/no-go decision on the naive path.** Doing so would have made this
project look infeasible when it is not.

Caveat: those are **forward-pass** costs. A real epoch costs more — plausibly 2–3× for the vectorised
path (still under an hour), and vastly more for parameter-shift, where 331 h is a floor. The gap in
*training* time is therefore wider than 970×, not narrower.

---

## 6. Training

No training code exists yet. This section explains what it must do, so the plan's vocabulary makes
sense.

### 6.1 The loss function

A single number measuring how wrong the model is. For classification the standard choice is
**cross-entropy**:

```
   L  =  - (1/N) * sum [ y*log(p) + (1-y)*log(1-p) ]
```

`y ∈ {0, 1}` is the true label, `p` the predicted probability of class 1.

**This is the negative log-likelihood of a Bernoulli trial.** You already minimise −log L when you
fit; ML renamed it. Two useful properties: zero when confidently right, divergent when confidently
wrong — so overconfidence is punished hard.

### 6.2 Gradient descent

With 10³–10⁵ parameters you cannot scan or invert anything. Start random, take repeated small
downhill steps:

```
   theta  <-  theta  -  eta * dL/dtheta
```

`η` is the **learning rate** — step size, and the single most important setting. Too large diverges;
too small takes forever. Usually the first thing swept.

An **optimizer** chooses steps more cleverly than plain descent, typically adapting step size per
parameter from recent gradient history. Adam and RMSProp are standard; our reference paper used
RMSProp with η = 0.01.

### 6.3 Automatic differentiation (the "gradient path")

You do not want to derive `∂L/∂θ` by hand for 10⁵ parameters.

**Automatic differentiation** solves it. As the forward pass runs, the framework records every
operation in a graph; each knows its own local derivative. Traverse the graph backwards multiplying
them out — the chain rule, executed mechanically — and you get exact derivatives with respect to
every input. Not finite differences; exact to machine precision.

The requirement is that the computation is built from operations the framework can differentiate.
That is why torch matters: our vectorised quanvolution is written in numpy, which does not track
derivatives. **Porting it to torch is what "implementing the gradient path" means** — the same matrix
multiplications, in a library that can differentiate them and run them on a GPU. The forward
algorithm is already written and validated, so this is largely mechanical.

### 6.4 Epochs and batches

Computing `∂L/∂θ` over all 292,603 patches for one step is wasteful. Estimate it from a random
subset instead:

| Term | Meaning |
|---|---|
| **batch** | a subset (say 256 patches) used for one gradient step |
| **step** | one parameter update from one batch |
| **epoch** | one full pass through the training set — many steps |

This is **stochastic gradient descent**: noisier gradients, far more steps, and the noise often helps
escape bad regions.

### 6.5 Overfitting and when to stop

Training loss falls monotonically. Validation loss falls, then rises — the model has begun fitting
features specific to the training set. That upturn is **overfitting**, and where it happens is where
you should have stopped.

**Early stopping** monitors validation performance and keeps the best parameters. **Regularization**
discourages overfitting directly: **weight decay** (a penalty on large parameters — ridge regression)
or **dropout** (randomly zeroing activations during training, preventing over-reliance on one path).

This matters for our comparison specifically. The published study applied dropout to the quantum
model but not the classical one, and trained them for different numbers of epochs. Any apparent
quantum advantage there could be a regularization difference. **Our arms must get identical
treatment.**

### 6.6 Seeds

Training starts from random parameters and shuffles batches randomly, so two runs of identical
configuration differ. The **seed** fixes that randomness.

The spread across seeds is often comparable to the differences between architectures being compared.
So **a single run is not a measurement.** Train with many seeds and report **mean ± standard
deviation over all of them**. Reporting best-of-N inflates the result, and inflates it more the
larger N is — a real trap, and one the published work falls into.

### 6.7 The metric: ROC AUC

The model outputs a continuous score; a decision needs a threshold. Sweep the threshold, plot
true-positive against false-positive rate, and you have a **ROC curve** — familiar from any
signal-versus-background study. **AUC** is the area under it: 0.5 random, 1.0 perfect. Equivalently,
the probability that a random shower pixel scores above a random track pixel.

Why AUC rather than **accuracy** (fraction correct):

1. **No threshold to tune**, so the comparison cannot be gamed by picking a favourable operating point.
2. **Insensitive to class balance.** With 80/20 classes, accuracy of 0.80 means nothing.
3. **Much better behaved at small sample size.** Not academic: the accuracies in our primary reference
   can only take values in steps of 2.5%, because its test set is ~40 images — one image moves the
   headline number by 2.5 points.

Report accuracy *alongside* AUC for the reproduction only, since the original paper quotes accuracy.

### 6.8 Parameters versus hyperparameters

| | Set by | Examples |
|---|---|---|
| **parameters** | the fit | circuit rotation angles, network weights |
| **hyperparameters** | you, by search | learning rate, layer count, filter count, batch size, normalisation mode, dropout rate |

Searching them is **HPO**. The fairness rule: **every arm gets the same HPO budget.** Tuning the
quantum model over 200 configurations and the classical one over 5 is the standard way these
comparisons go wrong.

### 6.9 Why the CNN comes first

The claim under test is *comparable performance at far fewer parameters*. That is a statement about
where the quantum point sits relative to the **classical curve** — so you need conventional CNN
results at ~10², ~10³ and ~10⁵ parameters before a quantum AUC of, say, 0.91 means anything. A single
classical number is not enough: you would not know whether the quantum model beat "a CNN" or just
*that* CNN.

The classical side is also entirely conventional work, fast on this hardware, and the natural place
for a new student to start.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **accuracy** | fraction classified correctly. Misleading under class imbalance |
| **ansatz** (`U_θ`) | the parameterised part of a quantum circuit — the adjustable bit |
| **AUC** | area under the ROC curve. 0.5 random, 1.0 perfect |
| **automatic differentiation** | exact derivatives obtained by recording operations and applying the chain rule backwards |
| **backpropagation** | automatic differentiation applied to a layered model |
| **batch** | subset of training data used for one parameter update |
| **channel** | one filter in a convolutional layer |
| **class imbalance** | unequal numbers of examples per class |
| **CNN** | convolutional neural network — the standard classical architecture for images |
| **cross-entropy** | the classification loss; identical to negative log-likelihood |
| **dropout** | randomly zeroing activations during training, to reduce overfitting |
| **early stopping** | halt when validation performance stops improving |
| **encoding** (`U_F`) | the fixed, data-dependent part of a quantum circuit |
| **epoch** | one full pass through the training set |
| **feature map** | the image produced by a convolutional or quanvolutional layer |
| **forward / backward pass** | evaluating the model / computing its derivatives |
| **gradient descent** | fit by repeated small downhill steps |
| **gradient path** | (our usage) a differentiable implementation — one that can produce `∂f/∂θ` |
| **head** | the final classical layers mapping features to a class score |
| **hyperparameter** | a setting chosen by search rather than by fitting |
| **HPO** | hyperparameter optimization |
| **leakage** | test data correlated with training data; inflates results |
| **learning rate** (η) | gradient-descent step size |
| **loss** | the scalar being minimised |
| **MLP** | multilayer perceptron — a plain fully-connected network |
| **optimizer** | the step-choosing recipe (Adam, RMSProp, SGD) |
| **overfitting** | fitting features specific to the training set rather than the physics |
| **parameter** | a number the fit adjusts (ML also says *weight*) |
| **parameter-shift rule** | exact derivative of a circuit expectation value from two extra circuit evaluations. Required on hardware, not on a simulator |
| **quanvolution** | a quantum circuit used as a sliding filter over an image |
| **regularization** | anything that discourages overfitting |
| **reuploading** | re-injecting the data between variational blocks |
| **ROC curve** | true-positive against false-positive rate as the threshold sweeps |
| **seed** | fixes random initialisation and shuffling for reproducibility |
| **semantic segmentation** | labelling every pixel of an image |
| **SGD** | stochastic gradient descent — steps from random batches |
| **sparse representation** | storing only nonzero entries, as coordinates plus values |
| **state vector simulation** | tracking all 2ⁿ complex amplitudes exactly |
| **train / validation / test** | fit on / choose settings on / quote once. Blind analysis |
| **variational circuit** | a quantum circuit with adjustable gate parameters |
| **weight decay** | penalty on large parameters; ridge regression |
