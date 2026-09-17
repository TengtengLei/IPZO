# MeZO on a Spiking MLP — Four Perturbation Random Sources, 10 Seeds Each

Trains a small spiking MLP on MNIST with **MeZO** (zeroth-order optimization — no backward pass) and compares **four sources of perturbation randomness**. Each source is run over **10 fixed seeds** so the comparison carries error bars rather than resting on a single run.

| Mode | Perturbation source |
|---|---|
| `Randn` | Gaussian, PyTorch native RNG |
| `Randint` | Uniform int8, PyTorch native RNG |
| `PGUXoR` | LFSR hardware RNG, XOR combination |
| `PGUReuse` | LFSR hardware RNG, shift-and-reuse |

MeZO estimates the gradient from the loss difference between two forward passes at `θ ± ε·u`, so the statistical quality of the perturbation vector `u` is the only thing that differs between these four runs. Everything else — model, data, seeds, schedule — is held fixed.

Beyond accuracy, the effective rank of the first-layer weight update `ΔW = W_final − W_init` is measured to show *how many independent directions* each random source actually explored.

---

## Requirements

```bash
conda create -n snn_env python=3.10
conda activate snn_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "spikingjelly>=0.0.0.0.14" galois numpy matplotlib tqdm
```

| Package | Why |
|---|---|
| `spikingjelly` **≥ 0.0.0.0.14** | `snn_mlp.py` imports `from spikingjelly.activation_based import neuron, layer, functional, surrogate`. The `activation_based` module was introduced in 0.0.0.0.14 (it replaced the older `clock_driven` package) — earlier versions fail with `ModuleNotFoundError`. |
| `galois` | The two LFSR generators use it to build degree-36 primitive polynomials |
| `torch`, `torchvision` | CUDA build; torchvision also supplies the MNIST loader |
| `numpy` | `rank.py` (SVD) and the `.npz` weight dumps |
| `matplotlib` | `plot_rank.py` |
| `tqdm` | Training progress bar |

A CUDA GPU is required. Memory use is small (the model is ~0.13 M parameters), but MeZO runs two forward passes per batch at `T=16` timesteps, so the job is compute-bound rather than memory-bound.

### Files that must be present

| Path | Size | Notes |
|---|---|---|
| `poly/primitives_degree36_10k.pkl` | 5.7 MB | Cache of 10k degree-36 primitive polynomials, required by `PGUXoR` and `PGUReuse`. **If missing, it is regenerated with `galois`, which is slow** — keep the file. |
| `data/` | 64 MB | MNIST. Auto-downloaded by torchvision (`download=True`) if absent. |

---

## Files

| File | Purpose |
|---|---|
| `run.sh` | **Entry point.** Set `RANDOM_MODE` at the top, then it loops over all 10 seeds sequentially |
| `snn_mlp.py` | **Everything for training, in one self-contained file**: the spiking MLP, the Poisson encoder, the `PGUXoR` / `PGUReuse` LFSR generators, and the `MeZO` optimizer. No local imports |
| `rank.py` | Effective-rank analysis. Reads `weights/*.npz`, prints a summary table |
| `plot_rank.py` | Produces the paper figure `fig_rank.png` (two panels: effective rank and `‖ΔW‖_F`). Data is hard-coded from `rank.py`'s output, so it reads no files |
| `output/` | **Per-epoch results** — 80 files, see [Outputs](#outputs) |
| `weights/` | 40 `.npz` files, each holding `W_init` and `W_final` of the first layer — the input to `rank.py` |
| `log/` | Raw stdout of the four runs, ~110 MB each. Mostly tqdm progress-bar redraws; only training metrics, no test metrics. The real results are in `output/` |
| `pgu.py` | Legacy standalone PGU implementation. Not imported by anything in the current pipeline; kept for reference |

---

## Running

Set the mode at the top of `run.sh`, then launch:

```bash
RANDOM_MODE="Randn"    # Randn | Randint | PGUXoR | PGUReuse
```

```bash
bash run.sh
```

One invocation runs **one mode across all 10 seeds, sequentially**:

```
seeds=(15 23 37 42 56 69 73 84 92 100)
```

Repeat four times, once per mode, to reproduce the full comparison.

**Runtime**: the logs show roughly 37 batch/s over 469 batches, i.e. ~13 s per training epoch. At 200 epochs that is ~45 min per seed, so **~7.5 h per mode** and ~30 h for all four if run sequentially. The four modes are independent and can be run in parallel on separate GPUs (`CUDA_VISIBLE_DEVICES`).

`run.sh` does not activate the conda environment — do that first.

## Model and hyperparameters

Fixed across all four modes. Changing any of these breaks comparability with the recorded results.

```
Architecture   1024 -> 128 -> 10, two LIF layers (tau=2.0, v_threshold=1.0,
               Sigmoid surrogate), Poisson-encoded input
Input          MNIST resized 28x28 -> 32x32 (bicubic), flattened to 1024
Timesteps      T = 16
Epochs         200
Batch size     128 (train) / 1024 (test)
Learning rate  lr = 2.5, cosine-annealed to eta_min = 0.1 over 200 epochs
MeZO           mu = 1.0, one perturbation per step
```

`lr = 2.5` looks large for a first-order optimizer but is normal for MeZO — the gradient estimate is a scalar loss difference times a unit-variance random direction, not a true gradient. All four perturbation sources are normalized to variance 1 so the comparison is fair.

**Seeding** is thorough, so runs are reproducible: `set_seed()` covers `random`, `numpy`, `torch` and `torch.cuda`, and the Poisson encoder, weight initialization and DataLoader each get their own seeded generator. The two LFSR generators use a *fixed* internal seed (`0x123456789`) in every run — so run-to-run variation comes from weight initialization and data shuffling order, not from the perturbation stream.

## Outputs

Per run (one mode, one seed), three artifacts:

```
output/Accuracy_MNIST_<MODE>_SEED<N>.txt   200 test accuracies, comma-separated, e.g. "42.54%, 56.67%, ..."
output/Loss_MNIST_<MODE>_SEED<N>.txt       200 training losses
weights/weights_<MODE>_SEED<N>.npz         W_init + W_final of fc1, each [128, 1024] float32
```

The accuracy file is one line of comma-separated values under an `All Test Accuracies:` header — the last value is the final-epoch accuracy, the maximum is the best. To pull the final accuracy of a run:

```bash
python -c "
import re
v = re.findall(r'([\d.]+)%', open('output/Accuracy_MNIST_PGUXoR_SEED15.txt').read())
print('final', v[-1], ' best', max(v, key=float))
"
```

Note that all three files are written **only after all 200 epochs finish** — killing a run early leaves nothing behind.

---

## Recorded results

MNIST, 10 seeds per mode, 200 epochs.

### Test accuracy

| | Randn | Randint | PGUXoR | PGUReuse |
|---|---:|---:|---:|---:|
| Final epoch, mean | 88.74 | 88.76 | 88.75 | **82.69** |
| Final epoch, std | 0.32 | 0.29 | **0.15** | 4.28 |
| Best, mean | 88.96 | 89.07 | 89.01 | **82.96** |
| Best, std | 0.27 | 0.19 | 0.19 | 4.35 |

`PGUXoR` matches both software baselines to within 0.02 points and has the **smallest spread of the four** (std 0.15 across 10 seeds, all between 88.52 and 89.03). `PGUReuse` is about 6 points lower and unstable — its worst seeds land at 71.93 and 79.01.

### Effective rank of `ΔW`

Reproduce with `python rank.py --per_seed`:

| mode | n | erank (mean ± std) | Roy-Vetterli | stable rank | `‖ΔW‖_F` |
|---|---:|---:|---:|---:|---:|
| Randn | 10 | 120.12 ± 0.09 | 125.89 ± 0.03 | 70.40 ± 0.73 | 1768.1 |
| Randint | 10 | 120.16 ± 0.14 | 125.90 ± 0.04 | 70.86 ± 1.29 | 1735.8 |
| PGUXoR | 10 | 120.08 ± 0.13 | 125.88 ± 0.04 | 71.04 ± 0.96 | 1735.0 |
| PGUReuse | 10 | **61.45 ± 0.19** | 63.35 ± 0.05 | 36.72 ± 2.15 | 1735.6 |

`ΔW` is 128×1024, so the maximum possible rank is 128. Primary metric is the entropy effective rank over the energy (`s²`) spectrum.

**This is the point of the analysis**: all four modes move the weights essentially the same total distance (`‖ΔW‖_F` = 1735–1768), but `PGUReuse` does so within **half the number of independent directions** (61.5 vs ~120). Reusing the LFSR stream by shifting it produces perturbation vectors that are not independent enough, which is the mechanism behind its accuracy gap.

### The figure

```bash
python plot_rank.py      # writes fig_rank.png, 600 dpi, 3.8 x 2.2 in
```

Two-panel bar chart: (a) effective rank, (b) `‖ΔW‖_F` in units of 10³, each with mean ± std over the 10 seeds. Requires the **Arial** font — without it matplotlib substitutes a wider font and the value labels collide. Change `rcParams['font.family']` at line 36 if Arial is unavailable.
