# MeZO Fine-tuning Experiments (`finetune/`)

This folder contains all code for the **MeZO (zeroth-order optimization) fine-tuning experiments**. It is separate from the upstream Spikingformer training pipeline — the other directories in this repo (`imagenet/`, `cifar10/`, `cifar100/`, etc.) are the upstream backprop training scripts. The MeZO work lives only here.

**What it does**: takes the ImageNet-pretrained Spikingformer-8-768 and fine-tunes it on CIFAR-10 with MeZO, comparing four sources of perturbation randomness.

| Mode | Description |
|---|---|
| `Randn` | Gaussian noise (original MeZO baseline) |
| `Randint` | Uniform integer noise |
| `PGUXoR` | LFSR-based hardware RNG, XOR combination |
| `PGUReuse` | LFSR-based hardware RNG, shift-and-reuse |

MeZO needs no backward pass — each step perturbs the parameters by ±ε and estimates the gradient from the loss difference between two forward passes (`mezo.py`). Because of that, *where the perturbation noise comes from* directly affects convergence, which is what these experiments compare.

---

## Files

| File | Purpose |
|---|---|
| `run.sh` | **Entry point.** Sets the GPU, mode and output directory, launches one `nohup` background process, writes the log into `logs/` |
| `config.yml` | Default hyperparameters (dataset, model shape, freeze ratio, MeZO params, epochs). Command-line arguments override these |
| `train.py` | Main flow: build dataset → build model + load pretrained weights → freeze parameters → MeZO training loop → validate each epoch |
| `mezo.py` | The MeZO optimizer. The perturbation logic for all four modes lives in `step()` |
| `pgu.py` | Implementations of the two LFSR generators, `PGUXoR` and `PGUReuse` |
| `pgu_pre.py` | Earlier version (`PGUST` / `PGUDT` / `PGUPeZO`). **No longer imported anywhere** — kept for reference |
| `logs/` | **This is where the results are** — 12 log files, see [Results](#results) |
| `data/` | CIFAR-10 / CIFAR-100 (auto-downloaded by torchvision, `download=True`) |
| `poly/` | `primitives_degree36_100k.pkl` — cache of 100k degree-36 primitive polynomials for the LFSRs, 61 MB. If absent, `pgu.py` regenerates them with `galois`, which is slow, so keep the file |

The checkpoint output directory is created automatically on each run (`train.py:215`, from `--output_dir`). No checkpoints are included in this folder: `train.py` keeps the best weights in memory and writes them to disk only after the full epoch loop finishes, and these runs were all stopped early. The accuracies below come from the logs.

### Two external dependencies (in the parent directory)

```
../imagenet/model.py          Spikingformer model definition.
                              train.py:3 adds it to sys.path, then `import model`
                              triggers timm's @register_model registration.
../checkpoint-284.pth.tar     ImageNet pretrained weights, 254 MB (Spikingformer-8-768).
                              Path is set in config.yml:10
```

When loading the weights, `head` is skipped automatically (ImageNet has 1000 classes, CIFAR-10 has 10) — see `build_model()` in `train.py`.

## Environment

Use the `spikingformer` conda environment:

```
torch 2.1.0+cu121    torchvision 0.16.0+cu121    timm 0.6.12
spikingjelly 0.0.0.0.12    galois 0.4.11    PyYAML 6.0.3
```

`timm` must be 0.6.x — `imagenet/model.py` imports `timm.models.layers` and `timm.models.registry`, both of which moved in timm 0.9+. `galois` is used by `pgu.py` to generate the primitive polynomials.

`run.sh` does not activate the environment, so run `conda activate spikingformer` first.

## Running

```bash
bash run.sh
```

The parts of `run.sh` you would change:

```bash
CUDA_VISIBLE_DEVICES=7          # which GPU
--random_mode Randint           # Randn | Randint | PGUXoR | PGUReuse
--output_dir ./randint          # where the checkpoint goes
> logs/cifar10_randint_num5_freeze75_hflip.log    # log filename
```

Everything else is read from `config.yml`. The parameters that matter:

```yaml
freeze_ratio: 0.75      # freeze the first 75% of parameters, train the last 25% (head included)
lr: 5e-6                # MeZO learning rate
epsilon: 1e-3           # perturbation magnitude
perturb_nums: 5         # 5 perturbations per step, averaged for the gradient estimate
epochs: 500
batch_size: 64
```

> `random_mode: PGUST` in `config.yml` is the old name; the current code only accepts `PGUXoR` / `PGUReuse`. `run.sh` always passes `--random_mode` explicitly and overrides it, so running via `run.sh` is fine — but a bare `python train.py` fails on the first step with `AttributeError: 'MeZO' object has no attribute 'pgu'`. Fixing that one line in `config.yml` resolves it.

## The three experiment variants

The same four modes were run under three settings. Only two things differ between them:

| Variant | `freeze_ratio` | Horizontal flip | Trainable params | Matching logs |
|---|---|---|---|---|
| A | `0.0` (all trainable) | no | 65,601,898 | `cifar10_<mode>_num5.log` |
| B | `0.75` | no | 14,208,778 | `cifar10_<mode>_num5_freeze75.log` |
| C | `0.75` | yes | 14,208,778 | `cifar10_<mode>_num5_freeze75_hflip.log` |

- `freeze_ratio` is set in `config.yml`
- **Horizontal flip has no command-line switch** — it is the `transforms.RandomHorizontalFlip()` line at `train.py:89`, inside `build_dataset()`. **It is currently enabled**, so the code as it stands corresponds to variant C. To run A or B, comment that line out.

`build_dataset()` also contains commented-out `RandomCrop`, `ColorJitter` and `RandomErasing` lines (`train.py:88,90,93`) — tried, but not used for any of these results.

## Results

All results live in the log files under `logs/` (there is no separate CSV or results file). Log format:

```
Epoch [194/500] Batch [0/782] Loss: 1.2117  Acc: 56.25%     ← every log_interval (782) batches
[Epoch 194] Val Loss: 0.9946  Val Acc: 66.87%               ← end of each epoch
  → 新最佳: 66.87%  (epoch 194)                              ← printed when a new best is reached
```

`新最佳` is Chinese for "new best" — the line is printed every time validation accuracy beats the previous best, so the last such line in a log gives that run's final result.

To pull the best accuracy out of a run:

```bash
grep "新最佳" logs/cifar10_pguxor_num5_freeze75.log | tail -1
```

### Recorded results (CIFAR-10, lr=5e-6, ε=1e-3, perturb_nums=5)

**Variant A: all parameters trainable**

| Mode | Best Val Acc | Best epoch | Stopped at epoch |
|---|---|---|---|
| Randn | 50.61% | 76 | 94 |
| Randint | 49.05% | 81 | 107 |
| PGUXoR | 50.98% | 82 | 88 |
| PGUReuse | 42.43% | 42 | 93 |

**Variant B: freeze 75%, no flip**

| Mode | Best Val Acc | Best epoch | Stopped at epoch |
|---|---|---|---|
| Randn | 76.59% | 252 | 254 |
| Randint | 76.53% | 241 | 259 |
| PGUXoR | 76.41% | 231 | 250 |
| PGUReuse | 66.85% | 247 | 250 |

**Variant C: freeze 75% + flip**

| Mode | Best Val Acc | Best epoch | Stopped at epoch |
|---|---|---|---|
| Randn | 75.75% | 186 | 213 |
| Randint | 76.27% | 210 | 213 |
| PGUXoR | 75.46% | 209 | 211 |
| PGUReuse | 66.87% | 194 | 213 |

None of the three variants ran the full 500 epochs — they were stopped manually once the convergence trend was clear.

### ⚠️ One log file is named after the old mode name

`logs/cifar10_pgust_num5_freeze75_hflip.log` — the filename says `pgust`, but the log itself prints `[MeZO] Random mode : PGUXoR`. **This is the PGUXoR run of variant C.** There is no separate `PGUST` result, and PGUXoR + flip is not missing. Every other log file's name matches the mode it ran.
