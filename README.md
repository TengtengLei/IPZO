# IPZO — Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers

Code and experiment artifacts for:

> **Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers**
> Tengteng Lei, Prabodh Katti, Rashi Dutt, Houssem Sifaou, Tan Peng, Osvaldo Simeone, Kai Xu, Bipin Rajendran
> [arXiv:2608.21223](https://arxiv.org/abs/2608.21223)

Zeroth-order (ZO) optimization estimates gradients from forward passes alone, which makes it a natural fit for fine-tuning non-differentiable, event-driven spiking neural networks. Deploying it on in-memory computing (IMC) accelerators, however, runs into two hardware costs: the read-modify-write traffic of explicitly perturbing every weight, and the area of an RNG array large enough to supply statistically independent per-weight perturbations.

**IPZO** avoids the first by folding perturbation sums — produced by an event-triggered **perturbation generation unit (PGU)** — into the weighted sums coming out of the IMC array, so weights stay stationary and are never rewritten for perturbation. It shrinks the second by exploiting spike sparsity: the PGU generates perturbation contributions only for spike-activated weight rows, which reduces the row dimension the RNG array has to cover.

That leaves one question, and it is what this repository measures: **how far can the RNG be cheapened before accuracy suffers?**

| Mode | Perturbation source | Role |
|---|---|---|
| `Randn` | Gaussian, software RNG | baseline (original MeZO) |
| `Randint` | Uniform int8, software RNG | baseline |
| **`PGU-XOR`** | LFSR array + address-driven XOR recombination | proposed |
| **`PGU-Reuse`** | LFSR array, stream reused by shifting | cheaper, but spatially correlated |

All four are normalized to unit variance so the comparison is fair. In the code the two PGU modes are spelled `PGUXoR` and `PGUReuse`.

## What is here

| Directory | Model | Task | Role in the paper |
|---|---|---|---|
| [`Spikingformer/finetune/`](Spikingformer/finetune/) | Spikingformer-8-768 (66 M) | CIFAR-10 | Main accuracy comparison |
| [`SpikeGPT/`](SpikeGPT/) | SpikeGPT-216M | WikiText-2 / WikiText-103 | Main perplexity comparison |
| [`Spikingformer/sparsity/`](Spikingformer/sparsity/) | Spikingformer-4-384 / -8-768 | CIFAR-10/100, ImageNet | Measures the spike sparsity the PGU's row reduction relies on |
| [`snn_mlp/`](snn_mlp/) | Spiking MLP (0.13 M) | MNIST | 10 seeds per mode, plus the effective-rank analysis of `ΔW` |

Each directory has its own README with environment requirements, exact hyperparameters, and recorded results.

The circuit-level results in the paper (TSMC 16-nm area and energy) come from a separate hardware flow and are not part of this repository.

## Reproducing the headline numbers

The two figures quoted in the abstract come from:

| Paper claim | Where | Setting |
|---|---|---|
| Spikingformer / CIFAR-10: **76.41%** (PGU-XOR) vs **76.53%** (software) | [`Spikingformer/finetune/`](Spikingformer/finetune/) | variant **B** — `freeze_ratio=0.75`, no horizontal flip |
| SpikeGPT / WikiText-2 PPL: **54.20** (PGU-XOR) vs **53.23** (software) | [`SpikeGPT/`](SpikeGPT/) | configuration **1** — `DATA_DIR=data/wikitext2`, `FREEZE_RATIO=0.0` |
| PGU-Reuse: **−9.56** accuracy points, **+11.8** PPL | both of the above | 66.85% and 66.01 respectively |

Both subproject READMEs list every variant, not just the reported one, so the numbers can be checked in context.

## Why PGU-Reuse degrades

`snn_mlp/` isolates the mechanism. Across 10 seeds on MNIST, all four modes move the first-layer weights essentially the same total distance (`‖ΔW‖_F` within 2% of each other), but the *effective rank* of `ΔW` splits cleanly in two:

| Mode | Effective rank of `ΔW` (max 128) | `‖ΔW‖_F` |
|---|---:|---:|
| Randn | 120.12 ± 0.09 | 1768.1 |
| Randint | 120.16 ± 0.14 | 1735.8 |
| PGU-XOR | 120.08 ± 0.13 | 1735.0 |
| **PGU-Reuse** | **61.45 ± 0.19** | 1735.6 |

Reusing one LFSR stream by shifting it produces perturbation vectors that are not mutually independent, so the optimizer explores roughly half as many directions for the same amount of movement. The XOR recombination in PGU-XOR removes that spatial correlation and restores full-rank exploration. Reproduce with `python rank.py --per_seed`.

## Provenance and licensing

Two directories are modified copies of existing projects. Their original licenses are preserved in place; modifications are confined to the optimizer path and the analysis scripts, and the model definitions are unchanged.

| Directory | Upstream | License |
|---|---|---|
| `SpikeGPT/` | [ridgerchu/SpikeGPT](https://github.com/ridgerchu/SpikeGPT) | BSD 2-Clause, © 2023 Ruijie Zhu — [`SpikeGPT/LICENSE`](SpikeGPT/LICENSE) |
| `Spikingformer/` | [zhouchenlin2096/Spikingformer](https://github.com/zhouchenlin2096/Spikingformer) | Apache 2.0 — [`Spikingformer/LICENSE`](Spikingformer/LICENSE) |
| `snn_mlp/` | original to this work | *(to be chosen)* |

Changes relative to upstream:

- **`SpikeGPT/`** — added `src/mezo.py` and `src/pgu.py` (the MeZO optimizer and the two PGU generators); `src/trainer.py` routes to MeZO; `train.py` gains the MeZO and `--freeze_ratio` arguments; `finetune.sh` and `eval_ppl.py` are new.
- **`Spikingformer/`** — added `finetune/` (MeZO fine-tuning on CIFAR-10) and `sparsity/` (firing-rate measurement). The upstream training code under `cifar10/`, `cifar100/`, `imagenet/` and the DVS directories is untouched.

## What is not in this repository

Model weights, datasets and raw run logs are excluded (see [`.gitignore`](.gitignore)) — together they are several GB, past GitHub's limits. Each subproject README says what to fetch and where to put it:

| Needed by | Item | Source |
|---|---|---|
| `SpikeGPT/` | `SpikeGPT-216M.pth` | [HuggingFace](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M) |
| `SpikeGPT/` | WikiText-2 / WikiText-103 as plain `.txt` | token-count fingerprints are given in its README for verification |
| `Spikingformer/` | `checkpoint-284.pth.tar` (ImageNet), `checkpoint-405.pth.tar` (CIFAR-10) | [upstream repo](https://github.com/zhouchenlin2096/Spikingformer) |
| `Spikingformer/sparsity/` | `checkpoint-cifar100.pth.tar` | not published upstream — train with `cifar100/train.py` |
| `snn_mlp/`, `Spikingformer/` | MNIST / CIFAR | auto-downloaded by torchvision |
| all | `poly/primitives_degree36_*.pkl` | auto-generated by `pgu.py` via `galois` on first run (slow — cache it) |

Two artifacts *are* committed, because they make the analysis reproducible in seconds instead of hours:

- `snn_mlp/output/` — per-epoch accuracy and loss for all 4 modes × 10 seeds (80 small text files)
- `snn_mlp/weights/` — `W_init` and `W_final` of the first layer for all 40 runs (41 MB), the input to `rank.py`. Without them, reproducing the effective-rank result means ~30 h of retraining.

## Citation

```bibtex
@article{lei2026ipzo,
        title   = {Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers},
        author  = {Lei, Tengteng and Katti, Prabodh and Dutt, Rashi and Sifaou, Houssem and Peng, Tan and
                   Simeone, Osvaldo and Xu, Kai and Rajendran, Bipin},
        journal = {arXiv preprint arXiv:2608.21223},
        year    = {2026},
        url     = {https://arxiv.org/abs/2608.21223}
}
```

Please also cite the upstream models this work builds on:

```bibtex
@article{zhu2023spikegpt,
        title   = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
        author  = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
        journal = {arXiv preprint arXiv:2302.13939},
        year    = {2023}
}

@article{zhou2023spikingformer,
        title   = {Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
        author  = {Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and
                   Zhou, Huihui and Tian, Yonghong},
        journal = {arXiv preprint arXiv:2304.11954},
        year    = {2023},
        url     = {https://arxiv.org/abs/2304.11954}
}
```
