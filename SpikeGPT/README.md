# SpikeGPT + MeZO: Comparing Perturbation Random Sources

MeZO (zeroth-order) fine-tuning of SpikeGPT-216M, comparing **four sources of perturbation randomness** on language modeling:

| Mode | Description |
|---|---|
| `Randn` | Gaussian noise (original MeZO baseline) |
| `Randint` | Uniform integer noise |
| `PGUXoR` | LFSR-based hardware RNG, XOR combination |
| `PGUReuse` | LFSR-based hardware RNG, shift-and-reuse |

`PGUXoR` / `PGUReuse` are implemented in [`src/pgu.py`](src/pgu.py) and wired into the optimizer in [`src/mezo.py`](src/mezo.py).

Upstream project: [SpikeGPT](https://github.com/ridgerchu/SpikeGPT) ([arXiv:2302.13939](https://arxiv.org/abs/2302.13939)). This repo replaces the optimizer path; the model itself is unchanged.

---

## Settings

Everything you need to change lives in one block at the top of [`finetune.sh`](finetune.sh):

```bash
DATA_FORMAT=bpe                    # char | bpe
DATA_DIR=data/wikitext103          # dataset dir; train/validation/test .txt live here
FREEZE_RATIO=0.5                   # 0.0 = all params trainable; 0.5 = freeze bottom half of the blocks
GPUS=(4 5 6 7)                     # one GPU per mode, in the order Randn Randint PGUXoR PGUReuse
OUT_ROOT=./runs                    # where checkpoints/logs/results go
CONDA_ENV_BIN=                     # optional: path to your env's bin/
```

`CONDA_ENV_BIN` exists because the WKV CUDA kernel is JIT-compiled and needs **both `python` and `ninja` on `PATH`**. Leave it empty if your environment is already activated; otherwise point it at your env's `bin/`. The script verifies both binaries are reachable and aborts with a clear message if not, rather than silently falling back to the system interpreter and failing later somewhere unrelated.

Output goes to `$OUT_ROOT/<format>_<dataset>_fz<ratio>/`, so the three experiments below land in separate directories automatically.

---

## Environment

```bash
conda create -n spikegpt python=3.10
conda activate spikegpt
pip install -r requirements.txt
```

Verified working combination (CUDA 12.1 / H100):

```
torch 2.5.1+cu121    transformers 5.9.0    datasets 5.0.0
numpy 1.26.4         galois 0.4.11         cupy-cuda12x 12.3.0
```

`transformers` and `datasets` are both on major version 5.x — other major versions are unlikely to work.

Two dependencies that are easy to miss:

- **`ninja`** — the WKV CUDA kernel ([`cuda/wkv_cuda.cu`](cuda/wkv_cuda.cu)) is JIT-compiled at [`src/model.py:53`](src/model.py#L53) via `torch.utils.cpp_extension.load()`, which requires `ninja` on `PATH`.
- **`cupy`** — [`src/model.py:316,318`](src/model.py#L316) hardcode the spiking neuron backend: `MultiStepLIFNode(..., backend='cupy')`. spikingjelly wraps `import cupy` in a try/except, so **a missing cupy does not fail at import time — it fails in the forward pass**. Match `cupy-cuda12x` to your CUDA major version (use `cupy-cuda11x` for CUDA 11.x).

## Files you need to supply

| File | Size | Source |
|---|---|---|
| `SpikeGPT-216M.pth` | 861 MB | [HuggingFace: ridger/SpikeGPT-OpenWebText-216M](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M) — place in the repo root |
| `data/wikitext2/{train,validation,test}.txt` | 13 MB | see [Data fingerprints](#data-fingerprints) to verify your copy |
| `data/wikitext103/{train,validation,test}.txt` | 524 MB | same |
| `poly/primitives_degree36_100k.pkl` | 61 MB | **auto-generated if absent** — `src/pgu.py` will compute 100k degree-36 primitive polynomials via `galois`, which is slow. Copying the file is strongly preferred. |

`20B_tokenizer.json` (GPT-NeoX BPE, vocab 50277) is included in the repo.

---

## The three experiment configurations

Switch by editing two variables at the top of [`finetune.sh`](finetune.sh). All four modes launch in parallel, one mode per GPU.

| # | `DATA_DIR` | `FREEZE_RATIO` | Description |
|---|---|---|---|
| 1 | `data/wikitext2` | `0.0` | WikiText-2, all parameters trainable |
| 2 | `data/wikitext103` | `0.0` | WikiText-103, all parameters trainable |
| 3 | `data/wikitext103` | `0.5` | Freeze the bottom half of the blocks (0–8); 107,699,712 / 215,399,424 params trainable |

Set `DATA_DIR` and `FREEZE_RATIO` in the settings block accordingly; `finetune.sh` ships configured for experiment 3.

Every other hyperparameter is identical across the three experiments. Do not change them — the existing results depend on these exact values:

```
--data_format bpe            --ctx_len 1024      --n_layer 18       --n_embd 768
--batch_size 8               --n_epoch 4000      --epoch_length 640
--optimizer_type mezo        --mezo_perturb_nums 5
--mezo_learning_rate 5e-6    --mezo_lr_schedule constant
--mezo_epsilon 1e-3          --mezo_seed 0x123456789    --seed 42
```

## Running

### All four modes at once

```bash
bash finetune.sh
```

The script launches four `train.py` processes in parallel — one mode per GPU, each backgrounded — and when a mode finishes training it automatically runs `eval_ppl.py` on that mode's `model_best.pth`.

Monitoring:

```bash
tail -f runs/<experiment>/log/out_PGUXoR.log
tail -f runs/<experiment>/result/results_val_PGUXoR.txt
```

### A single mode by hand

`finetune.sh` is only a launcher; this is exactly what it runs for one mode. Useful if you want to run modes on separate machines, or change one thing without editing the script:

```bash
CUDA_VISIBLE_DEVICES=0 RWKV_FLOAT_MODE=32 RWKV_LOAD_MODEL=True \
MEZO_LOG=runs/manual/results_val_PGUXoR.txt \
python train.py \
    --mode finetune --data_format bpe \
    --data_train data/wikitext2/train.txt \
    --data_valid data/wikitext2/validation.txt \
    --data_test  data/wikitext2/test.txt \
    --optimizer_type mezo --mezo_random_mode PGUXoR \
    --ctx_len 1024 --n_layer 18 --n_embd 768 \
    --batch_size 8 --n_epoch 4000 --epoch_length 640 \
    --pretrained_model SpikeGPT-216M.pth \
    --save_path runs/manual/checkpoint/PGUXoR --epoch_save_freq 500 \
    --seed 42 --mezo_seed 0x123456789 \
    --mezo_learning_rate 5e-6 --mezo_lr_schedule constant --mezo_lr_final 5e-6 \
    --mezo_epsilon 1e-3 --mezo_perturb_nums 5 \
    --freeze_ratio 0.0
```

Then score the checkpoint:

```bash
python eval_ppl.py \
    --ckpt runs/manual/checkpoint/PGUXoR/model_best.pth \
    --tag PGUXoR \
    --out runs/manual/test_ppl_PGUXoR.txt \
    --data_test data/wikitext2/test.txt
```

> **`MEZO_LOG` is not optional.** Set it to a different path per mode. See the note under [Output layout](#output-layout) for why.

## Output layout

```
$OUT/
├── checkpoint/<MODE>/model_best.pth   # overwritten whenever training loss improves,
│                                      # checked every epoch_save_freq (500) epochs
├── log/out_<MODE>.log                 # stdout (includes tqdm progress)
└── result/
    ├── results_val_<MODE>.txt         # one line per epoch
    └── test_ppl_<MODE>.txt            # test perplexity, written after training
```

`results_val_<MODE>.txt` — five space-separated fields, no header ([`src/trainer.py:240`](src/trainer.py#L240)):

```
epoch  train_loss  train_ppl  lr  timestamp
14     5.364360    213.6544   0.00000500  2026-07-27 16:12:47.301118
```

`test_ppl_<MODE>.txt` — a single tab-separated line ([`eval_ppl.py`](eval_ppl.py)):

```
PGUXoR	loss=4.430258	warm_test_ppl=83.9764	scored=300328	2026-07-16 ...
```

Evaluation protocol: window 1024, stride 512 — every scored token has at least 512 tokens of left context within the same forward pass, and the SNN state is reset per window. Per-token cross-entropy is read from a forward hook on `model.head`, so it never relies on the scalar loss the model returns.

> **`MEZO_LOG` must be set.** [`src/trainer.py:33`](src/trainer.py#L33) opens the results file at module level: `open(os.environ.get("MEZO_LOG", "wik8-0.01.txt"), "a")` — note **append** mode. Without it, all four modes write into the same `wik8-0.01.txt`, interleaved with whatever was there from previous runs. `finetune.sh` sets it per mode; remember it if you invoke `train.py` directly.

---

## Existing results

Raw result files are not included in this repo (checkpoints and logs are gitignored). Use these numbers to check a reproduction attempt.

**Configuration 2: WikiText-103, all parameters** (completed, 4000 epochs)

| Mode | Final train ppl | warm test ppl |
|---|---|---|
| Randn | 89.24 | 82.32 |
| Randint | 89.92 | 82.67 |
| PGUXoR | 85.91 | 83.98 |
| PGUReuse | 98.07 | 94.73 |

**Configuration 1: WikiText-2, all parameters** (completed, 4000 epochs)

| Mode | Final train ppl | warm test ppl |
|---|---|---|
| Randn | 78.42 | 53.07 |
| Randint | 78.86 | 53.23 |
| PGUXoR | 79.44 | 54.20 |
| PGUReuse | 98.10 | 66.01 |

**Configuration 3: WikiText-103 + freeze 0.5** — ⚠️ **incomplete.** It only reached epochs 14–21 out of 4000, train ppl was still 198–214, there is no test perplexity, and there are no checkpoints (`--epoch_save_freq 500` never triggered a first save). This configuration needs to be re-run from scratch.

---

## Repository layout

```
finetune.sh              Entry point: launches 4 modes in parallel + auto-evaluates
train.py                 Training driver (argparse → GPT + Trainer)
eval_ppl.py              Warm/strided test perplexity; run automatically by finetune.sh
requirements.txt

src/
├── model.py             SpikeGPT (GPT/GPTConfig); JIT-compiles the WKV kernel; LIF backend='cupy'
├── trainer.py           Training loop; :91 lazily imports MeZO; :33 opens the MEZO_LOG handle
├── mezo.py              MeZO optimizer + LR schedule (constant/linear/cosine)
├── pgu.py               PGUXoR / PGUReuse — LFSR random number generators
├── utils.py             Dataset / set_seed
├── binidx.py            MMapIndexedDataset (top-level import at train.py:14 — unused in these
│                        experiments, but removing it breaks the import)
└── spikingjelly/        Vendored SNN library (only clock_driven's neuron/functional/surrogate
                         are used, but the internal import graph makes pruning risky)

cuda/                    wkv_cuda.cu + wkv_op.cpp — WKV CUDA kernel source
20B_tokenizer.json       GPT-NeoX BPE, vocab 50277
poly/                    Primitive-polynomial cache for the LFSRs
data/                    wikitext2/ and wikitext103/ (train/validation/test .txt)
```

`data/wt103_wt2_train_ids.npy` (485 MB) is a pre-tokenized cache of the wt103+wt2 training sets (127,085,900 tokens, int32). **None of the three experiments use it** — `finetune.sh` passes the `.txt` files and re-tokenizes on every start. `train.py:133` has an `.npy` branch, so pointing `--data_train` at it would skip that startup cost, but the data loading path would then differ from the one that produced the existing results.

## Data fingerprints

Use these to confirm your data files match the ones that produced the results above (token counts are from `20B_tokenizer.json`):

| Dataset | train | validation | test |
|---|---|---|---|
| wikitext2 (BPE) | 2,435,985 | 256,959 | 293,904 |
| wikitext103 (BPE) | 124,649,982 | 262,450 | 300,329 |

To check:

```bash
python -c "
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file='20B_tokenizer.json')
for f in ['train','validation','test']:
    p = f'data/wikitext2/{f}.txt'
    print(p, len(tok(open(p).read())['input_ids']))
"
```

## Known gotchas

1. **Unset `MEZO_LOG` cross-contaminates all four modes' results.** [`src/trainer.py:33`](src/trainer.py#L33) opens the results file at module level in **append** mode. `finetune.sh` sets it per mode; remember it if you invoke `train.py` directly.
2. **`train.py --help` has a side effect** — because of that same module-level `open(..., "a")`, merely importing the module or printing help creates `wik8-0.01.txt` in the current directory.
3. **`eval_ppl.py` chdirs to its own directory** on startup so that `src/` imports and the WKV kernel sources resolve. Harmless, but it means relative `--data_test` paths are interpreted against the repo root, not your shell's cwd.
4. **`src/binidx.py` looks unused but cannot be deleted** — [`train.py:14`](train.py#L14) imports `MMapIndexedDataset` at the top level even though none of these experiments use it.
5. **Seeds are not echoed to the logs.** `--seed 42` and `--mezo_seed 0x123456789` are set in `finetune.sh` but never printed, so a log alone cannot confirm which seed a run used.

## Citation

```bibtex
@article{zhu2023spikegpt,
        title = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
        author = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
        journal = {arXiv preprint arXiv:2302.13939},
        year    = {2023}
}
```
