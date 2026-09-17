#!/bin/bash
# ============================================================================
# MeZO perturbation-source comparison on SpikeGPT-216M.
# Launches all four random modes at once, ONE mode per GPU (no sharing):
#   Randn (Gaussian)   Randint (uniform)   PGUXoR (LFSR)   PGUReuse (LFSR)
#
# The three reported experiments differ only in DATA_DIR and FREEZE_RATIO:
#   1. DATA_DIR=data/wikitext2    FREEZE_RATIO=0.0   WikiText-2,   all params
#   2. DATA_DIR=data/wikitext103  FREEZE_RATIO=0.0   WikiText-103, all params
#   3. DATA_DIR=data/wikitext103  FREEZE_RATIO=0.5   WikiText-103, bottom half frozen
# Everything else is held fixed -- see README.md.
#
# Outputs (under $OUT/):
#   checkpoint/<MODE>/model_best.pth     weights (single best, overwritten)
#   log/out_<MODE>.log                   stdout (tqdm per-step + per-epoch line)
#   result/results_val_<MODE>.txt        `epoch train_loss train_ppl lr time`
#   result/test_ppl_<MODE>.txt           warm/strided test perplexity
# Run:  bash finetune.sh
# ============================================================================

# ---- settings: edit these four, nothing else -------------------------------
DATA_FORMAT=bpe                    # char | bpe
DATA_DIR=data/wikitext103          # dataset dir; train/validation/test .txt live here
FREEZE_RATIO=0.5                   # 0.0 = all params trainable; 0.5 = freeze bottom half of the blocks
GPUS=(4 5 6 7)                     # one GPU per mode, in the order Randn Randint PGUXoR PGUReuse
OUT_ROOT=./runs                    # where checkpoints/logs/results go
CONDA_ENV_BIN=                     # optional: path to your env's bin/, e.g. /opt/conda/envs/spikegpt/bin
                                   # leave empty to use whatever `python` and `ninja` are already on PATH
# ----------------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1

# python + ninja must both be reachable (ninja compiles the WKV CUDA kernel).
if [ -n "$CONDA_ENV_BIN" ]; then
    [ -d "$CONDA_ENV_BIN" ] || { echo "ERROR: CONDA_ENV_BIN does not exist: $CONDA_ENV_BIN"; exit 1; }
    export PATH="$CONDA_ENV_BIN:$PATH"
fi
command -v python >/dev/null || { echo "ERROR: no python on PATH"; exit 1; }
command -v ninja  >/dev/null || { echo "ERROR: no ninja on PATH (needed to build the WKV CUDA kernel)"; exit 1; }
echo "using python: $(command -v python)"

OUT="$OUT_ROOT/${DATA_FORMAT}_$(basename "$DATA_DIR")_fz${FREEZE_RATIO}"
mkdir -p "$OUT/checkpoint" "$OUT/log" "$OUT/result"
echo "DATA_FORMAT=$DATA_FORMAT  ->  outputs in $OUT/"

EVAL=eval_ppl.py   # in the repo; cwd is set by the `cd` above. Warm/strided WT-2 test ppl.

launch () {
    MODE=$1
    GPU=$2
    mkdir -p "$OUT/checkpoint/$MODE"
    # train, then (on success) auto-run the warm/strided WT-2 test-ppl eval on this mode's checkpoint.
    # both steps share one backgrounded subshell pinned to $GPU.
    (
      CUDA_VISIBLE_DEVICES=$GPU RWKV_FLOAT_MODE=32 RWKV_LOAD_MODEL=True \
      MEZO_LOG="$OUT/result/results_val_${MODE}.txt" \
      python train.py \
        --mode "finetune" \
        --data_format "$DATA_FORMAT" \
        --data_train "$DATA_DIR/train.txt" \
        --data_valid "$DATA_DIR/validation.txt" \
        --data_test  "$DATA_DIR/test.txt" \
        --optimizer_type "mezo" \
        --ctx_len 1024 \
        --n_layer 18 \
        --n_embd 768 \
        --batch_size 8 \
        --n_epoch 4000 \
        --epoch_length 640 \
        --pretrained_model "SpikeGPT-216M.pth" \
        --save_path "$OUT/checkpoint/$MODE" \
        --epoch_save_freq 500 \
        --seed 42 \
        --mezo_random_mode "$MODE" \
        --mezo_learning_rate 5e-6 \
        --mezo_lr_schedule constant \
        --mezo_lr_final 5e-6 \
        --mezo_epsilon 1e-3 \
        --mezo_perturb_nums 5 \
        --mezo_seed "0x123456789" \
        --freeze_ratio "$FREEZE_RATIO" \
      && CUDA_VISIBLE_DEVICES=$GPU RWKV_FLOAT_MODE=32 RWKV_LOAD_MODEL=True \
         python "$EVAL" \
           --ckpt "$OUT/checkpoint/$MODE/model_best.pth" \
           --tag  "$MODE" \
           --out  "$OUT/result/test_ppl_${MODE}.txt" \
           --data_test "$DATA_DIR/test.txt"
    ) > "$OUT/log/out_${MODE}.log" 2>&1 &
    echo "launched $MODE on GPU $GPU  (pid $!)  train->$OUT/result/results_val_${MODE}.txt  test->$OUT/result/test_ppl_${MODE}.txt"
}

launch Randn    "${GPUS[0]}"
launch Randint  "${GPUS[1]}"
launch PGUXoR   "${GPUS[2]}"
launch PGUReuse "${GPUS[3]}"

echo ""
echo "all 4 launched ($DATA_FORMAT). watch a run:   tail -f $OUT/log/out_PGUXoR.log"
echo "watch the metric:                             tail -f $OUT/result/results_val_PGUXoR.txt"
