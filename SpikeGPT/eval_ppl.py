"""
Parameterized warm/strided WikiText-2 test perplexity for SpikeGPT-216M (BPE).
Paper-faithful token-level ppl on the WT-2 TEST set (compare to paper's 18.01).
Invoked automatically by finetune.sh after each mode finishes:
    python eval_ppl.py --ckpt <model_best.pth> --tag <MODE> --out <file>

Protocol: window=1024, stride=512 -> each scored token has >=512 tokens of
left context inside the same forward (SNN state reset per window). Per-token CE
is read off a forward hook on model.head, so we never depend on the scalar loss
the model returns. Does NOT modify any repo file.
"""
import os, sys, math, argparse, datetime

os.environ["RWKV_FLOAT_MODE"] = "32"
os.environ["RWKV_LOAD_MODEL"] = "True"
os.environ.setdefault("RWKV_NUM_GPUS", "1")

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)   # so `src` imports and the WKV cuda kernel (cuda/*.cpp/.cu) resolve

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from src.model import GPT, GPTConfig
from src.spikingjelly.clock_driven import functional

CTX = 1024
STRIDE = 512
N_LAYER, N_EMBD, VOCAB = 18, 768, 50277
TEST = "data/wikitext2/test.txt"
TOKF = "20B_tokenizer.json"

_logits = {}
def _hook(mod, inp, out):
    _logits["v"] = out


@torch.no_grad()
def warm_ppl(model, ids):
    N = len(ids)
    nll_sum, n_scored = 0.0, 0
    prev_end, begin = 0, 0
    while True:
        end = min(begin + CTX, N - 1)
        trg_len = end - prev_end
        x = torch.tensor(ids[begin:end], dtype=torch.long).cuda().unsqueeze(0)
        y = torch.tensor(ids[begin + 1:end + 1], dtype=torch.long).cuda().unsqueeze(0)
        functional.reset_net(model)
        model(x, y)
        ce = F.cross_entropy(_logits["v"][0], y[0], reduction="none")
        nll_sum += ce[-trg_len:].sum().item()
        n_scored += trg_len
        prev_end = end
        if end >= N - 1:
            break
        begin += STRIDE
    mean_ce = nll_sum / n_scored
    return mean_ce, math.exp(mean_ce), n_scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data_test", default=TEST,
                    help="test .txt to score (default: WT-2 test — keeps old runs backward-compatible)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[warm-eval] MISSING checkpoint: {args.ckpt}", flush=True)
        sys.exit(1)

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKF)
    ids = np.array(tok(open(args.data_test, encoding="utf-8").read())["input_ids"], dtype=np.int64)

    model = GPT(GPTConfig(VOCAB, CTX, model_type="RWKV", n_layer=N_LAYER, n_embd=N_EMBD)).cuda()
    model.head.register_forward_hook(_hook)
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))

    ce, ppl, ns = warm_ppl(model, ids)
    ts = datetime.datetime.now()
    line = f"{args.tag}\tloss={ce:.6f}\twarm_test_ppl={ppl:.4f}\tscored={ns}\t{ts}"
    print(f"[warm-eval] window={CTX} stride={STRIDE} warmup={CTX-STRIDE}", flush=True)
    print(f"[warm-eval] {line}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
