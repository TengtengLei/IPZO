#!/usr/bin/env python
"""Effective-rank analysis of the first-layer weight update  dW = W_final - W_init.

For each random_mode, over its seeds, report the effective rank of dW (fc1,
128x1024) as mean +/- std across runs. A low effective rank means the MeZO
optimization explored few independent directions in weight space.

Reads ./weights/weights_<MODE>_SEED<seed>.npz produced by snn_mlp.py.

Usage:
    python analyze_rank.py                 # summary table
    python analyze_rank.py --per_seed      # also list every run
"""
import os
import re
import glob
import argparse
import numpy as np


def _svals(dW):
    return np.linalg.svd(dW, compute_uv=False)


def effective_rank_energy(dW):
    """Entropy effective rank over the ENERGY spectrum: p_i = s_i^2 / sum(s^2),
    erank = exp(-sum p_i log p_i).  This is the metric used in the paper plan."""
    lam = _svals(dW) ** 2
    lam = lam[lam > 0]
    p = lam / lam.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def effective_rank_rv(dW):
    """Roy-Vetterli effective rank over the singular-value distribution (not
    squared): p_i = s_i / sum(s), erank = exp(-sum p_i log p_i)."""
    s = _svals(dW)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def stable_rank(dW):
    """Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s^2) / max(s)^2."""
    s = _svals(dW)
    return float((s ** 2).sum() / (s[0] ** 2))


def analyze(weights_dir):
    pat = re.compile(r'weights_(.+)_SEED(\d+)\.npz$')
    by_mode = {}
    for path in sorted(glob.glob(os.path.join(weights_dir, 'weights_*_SEED*.npz'))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        mode, seed = m.group(1), int(m.group(2))
        d = np.load(path)
        if 'W_init' not in d or 'W_final' not in d:
            print(f"  [skip] {os.path.basename(path)}: missing W_init/W_final")
            continue
        dW = d['W_final'].astype(np.float64) - d['W_init'].astype(np.float64)
        by_mode.setdefault(mode, []).append({
            'seed': seed,
            'erank': effective_rank_energy(dW),
            'erank_rv': effective_rank_rv(dW),
            'srank': stable_rank(dW),
            'dW_fro': float(np.linalg.norm(dW)),
            'shape': dW.shape,
        })
    return by_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights_dir', default='./weights')
    ap.add_argument('--order', default='Randn,Randint,PGUXoR,PGUReuse',
                    help='comma-separated display order for modes')
    ap.add_argument('--per_seed', action='store_true', help='also print each run')
    args = ap.parse_args()

    by_mode = analyze(args.weights_dir)
    if not by_mode:
        print(f"No weight files found in {args.weights_dir}")
        return

    order = [m for m in args.order.split(',') if m in by_mode]
    order += [m for m in sorted(by_mode) if m not in order]

    shape = by_mode[order[0]][0]['shape']
    print(f"\nEffective rank of dW = W_final - W_init  "
          f"(fc1, {shape[0]}x{shape[1]}, max rank {min(shape)})")
    print("Primary metric: entropy effective rank over the energy (s^2) spectrum.\n")

    if args.per_seed:
        for mode in order:
            print(f"[{mode}]")
            for r in sorted(by_mode[mode], key=lambda r: r['seed']):
                print(f"   seed {r['seed']:>3}: erank={r['erank']:6.2f}   "
                      f"(RoyVetterli={r['erank_rv']:6.2f}, stable={r['srank']:6.2f}, "
                      f"||dW||_F={r['dW_fro']:.3f})")
            print()

    hdr = f"{'mode':<10}{'n':>4}{'erank mean':>12}{'std':>8}{'RoyVetterli':>14}{'stable_rk':>12}{'||dW||_F':>11}"
    print(hdr)
    print('-' * len(hdr))
    for mode in order:
        recs = by_mode[mode]
        e = np.array([r['erank'] for r in recs])
        erv = np.array([r['erank_rv'] for r in recs])
        sr = np.array([r['srank'] for r in recs])
        fro = np.array([r['dW_fro'] for r in recs])
        print(f"{mode:<10}{len(recs):>4}{e.mean():>12.2f}{e.std():>8.2f}"
              f"{erv.mean():>10.2f}±{erv.std():<3.2f}{sr.mean():>8.2f}±{sr.std():<3.2f}"
              f"{fro.mean():>11.3f}")
    print()


if __name__ == '__main__':
    main()
