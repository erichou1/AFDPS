"""Aggregate relative-L2 across saved per-case result files (e.g. from multi-GPU shards).

Reuses the benchmark's canonical metric (eval.relative_l2) on the saved, unnormalized
(recon, target) pairs, so the number is identical to what main.py's NavierStokes2d
evaluator would report -- just gathered across shard directories.

Usage:
  python scripts/aggregate.py "exps/inference/navier-stokes-afdps-ds2/AFDPS/final_g10_shard*/result_*.pt"
  python scripts/aggregate.py <file1.pt> <file2.pt> ...
"""
import os
import re
import sys
import glob
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import relative_l2


def collect(patterns):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return files


def case_id(path):
    m = re.search(r"result_(.+)\.pt$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    files = collect(sys.argv[1:])
    if not files:
        print(f"No result files matched: {sys.argv[1:]}"); sys.exit(1)

    # de-duplicate by case id (a case should appear once across shards)
    by_id = {}
    for f in files:
        by_id[case_id(f)] = f

    rows = []
    for cid, f in by_id.items():
        d = torch.load(f, map_location="cpu")
        rel = relative_l2(d["recon"], d["target"])   # (num_samples,)
        rows.append((cid, float(rel.mean())))

    # sort by numeric case id when possible
    def keyfn(r):
        try:
            return (0, int(r[0]))
        except ValueError:
            return (1, r[0])
    rows.sort(key=keyfn)

    vals = np.array([v for _, v in rows])
    print(f"cases aggregated: {len(rows)}")
    for cid, v in rows:
        print(f"  case {cid:>4}: relative l2 = {v:.4f}")
    print(f"\nAGGREGATE relative l2:  mean = {vals.mean():.4f}   std = {vals.std():.4f}   "
          f"(n={len(vals)})")
    print("Baselines to beat (x2, sigma=0): EnKG ~ 0.12, DPG ~ 0.32")


if __name__ == "__main__":
    main()
