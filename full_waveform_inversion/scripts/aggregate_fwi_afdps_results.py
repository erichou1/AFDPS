#!/usr/bin/env python
"""Aggregate AFDPS x FWI per-case results into the InverseBench Table-7 row format.

Recomputes the three image metrics (Relative L2, PSNR, SSIM) directly from the saved
``result_*.pt`` files using the SAME definitions as ``eval.AcousticWave`` (so the numbers
match a single unsharded run), then averages over all cases as ``mean (std)`` -- exactly how
InverseBench Table 7 reports. The Devito-dependent "Data misfit" column is read from the run
logs (``--logs``) since recomputing it would require re-instantiating the wave solver.

Usage:
    python aggregate_fwi_afdps_results.py <result_glob...> [--logs <log_glob...>] [--label NAME]
Example:
    python aggregate_fwi_afdps_results.py results/fwi-afdps/AFDPS/final_shard*/result_*.pt \
        --logs final_shard*.log --label "AFDPS (final)"
"""
import argparse
import glob
import re
import sys

import numpy as np
import torch

# ---- metric definitions, identical to navier_stokes/eval.py:AcousticWave ----

def relative_l2(pred, target):
    diff = pred - target
    l2_norm = torch.linalg.norm(target.reshape(-1))
    return (torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1) / l2_norm)


def fwi_norm(x):
    return (x - 1.5) / 3.0


try:
    from piq import psnr as _piq_psnr, ssim as _piq_ssim

    def _psnr(p, t):
        return _piq_psnr(fwi_norm(p).clip(0, 1), fwi_norm(t).clip(0, 1), data_range=1.0, reduction='none')

    def _ssim(p, t):
        return _piq_ssim(fwi_norm(p).clip(0, 1), fwi_norm(t).clip(0, 1), data_range=1.0, reduction='none')
    _HAVE_PIQ = True
except Exception:  # piq not installed (e.g. aggregating on a laptop) -> PSNR by formula, SSIM N/A
    _HAVE_PIQ = False

    def _psnr(p, t):
        p, t = fwi_norm(p).clip(0, 1), fwi_norm(t).clip(0, 1)
        mse = (p - t).reshape(p.shape[0], -1).pow(2).mean(dim=1).clamp_min(1e-12)
        return 10.0 * torch.log10(1.0 / mse)

    def _ssim(p, t):
        return torch.full((p.shape[0],), float('nan'))


def _per_case(path):
    d = torch.load(path, map_location='cpu')
    recon = d['recon'].float()                      # (N, 1, H, W) physical km/s
    target = d['target'].float()                    # (1, 1, H, W)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    tt = target.expand_as(recon) if recon.shape[0] != target.shape[0] else target
    rl2 = relative_l2(recon, target).mean().item()
    ps = _psnr(recon, tt).mean().item()
    ss = _ssim(recon, tt).mean().item()
    return rl2, ps, ss


_MISFIT_RE = re.compile(r"'data misfit':\s*([0-9eE.+-]+)")


def _misfits_from_logs(log_globs):
    vals = []
    for g in log_globs:
        for lf in sorted(glob.glob(g)):
            try:
                with open(lf) as f:
                    for line in f:
                        if 'data misfit' in line:
                            m = _MISFIT_RE.search(line)
                            if m:
                                vals.append(float(m.group(1)))
            except OSError:
                pass
    return vals


def _fmt(vals):
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return "   N/A      "
    return f"{a.mean():.3f} ({a.std():.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='glob(s) for result_*.pt')
    ap.add_argument('--logs', nargs='*', default=[], help='glob(s) for run logs (for data misfit)')
    ap.add_argument('--label', default='AFDPS', help='row label')
    args = ap.parse_args()

    paths = []
    for g in args.results:
        paths.extend(sorted(glob.glob(g)))
    # de-duplicate while preserving order (case ids may repeat across overlapping globs)
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); uniq.append(p)
    paths = uniq
    if not paths:
        print("No result files matched.", file=sys.stderr)
        sys.exit(1)

    rl2s, psnrs, ssims = [], [], []
    for p in paths:
        try:
            rl2, ps, ss = _per_case(p)
            rl2s.append(rl2); psnrs.append(ps); ssims.append(ss)
        except Exception as e:
            print(f"  ! skipping {p}: {type(e).__name__}: {e}", file=sys.stderr)
    misfits = _misfits_from_logs(args.logs) if args.logs else []

    print(f"\nAggregated over {len(rl2s)} case(s)"
          + ("" if _HAVE_PIQ else "   [piq missing: PSNR by formula, SSIM=N/A]"))
    print("=" * 96)
    print(f"{'Method':<26}{'Relative L2 v':<18}{'PSNR ^':<18}{'SSIM ^':<18}{'Data misfit v':<18}")
    print("-" * 96)
    # InverseBench Table 7 baselines (noise-free, 10 cases) for side-by-side context.
    baselines = [
        ("Adam",        "0.333 (0.086)", "9.968 (2.083)",  "0.305 (0.120)", "115.14 (52.10)"),
        ("Adam\u2020",  "0.089 (0.021)", "21.273 (2.045)", "0.679 (0.073)", "15.89 (10.16)"),
        ("LBFGS\u2020", "0.070 (0.023)", "23.398 (2.749)", "0.704 (0.077)", "9.18 (6.47)"),
        ("DPS",         "0.250 (0.154)", "14.111 (6.820)", "0.491 (0.161)", "155.08 (92.17)"),
        ("LGD",         "0.244 (0.024)", "12.288 (0.889)", "0.341 (0.047)", "258.47 (26.40)"),
        ("DiffPIR",     "0.204 (0.129)", "16.113 (6.962)", "0.554 (0.191)", "88.53 (56.91)"),
        ("DAPS\u2020",  "0.201 (0.103)", "14.914 (4.184)", "0.321 (0.067)", "111.13 (71.33)"),
        ("PnP-DM",      "0.259 (0.075)", "11.983 (2.269)", "0.431 (0.073)", "308.84 (26.34)"),
        ("REDDiff",     "0.319 (0.102)", "10.372 (2.650)", "0.280 (0.108)", "94.67 (41.33)"),
    ]
    for name, a, b, c, d in baselines:
        print(f"{name:<26}{a:<18}{b:<18}{c:<18}{d:<18}")
    print("-" * 96)
    print(f"{args.label:<26}{_fmt(rl2s):<18}{_fmt(psnrs):<18}{_fmt(ssims):<18}{_fmt(misfits):<18}")
    print("=" * 96)
    print("(\u2020 = initialized from Gaussian-blurred ground truth; AFDPS uses the prior-only, "
          "un-daggered regime.)")


if __name__ == '__main__':
    main()
