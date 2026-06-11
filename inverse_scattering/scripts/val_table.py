"""Summarize a validation sweep: one row per run directory (config), mean PSNR/SSIM
over the validation cases, sorted by PSNR. Reads each run's saved config.yaml to label
the key AFDPS knobs. Used to pick the winning config per receiver count.

Usage (from inverse_scattering/):
  python scripts/val_table.py "exps/inference/inverse-scatter-afdps/AFDPS/val_*"
"""
import argparse
import glob
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'navier_stokes'))
import metrics_lib as M


def knobs(cfg_path):
    if not os.path.exists(cfg_path):
        return {}
    c = OmegaConf.load(cfg_path)
    m = c.algorithm.method
    sk = m.get('sampler_kwargs', {}) or {}
    return dict(R=c.problem.model.numRec, J=m.num_particles, steps=m.num_steps,
                gmode=sk.get('guidance_mode'), gstep=sk.get('guidance_step'),
                gamma=m.guidance_gamma, reduce=m.reduce, val=sk.get('value_coef'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_globs", nargs="+", help="glob(s) for run directories")
    args = ap.parse_args()

    rows = []
    dirs = []
    for g in args.run_globs:
        dirs.extend([d for d in glob.glob(g) if os.path.isdir(d)])
    for d in sorted(set(dirs)):
        by_id = M.collect([os.path.join(d, "result_*.pt")])
        if not by_id:
            continue
        ps, ss = [], []
        for cid in M.sort_case_ids(by_id):
            data = torch.load(by_id[cid], map_location='cpu')
            p, s = M.compute_psnr_ssim(data['recon'], data['target'])
            ps.append(p); ss.append(s)
        pm, _ = M.mean_std(ps); sm, _ = M.mean_std(ss)
        rows.append((pm, sm, len(ps), os.path.basename(d), knobs(os.path.join(d, "config.yaml"))))

    rows.sort(key=lambda r: -r[0])
    print(f"{'PSNR':>7} {'SSIM':>6} {'n':>3}  config")
    for pm, sm, n, name, k in rows:
        kk = f"R={k.get('R')} J={k.get('J')} steps={k.get('steps')} {k.get('gmode')}/{k.get('gstep')} " \
             f"gamma={k.get('gamma')} reduce={k.get('reduce')} val={k.get('val')}"
        print(f"{pm:7.3f} {sm:6.3f} {n:3d}  {name:28s} {kk}")
    if rows:
        print(f"\nWinner: {rows[0][3]}  (PSNR {rows[0][0]:.3f})")


if __name__ == "__main__":
    main()
