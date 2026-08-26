#!/usr/bin/env python3
"""
Weight-degeneracy diagnostic for the AFDPS scattering sampler.

WHY: a paired control at R=360 found reduce='best' bit-identical to
reduce='mean' on all 100 test cases. Two mechanisms explain that:

  (A) particle collapse   - the J particles are (nearly) the same image;
  (B) weight degeneracy   - the particles differ, but softmax(log_weight)
                            is one-hot, so sum_j w_j x_j == argmax particle.

They demand opposite fixes, so measure before prescribing. This script runs
the sampler directly (no reduction) and reports, per case:

  ESS            = 1 / sum_j w_j^2          (1.0 => a single particle carries all weight)
  max_weight     = max_j w_j
  logw_gap       = log_weight[top1] - log_weight[top2]
  particle_rmse  = RMSE between the top-2 particles (0 => genuinely identical)
  mean_vs_best   = max |weighted_mean - argmax particle|, before the fp32 cast

Reading the output:
  ESS ~ 1 and particle_rmse > 0   -> (B) weight degeneracy. Enabling ESS
                                     resampling makes the reported mean a real
                                     ensemble estimate and may change Table 3.
  ESS ~ 1 and particle_rmse ~ 0   -> (A) genuine collapse. Resampling changes
                                     nothing; the reduction question is moot.
  ESS >> 1                        -> neither; re-examine the control run.

USAGE (from inverse_scattering/, GPU box):

  python scripts/diag_ess.py --numRec 360 --cases 0-9
  python scripts/diag_ess.py --numRec 360 --cases 0-9 --resample     # with ESS resampling
  python scripts/diag_ess.py --numRec 15  --cases 0-9                # sparse control

Writes one JSON row per case to --out (default: exps/diag/ess_R<numRec>.jsonl)
and prints a summary table. Read-only w.r.t. the manuscript and existing runs.
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NS = os.path.join(os.path.dirname(_HERE), 'navier_stokes')
sys.path.insert(1, _NS)
sys.path.insert(1, _HERE)

import pickle
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from utils.helper import open_url


def parse_cases(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--numRec', type=int, default=360)
    ap.add_argument('--cases', type=str, default='0-9')
    ap.add_argument('--num-particles', type=int, default=512)
    ap.add_argument('--num-steps', type=int, default=200)
    ap.add_argument('--sigma-noise', type=float, default=1e-4)
    ap.add_argument('--resample', action='store_true',
                    help='enable AFDPS Algorithm-1 ESS resampling')
    ap.add_argument('--resample-threshold', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    os.chdir(_HERE)
    cases = parse_cases(args.cases)
    out_path = args.out or os.path.join(
        'exps', 'diag',
        f"ess_R{args.numRec}{'_resample' if args.resample else ''}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with initialize_config_dir(version_base='1.3',
                               config_dir=os.path.join(_HERE, 'configs')):
        cfg = compose(config_name='config', overrides=[
            'problem=inv-scatter-afdps',
            'algorithm=afdps',
            'pretrain=inv-scatter',
            f'problem.model.numRec={args.numRec}',
            f'problem.model.sigma_noise={args.sigma_noise}',
            f'problem.data.id_list={min(cases)}-{max(cases)}',
            f'algorithm.method.num_particles={args.num_particles}',
            f'algorithm.method.num_steps={args.num_steps}',
            f'algorithm.method.sampler_kwargs.resample={str(args.resample).lower()}',
            f'algorithm.method.sampler_kwargs.resample_threshold={args.resample_threshold}',
            'algorithm.method.sampler_kwargs.progress=false',
            'wandb=false',
        ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward_op = instantiate(cfg.problem.model, device=device)
    testset = instantiate(cfg.problem.data)

    ckpt_path = cfg.problem.prior
    try:
        with open_url(ckpt_path, 'rb') as f:
            net = pickle.load(f)['ema'].to(device)
    except Exception:
        net = instantiate(cfg.pretrain.model)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt['ema'] if 'ema' in ckpt else ckpt['net'])
        net = net.to(device)
    net.eval()

    algo = instantiate(cfg.algorithm.method, forward_op=forward_op, net=net)
    print(f"config: mode={cfg.algorithm.method.mode} "
          f"guidance_step={cfg.algorithm.method.sampler_kwargs.guidance_step} "
          f"guidance_mode={cfg.algorithm.method.sampler_kwargs.guidance_mode} "
          f"resample={cfg.algorithm.method.sampler_kwargs.resample}")

    rows = []
    with open(out_path, 'w') as fh, torch.no_grad():
        for i in range(len(testset)):
            data = testset[i]
            data_id = testset.id_list[i]
            if int(data_id) not in cases:
                continue
            if isinstance(data, dict):
                data = {k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                        for k, v in data.items()}
                target = data['target']
            else:
                target = data.to(device).unsqueeze(0)
                data = {'target': target}

            torch.manual_seed(args.seed + int(data_id))
            observation = forward_op(data)

            op = algo.forward_op
            op.set_observation(observation)
            res = net.img_resolution
            gt_dummy = torch.zeros(1, net.img_channels, res, res, device=device)

            class _N:
                def __init__(self, s): self.sigma = s
            out = algo.sampler(gt_dummy, observation, algo.num_particles, op,
                               _N(op.sigma_noise_eff))

            ens = out['ensemble'].to(torch.float64)      # (J, C, H, W)
            lw = out['log_weights'].to(torch.float64)    # (J,)
            w = torch.softmax(lw, dim=0)

            ess = float(1.0 / (w ** 2).sum())
            top = torch.argsort(lw, descending=True)
            best = ens[top[0]]
            second = ens[top[1]]
            wmean = (w.view(-1, 1, 1, 1) * ens).sum(dim=0)

            row = {
                'case': int(data_id),
                'numRec': args.numRec,
                'resample': bool(args.resample),
                'J': int(ens.shape[0]),
                'ESS': ess,
                'ESS_frac': ess / float(ens.shape[0]),
                'max_weight': float(w.max()),
                'logw_gap_top1_top2': float(lw[top[0]] - lw[top[1]]),
                'logw_spread': float(lw.max() - lw.min()),
                'n_finite_weights': int(torch.isfinite(lw).sum()),
                'particle_rmse_top1_top2': float(((best - second) ** 2).mean().sqrt()),
                'particle_rmse_top1_meanens': float(((best - ens.mean(0)) ** 2).mean().sqrt()),
                'mean_vs_best_maxabs_fp64': float((wmean - best).abs().max()),
                'mean_vs_best_identical_fp32': bool(torch.equal(
                    wmean.to(torch.float32), best.to(torch.float32))),
            }
            rows.append(row)
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(f"case {row['case']:>3}  ESS={row['ESS']:.3f}  "
                  f"maxw={row['max_weight']:.6f}  gap={row['logw_gap_top1_top2']:.2f}  "
                  f"p_rmse={row['particle_rmse_top1_top2']:.3e}  "
                  f"mean==best(fp32)={row['mean_vs_best_identical_fp32']}")

    if rows:
        n = len(rows)
        avg = lambda k: sum(r[k] for r in rows) / n
        print("\n=== summary ===")
        print(f"cases                 : {n}")
        print(f"mean ESS              : {avg('ESS'):.4f}  (of J={rows[0]['J']})")
        print(f"mean max_weight       : {avg('max_weight'):.8f}")
        print(f"mean top1-top2 gap    : {avg('logw_gap_top1_top2'):.3f} nats")
        print(f"mean particle RMSE    : {avg('particle_rmse_top1_top2'):.6e}")
        print(f"mean==best in fp32    : {sum(r['mean_vs_best_identical_fp32'] for r in rows)}/{n}")
        collapsed = avg('particle_rmse_top1_top2') < 1e-6
        degenerate = avg('ESS') < 1.05
        if degenerate and not collapsed:
            print("\nVERDICT: WEIGHT_DEGENERACY -- particles differ, weights are one-hot.")
            print("         Re-run with --resample to test whether an effective ensemble helps.")
        elif degenerate and collapsed:
            print("\nVERDICT: PARTICLE_COLLAPSE -- particles are genuinely identical.")
            print("         Reduction choice is moot; resampling will not change Table 3.")
        else:
            print("\nVERDICT: WEIGHTS_NON_DEGENERATE -- re-examine the paired control run.")
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
