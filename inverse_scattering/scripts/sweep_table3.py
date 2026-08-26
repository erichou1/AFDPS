#!/usr/bin/env python3
"""
Table 3 improvement sweep for AFDPS linear inverse scattering.

CONTEXT. The ESS diagnostic established that the reported "weighted mean" is
numerically a single-particle estimate: baseline ESS = 1.000 on 10/10 cases with
top1-top2 log-weight gaps of 1.5e3 to 6.8e5 nats, while the particles themselves
stay distinct (pairwise RMSE 0.036). ESS resampling raises ESS to 11.9 but fixes
the symptom by DUPLICATING particles (RMSE collapses 7.7x to 0.0047), and gains
SSIM (+0.0016, paired t=2.86) with no PSNR change (-0.12 dB, t=-1.0).

A second fact from that run matters: once weights are non-degenerate, the MEAN
reduction beats the best particle on 10/10 cases (PSNR +0.0135, t=4.65; SSIM
+0.0027, t=8.64). So the mean is the right estimator and the real problem is
that it was never actually averaging.

This sweep tests configurations that could make it a genuine average, plus two
guidance variants that have never been tuned for scattering.

ARMS (each is one 10-case run at R=360):
  baseline          reference, reproduces the diagnostic baseline
  temper_1e-2       weight_temper=1e-2   flatten FK weights, no duplication
  temper_1e-3       weight_temper=1e-3
  temper_1e-4       weight_temper=1e-4
  resample_0.9      resample_threshold=0.9 (fires earlier than 0.5)
  guidance_auto     guidance_mode=auto (isotropic PiGDM, gamma_e^2=lambda_bar)
  likelihood_denoised  likelihood_at=denoised (FWI/NS use this; scattering uses noisy)

CAVEAT ON TEMPERING. The measured gaps span 466x across cases, so a single fixed
alpha cannot hold ESS constant on every case. The three alphas bracket the range:
offline simulation on the measured gaps gives median ESS 1.8 / 12.3 / 108 for
1e-2 / 1e-3 / 1e-4. Treat this as a screen, not a tuned setting. If one alpha
wins, the follow-up is adaptive per-step tempering targeting a fixed ESS.

WHAT COUNTS AS A WIN. An arm must beat baseline on BOTH mean PSNR and mean SSIM
with a paired t-statistic above 2 on at least one, and neither metric degraded.
The gap to close at R=360 is -2.41 dB PSNR vs DDNM; AFDPS already leads SSIM.

USAGE (from inverse_scattering/, GPU box):

    python scripts/sweep_table3.py --cases 0-9                 # all arms
    python scripts/sweep_table3.py --cases 0-9 --arms baseline,temper_1e-3
    python scripts/sweep_table3.py --cases 0-9 --numRec 1440

Writes one JSON per arm plus a comparison table to --out (default exps/sweep/).
Read-only with respect to the manuscript and all existing run directories.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NS = os.path.join(os.path.dirname(_HERE), 'navier_stokes')
sys.path.insert(1, _NS)
sys.path.insert(1, _HERE)

import pickle
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from utils.helper import open_url


ARMS = {
    # --- reference ---
    'baseline':            {},
    # --- VARIANCE levers (sigma_p^2 -> sigma_p^2/J). Ceiling ~+1.95 dB. ---
    'uniform_mean':        {'method.reduce': 'uniform_mean'},
    'temper_1e-2':         {'sampler.weight_temper': 1e-2},
    'temper_1e-3':         {'sampler.weight_temper': 1e-3},
    'temper_1e-4':         {'sampler.weight_temper': 1e-4},
    'temper_1e-4_umean':   {'sampler.weight_temper': 1e-4, 'method.reduce': 'uniform_mean'},
    'resample_0.9':        {'sampler.resample': True, 'sampler.resample_threshold': 0.9},
    # --- BIAS levers (63.7% of the MSE; required to pass DDNM). ---
    'likelihood_denoised': {'method.likelihood_at': 'denoised'},
    'guidance_auto':       {'sampler.guidance_mode': 'auto'},
    'init_pinv':           {'problem.init_mode': 'pinv'},
    'steps_400':           {'method.num_steps': 400},
    # --- COMBINED: the only arms the error budget says can beat DDNM. ---
    'umean_denoised':      {'method.reduce': 'uniform_mean',
                            'method.likelihood_at': 'denoised'},
    'umean_pinv':          {'method.reduce': 'uniform_mean',
                            'problem.init_mode': 'pinv'},
    'umean_denoised_400':  {'method.reduce': 'uniform_mean',
                            'method.likelihood_at': 'denoised',
                            'method.num_steps': 400},
    # --- PROJECTION: exact range-space data consistency (zero extra cost). ---
    # Removes guidance bias on measured V-modes exactly; the ensemble supplies
    # the null-space. Directly targets the dominant bias term (63.7% of MSE).
    'projection':          {'method.final_projection': True},
    'umean_projection':    {'method.reduce': 'uniform_mean',
                            'method.final_projection': True},
    'umean_proj_tau1e-2':  {'method.reduce': 'uniform_mean',
                            'method.final_projection': True,
                            'method.projection_tau': 1e-2},
}


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


def paired_stats(deltas):
    n = len(deltas)
    if n < 2:
        return {'n': n, 'mean': deltas[0] if deltas else 0.0, 't': float('nan'), 'wins': 0}
    m = st.mean(deltas)
    sd = st.stdev(deltas)
    t = m / (sd / math.sqrt(n)) if sd > 0 else float('inf')
    return {'n': n, 'mean': m, 'sd': sd, 't': t, 'wins': sum(1 for d in deltas if d > 0)}


def build_overrides(arm, args):
    spec = ARMS[arm]
    # arm-level num_steps wins over the CLI default (avoids a duplicate override)
    num_steps = spec.get('method.num_steps', args.num_steps)
    ov = [
        'problem=inv-scatter-afdps', 'algorithm=afdps', 'pretrain=inv-scatter',
        f'problem.model.numRec={args.numRec}',
        f'problem.model.sigma_noise={args.sigma_noise}',
        f'problem.data.id_list={min(args.case_list)}-{max(args.case_list)}',
        f'algorithm.method.num_particles={args.num_particles}',
        f'algorithm.method.num_steps={num_steps}',
        'algorithm.method.sampler_kwargs.progress=false',
        'wandb=false',
    ]
    for key, val in spec.items():
        if key == 'method.num_steps':
            continue                       # already folded in above
        prefix, name = key.split('.', 1)
        base = {'sampler': 'algorithm.method.sampler_kwargs',
                'method': 'algorithm.method',
                'problem': 'problem.model'}[prefix]
        ov.append(f'{base}.{name}={str(val).lower() if isinstance(val, bool) else val}')
    return ov


def run_arm(arm, args, net_cache):
    with initialize_config_dir(version_base='1.3',
                               config_dir=os.path.join(_HERE, 'configs')):
        cfg = compose(config_name='config', overrides=build_overrides(arm, args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forward_op = instantiate(cfg.problem.model, device=device)
    testset = instantiate(cfg.problem.data)

    if net_cache.get('net') is None:
        try:
            with open_url(cfg.problem.prior, 'rb') as f:
                net = pickle.load(f)['ema'].to(device)
        except Exception:
            net = instantiate(cfg.pretrain.model)
            ck = torch.load(cfg.problem.prior, map_location=device, weights_only=False)
            net.load_state_dict(ck['ema'] if 'ema' in ck else ck['net'])
            net = net.to(device)
        net.eval()
        net_cache['net'] = net
    net = net_cache['net']

    algo = instantiate(cfg.algorithm.method, forward_op=forward_op, net=net)
    evaluator = instantiate(cfg.problem.evaluator, forward_op=forward_op)

    rows = []
    with torch.no_grad():
        for i in range(len(testset)):
            data = testset[i]
            data_id = int(testset.id_list[i])
            if data_id not in args.case_list:
                continue
            if isinstance(data, dict):
                data = {k: (v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                        for k, v in data.items()}
            else:
                data = {'target': data.to(device).unsqueeze(0)}
            target = data['target']

            torch.manual_seed(args.seed + data_id)
            observation = forward_op(data)
            recon = algo.inference(observation, num_samples=1)

            m = evaluator(pred=forward_op.unnormalize(recon).cpu(),
                          target=forward_op.unnormalize(target).cpu(),
                          observation=observation.cpu())
            psnr = float(m['psnr'] if not hasattr(m['psnr'], '__len__') else m['psnr'][0])
            ssim = float(m['ssim'] if not hasattr(m['ssim'], '__len__') else m['ssim'][0])
            rows.append({'case_id': data_id, 'psnr': psnr, 'ssim': ssim})
            print(f"  [{arm}] case {data_id:>3}  PSNR {psnr:7.4f}  SSIM {ssim:.6f}", flush=True)

    return {
        'arm': arm,
        'overrides': {k: v for k, v in ARMS[arm].items()},
        'config': {
            'numRec': args.numRec, 'sigma_y': args.sigma_noise,
            'num_particles': args.num_particles, 'num_steps': args.num_steps,
            'mode': str(cfg.algorithm.method.mode),
            'guidance_mode': str(cfg.algorithm.method.sampler_kwargs.guidance_mode),
            'guidance_step': str(cfg.algorithm.method.sampler_kwargs.guidance_step),
            'likelihood_at': str(cfg.algorithm.method.likelihood_at),
            'resample': bool(cfg.algorithm.method.sampler_kwargs.resample),
            'resample_threshold': float(cfg.algorithm.method.sampler_kwargs.resample_threshold),
            'weight_temper': float(cfg.algorithm.method.sampler_kwargs.get('weight_temper', 1.0)),
            'reduce': str(cfg.algorithm.method.reduce),
            'final_projection': bool(cfg.algorithm.method.get('final_projection', False)),
            'projection_tau': float(cfg.algorithm.method.get('projection_tau', 1e-3)),
            'init_mode': str(cfg.problem.model.get('init_mode', 'noise')),
            'seed': args.seed,
        },
        'cases': rows,
        'summary': {
            'psnr_mean': st.mean([r['psnr'] for r in rows]),
            'ssim_mean': st.mean([r['ssim'] for r in rows]),
            'psnr_std': st.stdev([r['psnr'] for r in rows]) if len(rows) > 1 else 0.0,
            'ssim_std': st.stdev([r['ssim'] for r in rows]) if len(rows) > 1 else 0.0,
            'n_cases': len(rows),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='0-9')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--numRec', type=int, default=360)
    ap.add_argument('--sigma-noise', dest='sigma_noise', type=float, default=1e-4)
    ap.add_argument('--num-particles', type=int, default=512)
    ap.add_argument('--num-steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    os.chdir(_HERE)
    args.case_list = parse_cases(args.cases)
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}. available: {list(ARMS)}")
    if 'baseline' not in arms:
        arms.insert(0, 'baseline')

    out_dir = args.out or os.path.join('exps', 'sweep', f'R{args.numRec}')
    os.makedirs(out_dir, exist_ok=True)

    net_cache = {}
    results = {}
    for arm in arms:
        print(f"\n=== arm: {arm}  overrides={ARMS[arm] or 'none'} ===", flush=True)
        res = run_arm(arm, args, net_cache)
        results[arm] = res
        with open(os.path.join(out_dir, f'{arm}.json'), 'w') as fh:
            json.dump(res, fh, indent=2, sort_keys=True)

    base = {r['case_id']: r for r in results['baseline']['cases']}
    table = []
    for arm in arms:
        cur = {r['case_id']: r for r in results[arm]['cases']}
        ids = sorted(set(base) & set(cur))
        dp = paired_stats([cur[i]['psnr'] - base[i]['psnr'] for i in ids])
        ds = paired_stats([cur[i]['ssim'] - base[i]['ssim'] for i in ids])
        win = (arm != 'baseline' and dp['mean'] > 0 and ds['mean'] > 0
               and max(abs(dp['t']), abs(ds['t'])) > 2.0)
        table.append({
            'arm': arm,
            'psnr_mean': results[arm]['summary']['psnr_mean'],
            'ssim_mean': results[arm]['summary']['ssim_mean'],
            'd_psnr': dp['mean'], 't_psnr': dp['t'], 'psnr_wins': dp['wins'],
            'd_ssim': ds['mean'], 't_ssim': ds['t'], 'ssim_wins': ds['wins'],
            'improves_both': bool(win),
        })

    print("\n" + "=" * 100)
    print(f"{'arm':<22}{'PSNR':>9}{'dPSNR':>9}{'t':>7}{'w':>4}"
          f"{'SSIM':>11}{'dSSIM':>10}{'t':>7}{'w':>4}   verdict")
    print("-" * 100)
    for r in table:
        verdict = 'reference' if r['arm'] == 'baseline' else (
            'IMPROVES BOTH' if r['improves_both'] else 'no')
        print(f"{r['arm']:<22}{r['psnr_mean']:>9.4f}{r['d_psnr']:>+9.4f}{r['t_psnr']:>7.2f}"
              f"{r['psnr_wins']:>4}{r['ssim_mean']:>11.6f}{r['d_ssim']:>+10.6f}"
              f"{r['t_ssim']:>7.2f}{r['ssim_wins']:>4}   {verdict}")
    print("=" * 100)

    winners = [r['arm'] for r in table if r['improves_both']]
    print(f"\nARMS IMPROVING BOTH METRICS: {winners or 'NONE'}")
    if winners:
        print("Next: rerun the winning arm on all 100 cases before any manuscript change.")
    else:
        print("No configuration improves both metrics. Table 3 stands; report this as a")
        print("negative result and keep the published settings.")

    with open(os.path.join(out_dir, 'comparison.json'), 'w') as fh:
        json.dump({'table': table, 'winners': winners,
                   'cases': args.case_list, 'numRec': args.numRec}, fh, indent=2)
    print(f"\nwrote {out_dir}/")


if __name__ == '__main__':
    main()
