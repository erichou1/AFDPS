"""Track A: end-to-end AFDPS on a synthetic Navier-Stokes inverse problem with the
analytic Gaussian prior (correctly specified). No neural net, no Hydra.

omega0 ~ N(0, (-Delta+9I)^{-4});  y = P L(omega0) + noise.
Run AFDPS (prior='gaussian') and report relative-L2 vs the true omega0 and the
data misfit vs the noise floor. With the verified adjoint + correct prior, the
reconstruction should beat the trivial prior-mean baseline (rel-L2 -> 1.0) and
drive the misfit toward the noise floor.

Usage: python verification/run_track_a.py [--res 32] [--steps 40] [--particles 8]
"""
import os
import sys
import math
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_problems.navier_stokes_afdps import AFDPSNavierStokes2d
from inverse_problems import ns_adjoint as A
from algo.afdps import AFDPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', type=int, default=32)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--particles', type=int, default=8)
    ap.add_argument('--forward_time', type=float, default=0.2)
    ap.add_argument('--delta_t', type=float, default=5e-3)
    ap.add_argument('--downsample', type=int, default=2)
    ap.add_argument('--sigma_noise', type=float, default=0.0)
    ap.add_argument('--amp', type=float, default=300.0)          # GRF prior amplitude -> O(1) fields
    ap.add_argument('--hutch_M', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    op = AFDPSNavierStokes2d(
        resolution=args.res, forward_time=args.forward_time, Re=200.0,
        downsample_factor=args.downsample, delta_t=args.delta_t, adaptive=False,
        sigma_noise=args.sigma_noise, unnorm_scale=1.0,   # Track A: normalized == physical
        hutchinson_M=args.hutch_M, hutchinson_eps=1e-3, grad_chunk=args.particles,
        device='cpu', dtype=torch.float32,
    )
    op.force = None   # Track A: unforced dynamics -> y depends entirely on omega0 (clean IC recovery)

    # --- synthetic ground truth + observation ---
    w_true = A.grf_sample(op.solver, 1, amp=args.amp).to(torch.float32)   # (1,s1,s2) ~ N(0, amp^2 C)
    field_std = w_true.std().item()
    x_true = w_true.unsqueeze(1)                                  # (1,1,s1,s2)
    y = op({'target': x_true})                                    # forward + noise, physical
    noise_floor = args.sigma_noise * math.sqrt(y.numel())
    print(f"field std={field_std:.4f}  |y|={y.norm().item():.4f}  res={args.res}  "
          f"obs_res={args.res // args.downsample}  noise_floor={noise_floor:.4e}")

    # --- schedule spanning the signal scale ---
    sigma_max = 8 * field_std
    sigma_min = 0.02 * field_std

    algo = AFDPS(net=None, forward_op=op, num_particles=args.particles,
                 num_steps=args.steps, sigma_min=sigma_min, sigma_max=sigma_max,
                 prior='gaussian', prior_amp=args.amp, reduce='best')
    algo.eff_net.img_channels = 1

    recon = algo.inference(y, num_samples=1)                      # (1,1,s1,s2), normalized==physical

    rel_l2 = (recon - x_true).norm().item() / x_true.norm().item()
    misfit = (op.forward(recon, unnormalize=False) - y).norm().item()
    # baselines
    rel_prior_mean = 1.0                                          # ||0 - w|| / ||w||
    print(f"\nAFDPS  rel-L2(recon, true) = {rel_l2:.4f}")
    print(f"data misfit ||P L(recon) - y|| = {misfit:.4e}   (noise floor {noise_floor:.4e})")
    print(f"baseline rel-L2 (prior mean 0) = {rel_prior_mean:.4f}")
    ok = rel_l2 < 0.9
    print(f"\n[{'PASS' if ok else 'FAIL'}] AFDPS beats prior-mean baseline (rel-L2 < 0.9)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
