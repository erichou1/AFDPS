"""Aggregate AFDPS inverse-scattering results into the InverseBench Table 3 format.

Reads the saved per-case result_*.pt files (across shard dirs), computes PSNR / SSIM
(matching eval.InverseScatter) and the relative measurement error (%), aggregates
mean (std) over cases, and prints a markdown + LaTeX table with the AFDPS row placed
next to the InverseBench Table 3 baselines for the chosen receiver count.

Usage (from inverse_scattering/):
  python scripts/aggregate_table3.py --numRec 360 \
      "exps/inference/inverse-scatter-afdps/AFDPS/*/result_*.pt"
  # add --numTrans 20 (default) and --no-measerr to skip rebuilding the operator.

The baseline numbers are transcribed VERBATIM from InverseBench (ICLR 2025), Table 3,
p.20 (noise level sigma_y = 1e-4). VERIFY against the PDF before publishing.
"""
import argparse
import os
import sys

import torch

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))            # scripts/
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # inverse_scattering/
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'navier_stokes'))

import metrics_lib as M


# InverseBench Table 3 (ICLR 2025, p.20), sigma_y = 1e-4. Keyed by receiver count.
# Each method -> (PSNR, PSNR_std, SSIM, SSIM_std, MeasErr%, MeasErr_std).
BASELINES = {
    360: {
        "FISTA-TV":  (32.126, 2.139, 0.979, 0.009, 1.23, 0.25),
        "DDRM":      (32.598, 1.825, 0.929, 0.012, 1.04, 0.26),
        "DDNM":      (36.381, 1.098, 0.935, 0.017, 0.78, 0.22),
        "PiGDM":     (27.925, 3.211, 0.889, 0.072, 2.74, 1.23),
        "DPS":       (32.061, 2.163, 0.846, 0.127, 4.35, 1.19),
        "LGD":       (27.901, 2.346, 0.812, 0.037, 1.17, 0.20),
        "DiffPIR":   (34.241, 2.310, 0.988, 0.006, 1.11, 0.24),
        "PnP-DM":    (33.914, 2.054, 0.988, 0.006, 1.21, 0.25),
        "DAPS":      (34.641, 1.693, 0.957, 0.006, 1.03, 0.25),
        "RED-diff":  (36.556, 2.292, 0.981, 0.005, 0.89, 0.23),
        "FPS":       (33.242, 1.602, 0.870, 0.026, 0.70, 0.01),
        "MCG-diff":  (30.937, 1.964, 0.751, 0.029, 0.70, 0.01),
    },
    180: {
        "FISTA-TV":  (26.523, 2.678, 0.914, 0.040, 2.65, 0.30),
        "DDRM":      (28.080, 1.516, 0.890, 0.019, 1.57, 0.39),
        "DDNM":      (35.024, 0.993, 0.895, 0.027, 0.58, 0.16),
        "PiGDM":     (26.412, 3.430, 0.816, 0.114, 3.66, 1.79),
        "DPS":       (31.798, 2.163, 0.862, 0.123, 4.28, 1.20),
        "LGD":       (27.837, 2.337, 0.803, 0.034, 1.06, 0.16),
        "DiffPIR":   (34.010, 2.269, 0.987, 0.006, 1.04, 0.23),
        "PnP-DM":    (31.817, 2.073, 0.981, 0.008, 1.42, 0.26),
        "DAPS":      (33.160, 1.704, 0.944, 0.009, 1.11, 0.25),
        "RED-diff":  (35.411, 2.166, 0.984, 0.004, 0.87, 0.21),
        "FPS":       (29.624, 1.651, 0.710, 0.040, 0.37, 0.01),
        "MCG-diff":  (28.057, 1.672, 0.631, 0.042, 0.38, 0.01),
    },
    60: {
        "FISTA-TV":  (20.938, 2.513, 0.709, 0.103, 6.05, 0.65),
        "DDRM":      (20.436, 1.210, 0.545, 0.037, 3.04, 0.92),
        "DDNM":      (29.235, 3.376, 0.917, 0.022, 0.28, 0.07),
        "PiGDM":     (20.074, 2.608, 0.540, 0.198, 6.90, 3.38),
        "DPS":       (27.372, 3.415, 0.813, 0.133, 4.53, 1.31),
        "LGD":       (20.491, 3.031, 0.552, 0.077, 1.45, 0.68),
        "DiffPIR":   (26.321, 3.272, 0.918, 0.028, 1.27, 0.23),
        "PnP-DM":    (24.715, 2.874, 0.909, 0.046, 2.20, 0.34),
        "DAPS":      (25.875, 3.110, 0.885, 0.030, 1.51, 0.25),
        "RED-diff":  (27.072, 3.330, 0.935, 0.037, 1.18, 0.23),
        "FPS":       (21.323, 1.445, 0.460, 0.030, 0.15, 0.02),
        "MCG-diff":  (21.004, 1.571, 0.445, 0.028, 0.21, 0.06),
    },
}


def evaluate(by_id, numRec, numTrans, want_measerr):
    forward_op = None
    if want_measerr:
        from inverse_problems.inverse_scatter import InverseScatter
        forward_op = InverseScatter(Nx=128, Ny=128, numRec=numRec, numTrans=numTrans,
                                    sigma_noise=1e-4, unnorm_shift=1.0, unnorm_scale=0.5,
                                    device='cpu', svd=False)
    psnrs, ssims, errs = [], [], []
    for cid in M.sort_case_ids(by_id):
        d = torch.load(by_id[cid], map_location='cpu')
        p, s = M.compute_psnr_ssim(d['recon'], d['target'])
        psnrs.append(p); ssims.append(s)
        if forward_op is not None:
            pred_meas = M.apply_forward(forward_op, d['recon'].to(torch.float64))
            errs.append(M.relative_meas_err_pct(pred_meas, d['observation']))
    return psnrs, ssims, errs


def fmt(mean, std):
    return f"{mean:.3f} ({std:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="glob(s) for result_*.pt")
    ap.add_argument("--numRec", type=int, required=True, choices=[360, 180, 60])
    ap.add_argument("--numTrans", type=int, default=20)
    ap.add_argument("--no-measerr", action="store_true", help="skip rebuilding the operator")
    ap.add_argument("--label", default="AFDPS (ours)")
    args = ap.parse_args()

    by_id = M.collect(args.patterns)
    if not by_id:
        print(f"No result files matched: {args.patterns}"); sys.exit(1)
    psnrs, ssims, errs = evaluate(by_id, args.numRec, args.numTrans, not args.no_measerr)
    n = len(psnrs)
    p_m, p_s = M.mean_std(psnrs)
    s_m, s_s = M.mean_std(ssims)
    e_m, e_s = M.mean_std(errs) if errs else (float('nan'), float('nan'))

    base = BASELINES[args.numRec]
    print(f"\n## Linear inverse scattering -- {args.numRec} receivers (sigma_y=1e-4), n={n} cases\n")
    print(f"| Method | PSNR | SSIM | Meas err (%) |")
    print(f"|---|---|---|---|")
    for name, (pp, pps, ss, sss, ee, ees) in base.items():
        print(f"| {name} | {pp:.3f} ({pps:.3f}) | {ss:.3f} ({sss:.3f}) | {ee:.2f} ({ees:.2f}) |")
    err_cell = "-" if not errs else fmt(e_m, e_s)
    print(f"| **{args.label}** | **{fmt(p_m, p_s)}** | **{fmt(s_m, s_s)}** | **{err_cell}** |")

    # LaTeX one-liner for the paper table
    print("\n% LaTeX row:")
    print(f"{args.label} & {p_m:.3f} ({p_s:.3f}) & {s_m:.3f} ({s_s:.3f}) & "
          f"{'--' if not errs else f'{e_m:.2f} ({e_s:.2f})'} \\\\")

    best_psnr = max(v[0] for v in base.values())
    print(f"\nBest baseline PSNR @ {args.numRec} recv: {best_psnr:.3f}  |  AFDPS: {p_m:.3f}  "
          f"({'AHEAD' if p_m >= best_psnr else 'behind'} by {abs(p_m - best_psnr):.3f} dB)")


if __name__ == "__main__":
    main()
