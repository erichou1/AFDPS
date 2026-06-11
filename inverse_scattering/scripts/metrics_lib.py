"""Shared metrics helpers for the AFDPS inverse-scattering results.

Pure functions (dedup, relative measurement error, table formatting) import no heavy
deps so they are unit-testable on CPU. PSNR/SSIM lazily import piq so they match the
benchmark's `eval.InverseScatter` exactly (clip [0,1], data_range=1) only when needed.
"""
import os
import re
import glob

import torch


# --------------------------------------------------------------------------- #
# Result file collection                                                       #
# --------------------------------------------------------------------------- #
def case_id(path):
    m = re.search(r"result_(.+)\.pt$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def collect(patterns):
    """Expand glob patterns and de-duplicate by case id (a case may appear once per
    shard directory; the last one wins). Returns {case_id: filepath}."""
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    by_id = {}
    for f in sorted(files):
        by_id[case_id(f)] = f
    return by_id


def sort_case_ids(ids):
    def keyfn(c):
        try:
            return (0, int(c))
        except (ValueError, TypeError):
            return (1, str(c))
    return sorted(ids, key=keyfn)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def relative_meas_err_pct(pred_meas, observation):
    """100 * ||pred_meas - observation||_2 / ||observation||_2  (complex-aware).
    `pred_meas` = A applied to the reconstruction; `observation` = recorded data.
    This is the 'Meas err (%)' column convention used in the comparison table.

    NOTE: confirm against the InverseBench appendix definition before publishing the
    comparison; the in-repo evaluator emits sqrt(data_misfit), not this normalized %."""
    diff = (pred_meas - observation).reshape(-1)
    num = torch.linalg.norm(torch.view_as_real(diff) if torch.is_complex(diff) else diff)
    o = observation.reshape(-1)
    den = torch.linalg.norm(torch.view_as_real(o) if torch.is_complex(o) else o)
    return float(100.0 * num / den.clamp_min(1e-30))


def compute_psnr_ssim(recon, target):
    """PSNR/SSIM matching eval.InverseScatter: piq on clip[0,1], data_range=1.
    recon (N,1,H,W), target (1,1,H,W) or (N,1,H,W). Returns (psnr, ssim) floats
    (mean over samples). Lazily imports piq."""
    from piq import psnr, ssim
    if target.shape != recon.shape:
        target = target.repeat(recon.shape[0], 1, 1, 1)
    r = recon.clip(0, 1).to(torch.float32)
    t = target.clip(0, 1).to(torch.float32)
    return (float(psnr(r, t, data_range=1.0, reduction='mean')),
            float(ssim(r, t, data_range=1.0, reduction='mean')))


def apply_forward(forward_op, recon_unnorm):
    """A @ f for an already-UNNORMALIZED reconstruction (physical permittivity).
    Loops samples (the benchmark forward is batch-1). Returns stacked measurements."""
    outs = []
    for j in range(recon_unnorm.shape[0]):
        f = recon_unnorm[j:j + 1]
        outs.append(forward_op.forward(f, unnormalize=False))
    return torch.cat(outs, dim=0)


def mean_std(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean()), float(t.std(unbiased=False))
