"""Build the Navier-Stokes qualitative comparison figure.

Reads the `result_<case>.pt` files that `main.py` writes and renders an
InverseBench-Figure-2-style grid: one row per acquisition regime, columns for the
ground truth, the measurement, each reconstruction, and the AFDPS posterior
spread. All panels in a row share one symmetric colour scale taken from the
ground truth, so panels are directly comparable.

Expected layout (see figures/README.md for the run commands):

    <root>/AFDPS/<tag>/result_<case>.pt
    <root>/EnKG/<tag>/result_<case>.pt
    <root>/EKI/<tag>/result_<case>.pt

Usage:
    python figures/make_ns_figure.py --root exps/figures --case 3
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Rows of the figure: the acquisition regimes, easiest first. `tag` must match the
# `exp_name` used for the runs.
REGIMES = [
    dict(tag='fig_ds2_s0', label='ds$=2$\n$\\sigma=0$'),
    dict(tag='fig_ds8_s0', label='ds$=8$\n$\\sigma=0$'),
    dict(tag='fig_ds8_s3', label='ds$=8$\n$\\sigma=3$'),
]

# Reconstruction columns, in display order. The directory name is the `name` field
# of the algorithm config (configs/algorithm/*.yaml). `ours` draws a highlight frame.
METHODS = [
    dict(dir='AFDPS', label='AFDPS (ours)', ours=True),
    dict(dir='EnKG', label='EnKG'),
    dict(dir='EKI', label='EKI'),
]

FIELD_CMAP = 'RdBu_r'   # signed vorticity -> diverging. 'turbo' matches InverseBench.
STD_CMAP = 'magma'
ERR_CMAP = 'inferno'
OURS_COLOR = '#1f4e79'


def relative_l2(pred, target):
    """Same definition as eval.relative_l2, so figure and tables agree."""
    diff = (pred - target).reshape(-1)
    return float(torch.linalg.norm(diff) / torch.linalg.norm(target.reshape(-1)))


def as_2d(x):
    """Squeeze a saved tensor down to (H, W)."""
    x = torch.as_tensor(x).float().squeeze()
    if x.ndim == 3:          # (N, H, W) -> first sample
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f'expected a 2-D field, got shape {tuple(x.shape)}')
    return x


def load_case(root, method_dir, tag, case):
    path = os.path.join(root, method_dir, tag, f'result_{case}.pt')
    if not os.path.exists(path):
        return None, path
    return torch.load(path, map_location='cpu'), path


def posterior_std(result):
    """Weighted per-pixel posterior std from the saved ensemble, or None."""
    ens = result.get('ensemble')
    if ens is None:
        return None
    ens = torch.as_tensor(ens).float()
    ens = ens.reshape(ens.shape[0], -1)
    lw = result.get('log_weights')
    if lw is None:
        var = ens.var(dim=0, unbiased=False)
    else:
        w = torch.softmax(torch.as_tensor(lw).float().reshape(-1), dim=0).unsqueeze(1)
        mean = (w * ens).sum(dim=0, keepdim=True)
        var = (w * (ens - mean) ** 2).sum(dim=0)
    side = int(round(var.numel() ** 0.5))
    return var.sqrt().reshape(side, side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='exps/figures',
                    help='directory holding <ALGO>/<tag>/result_<case>.pt')
    ap.add_argument('--case', type=int, default=3,
                    help='test-case id to display (3 = closest to every method mean)')
    ap.add_argument('--out', default='figures/fig_ns.pdf')
    ap.add_argument('--cmap', default=FIELD_CMAP)
    ap.add_argument('--no-std', action='store_true',
                    help='drop the posterior-spread column')
    ap.add_argument('--error-insets', action='store_true',
                    help='overlay |recon - truth| as a corner inset on each reconstruction')
    args = ap.parse_args()

    show_std = not args.no_std
    n_rows = len(REGIMES)
    # ground truth | observation | methods... | [std], then spacer/colourbar pairs.
    n_field_cols = 2 + len(METHODS) + (1 if show_std else 0)
    width_ratios = [1.0] * n_field_cols
    col_cbar = n_field_cols + 1                      # vorticity colourbar
    width_ratios += [0.10, 0.075]                    # spacer, colourbar
    col_cbar_std = None
    if show_std:
        col_cbar_std = n_field_cols + 3
        width_ratios += [0.42, 0.075]                # spacer (tick labels), colourbar

    panel = 1.12
    fig_w = panel * sum(width_ratios) + 0.75
    fig_h = panel * n_rows + 0.45
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(n_rows, len(width_ratios), width_ratios=width_ratios,
                          wspace=0.08, hspace=0.14,
                          left=0.075, right=0.955, top=0.93, bottom=0.02)

    missing, metric_log = [], []

    for r, regime in enumerate(REGIMES):
        tag = regime['tag']
        results = {}
        for m in METHODS:
            res, path = load_case(args.root, m['dir'], tag, args.case)
            if res is None:
                missing.append(path)
            results[m['dir']] = res

        ref = next((v for v in results.values() if v is not None), None)
        if ref is None:
            continue

        target = as_2d(ref['target'])
        # One symmetric scale per row, set by the ground truth.
        vmax = float(target.abs().max())
        kw = dict(cmap=args.cmap, vmin=-vmax, vmax=vmax, interpolation='nearest')

        # --- ground truth ---
        ax = fig.add_subplot(gs[r, 0])
        im = ax.imshow(target.numpy(), **kw)
        style(ax)
        ax.set_ylabel(regime['label'], fontsize=8, labelpad=6)
        if r == 0:
            ax.set_title('Ground truth', fontsize=8, pad=4)

        # --- observation (kept at its native resolution: the sparsity is the point) ---
        ax = fig.add_subplot(gs[r, 1])
        obs = as_2d(ref['observation'])
        ax.imshow(obs.numpy(), **kw)
        style(ax)
        ax.text(0.5, -0.04, f'{obs.shape[0]}$\\times${obs.shape[1]}',
                transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
        if r == 0:
            ax.set_title('Observation $y$', fontsize=8, pad=4)

        # --- reconstructions ---
        # One error scale for the whole row, so the insets compare across methods.
        recons = {m['dir']: as_2d(results[m['dir']]['recon'])
                  for m in METHODS if results[m['dir']] is not None}
        err_max = max((float((v - target).abs().reshape(-1).quantile(0.99))
                       for v in recons.values()), default=1.0)

        for c, m in enumerate(METHODS, start=2):
            ax = fig.add_subplot(gs[r, c])
            res = results[m['dir']]
            if res is None:
                ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7, color='0.5')
                style(ax, ours=m.get('ours'))
            else:
                recon = recons[m['dir']]
                ax.imshow(recon.numpy(), **kw)
                style(ax, ours=m.get('ours'))
                err = relative_l2(recon, target)
                ax.text(0.5, -0.04, f'{err:.3f}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=7,
                        fontweight='bold' if m.get('ours') else 'normal')
                metric_log.append((tag, m['dir'], err))
                if args.error_insets:
                    inset = ax.inset_axes([0.63, 0.03, 0.34, 0.34])
                    inset.imshow((recon - target).abs().numpy(), cmap=ERR_CMAP,
                                 vmin=0.0, vmax=err_max, interpolation='nearest')
                    inset.set_xticks([]); inset.set_yticks([])
                    for s in inset.spines.values():
                        s.set_linewidth(0.5)
                        s.set_color('w')
            if r == 0:
                ax.set_title(m['label'], fontsize=8, pad=4,
                             color=OURS_COLOR if m.get('ours') else 'black',
                             fontweight='bold' if m.get('ours') else 'normal')

        # --- AFDPS posterior spread ---
        if show_std:
            ax = fig.add_subplot(gs[r, 2 + len(METHODS)])
            std = posterior_std(results['AFDPS']) if results['AFDPS'] else None
            if std is None:
                ax.text(0.5, 0.5, 'no ensemble', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6.5, color='0.5')
                style(ax)
            else:
                im_std = ax.imshow(std.numpy(), cmap=STD_CMAP, interpolation='nearest')
                style(ax)
                bar(fig, im_std, fig.add_subplot(gs[r, col_cbar_std]),
                    label='std' if r == 0 else None)
            if r == 0:
                ax.set_title('AFDPS post. std', fontsize=8, pad=4)

        # --- shared vorticity colourbar for the row ---
        bar(fig, im, fig.add_subplot(gs[r, col_cbar]),
            label='$\\omega_0$' if r == 0 else None)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'wrote {args.out}')

    print('\nrelative L2 (cross-check against Table 3):')
    for tag, name, err in metric_log:
        print(f'  {tag:14s} {name:6s} {err:.4f}')
    if missing:
        print('\nmissing inputs (run these, or pass --no-std):')
        for p in missing:
            print('  ' + p)


def style(ax, ours=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1 if ours else 0.4)
        s.set_color(OURS_COLOR if ours else '0.6')


def bar(fig, im, cax, label=None):
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, width=0.4, pad=1.5)
    cb.outline.set_linewidth(0.4)
    if label:
        cax.set_title(label, fontsize=7, pad=4)


if __name__ == '__main__':
    main()
