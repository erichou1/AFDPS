"""Render every test case as a contact sheet, for picking the main-paper figure by eye.

Works for any of the three problems: it just globs `result_*.pt` under
<root>/<METHOD>/<tag>/ and lays out cases (rows) x methods (columns), with the
relative L2 printed on each panel.

This is a working file for choosing which case goes in the paper figure -- see
FIGURES.md section 1. It prints two rankings per method: lowest absolute error (the
best-looking panel for that method) and largest margin over the next-best method
(the best-looking comparison).

Examples:
  python figures/contact_sheet.py --root exps/figures --tag fig_ds8_s3 \
      --methods AFDPS EnKG EKI --cmap RdBu_r --out figures/sheet_ns_ds8_s3.pdf

  python figures/contact_sheet.py --root ../full_waveform_inversion/results \
      --tag fig_main --methods AFDPS DiffPIR DPS --cmap turbo --diverging false \
      --out figures/sheet_fwi.pdf
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def relative_l2(pred, target):
    diff = (pred - target).reshape(-1)
    return float(torch.linalg.norm(diff) / torch.linalg.norm(target.reshape(-1)))


def as_2d(x):
    x = torch.as_tensor(x).float().squeeze()
    if x.ndim == 3:
        x = x[0]
    return x


def discover_cases(root, methods, tag):
    """Case ids present for *every* requested method, so rows are complete."""
    per_method = []
    for m in methods:
        ids = set()
        for p in glob.glob(os.path.join(root, m, tag, 'result_*.pt')):
            mo = re.search(r'result_(\d+)\.pt$', p)
            if mo:
                ids.add(int(mo.group(1)))
        per_method.append(ids)
    common = set.intersection(*per_method) if per_method else set()
    return sorted(common), [sorted(s) for s in per_method]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tag', required=True, help='exp_name used for the runs')
    ap.add_argument('--methods', nargs='+', required=True,
                    help='directory names, e.g. AFDPS EnKG EKI (first one is highlighted)')
    ap.add_argument('--cmap', default='RdBu_r')
    ap.add_argument('--diverging', default='true',
                    help='true -> symmetric limits about 0 (vorticity); false -> min/max (velocity)')
    ap.add_argument('--out', default='figures/contact_sheet.pdf')
    args = ap.parse_args()

    diverging = str(args.diverging).lower() in ('1', 'true', 'yes')
    cases, per_method = discover_cases(args.root, args.methods, args.tag)
    if not cases:
        print(f'no cases common to all methods under {args.root}/*/{args.tag}/')
        for m, ids in zip(args.methods, per_method):
            print(f'  {m}: {ids}')
        return

    n_rows, n_cols = len(cases), len(args.methods) + 1   # +1 for ground truth
    panel = 1.05
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel * n_cols + 0.6, panel * n_rows + 0.35),
                             squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.955, bottom=0.01,
                        wspace=0.04, hspace=0.06)

    ranking = {m: [] for m in args.methods}

    for r, case in enumerate(cases):
        results = {}
        for m in args.methods:
            path = os.path.join(args.root, m, args.tag, f'result_{case}.pt')
            results[m] = torch.load(path, map_location='cpu')

        target = as_2d(results[args.methods[0]]['target'])
        if diverging:
            v = float(target.abs().max())
            kw = dict(cmap=args.cmap, vmin=-v, vmax=v, interpolation='nearest')
        else:
            kw = dict(cmap=args.cmap, vmin=float(target.min()),
                      vmax=float(target.max()), interpolation='nearest')

        ax = axes[r][0]
        ax.imshow(target.numpy(), **kw)
        blank(ax)
        ax.set_ylabel(f'case {case}', fontsize=7.5)
        if r == 0:
            ax.set_title('Ground truth', fontsize=8)

        for c, m in enumerate(args.methods, start=1):
            ax = axes[r][c]
            recon = as_2d(results[m]['recon'])
            ax.imshow(recon.numpy(), **kw)
            blank(ax, highlight=(c == 1))
            err = relative_l2(recon, target)
            ranking[m].append((err, case))
            ax.text(0.03, 0.03, f'{err:.3f}', transform=ax.transAxes,
                    fontsize=6.5, va='bottom', ha='left', color='w',
                    bbox=dict(facecolor='black', alpha=0.55, pad=1.0, edgecolor='none'))
            if r == 0:
                ax.set_title(m, fontsize=8,
                             fontweight='bold' if c == 1 else 'normal')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}  ({n_rows} cases x {len(args.methods)} methods)\n')

    # Per-method summary + the two selection criteria described in FIGURES.md.
    primary = args.methods[0]
    means = {m: sum(e for e, _ in v) / len(v) for m, v in ranking.items()}
    for m in args.methods:
        print(f'{m:10s} mean {means[m]:.4f}   best {min(ranking[m])[1]} '
              f'({min(ranking[m])[0]:.4f})   worst {max(ranking[m])[1]} '
              f'({max(ranking[m])[0]:.4f})')

    print(f'\nclosest to every method\'s own mean (a mid-range, unremarkable case):')
    rep = []
    for i, case in enumerate(cases):
        dev = sum(abs(ranking[m][i][0] - means[m]) / max(means[m], 1e-12)
                  for m in args.methods) / len(args.methods)
        rep.append((dev, case))
    for dev, case in sorted(rep)[:3]:
        print(f'  case {case}: mean abs. rel. deviation {dev:.1%}')

    print(f'\nbest comparison for {primary} (largest margin over the next best) -- '
          f'cross-check against the "best" case above and pick the figure:')
    marg = []
    for i, case in enumerate(cases):
        others = [ranking[m][i][0] for m in args.methods[1:]]
        if others:
            marg.append((min(others) - ranking[primary][i][0], case))
    for gap, case in sorted(marg, reverse=True)[:3]:
        print(f'  case {case}: {primary} better by {gap:+.3f} rel. L2')


def blank(ax, highlight=False):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1 if highlight else 0.4)
        s.set_color('#1f4e79' if highlight else '0.6')


if __name__ == '__main__':
    main()
