# Navier–Stokes qualitative figure

Reproduces the InverseBench-Figure-2-style comparison for the NS
initial-vorticity problem: one row per acquisition regime, columns for ground
truth, observation, each reconstruction, and the AFDPS posterior spread.

## Case selection

All panels use **test case 3**. It was not hand-picked: scoring each of the 10
test fields by mean relative deviation from every method's own 10-case mean,
case 3 is the most representative field at every regime — within 2% of all
three methods' means at `ds=8`.

| regime | AFDPS | EnKG | EKI | mean abs. rel. deviation |
|---|---|---|---|---|
| ds8, σ=3 | 0.824 | 1.010 | 1.274 | 1.7% |
| ds8, σ=5 | 1.023 | 1.313 | 1.487 | 1.1% |
| ds2, σ=0 | 0.557 | 0.143 | 0.544 | 11% |
| ds8, σ=0 | 0.642 | 0.486 | 0.857 | 20% |

## Solver consistency (important)

AFDPS uses `AFDPSNavierStokes2d` with `adaptive: False, delta_t=0.002` (a
fixed-length trajectory is required by the adjoint), while EnKG/EKI default to
`ForwardNavierStokes2d` with `adaptive: True`. Different dynamics *and*
independent noise draws mean each method would otherwise see a different `y` —
harmless for 10-case table means, but wrong for a side-by-side figure of one
field.

The commands below therefore

1. run the baselines with `problem.model.adaptive=False
   problem.model.delta_t=0.002` so all methods share identical dynamics
   (dt=0.002 is sub-CFL at Re=200, so the baselines are not disadvantaged), and
2. set `problem.observation_dir` so the first run writes `y` and the rest load
   the same tensor.

Sanity check after running: the printed EnKG rel-L2 at ds8/σ=3 should land near
the table's 1.010. A large gap means the solver change moved the problem and
needs looking at before the figure goes in the paper.

## Runs

Hyperparameters are the tuned values from `AFDPS_raw_data.xlsx` (AFDPS sheet,
`γ` and per-`ds` `floor` columns).

```bash
cd navier_stokes
ROOT=exps/figures

# ---- row 1: ds=2, sigma=0  (dense, noiseless; EnKG's best corner) ----
TAG=fig_ds2_s0
python main.py problem=navier-stokes-afdps algorithm=afdps \
  problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
  problem.data.id_list=3 \
  problem.model.downsample_factor=2 problem.model.sigma_noise=0.0 \
  problem.model.sigma_floor=3.0 algorithm.method.guidance_gamma=5.25
for A in enkg eki; do
  python main.py problem=navier-stokes algorithm=$A \
    problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
    problem.data.id_list=3 \
    problem.model.downsample_factor=2 problem.model.sigma_noise=0.0 \
    problem.model.adaptive=False problem.model.delta_t=0.002
done

# ---- row 2: ds=8, sigma=0  (sparse, noiseless) ----
TAG=fig_ds8_s0
python main.py problem=navier-stokes-afdps algorithm=afdps \
  problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
  problem.data.id_list=3 \
  problem.model.downsample_factor=8 problem.model.sigma_noise=0.0 \
  problem.model.sigma_floor=1.75 algorithm.method.guidance_gamma=1.1
for A in enkg eki; do
  python main.py problem=navier-stokes algorithm=$A \
    problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
    problem.data.id_list=3 \
    problem.model.downsample_factor=8 problem.model.sigma_noise=0.0 \
    problem.model.adaptive=False problem.model.delta_t=0.002
done

# ---- row 3: ds=8, sigma=3  (sparse + noisy; AFDPS wins) ----
TAG=fig_ds8_s3
python main.py problem=navier-stokes-afdps algorithm=afdps \
  problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
  problem.data.id_list=3 \
  problem.model.downsample_factor=8 problem.model.sigma_noise=3.0 \
  problem.model.sigma_floor=1.75 algorithm.method.guidance_gamma=0.45
for A in enkg eki; do
  python main.py problem=navier-stokes algorithm=$A \
    problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$TAG exp_name=$TAG \
    problem.data.id_list=3 \
    problem.model.downsample_factor=8 problem.model.sigma_noise=3.0 \
    problem.model.adaptive=False problem.model.delta_t=0.002
done
```

Nine single-case runs, roughly one tenth of one row of the NS table.

`reduce`: the reported numbers must be reproduced with the same setting the
sweep used (`configs/algorithm/afdps.yaml` defaults to `reduce: best`, the
highest-log-weight particle). Add `algorithm.method.reduce=mean` only if the
tables were produced that way.

## Render

```bash
python figures/make_ns_figure.py --root exps/figures --case 3 --out figures/fig_ns.pdf
```

The script prints the rel-L2 it computed per panel (via `eval.relative_l2`, the
same function the tables use) so figure and Table 3 can be cross-checked. Flags:

- `--error-insets` overlays $|\hat\omega_0-\omega_0|$ as a corner inset on each
  reconstruction, on one scale per row (the row's 99th percentile) so methods are
  comparable. Worth it: at `\linewidth` the vorticity panels are ~0.85in and all
  look broadly similar, whereas the insets separate the methods immediately.
  Check legibility on the real fields before committing to it — the inset is only
  ~0.3in in the final PDF.
- `--no-std` drops the posterior-spread column
- `--cmap turbo` matches InverseBench's colour scheme; the default `RdBu_r` is
  diverging (correct for signed vorticity) and colourblind-safer. For a *velocity*
  field (FWI) use `turbo`, not `RdBu_r` — velocity is positive, and the seismic
  community expects a rainbow map. `turbo` is jet's perceptually-uniform
  replacement, so it keeps the expected look without jet's false banding.
- edit `REGIMES` at the top of the script to change or add rows

The AFDPS column is drawn with a highlight frame and bold metric (`ours=True` in
`METHODS`), so the reader's eye lands on it without a caption instruction.

## LaTeX

```latex
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/fig_ns.pdf}
  \caption{Navier--Stokes initial-vorticity recovery on test case~3, the field
  closest to every method's 10-case mean. Rows sweep acquisition difficulty from
  dense and noiseless to sparse and noisy; numbers under each panel are relative
  $L_2$. All panels in a row share one symmetric colour scale set by the ground
  truth, and all methods see an identical measurement. EnKG is far ahead in the
  benign regime (top), while AFDPS degrades most gracefully and leads once the
  measurement is both subsampled and noisy (bottom). The last column is the
  AFDPS per-pixel posterior standard deviation over the $J$-particle ensemble,
  which the single-trajectory diffusion samplers cannot produce; it concentrates
  on the vortex filaments, where the sparse measurement is least informative.}
  \label{fig:ns}
\end{figure}
```
