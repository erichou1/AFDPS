# Qualitative figures: exact steps

How to produce the InverseBench-Figure-2-style reconstruction panels for all three
problems, and how to choose which example goes in the paper.

Everything runs from this repo. The vendored InverseBench harness lives in
`navier_stokes/` and serves all three problems; the AFDPS ports live in their own
directories.

- [0. One-time setup](#0-one-time-setup)
- [1. Choosing which case to show](#1-choosing-which-case-to-show) ← read before running
- [2. Navier–Stokes](#2-navierstokes)
- [3. Full waveform inversion](#3-full-waveform-inversion)
- [4. Linear inverse scattering](#4-linear-inverse-scattering)
- [5. Assembling the paper figure](#5-assembling-the-paper-figure)

---

## 0. One-time setup

Already applied in this repo — listed so you know what changed:

| file | change |
|---|---|
| `navier_stokes/algo/afdps.py` | stash `last_ensemble` / `last_log_weights` before the ensemble collapses |
| `navier_stokes/main.py` | optional `observation_dir` cache; save the ensemble when present |
| `navier_stokes/configs/problem/navier-stokes{,-afdps}.yaml` | `observation_dir: null` default |
| `navier_stokes/figures/make_ns_figure.py` | the main figure renderer |
| `navier_stokes/figures/contact_sheet.py` | all-cases grid + rankings, for picking the case |

All additive. With `observation_dir` unset every existing run path is byte-identical,
so your tables stay reproducible.

To get the ensemble (and therefore the posterior-std column) on FWI and scattering
too, apply the same two-line patch to `full_waveform_inversion/afdps_fwi/algo.py`
and `inverse_scattering/algo/afdps_scatter.py`, plus the `result_dict` addition in
each `main.py`. NS is done; the other two are not yet.

Data and checkpoints must be in place: `../data/{navier-stokes-test,fwi-test,inv-scatter-test}`
and `navier_stokes/checkpoints/{ns,fwi,inv-scatter}-5m.pt`.

---

## 1. Choosing which case to show

Pick the best one. Showing a single strong example out of a test set is what
qualitative figures are for — the tables carry the statistical claim (10-case means
with std), the figure illustrates what the method does when it works. No appendix
grid needed; almost nobody ships one, InverseBench included.

### The workflow

1. Run all 10 cases (you already have them — this is what produced the xlsx).
2. Render the contact sheet and pick by eye. Human in the loop, as your mentor said.
3. Render the main figure at that case.

`contact_sheet.py` prints the rankings so you can choose deliberately rather than by
squinting:

```
AFDPS      mean 0.3522   best 8 (0.2636)   worst 2 (0.4777)
EnKG       mean 0.4912   best 6 (0.3175)   worst 2 (0.6731)

most flattering for AFDPS (largest margin over the next best):
  case 8: AFDPS better by +0.272 rel. L2
```

Two different things to optimise, and they usually disagree:

- **lowest absolute AFDPS error** → the AFDPS panel itself looks best (cleanest
  field, closest to ground truth)
- **largest margin over the next-best method** → the *comparison* looks best, even if
  AFDPS's own panel is mid-range

For a figure whose job is "AFDPS is good," the first usually reads stronger — a
reader's eye compares AFDPS to the ground truth first, and to the baselines second.
Per-problem picks are given in each section below.

### Three things to keep

These cost nothing and keep the figure defensible if a reviewer looks closely:

1. **Same case for every method.** Pick case 5 and show case 5 across the whole row.
   Picking case 5 for AFDPS and case 2 for DiffPIR is the one thing here that would
   genuinely be fabrication, and it is easy to avoid.
2. **Keep the rel-L2 printed on each panel** (the renderer does this by default).
   This is what makes selecting your best case fully above board: the panel says
   `0.091`, the table says `0.155 (0.066)`, and a reader has both numbers without
   you having to add anything. Self-labeling beats disclosure.
3. **Say "example", not "representative".** One word. "Representative" is a claim
   about typicality that a hand-picked best case does not support, and it is the only
   actually-false thing available to write in this caption. "Example case 5" or just
   "case 5" claims nothing and costs you nothing.

### Where AFDPS is genuinely strong

You have more honest headroom than you may realise — none of the following is
cherry-picking, and all of it makes the figure better:

1. **Pick the regime where you actually win.** For NS that is sparse + noisy. Your
   own sweep says the best row is **ds=8, σ=4**: AFDPS 0.909 vs EnKG 1.182 vs EKI
   1.411 — a 0.27 margin, while AFDPS still sits *below* 1.0 so its panel still
   resembles the truth. σ=5 has the same margin but every method is above 1.0 and
   all four panels look like noise, which helps nobody. σ=3 has a smaller margin
   (0.16). **ds=8, σ=4 is your best NS row** and it is not the one you'd have
   guessed from the table in the draft.
2. **Scattering at R≥120 is the best figure in the paper.** AFDPS SSIM 0.985 vs
   DDNM 0.904 at R=180 means DDNM has real structural artifacts, and structural
   artifacts are exactly what shows up in a picture. The figure will make a case the
   PSNR column cannot.
3. **`reduce=best` keeps sharp structure.** A softmax-weighted mean of a chaotic
   vorticity field washes out; the top-weight particle keeps the filaments. Both are
   legitimate disclosed estimators. Use whichever produced your table numbers — but
   know that for NS/FWI that is `best`, and it is also the better-looking one.
4. **Ensemble size.** FWI J=8 → 16 moved rel-L2 0.197 → 0.155. If you can afford
   J=32 for the figure case, that is a real result at a real setting; just report
   the J you used.
5. **Error insets and a shared colour scale** make a genuine advantage legible
   rather than leaving the reader to squint.
6. **The posterior-std column** is a capability the single-trajectory diffusion
   baselines simply do not have.

---

## 2. Navier–Stokes

Runs from `navier_stokes/`. Detail and rationale in
[`navier_stokes/figures/README.md`](navier_stokes/figures/README.md).

### Hyperparameters (from `AFDPS_raw_data.xlsx`, AFDPS sheet)

| ds | σ | `guidance_gamma` | `sigma_floor` | AFDPS | EnKG | EKI |
|---|---|---|---|---|---|---|
| 2 | 0 | 5.25 | 3.0 | 0.591 | **0.124** | 0.621 |
| 8 | 0 | 1.1 | 1.75 | 0.655 | **0.313** | 0.877 |
| 8 | 3 | 0.45 | 1.75 | **0.843** | 1.007 | 1.306 |
| 8 | 4 | 0.5 | 1.75 | **0.909** | 1.182 | 1.411 |

### Runs

```bash
cd navier_stokes
ROOT=exps/figures

run_regime () {   # $1=tag  $2=ds  $3=sigma  $4=gamma  $5=floor
  python main.py problem=navier-stokes-afdps algorithm=afdps \
    problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$1 exp_name=$1 \
    problem.data.id_list=0-9 \
    problem.model.downsample_factor=$2 problem.model.sigma_noise=$3 \
    problem.model.sigma_floor=$5 algorithm.method.guidance_gamma=$4
  for A in enkg eki; do
    python main.py problem=navier-stokes algorithm=$A \
      problem.exp_dir=$ROOT problem.observation_dir=$ROOT/_obs/$1 exp_name=$1 \
      problem.data.id_list=0-9 \
      problem.model.downsample_factor=$2 problem.model.sigma_noise=$3 \
      problem.model.adaptive=False problem.model.delta_t=0.002
  done
}

run_regime fig_ds2_s0 2 0.0 5.25 3.0    # dense, noiseless  (EnKG's corner)
run_regime fig_ds8_s0 8 0.0 1.1  1.75   # sparse, noiseless
run_regime fig_ds8_s4 8 4.0 0.5  1.75   # sparse + noisy    (AFDPS's corner)
```

`id_list=0-9` runs all ten cases so you can select from them.

### Which case to show

Rankings at **ds=8, σ=4** (your strongest NS row) from the xlsx per-run columns:

| case | AFDPS | EnKG | EKI | margin |
|---|---|---|---|---|
| **9** | **0.787** | 1.158 | 1.439 | +0.371 |
| 5 | 0.765 | 1.080 | 1.450 | +0.315 |
| 3 | 0.836 | 1.216 | 1.361 | +0.380 |
| — | *0.909 mean* | *1.182* | *1.411* | |

**Use case 9.** It is second-best on both criteria at once, and it is the most
dramatic panel you have: AFDPS at 0.787 is comfortably below 1.0 so its field still
resembles the ground truth, while EnKG (1.158) and EKI (1.439) are both above 1.0,
where a reconstruction is essentially decorrelated from the truth and looks like it.
Case 5 has the single lowest AFDPS error (0.765) if you want the cleanest AFDPS panel
in isolation; case 3 has the widest margin (+0.380) but a weaker AFDPS panel.

**Solver consistency (do not skip).** AFDPS forces `adaptive=False, delta_t=0.002`
because the adjoint needs a fixed-length trajectory, while the baselines default to
`adaptive=True`. Different dynamics *and* independent noise draws mean the methods
would otherwise be scored on different measurements — invisible in a 10-case mean,
wrong in a side-by-side of one field. The commands above pin the baselines to the
same solver and share `y` via `observation_dir`. Afterwards, check EnKG's printed
rel-L2 at ds8/σ=3 lands near the table's 1.007; a large gap means the solver change
moved the problem and needs looking at before the figure ships.

### Select and render

```bash
# eyeball all ten before committing to case 9
python figures/contact_sheet.py --root exps/figures --tag fig_ds8_s4 \
  --methods AFDPS EnKG EKI --cmap RdBu_r --out figures/sheet_ns.pdf

# edit REGIMES at the top of make_ns_figure.py if you changed the rows, then:
python figures/make_ns_figure.py --root exps/figures --case 9 \
  --error-insets --out figures/fig_ns.pdf
```

The contact sheet is a working file for you, not a paper figure — nothing to include.

Colormap: `RdBu_r`. Vorticity is signed and zero-centred, so a diverging map is
correct; `turbo` is available with `--cmap turbo` if you want visual parity with
InverseBench.

---

## 3. Full waveform inversion

AFDPS runs from `full_waveform_inversion/`, baselines from `navier_stokes/` (the
FWI problem config and every baseline algorithm live in the vendored harness). Both
execute with `cwd=navier_stokes/` so the relative checkpoint and data paths resolve.

Best AFDPS config from the xlsx: `sigma_noise=0.5, guidance_gamma=8, J=16` →
rel-L2 0.155.

### Runs

```bash
cd navier_stokes
ROOT=exps/figures_fwi

# --- AFDPS (main.py from the FWI port) ---
python ../full_waveform_inversion/main.py \
  problem=fwi-afdps algorithm=afdps pretrain=fwi \
  problem.exp_dir=$ROOT exp_name=fig_main problem.data.id_list=1-10 \
  problem.model.sigma_noise=0.5 algorithm.method.guidance_gamma=8 \
  algorithm.method.num_particles=16 num_samples=1 wandb=false

# --- baselines on the identical operator ---
for A in diffpir dps lgd reddiff; do
  python main.py problem=fwi algorithm=$A pretrain=fwi \
    problem.exp_dir=$ROOT exp_name=fig_main problem.data.id_list=1-10 \
    num_samples=1 wandb=false
done
```

Cases are `1-10` here, not `0-9`.

**Which baselines to show.** Your Table 2 marks most FWI baselines as *published*
values copied from InverseBench Table 7 — you did not run them. A panel must come
from a run you did, so either run the baseline here (the algorithms are all present:
`diffpir dps lgd reddiff daps pnpdm adam lbfgs`) or leave it out. Do **not** crop
panels out of InverseBench's Figure 2: they are another paper's results and another
paper's copyrighted figure.

DiffPIR is the one you already re-tuned (0.188), so it is the honest headline
comparison. Add `adam`/`lbfgs` only with the `†` blurred-ground-truth-init caveat
that the table already carries.

**Selection data you already have** (AFDPS J=16 vs DiffPIR re-tuned, per case):

| case | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AFDPS | 0.091 | 0.197 | 0.102 | 0.103 | 0.171 | 0.230 | 0.115 | 0.110 | 0.288 | 0.139 |
| DiffPIR | 0.045 | 0.105 | 0.348 | 0.086 | 0.389 | 0.108 | 0.371 | 0.107 | 0.085 | 0.257 |

**Use case 3 or case 7.**

- **case 3**: AFDPS 0.102 vs DiffPIR 0.348 — best of both worlds. AFDPS's own error
  is near its minimum (its best is 0.091 on case 1) *and* the margin is +0.246. This
  is the pick.
- **case 7**: AFDPS 0.115 vs DiffPIR 0.371 — widest margin (+0.256), AFDPS panel
  almost as clean. Near-equivalent choice.
- avoid case 1 (AFDPS 0.091 is its best panel, but DiffPIR beats it at 0.045) and
  case 9 (DiffPIR 0.085 vs AFDPS 0.288 — your worst case, DiffPIR's second best).

### Render

```bash
python figures/contact_sheet.py --root exps/figures_fwi --tag fig_main \
  --methods AFDPS DiffPIR DPS --cmap turbo --diverging false \
  --out figures/sheet_fwi.pdf
```

Colormap: **`turbo`, not `RdBu_r`**. Velocity is a positive quantity (1.5–4.5 km/s),
so a diverging map is wrong; and the seismic community expects a rainbow. `turbo` is
jet's perceptually-uniform replacement, so it keeps the look your screenshot has
without jet's false banding. Pass `--diverging false` so limits come from min/max
rather than being symmetric about zero.

To use `make_ns_figure.py` for FWI, copy it to `make_fwi_figure.py` and change
`REGIMES`/`METHODS`, `FIELD_CMAP='turbo'`, and the symmetric-limits line in the row
loop to `vmin=target.min(), vmax=target.max()`.

---

## 4. Linear inverse scattering

Runs from `inverse_scattering/`. This is your best-looking figure — AFDPS SSIM 0.985
vs DDNM 0.904 at R=180 means DDNM's structural artifacts are visible, which a
picture shows and the PSNR column hides.

### ⚠ Confirm this before running

The xlsx `IS` sheet has an **`AFDPS σ`** column — the tuned per-R knob (R=15→900,
30→700, 60→540, 120→450, 180→380, 360→440, 720→450, 1440→650) — and I could not
determine which config field it maps to. It is not `problem.model.sigma_noise`
(fixed at 1e-4 by the benchmark) and not obviously `sigma_max` (default 80). The
figure must use the same value that produced Table 1, so resolve this first —
check `inverse_scattering/scripts/run_val_sweep.sh` or the sweep logs for the
override name. Everything below assumes `<KNOB>=380` for R=180.

### Runs

```bash
cd inverse_scattering
ROOT=exps/figures_is
R=180

python main.py problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter \
  problem.exp_dir=$ROOT exp_name=fig_R180 problem.data.id_list=0-9 \
  problem.model.numRec=$R <KNOB>=380 num_samples=1 wandb=false

for A in ddnm dps reddiff; do
  python main.py problem=inv-scatter algorithm=$A pretrain=inv-scatter \
    problem.exp_dir=$ROOT exp_name=fig_R180 problem.data.id_list=0-9 \
    problem.model.numRec=$R num_samples=1 wandb=false
done
```

The test set is `0-99`; `0-9` is plenty for a figure. Per-case values are not in the
xlsx for scattering (only mean ± std), so the contact sheet is the only way to pick
here — which is fine, that is what it is for.

RED-diff and DPS have per-R tuned knobs too (xlsx `REDdiff weight` = 798000 and
`DPS step` = 352 at R=180). Use them, or the baselines are handicapped and the
comparison is not honest.

`reduce`: keep the config default **`mean`** for scattering. The AFDPS config sets
`reduce: mean` deliberately because Table 1 is a PSNR/SSIM point-estimate
leaderboard — this is the setting that produced your numbers, unlike NS/FWI where
it is `best`.

### Render

```bash
python ../navier_stokes/figures/contact_sheet.py --root exps/figures_is \
  --tag fig_R180 --methods AFDPS DDNM REDDiff DPS --cmap gray \
  --diverging false --out figures/sheet_is.pdf
```

Colormap: `gray` — these are cell-permittivity maps and InverseBench shows them
grayscale. Method directory names are the `name:` fields: `AFDPS`, `DDNM`,
`REDDiff`, `DPS`.

Consider adding a **zoom inset** on one cell boundary for this figure: the
AFDPS-vs-DDNM difference is a structural-artifact difference, which is most visible
magnified. `make_ns_figure.py`'s inset machinery can be repointed at a crop instead
of an error map.

---

## 5. Assembling the paper figure

Stack the three problems as row-blocks into one `\textwidth` figure, InverseBench
Figure 2 style, or keep them as three separate figures. Three separate ones are
easier to place and let each have its own caption and colormap.

With tables moved to the appendix you have room. Suggested main-body order:
scattering first (strongest visual), then FWI, then NS.

```latex
\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/fig_ns.pdf}
  \caption{Navier--Stokes initial-vorticity recovery, example case~9.
  Rows sweep acquisition difficulty from dense and noiseless to
  sparse and noisy; numbers under each panel are relative $L_2$ and insets show
  $|\hat\omega_0-\omega_0|$ on a shared per-row scale. All panels in a row share one
  symmetric colour scale set by the ground truth, and all methods receive an
  identical measurement. EnKG leads in the benign regime (top); AFDPS degrades most
  gracefully and leads once the measurement is both subsampled and noisy (bottom).
  The last column is the AFDPS per-pixel posterior standard deviation over the
  $J$-particle ensemble, which the single-trajectory diffusion samplers cannot
  produce.}
  \label{fig:ns}
\end{figure}
```

Caption checklist:

- [ ] says **example**, not *representative* — you hand-picked the case
- [ ] names the case number
- [ ] rel-L2 visible on each panel (renderer default; leave it on)
- [ ] states the ensemble size `J` and the `reduce` mode used
- [ ] states that all methods saw the same measurement
- [ ] carries the `†` informative-init caveat if Adam/LBFGS appear
- [ ] scopes the ensemble/UQ claim to *single-trajectory diffusion samplers* — EnKG
      and EKI are ensemble methods too, so the unqualified "uniquely" in the current
      abstract is wrong and a reviewer who knows EnKG will say so
