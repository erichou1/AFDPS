# AFDPS for InverseBench Full Waveform Inversion (FWI)

This directory applies **AFDPS** — *Approximation-Free Diffusion Posterior Sampling*
(Chen, Ren, Min, Ying, Izzo, TMLR 2026; `TMLR_Rebuttal-main/6054_Solving_Inverse_Problems_.pdf`) —
to the **full waveform inversion** benchmark of **InverseBench**
(ICLR 2025; `TMLR_Rebuttal-main/9220_InverseBench_Benchmarking.pdf`), so that AFDPS can be
compared head-to-head against the methods in InverseBench **Table 7**.

All new code lives here; it **reuses the InverseBench harness in `../navier_stokes/`** as a
library (the FWI Devito forward operator, the evaluator, the dataset loader and the
pretrained EDM prior are the benchmark's own, untouched) so the comparison is apples-to-apples.

```
full_waveform_inversion/
  main.py                          # Hydra entry point (FWI tree + reused navier_stokes harness)
  afdps_fwi/
    __init__.py                    # lazy package (sampler/algo import w/o Devito)
    sampler.py                     # vendored annealed-SDE Feynman-Kac ensemble sampler (verbatim)
    algo.py                        # AFDPS(Algo) inference wrapper
    operator.py                    # AFDPSAcousticWave(AcousticWave): the FWI operator API
    devito_adjoint.py              # data-space adjoint probes for the Gauss-Newton Laplacian
  configs/
    config.yaml                    # top-level defaults (pretrain=fwi, problem=fwi-afdps, algorithm=afdps)
    problem/fwi-afdps.yaml         # benchmark FWI settings + AFDPS operator + knobs
    algorithm/afdps.yaml           # AFDPS sampler config + knobs
    pretrain/fwi.yaml              # EDM prior architecture (fallback instantiation)
  scripts/
    run_fwi_afdps_single.sh        # one process over the 10 test cases (or a range)
    run_fwi_afdps_sharded.sh       # shard cases across GPUs (bit-for-bit == unsharded)
    run_fwi_afdps_gb200.sh         # GB200-tuned big-ensemble run
    aggregate_fwi_afdps_results.py # -> Table-7-format row (Relative L2 / PSNR / SSIM / Data misfit)
  tests/
    test_sampler_cpu.py            # Devito-free CPU smoke test (sampler + Hutchinson trace + wrapper)
  results/                         # runs land in results/fwi-afdps/AFDPS/<exp_name>/
```

---

## 1. How AFDPS was formulated for FWI, and why

**Unknown / forward model (verified against InverseBench §3, App. B.4, Table 2).** FWI
recovers the compressional **velocity** map `v(x)` (a 128×128 field, normalized to ~[-1.5, 1.5]
for the diffusion prior) from recorded waveforms `y = P u`, where `u` solves the acoustic wave
equation `(1/v²)∂²ₜu − ∇²u = q`. The benchmark uses **Devito** with 16 sources @ 1270 m depth,
129 receivers @ 10 m depth, a 5 Hz Ricker wavelet, `dt = 1 ms` (CFL), a 1 s record, and an
80-cell absorbing boundary. Three facts drive the formulation and make FWI *unlike* the
Navier–Stokes and linear-inverse-scattering ports:

| Property | FWI | Consequence for AFDPS |
|---|---|---|
| **Nonlinear**, no closed-form, no SVD | `v ↦ y` is a wave solve | Cannot reuse the linear inverse-scattering formulation; the likelihood is a nonlinear least-squares misfit. |
| **Gradient access** via adjoint state | Devito returns `∇` of the misfit | The likelihood gradient is exact & cheap; **no autograd through the C kernels**. |
| **Noise-free** measurement (Table 2) | `y = A(z*)` exactly | `σ_y` is **not** a noise level but a **likelihood temperature** — the dominant tuning knob. |
| **CFL-fragile** solver (Fig. 3) | noisy `v` ⇒ solver diverges | The AFDPS-SDE noise must be guarded (Tweedie evaluation + velocity clamp). |

With the paper's negative-log-likelihood convention,
`μ_y(x) = (1/2σ_y²)‖A(x) − y‖₂²` and `r(x) = A(x) − y`. AFDPS evolves a **weighted particle
ensemble** through the annealed posterior `q̂_{α,y}(x,t) ∝ p̂_t(x) e^{−α_t μ_y(x)}` (Eq. 3.1),
simulating the weighted-particle dynamics **Eq. (3.6)** with **Algorithm 2 (AFDPS-SDE)** and
optional **Algorithm 1 (ESS resampling)**. The sampler needs exactly two oracles from the
operator — `∇_x μ_y` and `Tr(∇²_x μ_y)` — plus the Feynman–Kac potential value `μ_y`.

The **same vendored sampler** the Navier–Stokes AFDPS results were produced with is reused
**verbatim** (`afdps_fwi/sampler.py`); only the operator behind it is FWI-specific.

---

## 2. Likelihood gradient (adjoint-state) and Laplacian

### Gradient — exact adjoint-state, reused from the benchmark operator
`grad_x μ_y(x) = (1/σ_y²) Jᵀ r`. The parent `AcousticWave.gradient()` already computes
`Jᵀ r` by the adjoint-state method: Devito back-propagates the data residual, cross-correlates
it with the forward wavefield to get the gradient w.r.t. **squared slowness** `m = 1/v²`, then
chain-rules `m → v` (`−2·∇_m/v³`) and `v → x` (`× unnorm_scale`). So:

```python
likelihood_gradient(x, y, σ) = (1/σ²) · AcousticWave.gradient(clamp(x), y)   # per particle
```

### Laplacian — estimated (FWI has no cheap exact Hessian trace)
`Tr(∇²_x μ_y) = (1/σ_y²)[ ‖J‖_F² + Σ_k r_k Tr(∇²A_k) ]`. Two estimators are provided
(`problem.model.laplacian_mode`):

- **`fd_divergence` (default) — EXACT full trace, lowest risk.** Hutchinson
  finite-difference divergence of the misfit gradient,
  `Tr ≈ E_v[ vᵀ(∇μ_y(x+εv) − ∇μ_y(x−εv))/2ε ]`, `v` Rademacher. It **reuses only the
  validated `likelihood_gradient`** (no new Devito code) and is the *same estimator the
  Navier–Stokes port validated*. `hutchinson_scheme='forward'` reuses the drift gradient
  `g0` (1 solve/probe instead of 2). Captures the residual-curvature term exactly.
- **`gn_hutchinson` — Gauss–Newton `(1/σ²)‖J‖_F²`, cheaper & PSD.** Data-space identity
  `‖J‖_F² = E_w‖Jᵀw‖²`, `w` a synthetic residual probe back-propagated by one Devito adjoint
  (`afdps_fwi/devito_adjoint.py`, reusing one forward wavefield across probes). It drops the
  residual-curvature term (vanishes as `r → 0`), is **PSD → maximally stable FK weights**, and
  is the standard FWI Hessian approximation. It touches Devito internals, so it is **off by
  default and must be smoke-tested on-device** before trusting at scale.
- **`zero`** — drop the curvature term entirely (ablation; what plain guided-diffusion does).

**Why the default is exact-trace FD rather than GN:** faithfulness to the AFDPS Feynman–Kac
term (`‖grad‖² − Tr(Hess)`), reproducibility (identical to the NS port), and zero new untested
Devito surface. GN is offered as the efficient PSD alternative. **The CPU smoke test verifies
the FD-divergence estimator reproduces the exact `‖W‖_F²/σ²` of a linear mock operator to
within 0.4 %** (`tests/test_sampler_cpu.py`).

### CFL stability (central to the adaptation, per Fig. 3)
Two guards prevent the AFDPS-SDE noise from breaking the wave solver: (1) the likelihood is
evaluated at the **smooth Tweedie estimate** `x̂₀` (`likelihood_at='denoised'`), and (2) the
physical velocity is **clamped to `[vel_min_kms, vel_max_kms] = [1.5, 4.5]`** before *every*
Devito call. A particle whose solve still fails has its gradient zeroed / value set to `+inf`,
so one CFL blow-up cannot poison the ensemble (the sampler additionally `nan_to_num`s and
quarantines non-finite particles).

---

## 3. Config & harness wiring

- **Operator** `afdps_fwi.operator.AFDPSAcousticWave` subclasses the benchmark
  `inverse_problems.acoustic.AcousticWave` (from `../navier_stokes/`), inheriting the exact
  forward model, geometry, normalization (`v_kms = (x + 3.0)·1.0`) and adjoint-state gradient.
- **Algorithm** `afdps_fwi.algo.AFDPS` wraps the vendored sampler behind
  `Algo.inference(observation, num_samples)` and stashes `op._y = observation` (the Devito
  Receiver shot-gather list) for the Laplacian/init.
- **Evaluator** is the benchmark's own `eval.AcousticWave` with `data_misfit: true` (so the
  Table-7 "Data misfit" column is produced) — *metrics are not reimplemented.*
- **`main.py`** mirrors `navier_stokes/main.py` (per-global-case-id seeding ⇒ sharding is
  bit-for-bit identical to an unsharded run) and puts `../navier_stokes/` on `sys.path`. The
  FWI package is named `afdps_fwi` (not `algo`/`inverse_problems`) so it never shadows the
  reused harness packages.

Run (cwd must be `navier_stokes/` so the relative `checkpoints/fwi-5m.pt` and `../data/fwi-test`
resolve — the scripts handle this):

```bash
bash full_waveform_inversion/scripts/run_fwi_afdps_single.sh baseline 1-10
```

---

## 4. Tuning

`σ_y` (the likelihood temperature) is the dominant FWI knob because the problem is noise-free.
Sweep, in rough priority order, via Hydra overrides:

| Knob | Override | Note |
|---|---|---|
| Likelihood temperature `σ_y` | `problem.model.sigma_noise=` | **dominant**; paper range [1e-2, 1e1] |
| Guidance strength | `algorithm.method.guidance_gamma=` | smaller ⇒ stronger/earlier data fit |
| # particles / # steps | `algorithm.method.num_particles= num_steps=` | quality vs solver cost |
| ESS resampling | `algorithm.method.sampler_kwargs.resample=true resample_threshold=` | Algorithm 1 on/off |
| FK value term | `algorithm.method.sampler_kwargs.use_value=true` | score particles by real misfit |
| Estimator | `algorithm.method.reduce=best|mean` | MAP particle vs FK posterior mean |
| Laplacian | `problem.model.laplacian_mode=fd_divergence|gn_hutchinson|zero` + `hutchinson_M=` | trace cost/variance |
| Clamp / init | `problem.model.vel_min_kms= vel_max_kms= init_mode=` | CFL margin |
| Schedule | `algorithm.method.sigma_max= rho=` | prior noise schedule |

Each run logs per-case Relative L2 / PSNR / SSIM / Data misfit, peak GPU memory, and (with
`resample`) ESS / resampling events.

---

## 5. GB200 / efficiency setup

**FWI's bottleneck is the CPU, not the GPU** — the wave forward/adjoint solves run on CPU via
Devito's compiled C kernels + a dask `LocalCluster` (the parent operator already fans the 16
shots across cores). The GB200 strategy therefore drives **both** processors:

- **GPU**: a large particle ensemble batches the EDM prior + Hutchinson/weight algebra;
  `tf32=true`, `compile=true`. Static solver/geometry tensors are built **once** in
  `__init__` and reused.
- **CPU (Grace)**: `DEVITO_LANGUAGE=openmp`, `OMP_NUM_THREADS=$(nproc)`; the 16 shots run in
  parallel on the dask cluster. `run_fwi_afdps_gb200.sh` sets these and a large ensemble.
- **Sharding**: split the 10 cases by global id across GPUs (`run_fwi_afdps_sharded.sh`);
  per-case seeding makes it byte-identical to a single run.
- **Trade-off knobs** for the solver-bound inner loop: `laplacian_mode` (exact FD = 2 extra
  solves/step at `M=1` forward-scheme = 1; GN = 1 forward + M adjoints; `zero` = none),
  `hutchinson_M`, `grad_chunk`.

> Note: the InverseBench FWI operator is CPU-Devito, so "100 % GPU utilization" is not the
> right target for FWI (it is for the differentiable-GPU Navier–Stokes port). The goal is to
> keep the GPU prior step and the CPU solver both busy and overlapped, and to parallelize
> shots/cases. A Devito GPU/OpenACC backend exists but the benchmark operator uses CPU; any
> GPU-offload experiment must be verified not to change the numbers.

---

## 6. Results — InverseBench Table 7 format

Metrics (mean (std) over the **10** CurveFaultB test cases, noise-free): **Relative L2 ↓,
PSNR ↑, SSIM ↑, Data misfit ↓**. AFDPS belongs in the *PnP-diffusion-prior, prior-only*
(un-daggered) group — alongside DPS / LGD / DiffPIR / PnP-DM / REDDiff (DDNM and ΠGDM are
absent from Table 7 because they are linear-only and cannot handle the nonlinear FWI forward).

| Method | Relative L2 ↓ | PSNR ↑ | SSIM ↑ | Data misfit ↓ |
|---|---|---|---|---|
| Adam | 0.333 (0.086) | 9.968 (2.083) | 0.305 (0.120) | 115.14 (52.10) |
| Adam† | 0.089 (0.021) | 21.273 (2.045) | 0.679 (0.073) | 15.89 (10.16) |
| LBFGS† | 0.070 (0.023) | 23.398 (2.749) | 0.704 (0.077) | 9.18 (6.47) |
| DPS | 0.250 (0.154) | 14.111 (6.820) | 0.491 (0.161) | 155.08 (92.17) |
| LGD | 0.244 (0.024) | 12.288 (0.889) | 0.341 (0.047) | 258.47 (26.40) |
| DiffPIR | 0.204 (0.129) | 16.113 (6.962) | 0.554 (0.191) | 88.53 (56.91) |
| DAPS† | 0.201 (0.103) | 14.914 (4.184) | 0.321 (0.067) | 111.13 (71.33) |
| PnP-DM | 0.259 (0.075) | 11.983 (2.269) | 0.431 (0.073) | 308.84 (26.34) |
| REDDiff | 0.319 (0.102) | 10.372 (2.650) | 0.280 (0.108) | 94.67 (41.33) |
| **AFDPS (ours)** | _run to fill_ | _run to fill_ | _run to fill_ | _run to fill_ |

> **The AFDPS row is intentionally left blank.** This implementation was authored and
> validated for correctness on a machine **without Devito, a GPU, the `fwi-5m.pt` checkpoint,
> or the CurveFaultB data**, so producing the numbers here would mean fabricating them. Fill
> the row by running the provided scripts on the GB200 (or any Devito+CUDA box that has the
> InverseBench FWI checkpoint and data):
>
> ```bash
> # 10 cases, sharded across 4 GPUs, then auto-aggregate into this exact table:
> bash full_waveform_inversion/scripts/run_fwi_afdps_sharded.sh final 4 10
> # single GB200, large ensemble:
> bash full_waveform_inversion/scripts/run_fwi_afdps_gb200.sh gb200 48 200
> ```
> `aggregate_fwi_afdps_results.py` prints the table above with the AFDPS row filled in, using
> the *same* metric definitions as `eval.AcousticWave`. **A claim of a fair comparison is only
> valid if the run used this forward model, the 10-case CurveFaultB test set, the noise-free
> setting, the benchmark normalization, and the `eval.AcousticWave` evaluator — which this
> harness enforces by construction.**

---

## 7. Failure modes, approximations & caveats

- **CFL blow-ups.** Mitigated by Tweedie-evaluation + velocity clamp + per-particle
  quarantine. If they persist, tighten `vel_min/max_kms`, lower `guidance_gamma`, raise
  `num_steps`, or reduce `sigma_max`. (InverseBench excludes one case for DAPS for exactly
  this instability; AFDPS's clamp is designed to avoid that.)
- **Laplacian approximation.** `fd_divergence` is the exact trace but stochastic (variance
  ↓ with `hutchinson_M`, common-random/antithetic probes). `gn_hutchinson` is biased (drops
  residual curvature) but PSD/stable/cheaper. `zero` removes the FK curvature term entirely.
- **Weight degeneracy.** On a stiff nonlinear misfit the Feynman–Kac weights can collapse to
  one particle; enable Algorithm 1 (`sampler_kwargs.resample=true`) and/or raise
  `num_particles`. ESS is logged.
- **`gn_hutchinson` / `guidance_mode='auto'`** depend on `devito_adjoint.py`, which mirrors
  the parent's proven `gradient_single_shot` but injects a synthetic residual and reuses the
  forward wavefield across probes; **validate on-device** before relying on it.
  `guidance_mode='full'` (anisotropic PiGDM) is **not supported** for the Devito operator
  (it would need matrix-free `J/Jᵀ` products through the solver).
- **Initialization regime.** AFDPS starts from a constant background (`init_mode='zeros'`),
  the fair comparison point for the un-daggered PnP-diffusion baselines. The †-baselines
  (Adam†/LBFGS†/DAPS†) instead start from a blurred ground truth; matching that would require
  plumbing the target into the operator and is intentionally not the default.
- **Environment.** Runs require Devito (FWI operator), a CUDA GPU (prior), the `fwi-5m.pt`
  checkpoint and the CurveFaultB LMDB data under `navier_stokes/`. Offline-checkable logic is
  covered by `tests/test_sampler_cpu.py` (passes: trace within 0.4 %, sampler reduces misfit
  114.7→20.7, wrapper reductions correct).
```
