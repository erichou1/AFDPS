# AFDPS on InverseBench Linear Inverse Scattering

## Context

AFDPS (Approximation-Free Diffusion Posterior Sampling, TMLR 2026) has been ported to the
Navier–Stokes problem in this repo. The task: apply AFDPS to **linear inverse scattering**
(InverseBench, ICLR 2025) so its PSNR / SSIM / Meas-err can sit next to Table 3 (p. 20) at
360/180/60 receivers, σ_y = 1e-4. All new code goes in `inverse_scattering/`; the benchmark
forward model, dataset, and evaluator in `navier_stokes/` are reused **by import, untouched**
(fairness).

**Scope (user-confirmed)**: implement the full port here and verify everything verifiable on
CPU (operator math, sampler, configs, aggregator — via the tests below). The GB200 is NOT
reachable from this machine; the user runs the prepared scripts there themselves. Deliverable:
ready-to-run scripts + README with the exact run order (download → precompute → smoke → val
sweep → test → aggregate), and the aggregator that emits the final Table-3-format report.
The asset download script IS in scope (extend the NS pattern).

Key verified facts (read from code/PDFs this session):
- `navier_stokes/inverse_problems/inverse_scatter.py::InverseScatter` — forward is **linear**
  in the permittivity `f` (first Born, incident field as total field), unbatched (`f[0,0]`),
  complex128, returns `(1, numTrans=20, numRec=R)`. `compute_svd()` caches a real stacked
  matrix `A` of shape `(2m, n)`, m = 20·R, n = 16384, rows interleaved [Re;Im] per measurement
  (matches `view_as_real(y.flatten()).flatten()`), plus `U (2m,2m)`, `Sigma (2m,)`, `V_t (2m,n)`
  in `cache/inv-scatter_numT_20_numR_{R}/` (CWD-relative, 10–20 min first time, **computed at
  `__init__`** — race hazard for parallel shards).
- Noise: `__call__` adds `sigma_noise * randn_like(out)` on a complex tensor → CN(0,1):
  per-real-component variance σ_y²/2 (re-verify numerically at implementation time).
- Normalization: model space x∈[−1,1]; `unnormalize(x) = (x+1)·0.5` (c = 0.5).
- Prior: EDMPrecond/DhariwalUNet 128×128×1ch (~26.8M params), `checkpoints/inv-scatter-5m.pt`
  (GitHub release asset may be named `in-scatter-5m.pt` — handle both). Data: LMDB
  `../data/inv-scatter-{test,val}` (100 / 10 images) on CaltechDATA record `zg89b-mpv16`
  (token pattern in `navier_stokes/scripts/download_assets.sh`). **Neither is on this machine.**
- `navier_stokes/main.py` seeds per global case id (`torch.manual_seed(seed + int(data_id))`)
  *before* observation generation → bit-identical multi-shard runs.
- AFDPS reference (`inverse_scattering/TMLR_Rebuttal-main/.../ensemble_denoiser_edm.py`):
  likelihood evaluated **at the particle** x_t, linear annealing α_t=(t0−t)/t0, value term in
  FK weights, best-of-N reduce. NS vendored sampler
  (`navier_stokes/algo/afdps_core/ensemble_denoiser_edm.py`): same skeleton, annealing moved
  into effective noise r_t = √(σ_y² + (γσ_t)²) — identically an α-curve α_t = σ̃²/r_t².
- Table 3 metrics: PSNR/SSIM via `eval.InverseScatter` (piq, clip[0,1], data_range=1, on
  unnormalized recon); Meas err (%) = 100·‖A(x̂)−y‖₂/‖y‖₂ (computed in our aggregator;
  the in-repo evaluator doesn't produce it). Baselines were tuned per receiver count on the
  10-image val set (Table 12: e.g. DPS guidance 280/380/625). Leaders to beat: RED-diff 36.56 /
  DDNM 36.38 PSNR @360; DDNM 29.24 @60; DiffPIR SSIM 0.988.

## Review revisions (2026-06-11)

Folded in after a full read-through + numeric re-verification (√2 noise constant confirmed:
`var(real)=0.5003` → σ̃ = σ_y/(√2·c) = √2·1e-4). Strategic decisions that change the build:

1. **Table-3 is an MMSE/point-estimate leaderboard; AFDPS is a sampler.** The leaders
   (RED-diff 36.56, DDNM 36.38) are effectively posterior-mean estimators → a single
   highest-weight particle will lose PSNR/SSIM at EVERY receiver count, not just R=60.
   **Default `reduce='mean'` (weighted ensemble mean) for the headline number**; keep
   best/sample draws as the uncertainty-quantification story.
2. **σ_y=1e-4 is near-noiseless → conditioning-dominated.** Hard data-consistency on
   well-measured modes + prior on the null space wins. The plan's own analysis says the
   **exact ΠGDM-full** path gives a DDNM-like contraction on measured modes → **promote
   ΠGDM-full + `exact_linear` to the PRIMARY configuration**; demote γ-annealed SDE to the
   paper-faithful ablation. (Both share the same exact_linear update form; only φ_i differs.)
3. **Real-checkpoint interface smoke is GB200 step 0**: load `inv-scatter-5m.pt`, assert
   `net(x/s,σ)→D_x`, `.sigma_min/max`, `.round_sigma`, `img_channels=1`, `img_resolution=128`
   BEFORE any sweep (the CPU tests only exercise a stub net).
4. **Linear-Gaussian twin is the acceptance gate**: analytic Gaussian denoiser → closed-form
   SVD-basis posterior; AFDPS weighted mean must match within MC error. Only check that
   validates state dynamics + FK weights JOINTLY. Treat a pass as the green light to sweep.
5. **Pin Meas-err (%) to InverseBench's definition** before reporting (the in-repo evaluator
   emits `sqrt(loss)`, not normalized ×100). Cross-check against the paper appendix.
6. **Reword "bit-identical" sharding**: per-case seeding reproduces INPUTS; CUDA GEMM/conv
   nondeterminism breaks bitwise equality unless `use_deterministic_algorithms(True)` (slow).
   Realistic guarantee = statistically identical mean PSNR. **Benchmark single-large-batch GPU
   util FIRST**; only add co-tenant shards if a real utilization gap remains (each shard
   reloads A_inv/U/V at multi-GB and contends for SMs).

## Formulation (the math)

**Fold the affine normalization into the observation once per case** (fp64):
- `y_real = view_as_real(observation.flatten()).flatten()` ∈ R^{2m} (ordering == rows of `A`)
- `ỹ = y_real/c − shift·(A @ 1)` so `ỹ = A x_norm + ẽ`, ẽ ~ N(0, σ̃² I) with
  **σ̃ = σ_y/(√2·c) = √2·1e-4** (pin the √2 with the numeric noise-convention test).
- Then μ_y(x) = ‖A x − ỹ‖²/(2σ̃²) with **no chain-rule factors anywhere**; all ops factored
  through the SVD (`A x = U·(S·(V_t x))`), batched over particles as GEMMs — no autograd,
  no UNet backprop (`likelihood_at='noisy'`, the paper-faithful choice for linear ops).

**Exact operator API** (everything analytic; per-particle batched):
- `likelihood_value(x,·,σ) = ‖Ax−ỹ‖²/(2σ²)`  (J,) fp64
- `likelihood_gradient(x,·,σ) = Aᵀ(Ax−ỹ)/σ²` via SVD factors, (J,1,128,128)
- `likelihood_laplacian = Σᵢsᵢ²/σ²` — **exact constant**, no Hutchinson
- `jacobian_trace = λ̄ = Σᵢsᵢ²/(2m)` — exact, free → `guidance_mode='auto'` is exact
- `likelihood_gradient_pigdm` — exact anisotropic ΠGDM `Aᵀ(σ̃²I+ς²AAᵀ)⁻¹r`, diagonal in
  U-basis, **no CG** (rel_residual = 0)
- `initialize_ensemble`: `σ_max·randn` (exact: α_0 ≈ 0 for the annealed curve). Optional
  `init_mode='tilted'`: exact sample of N(0,σ_max²I)·e^{−α₀μ_y} in the SVD basis (for α≡1 ablation).
- `value_coef(t) = 2γ_e²t/(σ̃²+γ_e²t²)` — the **exact** −α'_t coefficient for the FK value term
  (replaces the reference's 1/t0 heuristic; γ_e = γ for fixed mode, √λ̄ for auto).

**Primary sampler config** (revised — see Review revisions #1,#2): **ΠGDM-full guidance +
`guidance_step='exact_linear'`**, EDM linear schedule σ(t)=t, s≡1, prior drift via Euler, FK
weights `Δlogβ = (t_cur−t_next)·[t·(‖g‖²−Δ) + d_cur·g − κ(t)·μ̃_t(x)]` in fp64, ESS logged,
`resample` opt-in (Algorithm 1), **`reduce='mean'` (weighted ensemble mean) for the Table-3
number** (best/sample for UQ). The γ-annealed AFDPS-SDE (`guidance_mode='fixed'`, Euler) is
retained as the **paper-faithful ablation**, not the headline config.

**Key upgrade — `guidance_step='exact_linear'`**: the guidance drift is linear in x, so per step
integrate it **exactly in the V-basis** (operator-split): `u = V_t x`,
`φᵢ = (sᵢ²/γ_e²)·ln(r²(t_cur)/r²(t_next))`, `u ← u·e^{−φ} + (ỹ_U/S)·(−expm1(−φ))` (expm1-safe,
no bare 1/s), `x ← x + V_tᵀ(u_new−u_old)`. Unconditionally stable → removes the 1/σ̃² ≈ 5e7
stiffness; at the final step with the ΠGDM variant the contraction `e^{−φᵢ} = σ̃²/(σ̃²+sᵢ²t²)`
approaches a pinv-replacement on well-measured directions (DDNM-like data consistency) while
keeping the diffusion prior in weak/null directions. Euler mode kept for the faithful ablation.

**Paper-faithful ablation (γ-annealed)**: `guidance_mode='fixed'` (hand-set γ) or `'auto'`
(isotropic, γ_e²=λ̄), Euler guidance step — the byte-faithful AFDPS-SDE for comparison against
the promoted ΠGDM-full primary. Tweedie variant `ς² = t²σ_d²/(t²+σ_d²)` with
`likelihood_at='denoised'` (note: exact_linear is only EXACT for `likelihood_at='noisy'`;
with denoised it is an O(Δt) approximation — keep as ablation).

## Repo layout (all new code under `inverse_scattering/`)

```
inverse_scattering/
├── main.py                      # near-verbatim copy of navier_stokes/main.py; diffs:
│                                #  sys.path bootstrap; observation.cpu(); torch.load(weights_only=False)
├── inverse_problems/            # NAMESPACE package — NO __init__.py (merges with NS's)
│   └── inverse_scatter_afdps.py # AFDPSInverseScatter(InverseScatter): the API above;
│                                # never overrides forward/__call__/loss/normalize (fairness)
├── algo/                        # NAMESPACE package — NO __init__.py
│   ├── afdps_scatter.py         # Algo wrapper (modeled on navier_stokes/algo/afdps.py);
│   │                            # passes _Noiser(op.sigma_noise_eff); reduce best/mean
│   └── afdps_core_scatter/      # vendored sampler (from NS afdps_core) + new opts:
│       └── ensemble_denoiser_edm.py   # value_coef='t0'|'exact', guidance_step='euler'|'exact_linear',
│                                      # fp64 log-weights; NS tree untouched
├── configs/
│   ├── config.yaml              # hydra.searchpath: file://../navier_stokes/configs
│   ├── problem/inv-scatter-afdps.yaml   # defaults: [inv-scatter, _self_] → physics single-sourced
│   └── algorithm/afdps.yaml     # _target_: algo.afdps_scatter.AFDPSScatter
├── scripts/
│   ├── download_assets.sh       # CaltechDATA file-listing query (token) + GitHub release asset
│   │                            # discovery (in-scatter-5m.pt vs inv-scatter-5m.pt) → checkpoints/inv-scatter-5m.pt
│   ├── precompute_svd.py        # builds all 3 SVD caches BEFORE parallel launches (race guard:
│   │                            # gate on matrix_inv.pt, the last file written)
│   ├── smoke_gb200.sh           # 1 case, J=64/50 steps; then timed J∈{512,1024,2048} step bench
│   ├── run_val_sweep.sh         # per R: sweep grid on val ids 0-9, CONC jobs on one GPU
│   ├── run_test.sh              # per R: 100 cases, NSHARD co-tenant shards (id ranges) on the GPU,
│   │                            # gpu_log.sh start/stop; run_test_all.sh chains R=360→180→60
│   ├── gpu_log.sh               # nvidia-smi CSV sampler + summarize (mean/median/p10 util)
│   ├── metrics_lib.py           # collect/dedup results; PSNR/SSIM via eval.InverseScatter(forward_op=None);
│   │                            # meas-err: op(svd=False, device='cpu').forward(recon, unnormalize=False)
│   ├── val_table.py             # per-(R, config) winner table from val runs
│   └── aggregate_table3.py      # Table-3-format markdown+LaTeX: 12 baseline rows hardcoded
│                                # (extract verbatim from the PDF via pdftotext, not by eye) + AFDPS row
├── tests/                       # CPU-only, runnable on this MacBook (no data/GPU)
│   ├── conftest.py              # sys.path bootstrap + chdir-to-tmp (cache/ lands in tmp)
│   ├── test_operator_cpu.py     # tiny op (Nx=Ny=16, numTrans=2, numRec=4, device='cpu')
│   ├── test_configs.py          # hydra compose + _target_ resolution (catches searchpath breakage)
│   └── test_aggregate.py        # dedup rule, meas-err formula, table formatting
└── README.md                    # run order; CWD convention (everything runs from inverse_scattering/)
```

sys.path bootstrap (top of main.py / conftest.py / scripts): insert
`…/navier_stokes` at position 1 → `utils`, `training`, `models`, `eval` resolve from NS;
`inverse_problems` and `algo` namespace-merge across both dirs. **Never add `__init__.py`**
to the two local namespace dirs (would shadow the benchmark modules).

CWD convention: every entry point `cd`s to `inverse_scattering/` so `../data`,
`checkpoints/`, `cache/`, `exps/` resolve; data dir is shared with the NS configs' `../data`.

## Verification checklist (CPU tests now; GB200 smoke later)

1. Noise convention: complex `randn_like` per-real-component variance = σ_y²/2 (numeric).
2. Matrix ≡ forward(): `A @ f.flatten()` == `view_as_real(op.forward(f, unnormalize=False))
   .permute(0,2,1).flatten()`; folding identity ‖A_real f − y_real‖ = c·‖A x − ỹ‖.
3. `likelihood_gradient` vs `torch.autograd.grad` through the **benchmark** `op.loss` (rel < 1e-8);
   Laplacian vs Hutchinson probe estimate; ΠGDM solve residual < 1e-10.
4. `value_coef`: finite-difference ∂_t[misfit/(2r_t²)] vs −κ(t)·value.
5. `exact_linear` substep vs 2e4-step Euler on guidance-only dynamics (rel < 1e-3); plus an
   end-to-end **linear-Gaussian twin**: analytic Gaussian denoiser ⇒ closed-form posterior in the
   SVD basis; AFDPS weighted mean must match within MC error (validates dynamics + FK weights jointly).
6. 5-step sampler run (stub net, J=8): finite ensemble/weights; determinism: same seed ⇒ same bits.
7. Fairness guard: `AFDPSInverseScatter.forward is InverseScatter.forward` (and `__call__`, `loss`).
8. Hydra compose test; aggregator unit test.
9. GB200 smoke: download → precompute(R=360) → 1 case J=64 → timed step bench → commit defaults.

## Tuning & run plan (GB200)

- Stage 0 (free): SVD caches; log s_max, λ̄, spectrum; sanity E[μ_y(truth)] ≈ m.
- Validation (per R ∈ {360,180,60}, val ids 0-9, J≈512): coordinate-descent —
  (1) guidance family: auto, fixed γ ∈ √λ̄·{¼,½,1,2,4}, ΠGDM-full(ς=σ_t), ΠGDM-full(Tweedie,denoised),
      × guidance_step ∈ {exact_linear, euler} × steps ∈ {100,200};
  (2) steps × particles: {50,…,400} × {256,512,1024} around the winner;
  (3) FK machinery: use_value {exact-κ, off, t0} × resample {off, on@0.5} × reduce {best, mean};
  (4) sigma_max {20,80}, likelihood_at {noisy, denoised}.
  Score by PSNR, meas-err tiebreak. Expected: ΠGDM-full+exact_linear wins @360/180;
  mean-reduce + more particles + resampling wins @60.
- Final: 3 settings × 100 test cases, J=1024, winner config per R, NSHARD=2 co-tenant shards
  (per-case seeding ⇒ reproducible INPUTS → statistically identical mean PSNR; NOT bitwise
  unless `use_deterministic_algorithms(True)`), GPU util logged → report number. Benchmark
  single-large-batch util first; only shard if a real gap remains.
- Estimates: UNet dominates (~50 GFLOPs/sample fwd); J=1024 ≈ 0.1–0.25 s/step → 3×100 cases
  ≈ 4–9 h; val sweeps ≈ 1–2.5 h; SVD precompute 30–60 min once.

## Numerics & failure modes

- Operator linear algebra fp64 masters on GPU (R=360: ~4 GB — trivial on GB200) with fp32 working
  copies for the hot GEMMs (knob `likelihood_dtype`); FK reductions and log-weights fp64.
  Keep TF32 away from residual GEMMs (residual scale ~σ̃ = 1.4e-4).
- Stiffness: eliminated by exact_linear substep; Euler ablation logs the stability factor
  `2tΔt·s_max²/r_t²` and aborts > 2.
- Weight degeneracy: max-subtraction + −inf quarantine (already in vendored sampler), ESS logging,
  opt-in resampling. No bare 1/sᵢ anywhere (expm1-safe forms); never use the `op.M` mask.

## Deliverables / final report

(1) formulation write-up (this doc → README); (2) config + harness wiring; (3) Table-3-format
results table with AFDPS rows beside the 12 baselines at 360/180/60 (mean (std) over 100 cases);
(4) best AFDPS tuning per receiver count (from val sweeps); (5) GB200 efficiency: J/batching,
sharding, measured GPU utilization from gpu_log.sh.

## Risks

- Asset names/tokens: release asset `in-scatter-5m.pt` vs config `inv-scatter-5m.pt`; CaltechDATA
  share token may expire → download script queries listings and tries both names.
- **Val LMDB existence**: configs ship only `inv-scatter-test` (100); confirm a separate
  `inv-scatter-val` (10) downloadable asset exists before relying on it for per-R tuning.
- **numTrans**: config says 20; `construct_parameters` defaults to 60. Confirm 20 matches the
  InverseBench Table-3 setup or the comparison shifts.
- **A_inv memory**: `pinv(A)` at R=360 adds ~1.9 GB on top of A/U/V → budget ~6-8 GB (not ~4).
- SVD cache: computed at operator `__init__` (race for parallel shards; `torch.load` without
  map_location) → mandatory precompute step + completeness gating.
- `forward()` silently batch-1 → all batched math goes through the SVD factors; aggregator loops samples.
- Hydra relative searchpath / CWD soup → pinned by test_configs.py + scripts `cd` to inverse_scattering/.
- torch ≥ 2.6 `weights_only` default breaks the pickled-`ema` fallback → handled in our main.py copy.
