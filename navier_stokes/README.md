# AFDPS for the Navier–Stokes Inverse Problem

Applies **AFDPS** — *Approximation-Free Diffusion Posterior Sampling* (Chen, Ren, Min, Ying, Izzo, TMLR 2026) — to the **2D Navier–Stokes inverse problem** from the [InverseBench](https://github.com/devzhk/InverseBench) benchmark: recover the initial vorticity field `ω₀` from a noisy, spatially-subsampled observation of the final-time vorticity `ω(T)`.

This repo is a fork of InverseBench (its Hydra harness, dataset loaders, evaluator, and baselines are kept so results are directly comparable) with the AFDPS sampler vendored in and a new **adjoint-state gradient engine** added.

## Why this is a contribution

InverseBench treats the NS forward map as a **black box** — "automatic differentiation through the numerical solver is challenging due to the extensive computation graph," so its diffusion-posterior methods use *zeroth-order smoothed gradients* (DPS-fGSG/cGSG → **1.2–2.2** rel-ℓ2, worse than the mean field). The best baselines are derivative-free ensemble methods: **EnKG ≈ 0.12**, **DPG ≈ 0.32** rel-ℓ2 (×2 subsampling, σ=0).

AFDPS needs exactly what InverseBench lacked: (1) an accurate **likelihood gradient** `∇μ_y`, and (2) the **log-likelihood Laplacian** `Δμ_y = Tr(∇²μ_y)` for its Feynman–Kac particle reweighting. We supply both:

- **`∇μ_y` via the discrete adjoint** — the InverseBench pseudo-spectral solver (`NavierStokes2d.solve`) is built from differentiable `torch.fft` ops; with **fixed Δt** (`adaptive=False`) it has a fixed-length graph, so reverse-mode autograd through it *is* the exact discrete adjoint (including the correct `rfft2` adjoint). A hand-coded **continuous adjoint** PDE solver (mentor's eqs 32–45) is included as an O(Δt) cross-check.
- **`Δμ_y` via Hutchinson trace estimation** — `(1/M)Σ ⟨∇μ_y(ω₀+εξ)−∇μ_y(ω₀−εξ), ξ⟩/(2ε)`, reusing the gradient routine (the *estimated* second-order term the mentor's note calls for; it has no closed form for a nonlinear forward).

## Layout (new/changed files)

```
algo/afdps.py                              AFDPS(Algo): wires net + NS adapter into the sampler; prior switch
algo/afdps_core/ensemble_denoiser_edm.py   vendored+cleaned AFDPS SMC sampler (Feynman-Kac reweighting)
inverse_problems/navier_stokes_afdps.py    AFDPSNavierStokes2d: operator API on the verified gradient engine
inverse_problems/ns_adjoint.py             gradient engine: diff. forward, discrete+continuous adjoint, P*,
                                           Hutchinson Laplacian, analytic GRF prior
configs/problem/navier-stokes-afdps.yaml   forward op (adjoint-enabled, adaptive=False)
configs/algorithm/afdps.yaml               AFDPS hyperparameters
configs/sweep/navier-stokes/afdps.yaml     wandb sweep (particles / steps / guidance_gamma / M)
verification/                              Track A harness + numerical verification ladder
```

## Install

```bash
# Clone (public repo) -- works on any machine, including the GPU box:
git clone https://github.com/erichou1/AFDPS.git
cd AFDPS

# CPU dev / verification (Track A + ladder run fine on CPU):
python -m venv .venv && source .venv/bin/activate
pip install torch numpy pyyaml tqdm omegaconf hydra-core lmdb piq requests
# Full env (GPU box, Track B): `uv sync` (this repo's pyproject).
# wandb is OPTIONAL -- only for the Step-8 hyperparameter sweep (`pip install wandb`).
# Inference/verification need neither wandb nor an account (wandb=false is the default).
```

See **SETUP_GUIDE.pdf** for the full step-by-step (clone -> env -> download assets -> verify -> run -> sweep).

## Track A — numerical verification (no GPU, no checkpoint)

The whole value of the gradient engine is that it's *correct*. Run the ladder (float64, small grids):

```bash
python verification/checks.py        # 15 checks; expect "15/15 checks passed."
```

It verifies (with pass tolerances): `irfft2∘rfft2=I` and the `P/P*` adjoint (exact); forward enstrophy decay, mean conservation, 2nd-order Δt convergence; **finite-difference gradient check** (autograd ∇μ vs central FD → ~1e-10; Taylor-remainder slope ≈2); continuous-adjoint O(Δt) agreement with the discrete adjoint; **Hutchinson Laplacian vs brute-force Hessian trace** — for the original loop, the **batched** estimator, and the **forward-difference** scheme (Hessian symmetric, traces within tolerance); GRF prior score vs autograd of `−½⟨ω,C⁻¹ω⟩` (exact) and DC-mode consistency.

End-to-end AFDPS on a synthetic problem with the **correctly-specified** analytic Gaussian prior (any failure here is the adjoint/sign/normalization, not the prior):

```bash
python verification/run_track_a.py --res 32 --steps 150 --particles 8
# -> AFDPS rel-L2(recon, true) ~ 0.33 (vs prior-mean baseline 1.0); misfit -> noise floor.
```

## Track B — the InverseBench benchmark (GPU + checkpoint)

1. Download assets (see `scripts/download_assets.sh`): the prior `checkpoints/ns-5m.pt` and the NS LMDB data into a sibling `../data/`.
2. Validate the code path with a mock net (no checkpoint needed):
   ```bash
   python verification/smoke_track_b.py    # -> PASS (config wiring + diffusion-prior path execute)
   ```
3. First real run:
   ```bash
   python main.py problem=navier-stokes-afdps algorithm=afdps pretrain=navier-stokes \
       num_samples=1 wandb=false
   ```
4. Tune + sweep (the **`guidance_gamma`** annealed-guidance strength is the key knob — it must be scaled to the data; the sweep covers it), then run the full grid by overriding `problem.model.downsample_factor ∈ {2,4,8}` and `problem.model.sigma_noise ∈ {0,1,2}`. **Target: beat EnKG (≈0.12) / DPG (≈0.32).**

## Key design notes (and the bugs the verification caught)

- **Likelihood at the denoised estimate.** The reference sampler evaluates the likelihood at the *noisy* particle (fine for linear image operators). Running the NS solver from a noise-dominated field violates CFL and diverges, so for the nonlinear forward we evaluate `∇μ_y`/`Δμ_y` at the Tweedie estimate `D_x` (`likelihood_at='denoised'`), which is smooth and shrinks to ≈0 at high noise.
- **Annealed guidance.** `r_t = √(σ_y² + (γ·σ(t))²)` tempers the stiff `1/σ_y²` guidance at high diffusion noise and recovers the true measurement noise as σ(t)→0. `guidance_gamma` (γ) is tuned per data scale.
- **Normalization chain rule.** The sampler/prior live in the normalized domain `x`; the solver consumes `ω₀ = unnorm_scale·x`. `likelihood_gradient` applies the `unnorm_scale` factor exactly once after the physical-domain adjoint.
- **`adaptive=False` mandatory** for a fixed-length differentiable trajectory; **σ floored** to avoid `1/σ²` blow-up at the σ=0 grid cell; **NaN/Inf guards** quarantine a diverged particle so it can't poison the ensemble.
- **Sweep efficiency knobs** (the per-step cost is `(1+2M)·particles` forward+adjoint solves): the Hutchinson probes are **batched** into one chunked solve; `hutchinson_scheme: forward` reuses the drift gradient to cut `2M→M` solves (O(ε) vs O(ε²) — fine since the Laplacian only enters the FK weight); raise `grad_chunk` to fill the GPU.
- **Open items for the GPU run:** confirm `delta_t=0.002` is sub-CFL at the full 128² resolution (the dataset was generated with adaptive stepping); reduce if the forward diverges. The sweep keys on `"relative l2"` (the only metric logged on the AFDPS path — *not* `data_fitting_loss`, which only the baselines log).

See `README_InverseBench.md` for the upstream benchmark docs, and `../AFDPS_PDE_Inverse_Problem (3).pdf` for the full mathematical recipe.
