"""AFDPS-compatible forward operator for InverseBench full waveform inversion (FWI).

`AFDPSAcousticWave` subclasses InverseBench's Devito `AcousticWave`
(`inverse_problems.acoustic.AcousticWave`, imported from the reused `navier_stokes/`
harness) so the forward wave-equation model, the source/receiver geometry, the
adjoint-state gradient and the normalization are identical to the benchmark -- the
observation, the dataset and every other method's evaluation stay byte-for-byte
unchanged. It *adds* the operator API the AFDPS ensemble sampler calls:

    initialize_ensemble(gt, J)                 -> (J, 1, H, W)
    proximal_generator(x_ref, y, sigma, scale) -> perturbed particles
    likelihood_gradient(x, y, sigma)           -> (J, 1, H, W)   (= + grad mu_y)
    likelihood_laplacian(x, sigma, g0=None)    -> (J,)           (Tr Hessian estimate)
    likelihood_value(x, y, sigma)              -> (J,)           (FK potential mu_y)

================================  FWI formulation  ============================
Unknown: the compressional velocity field. The AFDPS sampler / EDM prior live in the
*normalized* domain x (CurveFaultB normalized to ~[-1.5, 1.5]); the Devito solver
consumes the physical velocity v = unnormalize(x) = (x + unnorm_shift) * unnorm_scale
in km/s. With the paper's negative-log-likelihood convention and Gaussian/quadratic
misfit,

    r(x) = A(x) - y ,        mu_y(x) = (1 / 2 sigma_y^2) || A(x) - y ||_2^2 .

The likelihood gradient is the adjoint-state gradient already implemented by the parent
operator (Devito back-propagates the data residual, cross-correlates with the forward
wavefield to get d/d(slowness), then the parent chain-rules slowness m = 1/v^2 ->
velocity -> normalized x). Hence

    grad_x mu_y(x) = (1 / sigma_y^2) * AcousticWave.gradient(x, y) = (1 / sigma_y^2) J^T r.

FWI is NONLINEAR (no closed-form / no SVD), so there is no cheap exact Hessian trace.
The Laplacian Tr(grad^2_x mu_y) is therefore estimated (see `likelihood_laplacian`).

==============================  Noise-free benchmark  ========================
The InverseBench FWI measurement is NOISE-FREE (Table 2). `sigma_y` (carried as
`sigma_noise`) is therefore NOT a physical noise level but a likelihood *temperature* /
regularization knob -- the dominant FWI hyperparameter (paper search range [1e-2, 1e1]).
`sigma_floor` guards the 1/sigma_y^2 factor; the sampler additionally tempers it with the
annealed effective std r_t = sqrt(sigma_y^2 + (gamma sigma(t))^2).

==============================  CFL stability guard  =========================
The Devito wave solver diverges if its velocity input violates the CFL condition
(InverseBench Fig. 3: this is exactly how the noise-injecting DAPS / PnP-DM fail). AFDPS
is also an SDE, so two guards are applied: (1) the sampler evaluates the likelihood at the
smooth Tweedie estimate x_hat (`likelihood_at='denoised'`); (2) every velocity handed to
Devito is clamped to the physical range [vel_min_kms, vel_max_kms]. A particle whose solve
still fails has its gradient/value zeroed/penalized so it can never poison the ensemble.
"""
import numpy as np
import torch

from inverse_problems.acoustic import AcousticWave  # reused navier_stokes/ harness (on sys.path)


class AFDPSAcousticWave(AcousticWave):
    def __init__(self,
                 # ---- AFDPS-specific knobs (everything else is forwarded to AcousticWave) ----
                 laplacian_mode='fd_divergence',  # 'fd_divergence' | 'gn_hutchinson' | 'zero'
                 hutchinson_M=1,                   # # stochastic trace probes
                 hutchinson_eps=1e-3,              # finite-difference step (normalized units)
                 hutchinson_scheme='forward',      # 'forward' (reuse drift g0, 1 solve/probe) | 'central' (2 solves/probe)
                 grad_chunk=1,                     # particles per gradient pass (Devito adjoint is per-particle)
                 sigma_floor=1e-3,                 # floor for the likelihood temperature sigma_y
                 vel_min_kms=1.5,                  # CFL-safe physical velocity clamp (km/s)
                 vel_max_kms=4.5,
                 init_mode='zeros',                # 'zeros' (constant background) | 'mean'
                 **kwargs):
        # capture the velocity-grid shape before it is consumed by the parent constructor
        self._res = tuple(kwargs.get('shape', (128, 128)))
        super().__init__(**kwargs)
        self.laplacian_mode = laplacian_mode
        self.hutchinson_M = int(hutchinson_M)
        self.hutchinson_eps = float(hutchinson_eps)
        self.hutchinson_scheme = hutchinson_scheme
        self.grad_chunk = max(1, int(grad_chunk))
        self.sigma_floor = float(sigma_floor)
        self.vel_min_kms = float(vel_min_kms)
        self.vel_max_kms = float(vel_max_kms)
        self.init_mode = init_mode
        # Normalized-domain clamp bounds equivalent to the physical km/s range:
        #   v = (x + unnorm_shift) * unnorm_scale  =>  x = v / unnorm_scale - unnorm_shift.
        self._x_min = self.vel_min_kms / self.unnorm_scale - self.unnorm_shift
        self._x_max = self.vel_max_kms / self.unnorm_scale - self.unnorm_shift
        # Observation stashed by the algorithm before sampling (the Devito Receiver list);
        # likelihood_laplacian / jacobian_trace read it (they get no explicit y argument).
        self._y = None

    # ------------------------------------------------------------------ #
    # CFL-safe clamp                                                      #
    # ------------------------------------------------------------------ #
    def _clamp_norm(self, x):
        """Clamp normalized velocity x so the physical velocity stays in the CFL-safe
        [vel_min_kms, vel_max_kms] range before any Devito solve."""
        return x.clamp(self._x_min, self._x_max)

    # ------------------------------------------------------------------ #
    # Negative-log-likelihood gradient  grad_x mu_y = (1/sigma^2) J^T r   #
    # ------------------------------------------------------------------ #
    def likelihood_gradient(self, x, y, sigma):
        """+grad of the data misfit for normalized x. Returns (J, 1, H, W).

        Uses the parent operator's adjoint-state gradient (one Devito multi-shot adjoint
        per particle; the 16 shots run in parallel on the dask cluster). The velocity is
        CFL-clamped first. A particle whose solve raises (e.g. residual NaN) contributes a
        zero gradient instead of aborting the whole ensemble."""
        sigma_eff = max(float(sigma), self.sigma_floor)
        inv = 1.0 / (sigma_eff ** 2)
        xc = self._clamp_norm(x)
        g = torch.zeros_like(x)
        for j in range(x.shape[0]):
            try:
                gj = self.gradient(xc[j:j + 1], y, unnormalize=True)  # (1,1,H,W) = J^T r in x-units
                g[j:j + 1] = inv * gj
            except Exception as e:  # CFL blow-up / NaN residual on this particle only
                print(f"[AFDPSAcousticWave] likelihood_gradient: particle {j} solve failed "
                      f"({type(e).__name__}: {e}); zeroing its gradient.", flush=True)
        return g

    # ------------------------------------------------------------------ #
    # Negative-log-likelihood VALUE  mu_y = (1/2 sigma^2) ||A(x)-y||^2    #
    # ------------------------------------------------------------------ #
    def likelihood_value(self, x, y, sigma):
        """mu_y(x) per particle (the Feynman-Kac potential). Returns (J,).

        Reuses the parent's 0.5||A(x)-y||^2 loss (so mu_y = loss / sigma^2). Per-particle
        loop with isolation: a failed solve yields +inf (the sampler down-weights it)."""
        sigma_eff = max(float(sigma), self.sigma_floor)
        inv = 1.0 / (sigma_eff ** 2)
        xc = self._clamp_norm(x)
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        for j in range(x.shape[0]):
            try:
                lj = self.loss(xc[j:j + 1], y, unnormalize=True)  # (1,) = 0.5||A-y||^2
                out[j] = inv * lj.reshape(()).to(torch.float32)
            except Exception:
                out[j] = float('inf')
        return out

    # ------------------------------------------------------------------ #
    # Laplacian Tr(grad^2_x mu_y)                                         #
    # ------------------------------------------------------------------ #
    def likelihood_laplacian(self, x, sigma, g0=None):
        """Estimate Tr(Hessian_x mu_y) per particle. Returns (J,).

        Modes (config `laplacian_mode`):
          'fd_divergence' (default): EXACT full Hessian trace via Hutchinson finite-difference
              divergence of the misfit gradient,
                  Tr(grad^2 mu_y) ~ E_v[ v^T (grad mu_y(x+eps v) - grad mu_y(x-eps v)) / (2 eps) ],
              with v Rademacher. Reuses `likelihood_gradient` only (NO new Devito code), so it
              is the same estimator the Navier-Stokes AFDPS port validated. `scheme='forward'`
              reuses the drift gradient g0 (1 solve/probe instead of 2).
          'gn_hutchinson': Gauss-Newton approximation Tr ~ (1/sigma^2) ||J||_F^2 via data-space
              adjoint probes (PSD -> very stable FK weights; cheaper -- 1 forward + M adjoints).
              Drops the residual-curvature term (vanishes as r -> 0). Touches Devito internals
              (see `devito_adjoint.py`); validate on-device before trusting at scale.
          'zero': drop the curvature term entirely (ablation; many guided-diffusion methods do).
        """
        if self.laplacian_mode == 'zero':
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        if self.laplacian_mode == 'gn_hutchinson':
            return self._gn_laplacian(x, sigma)
        # default: finite-difference divergence (exact trace)
        grad_fn = lambda xx: self.likelihood_gradient(xx, self._y, sigma)
        return self._hutchinson_fd(grad_fn, x, self.hutchinson_M, self.hutchinson_eps,
                                   scheme=self.hutchinson_scheme, g0=g0)

    @staticmethod
    def _hutchinson_fd(grad_fn, x, M, eps, scheme='forward', g0=None):
        """Vectorized Hutchinson finite-difference trace estimate, per sample. (J,).

        All M probes are stacked onto the batch dim and pushed through `grad_fn` in one call
        (grad_fn loops/chunks internally). 'central': 2M grad evals, O(eps^2). 'forward': M
        grad evals reusing g0 = grad_fn(x), O(eps). Mathematically identical to the
        Navier-Stokes port's `hutchinson_laplacian_batched`."""
        xi = (torch.randint(0, 2, (M, *x.shape), device=x.device).to(x.dtype) * 2 - 1)
        if scheme == 'forward':
            assert g0 is not None, "scheme='forward' requires g0 = grad_fn(x)"
            Xp = (x.unsqueeze(0) + eps * xi).reshape(M * x.shape[0], *x.shape[1:])
            gp = grad_fn(Xp).reshape(M, *x.shape)
            hvp = (gp - g0.unsqueeze(0)) / eps
        else:
            assert scheme == 'central'
            Xp = (x.unsqueeze(0) + eps * xi)
            Xm = (x.unsqueeze(0) - eps * xi)
            X = torch.cat([Xp, Xm], dim=0).reshape(2 * M * x.shape[0], *x.shape[1:])
            g = grad_fn(X).reshape(2, M, *x.shape)
            hvp = (g[0] - g[1]) / (2 * eps)
        return (hvp * xi).flatten(start_dim=2).sum(dim=2).mean(dim=0)  # (M,J) -> (J,)

    def _gn_laplacian(self, x, sigma):
        """Gauss-Newton Tr ~ (1/sigma^2) ||J||_F^2 via data-space Hutchinson adjoint probes.
        ||J||_F^2 = Tr(J J^T) = E_w ||J^T w||^2, w ~ N(0, I) in receiver/data space; each
        J^T w is one Devito adjoint (probe injected as a synthetic residual). Best-effort
        Devito path -- validate on-device."""
        from .devito_adjoint import jtw_norm_sq  # lazy: only import Devito-internals when used
        sigma_eff = max(float(sigma), self.sigma_floor)
        inv = 1.0 / (sigma_eff ** 2)
        xc = self._clamp_norm(x)
        out = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        for j in range(x.shape[0]):
            try:
                fro2 = jtw_norm_sq(self, xc[j:j + 1], self._y, self.hutchinson_M)  # mean_w ||J^T w||^2
                out[j] = inv * float(fro2)
            except Exception as e:
                print(f"[AFDPSAcousticWave] _gn_laplacian: particle {j} failed "
                      f"({type(e).__name__}: {e}); using 0.", flush=True)
        return out

    # ------------------------------------------------------------------ #
    # Mean Jacobian eigenvalue lambda_bar = tr(J J^T)/n_meas (guidance='auto')   #
    # ------------------------------------------------------------------ #
    def jacobian_trace(self, x, trace_M=1):
        """Ensemble-mean tr(J J^T)/n_meas in normalized-x units (for the sampler's
        J-aware 'auto' guidance). Best-effort -- reuses the `gn_hutchinson` adjoint path."""
        from .devito_adjoint import jtw_norm_sq
        n_meas = self.nshots * self.num_time_steps * self.nreceivers
        xc = self._clamp_norm(x)
        total, count = 0.0, 0
        for j in range(x.shape[0]):
            total += float(jtw_norm_sq(self, xc[j:j + 1], self._y, trace_M)) / n_meas
            count += 1
        return total / max(count, 1)

    def likelihood_gradient_pigdm(self, *args, **kwargs):
        raise NotImplementedError(
            "guidance_mode='full' (anisotropic PiGDM) is not supported for the Devito FWI "
            "operator: the matrix-free measurement-space CG would need J/J^T products through "
            "the wave solver. Use guidance_mode='fixed' (default) or 'auto'.")

    # ------------------------------------------------------------------ #
    # Ensemble initialization & top-noise perturbation                   #
    # ------------------------------------------------------------------ #
    def initialize_ensemble(self, gt, num_particles):
        """Seed the ensemble in normalized units. Default 'zeros' = a constant background
        velocity (x=0 -> v = unnorm_shift * unnorm_scale km/s, the dataset mean). This matches
        the *un-daggered* PnP-diffusion baselines (DPS / DiffPIR / PnP-DM / REDDiff), which
        start from no model knowledge and lean on the prior -- the fair comparison for AFDPS.
        (The daggered Adam/LBFGS/DAPS baselines instead start from a blurred ground truth; that
        would require plumbing the target into the operator and is intentionally not the
        default, to keep AFDPS in the prior-only regime.)"""
        shape = (num_particles, 1, self._res[0], self._res[1])
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def proximal_generator(self, x_ref, y, sigma, scale):
        # A closed-form proximal map is impossible for the nonlinear wave-equation forward,
        # so we seed particles at the EDM prior's top noise level: x_ref + scale * N(0, I).
        return x_ref + scale * torch.randn_like(x_ref)
