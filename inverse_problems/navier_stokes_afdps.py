"""AFDPS-compatible forward operator for the 2D Navier-Stokes inverse problem.

`AFDPSNavierStokes2d` subclasses InverseBench's `ForwardNavierStokes2d` (so the
forward map, forcing and normalization are identical -- observation generation and
all baselines remain byte-for-byte unchanged) and *adds* the operator API the
AFDPS ensemble sampler calls:

    initialize_ensemble(gt, J)              -> (J, 1, res, res)
    proximal_generator(x_ref, y, sigma, scale) -> perturbed particles
    likelihood_gradient(x, y, sigma)        -> (J, 1, res, res)   (= + grad mu_y)
    likelihood_laplacian(x, sigma)          -> (J,)               (Hutchinson Tr Hessian)

Domains. The sampler/prior live in the *normalized* domain x; the physical solver
consumes omega0 = unnorm_scale * x. The gradient engine (`ns_adjoint`) works in
physical units, so `likelihood_gradient` applies the chain-rule factor `unnorm_scale`
exactly once. The observation `y` is in physical units (InverseBench's `forward`
unnormalizes internally and never re-normalizes), so the residual and P* are physical.
"""
import math
import torch

from .navier_stokes import ForwardNavierStokes2d
from . import ns_adjoint as A


class AFDPSNavierStokes2d(ForwardNavierStokes2d):
    def __init__(self,
                 hutchinson_M=1,
                 hutchinson_eps=1e-3,
                 hutchinson_scheme='central', # 'central' (O(eps^2), 2M solves) | 'forward' (O(eps), M solves, reuses g0)
                 grad_chunk=16,               # particles per adjoint sub-solve; raise on GPU to fill the device
                 sigma_floor=1e-3,
                 adjoint_mode='autodiff',     # 'autodiff' (discrete adjoint) | 'continuous'
                 init_mode='zeros',           # 'zeros' | 'Pstar_y'
                 grad_smooth_M=1,             # randomized smoothing: average the adjoint over M perturbed solves (1 = off)
                 grad_smooth_eps=0.0,         # perturbation std (physical units) for randomized smoothing
                 grad_lowpass_frac=None,      # spectral low-pass cutoff as a fraction of Nyquist (None = off)
                 pigdm_chunk=8,               # particles per CG sub-solve for guidance_mode='full' (2nd-order graph -> keep small)
                 **kwargs):
        # Force fixed dt: a fixed-length differentiable trajectory is required for
        # the adjoint. Default to non-adaptive even if the config forgets.
        kwargs.setdefault('adaptive', False)
        super().__init__(**kwargs)
        self.hutchinson_M = hutchinson_M
        self.hutchinson_eps = hutchinson_eps
        self.hutchinson_scheme = hutchinson_scheme
        self.grad_chunk = grad_chunk
        self.sigma_floor = sigma_floor
        self.adjoint_mode = adjoint_mode
        self.init_mode = init_mode
        # gradient variance reduction (the chaotic Re=200 adjoint is spiky; both default off)
        self.grad_smooth_M = grad_smooth_M
        self.grad_smooth_eps = grad_smooth_eps
        self.grad_lowpass_frac = grad_lowpass_frac
        self._lowpass_mask = None
        self.pigdm_chunk = pigdm_chunk  # particles per CG sub-solve (full PiGDM; 2nd-order graph -> keep small)
        # observation stashed by the algorithm before sampling (laplacian gets no y arg)
        self._y = None

    # ----- measurement operator -----
    def P(self, full):
        return A.downsample(full, self.downsample_factor)

    def Pstar(self, obs):
        return A.upsample_adjoint(obs, self.downsample_factor, self.solver.s1, self.solver.s2)

    # ----- gradient of the negative log-likelihood (physical -> normalized) -----
    def _phys_grad(self, omega0_phys, y_phys, sigma_eff):
        """d mu_y / d omega0 (physical), shape (B, s1, s2)."""
        if self.adjoint_mode == 'continuous':
            return A.continuous_adjoint_gradient(
                self.solver, omega0_phys, self.force, self.forward_time, self.Re,
                self.delta_t, y_phys, self.downsample_factor, sigma_eff)
        return A.grad_misfit_autograd(
            self.solver, omega0_phys, self.force, self.forward_time, self.Re,
            self.delta_t, y_phys, self.downsample_factor, sigma_eff)

    def likelihood_gradient(self, x, y, sigma):
        """grad_x mu_y for normalized x. Returns (B, 1, res, res). Matches the
        reference sign convention (+grad of the data misfit).

        Variance reduction for the chaotic Re=200 adjoint (both opt-in, default off):
          * randomized smoothing -- average the gradient over grad_smooth_M solves at
            x + N(0, grad_smooth_eps^2), i.e. grad of a Gaussian-smoothed misfit. Tames
            the intermittent gradient spikes at the cost of M extra adjoint solves.
          * spectral low-pass -- zero the gradient's high-wavenumber modes (above
            grad_lowpass_frac * Nyquist), which on a chaotic flow are dominated by
            untrustworthy noise; the recoverable signal lives at low/mid k.
        """
        sigma_eff = max(float(sigma), self.sigma_floor)
        y_phys = y.squeeze(1)                       # (1, res/f, res/f), physical units
        M = max(1, int(self.grad_smooth_M))
        g = torch.zeros_like(x)
        for m in range(M):
            xm = x if (M == 1 or self.grad_smooth_eps <= 0) else x + self.grad_smooth_eps * torch.randn_like(x)
            gm = torch.empty_like(x)
            for sl in torch.split(torch.arange(x.shape[0]), self.grad_chunk):
                chunk = xm[sl]
                w0 = self.unnormalize(chunk).squeeze(1)  # physical vorticity (b, s1, s2)
                g_phys = self._phys_grad(w0, y_phys, sigma_eff)
                gm[sl] = (self.unnorm_scale * g_phys).unsqueeze(1)   # chain rule x -> omega0
            g = g + gm
        g = g / M
        if self.grad_lowpass_frac is not None:
            g = self._lowpass(g)
        return g

    def _lowpass(self, g):
        """Zero spatial-frequency modes above grad_lowpass_frac * Nyquist (rfft2 grid)."""
        if self._lowpass_mask is None:
            s1, s2 = self.solver.s1, self.solver.s2
            k1 = torch.fft.fftfreq(s1, d=1.0 / s1, device=g.device).abs().view(-1, 1)
            k2 = torch.fft.rfftfreq(s2, d=1.0 / s2, device=g.device).abs().view(1, -1)
            cut = self.grad_lowpass_frac * (min(s1, s2) / 2.0)
            self._lowpass_mask = ((k1 <= cut) & (k2 <= cut)).to(g.dtype)
        gh = torch.fft.rfft2(g.squeeze(1))
        out = torch.fft.irfft2(gh * self._lowpass_mask, s=(self.solver.s1, self.solver.s2))
        return out.unsqueeze(1).to(g.dtype)

    # ----- negative-log-likelihood VALUE (Feynman-Kac potential, from the TMLR refinement) -----
    def likelihood_value(self, x, y, sigma):
        """mu_y(x) = ||P L(unnorm_scale x) - y||^2 / (2 sigma^2) per particle. Returns (B,).
        Used in the Feynman-Kac log-weight so particles are scored by their actual data
        misfit level, not only the gradient/curvature terms."""
        sigma_eff = max(float(sigma), self.sigma_floor)
        y_phys = y.squeeze(1)
        out = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        for sl in torch.split(torch.arange(x.shape[0]), self.grad_chunk):
            w0 = self.unnormalize(x[sl]).squeeze(1)
            wT = A.forward_solve(self.solver, w0, self.force, self.forward_time, self.Re, self.delta_t)
            r = A.downsample(wT, self.downsample_factor) - y_phys
            out[sl] = 0.5 * r.pow(2).flatten(start_dim=1).sum(dim=1) / (sigma_eff ** 2)
        return out

    # ----- mean eigenvalue of J J^T (J-aware / isotropic-PiGDM guidance) -----
    def jacobian_trace(self, x, trace_M=1):
        """Ensemble-mean lambda_bar = tr(J J^T)/n_meas for J = d(P L(unnorm_scale x))/dx,
        in NORMALIZED-x units (so it is directly comparable to gamma^2: the guidance
        denominator becomes sigma_y^2 + sigma(t)^2 * lambda_bar). The chain rule x -> omega0
        contributes a factor unnorm_scale^2 on top of the physical trace. Returns a Python
        float. One forward solve + trace_M VJP backprops per chunk (chunked like the
        gradient to bound memory)."""
        total, count = 0.0, 0
        for sl in torch.split(torch.arange(x.shape[0]), self.grad_chunk):
            w0 = self.unnormalize(x[sl]).squeeze(1)              # physical vorticity
            lam = A.jac_mean_eig(self.solver, w0, self.force, self.forward_time,
                                 self.Re, self.delta_t, self.downsample_factor, M=trace_M)
            total += ((self.unnorm_scale ** 2) * lam).sum().item()
            count += w0.shape[0]
        return total / max(count, 1)

    # ----- full anisotropic PiGDM guidance (the principled Bayesian guidance, no gamma) -----
    def likelihood_gradient_pigdm(self, x, y, sigma_y, sigma_t, cg_iters=5, pigdm_scale=1.0, return_diag=False):
        """PiGDM guidance for normalized x: g = scale * J^T (sigma_y^2 I + sigma_t^2 J J^T)^{-1}(P L - y),
        solved matrix-free by CG in measurement space. Returns (B,1,res,res), same +grad sign /
        units as likelihood_gradient (x->omega0 chain-rule via unnorm_scale applied once).
        sigma_t = diffusion noise level (normalized-x units); the J J^T weight is propagated to
        physical units by unnorm_scale^2. There is NO guidance_gamma here: the guidance is fully
        determined by the known measurement noise sigma_y and the schedule sigma_t (gamma == 1).
        pigdm_scale is a SINGLE GLOBAL magnitude calibration (the PiGDM gradient is the matrix-inverse
        solution, whose scale differs from the fixed-mode 1/r_t^2 gradient the SDE drift coefficient
        2 s^2 sigma' sigma was tuned for; default 1.0). Chunked by pigdm_chunk (CG holds a 2nd-order graph).
        With return_diag, also returns the mean CG relative residual for convergence monitoring."""
        sy2 = max(float(sigma_y), self.sigma_floor) ** 2
        jjt = (float(sigma_t) * self.unnorm_scale) ** 2
        y_phys = y.squeeze(1)
        g = torch.empty_like(x)
        rels = []
        chunk = max(1, int(self.pigdm_chunk))
        for sl in torch.split(torch.arange(x.shape[0]), chunk):
            w0 = self.unnormalize(x[sl]).squeeze(1)
            gp, rel = A.pigdm_gradient(self.solver, w0, self.force, self.forward_time, self.Re,
                                       self.delta_t, y_phys, self.downsample_factor, sy2, jjt,
                                       cg_iters=cg_iters)
            g[sl] = (float(pigdm_scale) * self.unnorm_scale * gp).unsqueeze(1)   # chain rule x -> omega0
            rels.append(rel)
        if return_diag:
            return g, (sum(rels) / max(len(rels), 1))
        return g

    # ----- Hutchinson trace estimate of the likelihood Laplacian -----
    def likelihood_laplacian(self, x, sigma, g0=None):
        """Tr(Hessian_x mu_y) per particle via batched Hutchinson estimation.
        Returns (B,). Uses the stashed observation self._y. All M probes are pushed
        through likelihood_gradient in one (internally chunked) call. With
        hutchinson_scheme='forward', reuses the precomputed drift gradient g0 to halve
        the per-step solve count (O(eps) vs the central scheme's O(eps^2))."""
        assert self._y is not None, "set operator._y to the observation before sampling"
        y = self._y

        def grad_fn(xx):
            return self.likelihood_gradient(xx, y, sigma)

        return A.hutchinson_laplacian_batched(
            grad_fn, x, self.hutchinson_M, self.hutchinson_eps,
            scheme=self.hutchinson_scheme, g0=g0)

    # ----- ensemble initialization & top-noise perturbation -----
    def initialize_ensemble(self, gt, num_particles):
        shape = (num_particles, 1, self.solver.s1, self.solver.s2)
        # Warm start (improvement #2): seed from a pre-computed estimate (e.g. the EnKG
        # ensemble mean) stashed on self._warm_start by the hybrid algo. Normalized units.
        if self.init_mode == 'warm' and getattr(self, '_warm_start', None) is not None:
            ws = self._warm_start
            if ws.dim() == 3:
                ws = ws.unsqueeze(0)
            if ws.shape[0] == 1:
                ws = ws.repeat(num_particles, 1, 1, 1)
            return ws.to(self.device, torch.float32)
        if self.init_mode == 'Pstar_y' and self._y is not None:
            base = self.normalize(self.Pstar(self._y.squeeze(1)).unsqueeze(1))
            return base.repeat(num_particles, 1, 1, 1)
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def proximal_generator(self, x_ref, y, sigma, scale):
        # Closed-form prox is impossible for the nonlinear NS forward; seed particles
        # at the EDM prior of the top noise level: x_ref + scale * N(0, I).
        return x_ref + scale * torch.randn_like(x_ref)
