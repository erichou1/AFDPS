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
        reference sign convention (+grad of the data misfit)."""
        sigma_eff = max(float(sigma), self.sigma_floor)
        y_phys = y.squeeze(1)                       # (1, res/f, res/f), physical units
        g = torch.empty_like(x)
        for sl in torch.split(torch.arange(x.shape[0]), self.grad_chunk):
            chunk = x[sl]
            w0 = self.unnormalize(chunk).squeeze(1)  # physical vorticity (b, s1, s2)
            g_phys = self._phys_grad(w0, y_phys, sigma_eff)
            g[sl] = (self.unnorm_scale * g_phys).unsqueeze(1)   # chain rule x -> omega0
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
        if self.init_mode == 'Pstar_y' and self._y is not None:
            base = self.normalize(self.Pstar(self._y.squeeze(1)).unsqueeze(1))
            return base.repeat(num_particles, 1, 1, 1)
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def proximal_generator(self, x_ref, y, sigma, scale):
        # Closed-form prox is impossible for the nonlinear NS forward; seed particles
        # at the EDM prior of the top noise level: x_ref + scale * N(0, I).
        return x_ref + scale * torch.randn_like(x_ref)
