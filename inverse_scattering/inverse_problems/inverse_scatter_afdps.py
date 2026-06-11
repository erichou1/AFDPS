"""AFDPS-compatible forward operator for the LINEAR inverse scattering problem.

`AFDPSInverseScatter` subclasses InverseBench's `InverseScatter` (so the forward
map, Green's functions, noise model and normalization are identical -- observation
generation and every baseline remain byte-for-byte unchanged) and *adds* the
operator API the AFDPS ensemble sampler calls:

    set_observation(observation)                 -> caches the folded data terms
    initialize_ensemble(gt, J)                   -> (J, 1, Ny, Nx)
    proximal_generator(x_ref, y, sigma, scale)   -> perturbed particles
    likelihood_value(x, y, sigma)                -> (J,)   mu_y per particle
    likelihood_gradient(x, y, sigma)             -> (J, 1, Ny, Nx)  (= +grad mu_y)
    likelihood_laplacian(x, sigma, g0=None)      -> (J,)   exact Tr(Hessian)
    jacobian_trace(x, trace_M=1)                 -> float  exact lambda_bar
    likelihood_gradient_pigdm(...)               -> (J,1,Ny,Nx)  exact anisotropic PiGDM
    exact_linear_substep(x, t_cur, t_next, ...)  -> x_new  exact guidance integration

Why scattering is EASIER than the Navier-Stokes port. NS has a *nonlinear* PDE
forward, so its gradient needs a discrete-adjoint solve and its log-likelihood
Laplacian needs a Hutchinson estimate. Scattering's forward is *linear* in the
permittivity (first Born approximation), and the base operator already caches a
real SVD of the forward matrix  A = U diag(S) V^T  (`self.U, self.Sigma, self.V_t`).
So EVERYTHING here is exact and closed-form, with no autograd, no PDE solve, and no
stochastic trace estimate:
    grad mu_y  = A^T (A x - y~) / sigma^2                       (one V/U projection)
    Tr(Hess)   = ||A||_F^2 / sigma^2 = sum_i s_i^2 / sigma^2    (a constant)
    lambda_bar = sum_i s_i^2 / n_meas                            (a constant)
    PiGDM      = A^T (sigma_y^2 I + sigma_t^2 A A^T)^-1 (A x - y~)  (diagonal in U-basis)

Domains and the affine-normalization fold. The sampler/prior live in the
*normalized* model domain x in [-1, 1]; the physical permittivity consumed by the
benchmark forward is  f = unnormalize(x) = (x + shift) * scale  (shift=1, scale=0.5).
The cached matrix A maps the *physical* f.flatten() to the real (Re/Im-stacked)
measurement vector. We fold the affine map into the observation ONCE per case so
the likelihood is a clean least-squares in the normalized variable:

    A f = scale * (A x) + scale * shift * (A 1)
    => mu_y(x) = || A x - y~ ||^2 / (2 sigma~^2),
       y~ = y_real / scale - shift * (A 1),
       sigma~ = sigma_y / (sqrt(2) * scale).

The sqrt(2) is the complex-noise convention: the benchmark adds
`sigma_y * randn_like(complex)`, i.e. CN(0, sigma_y^2) whose real/imag parts each
have variance sigma_y^2 / 2 (verified numerically: var(real)=0.5003). After the
fold there are NO chain-rule factors anywhere; the sampler is handed `sigma~` as
the effective measurement noise so its r_t schedule is already in the folded scale.
"""
import math
import torch

from .inverse_scatter import InverseScatter


class AFDPSInverseScatter(InverseScatter):
    def __init__(self,
                 sigma_floor=1e-8,        # floor on the effective noise std (avoids 1/sigma^2 blow-up)
                 init_mode='noise',       # 'noise' | 'pinv' (seed particles from A^+ y~)
                 svd=True,                # MUST be True: the whole engine runs on the cached SVD
                 **kwargs):
        assert svd, "AFDPSInverseScatter requires the cached SVD (svd=True)."
        super().__init__(svd=True, **kwargs)
        self.init_mode = init_mode
        self.sigma_floor = float(sigma_floor)

        # Pin the SVD factors to the operator device in fp64 (guards the base class's
        # torch.load-from-cache without map_location, which can land them on the wrong
        # device / dtype for a parallel shard).
        self.U = self.U.to(self.device, torch.float64)        # (2m, k), k = min(2m, n); square+orthogonal when 2m<=n
        self.Sigma = self.Sigma.to(self.device, torch.float64)  # (k,)
        self.V_t = self.V_t.to(self.device, torch.float64)      # (k, n), n = Ny*Nx

        self._S = self.Sigma                                  # (k,)
        self._S2 = self.Sigma ** 2                            # (k,)
        self._S_safe = self.Sigma.clamp_min(1e-12)            # for the data target y~_U / s
        self._n = self.Nx * self.Ny
        self._n_meas = self.U.shape[0]                        # 2m (number of real measurements)
        self._S2_sum = float(self._S2.sum())                  # ||A||_F^2 (a scalar)

        # Folded effective measurement-noise std (normalized-x scale). The sqrt(2) is the
        # complex CN(0, sigma^2) -> per-real-component variance sigma^2/2 convention.
        self.sigma_noise_eff = max(self.sigma_noise / (math.sqrt(2.0) * self.unnorm_scale),
                                   self.sigma_floor)

        # A @ 1  (sum of columns): the constant shift contribution of the affine fold,
        # computed through the SVD factors as A 1 = U (S * (V^T 1)).
        ones_n = torch.ones(1, self._n, device=self.device, dtype=torch.float64)
        self._A1 = self._A_apply(ones_n).squeeze(0)         # (2m,)

        # caches set by set_observation()
        self._y = None         # raw observation (for reference / NS-style API parity)
        self._y_tilde = None   # (2m,) folded real observation
        self._yU = None        # (k,) = U^T y~ (the only form the engine needs)

    # ------------------------------------------------------------------ #
    # SVD projections (fp64, batched over particles). We use the matrices #
    # directly rather than the base Vt/V helpers (which cast to float32). #
    # ------------------------------------------------------------------ #
    def _to_coeff(self, x):
        """x (J,1,Ny,Nx) -> u = V^T x  (J, k) fp64."""
        return x.reshape(x.shape[0], -1).to(torch.float64) @ self.V_t.T

    def _from_coeff(self, u):
        """u (J, k) fp64 -> x = V u  (J,1,Ny,Nx) fp64."""
        return (u @ self.V_t).reshape(-1, 1, self.Ny, self.Nx)

    def _A_apply(self, x):
        """A x in real measurement space. x (J,1,Ny,Nx) or (J,n) -> (J, 2m) fp64."""
        if x.dim() > 2:
            u = self._to_coeff(x)
        else:
            u = x.to(torch.float64) @ self.V_t.T
        return (self._S * u) @ self.U.T

    # ------------------------------------------------------------------ #
    # Observation handling: fold the affine normalization in ONCE.        #
    # ------------------------------------------------------------------ #
    def set_observation(self, observation):
        """Cache the folded data terms for the current case. `observation` is the
        complex scattered field (1, numTrans, numRec) as produced by the benchmark
        forward operator. After this, mu_y(x) = ||A x - y~||^2 / (2 sigma^2) with
        y~_U = U^T y~ the only quantity the engine reads."""
        obs = observation.to(self.device)
        y_real = torch.view_as_real(obs.flatten()).flatten().to(torch.float64)  # (2m,)
        # affine fold: y~ = y_real / scale - shift * (A 1)
        y_tilde = y_real / self.unnorm_scale - self.unnorm_shift * self._A1
        self._y = observation
        self._y_tilde = y_tilde
        self._yU = self.U.T @ y_tilde                       # (k,)
        return self

    def _residual_coeff(self, x):
        """rho = U^T (A x - y~) = S * (V^T x) - y~_U,  (J, k) fp64."""
        assert self._yU is not None, "call set_observation(observation) before sampling"
        u = self._to_coeff(x)
        return self._S * u - self._yU

    # ------------------------------------------------------------------ #
    # Exact likelihood quantities (all closed-form via the SVD).          #
    # ------------------------------------------------------------------ #
    def likelihood_value(self, x, y, sigma):
        """mu_y(x) = ||A x - y~||^2 / (2 sigma^2) per particle. Returns (J,) fp64.
        `sigma` is the effective (folded) noise std handed in by the sampler."""
        s2 = max(float(sigma), self.sigma_floor) ** 2
        rho = self._residual_coeff(x)
        return 0.5 * rho.pow(2).sum(dim=1) / s2

    def likelihood_gradient(self, x, y, sigma):
        """grad_x mu_y = A^T (A x - y~) / sigma^2. Returns (J,1,Ny,Nx) float32, the
        reference +grad sign. Exact discrete adjoint via the SVD (no autograd)."""
        s2 = max(float(sigma), self.sigma_floor) ** 2
        rho = self._residual_coeff(x)
        g = self._from_coeff(self._S * rho / s2)            # (J,1,Ny,Nx) fp64
        return g.to(torch.float32)

    def likelihood_laplacian(self, x, sigma, g0=None):
        """Tr(Hessian_x mu_y) = ||A||_F^2 / sigma^2 = (sum_i s_i^2) / sigma^2.
        EXACT constant for a linear forward -- no Hutchinson estimate. Returns (J,).
        `g0` is accepted for API parity with the NS operator and ignored."""
        s2 = max(float(sigma), self.sigma_floor) ** 2
        val = self._S2_sum / s2
        return torch.full((x.shape[0],), val, device=x.device, dtype=torch.float64)

    def jacobian_trace(self, x, trace_M=1):
        """lambda_bar = tr(A A^T) / n_meas = (sum_i s_i^2) / (2m), EXACT and free.
        Used by guidance_mode='auto' (isotropic PiGDM): gamma_e^2 = lambda_bar."""
        return self._S2_sum / self._n_meas

    def likelihood_gradient_pigdm(self, x, y, sigma_y, sigma_t, cg_iters=0,
                                  pigdm_scale=1.0, return_diag=False):
        """Anisotropic PiGDM guidance, EXACT in closed form (diagonal in the U-basis):
            g = scale * A^T (sigma_y^2 I + sigma_t^2 A A^T)^{-1} (A x - y~)
              = scale * V [ S * rho / (sigma_y^2 + sigma_t^2 S^2) ].
        No CG is needed (A A^T = U S^2 U^T), so the relative residual is exactly 0.
        Same +grad sign / units as likelihood_gradient. Returns (J,1,Ny,Nx) float32
        (and the 0.0 residual when return_diag)."""
        sy2 = max(float(sigma_y), self.sigma_floor) ** 2
        st2 = float(sigma_t) ** 2
        rho = self._residual_coeff(x)
        coeff = self._S * rho / (sy2 + st2 * self._S2)
        g = float(pigdm_scale) * self._from_coeff(coeff)
        g = g.to(torch.float32)
        if return_diag:
            return g, 0.0
        return g

    # ------------------------------------------------------------------ #
    # Exact integration of the (linear) guidance drift over one step.     #
    # ------------------------------------------------------------------ #
    def exact_linear_substep(self, x, t_cur, t_next, mode='pigdm',
                             gamma_e2=None):
        """Integrate the guidance-only ODE  du_i/dt = a_i(t) (u_i - c_i)  EXACTLY over
        [t_cur, t_next] in the V-basis, where c_i = y~_U,i / s_i is the per-mode data
        target. The guidance drift is linear in x, so the exact solution is

            u_i <- u_i * exp(-phi_i) + c_i * (1 - exp(-phi_i)),

        which is unconditionally stable -- it removes the stiff ~1/sigma~^2 (~5e7)
        guidance coefficient that would force tiny Euler steps. The two modes differ
        only in the per-mode rate phi_i (>= 0):

          mode='fixed'/'auto' (isotropic, r_t^2 = sigma~^2 + gamma_e^2 t^2):
              phi_i = (s_i^2 / gamma_e^2) * log( r^2(t_cur) / r^2(t_next) )
          mode='pigdm' (anisotropic, effective per-mode variance sigma~^2 + t^2 s_i^2):
              phi_i = log( (sigma~^2 + t_cur^2 s_i^2) / (sigma~^2 + t_next^2 s_i^2) )

        At the final step the pigdm contraction exp(-phi_i) -> sigma~^2/(sigma~^2+t^2 s_i^2)
        enforces hard data consistency on well-measured modes (s_i large) while leaving
        weak/null modes (s_i ~ 0, phi_i ~ 0) to the diffusion prior -- a DDNM-like split.
        Returns the updated x (same dtype as input)."""
        sig2 = self.sigma_noise_eff ** 2
        tc, tn = float(t_cur), float(t_next)
        if mode == 'pigdm':
            phi = torch.log((sig2 + (tc ** 2) * self._S2) / (sig2 + (tn ** 2) * self._S2))
        else:
            assert gamma_e2 is not None and gamma_e2 > 0, "isotropic exact_linear needs gamma_e2 > 0"
            r2c = sig2 + gamma_e2 * (tc ** 2)
            r2n = sig2 + gamma_e2 * (tn ** 2)
            phi = (self._S2 / gamma_e2) * math.log(r2c / r2n)
        phi = phi.clamp_min(0.0)                            # t_cur > t_next => phi >= 0
        em = torch.exp(-phi)                                # (k,)
        u = self._to_coeff(x)                               # (J, k) fp64
        c = self._yU / self._S_safe                         # (k,)
        u_new = u * em + c * (-torch.expm1(-phi))           # expm1-safe (1 - e^{-phi})
        dx = self._from_coeff(u_new - u)                    # (J,1,Ny,Nx) fp64
        return x + dx.to(x.dtype)

    # ------------------------------------------------------------------ #
    # Ensemble init & top-noise perturbation (mirrors the NS port).       #
    # ------------------------------------------------------------------ #
    def initialize_ensemble(self, gt, num_particles):
        shape = (num_particles, 1, self.Ny, self.Nx)
        if self.init_mode == 'pinv' and self._y is not None:
            base = self.pseudo_inverse(self._y).reshape(1, 1, self.Ny, self.Nx).to(self.device, torch.float32)
            return base.repeat(num_particles, 1, 1, 1)
        return torch.zeros(*shape, device=self.device, dtype=torch.float32)

    def proximal_generator(self, x_ref, y, sigma, scale):
        # No closed-form prox is needed: seed particles at the EDM prior of the top
        # noise level, x_ref + scale * N(0, I) (scale = s(t0) sigma(t0) = sigma_max).
        return x_ref + scale * torch.randn_like(x_ref)
