"""Vendored AFDPS ensemble sampler, extended for the LINEAR inverse-scattering port.

This subclasses the Navier-Stokes port's verified `Ensemble_Denoiser_EDM`
(`navier_stokes/algo/afdps_core/ensemble_denoiser_edm.py`, resolved via the shared
`algo` namespace package) so the EDM schedule construction is reused byte-for-byte.
Only the per-step loop is reimplemented, to add two scattering-specific upgrades that
exploit the *linear* forward operator:

  * `guidance_step='exact_linear'` -- the guidance drift is linear in x, so instead of
    an explicit Euler nudge `+(t_next-t_cur)*(2 s^2 sigma' sigma) grad mu_y` (which is
    stiff: the ~1/sigma~^2 ~ 5e7 coefficient forces tiny steps) we integrate the
    guidance ODE EXACTLY in the operator's V-basis via `operator.exact_linear_substep`.
    Unconditionally stable; at the final step it becomes a DDNM-like hard data
    projection on well-measured singular directions. (`'euler'` keeps the faithful
    AFDPS-SDE step for the paper-comparison ablation.)
  * fp64 Feynman-Kac log-weights, and the option `value_coef='exact'` for the
    annealed value term (vs the NS port's `'t0'` heuristic coefficient).

The Feynman-Kac reweighting itself is kept in its stable ISOTROPIC form for every
guidance mode (the weight is a particle-selection heuristic; the guidance *direction*
is what 'full'/PiGDM changes).
"""
import numpy as np
import torch
from tqdm import tqdm

from algo.afdps_core.ensemble_denoiser_edm import Ensemble_Denoiser_EDM as _BaseSampler


class Ensemble_Denoiser_EDM(_BaseSampler):
    def __init__(self, *args, guidance_step='euler', value_coef='t0', **kwargs):
        """`guidance_step`: 'euler' (faithful AFDPS-SDE) | 'exact_linear' (exact guidance
        integration, the scattering primary). `value_coef`: 't0' (NS heuristic 1/t0) |
        'exact' (annealed kappa(t) = 2 gamma_e^2 t / r_t^2). All other arguments are
        forwarded to the base sampler unchanged (schedule, guidance_mode, trace_*, ...)."""
        super().__init__(*args, **kwargs)
        assert guidance_step in ('euler', 'exact_linear')
        assert value_coef in ('t0', 'exact')
        self.guidance_step = guidance_step
        self.value_coef = value_coef

    @torch.no_grad()
    def __call__(self, gt, x_noisy, num_particles, operator, noiser, return_trajectory=False):
        """Run the ensemble sampler. Returns dict('ensemble', 'log_weights', 'best_traj').

        `noiser.sigma` is the *folded* effective measurement noise sigma~ (set by the
        Algo wrapper), so every r_t = sqrt(sigma~^2 + gamma_e^2 sigma(t)^2) here is
        already in the operator's normalized-x scale -- no chain-rule factors."""
        sigma_y = float(noiser.sigma)
        t0 = self.t_steps[0]
        scale = self.s(t0) * self.sigma(t0)

        x_track = operator.proximal_generator(
            operator.initialize_ensemble(gt, num_particles), x_noisy, sigma_y, scale)
        x_return = (torch.empty(len(self.t_steps) - 1, *x_track[0].shape, device=self.device)
                    if return_trajectory else None)
        log_weight = torch.zeros(num_particles, device=self.device, dtype=torch.float64)
        self._lambda_bar = None

        steps = list(zip(self.t_steps[:-1], self.t_steps[1:]))
        for i, (t_cur, t_next) in enumerate(tqdm(steps, desc='AFDPS-scatter', disable=not self.progress)):
            denoised = self.net(x_track / self.s(t_cur), self.sigma(t_cur)).to(torch.float32)
            x_eval = denoised if self.likelihood_at == 'denoised' else x_track
            sigma_t = float(self.sigma(t_cur))

            # ---- effective guidance strength gamma_e^2 (sets the isotropic r_t) ----
            if self.guidance_mode == 'fixed':
                gamma_e2 = float(self.guidance_gamma) ** 2
            else:  # 'auto' or 'full' -> use the measured (exact, free) Jacobian trace
                if (self._lambda_bar is None) or (i < self.trace_dense_until) or (i % self.trace_every == 0):
                    self._lambda_bar = float(operator.jacobian_trace(x_eval, trace_M=self.trace_M))
                gamma_e2 = self._lambda_bar
            r_t = (sigma_y ** 2 + gamma_e2 * (sigma_t ** 2)) ** 0.5

            # ---- Feynman-Kac weight terms: isotropic gradient + exact Laplacian ----
            grad_fk = torch.nan_to_num(operator.likelihood_gradient(x_eval, x_noisy, r_t))
            laplacian = torch.nan_to_num(operator.likelihood_laplacian(x_eval, r_t, g0=grad_fk))

            # ---- prior (probability-flow) drift + diffusion noise ----
            d_cur = (2 * (self.sigma_deriv(t_cur) / self.sigma(t_cur)) + self.s_deriv(t_cur) / self.s(t_cur)) * x_track \
                - self.sigma_deriv(t_cur) * self.s(t_cur) / self.sigma(t_cur) * (2 * denoised)
            n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur) * self.sigma(t_cur)) * torch.randn_like(x_track)

            # ---- guidance move ----
            if self.guidance_step == 'exact_linear':
                # Lie split: prior + noise (Euler), then EXACT guidance integration.
                x_track = x_track + (t_next - t_cur) * d_cur + torch.sqrt(t_cur - t_next) * n_cur
                substep_mode = 'pigdm' if self.guidance_mode == 'full' else 'iso'
                x_track = operator.exact_linear_substep(
                    x_track, t_cur, t_next, mode=substep_mode, gamma_e2=gamma_e2)
            else:  # 'euler' -- the faithful explicit AFDPS-SDE step
                if self.guidance_mode == 'full':
                    grad_move = torch.nan_to_num(operator.likelihood_gradient_pigdm(
                        x_eval, x_noisy, sigma_y, sigma_t,
                        cg_iters=self.cg_iters, pigdm_scale=self.pigdm_scale))
                else:
                    grad_move = grad_fk
                grad_cur = ((2 * (self.s(t_cur) ** 2)) * (self.sigma_deriv(t_cur) * self.sigma(t_cur))) * grad_move
                x_track = x_track + (t_next - t_cur) * d_cur + (t_next - t_cur) * grad_cur \
                    + torch.sqrt(t_cur - t_next) * n_cur

            # ---- Feynman-Kac log-weight increment (fp64, isotropic potential) ----
            g64 = grad_fk.to(torch.float64)
            reweight = ((self.s(t_cur) ** 2) * (self.sigma_deriv(t_cur) * self.sigma(t_cur))) \
                * ((g64 ** 2).sum(dim=(1, 2, 3)) - laplacian) \
                + (d_cur.to(torch.float64) * g64).sum(dim=(1, 2, 3))
            if self.use_value:
                value = torch.nan_to_num(operator.likelihood_value(x_eval, x_noisy, r_t))
                if self.value_coef == 'exact':
                    kappa = 2.0 * gamma_e2 * sigma_t / (r_t ** 2)   # = -d/dt log alpha_t
                else:
                    kappa = 1.0 / float(self.t_steps[0])
                reweight = reweight - kappa * value
            log_weight = log_weight + (t_cur - t_next).to(torch.float64) * reweight

            # ---- quarantine diverged particles; stabilize weights ----
            bad = ~torch.isfinite(x_track).flatten(1).all(dim=1)
            if bad.any():
                x_track = torch.nan_to_num(x_track)
                log_weight = log_weight.masked_fill(bad, float('-inf'))
            mx = log_weight.max()
            if torch.isfinite(mx):
                log_weight = log_weight - mx

            # ---- optional ESS resampling (AFDPS Algorithm 1) ----
            if self.resample and i < len(steps) - 1:
                w = torch.softmax(log_weight, dim=0)
                ess_norm = 1.0 / (num_particles * (w ** 2).sum().clamp_min(1e-30))
                n_finite = int(torch.isfinite(log_weight).sum())
                if n_finite > 1 and torch.isfinite(ess_norm) and float(ess_norm) < self.resample_threshold:
                    idx = torch.multinomial(w, num_particles, replacement=True)
                    x_track = x_track[idx]
                    log_weight = torch.zeros_like(log_weight)

            if return_trajectory:
                x_return[i] = x_track[torch.argmax(log_weight)]

        return {'ensemble': x_track, 'log_weights': log_weight, 'best_traj': x_return}
