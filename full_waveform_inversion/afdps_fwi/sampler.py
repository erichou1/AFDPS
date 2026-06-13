"""Annealed-SDE diffusion posterior sampler with Feynman-Kac particle reweighting.

This is the core of AFDPS ("Approximation-Free Diffusion Posterior Sampling").
An ensemble of `num_particles` particles is evolved through an annealed (EDM) SDE.
At every step the update combines

    * the prior score, supplied by a denoiser  net(x/s, sigma)  ->  D_x   (Tweedie);
    * the likelihood drift, supplied by  operator.likelihood_gradient(x, y, sigma);

and the particles carry Feynman-Kac importance weights whose log-increment uses

    reweight = (s^2 * sigma' * sigma) * (||grad||^2 - laplacian) + (d_cur . grad)

where `laplacian = operator.likelihood_laplacian(x, sigma)` is Tr(Hessian of the
negative log-likelihood). For a *nonlinear* forward operator (Navier-Stokes, FWI)
this Laplacian has no closed form and must be estimated by the operator.

`operator` must expose:
    initialize_ensemble(gt, num_particles) -> (num_particles, C, H, W)
    proximal_generator(x_ref, y, sigma_noise, scale) -> perturbed particles
    likelihood_gradient(x, y, sigma_noise) -> (num_particles, C, H, W)
    likelihood_laplacian(x, sigma_noise) -> (num_particles,)
`noiser` must expose `.sigma` (the measurement-noise std).
`net` must expose `.sigma_min`, `.sigma_max`, `.round_sigma(sigma)` and be callable
as `net(x, sigma) -> denoised`.

NOTE: this file is vendored verbatim from the Navier-Stokes AFDPS port
(`navier_stokes/algo/afdps_core/ensemble_denoiser_edm.py`). It is problem-agnostic:
all problem-specific behaviour lives behind the `operator` API above. Keeping it
byte-identical guarantees the FWI port runs the *same* AFDPS sampler the paper and
the Navier-Stokes results were produced with.
"""
import numpy as np
import torch
from tqdm import tqdm


class Ensemble_Denoiser_EDM:
    def __init__(
        self,
        net,
        device,
        num_steps=18,
        sigma_min=None,
        sigma_max=None,
        rho=7,
        solver='euler',
        discretization='edm',
        schedule='linear',
        scaling='none',
        epsilon_s=1e-3,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        alpha=1,
        S_churn=0,
        S_min=0,
        S_max=float('inf'),
        S_noise=1,
        mode='sde',
        likelihood_at='denoised',
        guidance_gamma=1.0,
        guidance_mode='fixed',
        trace_M=1,
        trace_every=10,
        trace_dense_until=0,
        cg_iters=30,
        pigdm_scale=1.0,
        use_value=False,
        resample=False,
        resample_threshold=0.5,
        progress=True,
    ):
        self.net = net
        self.device = device
        self.progress = progress   # show a per-sample tqdm bar (steps + live ETA)
        # Annealed-guidance schedule (PiGDM/DPS-style). The likelihood is evaluated
        # at the denoised estimate x_hat, whose residual uncertainty scales with the
        # current diffusion noise sigma(t). Using an effective data std
        #   r_t = sqrt(sigma_y^2 + (gamma * sigma(t))^2)
        # tempers the otherwise stiff ~1/sigma_y^2 guidance at high noise and recovers
        # the true measurement noise sigma_y as sigma(t) -> 0. gamma is the guidance
        # strength (smaller -> stronger/earlier data fit). Without this, the explicit
        # SDE step requires thousands of steps to remain stable for small sigma_y.
        self.guidance_gamma = guidance_gamma
        # Guidance mode. 'fixed' (default) uses the hand-set guidance_gamma and is
        # byte-identical to before. 'auto' is J-aware (isotropic PiGDM): it ignores
        # guidance_gamma and sets the guidance strength from the MEASURED Jacobian trace
        #   lambda_bar = tr(J J^T) / n_meas   ->   r_t^2 = sigma_y^2 + sigma(t)^2 * lambda_bar,
        # so the gamma~10/ds rescaling is replaced by the operator's own sensitivity
        # (sqrt(lambda_bar) plays the role of gamma). Amortized: lambda_bar is recomputed
        # every `trace_every` steps -- and every step while i < `trace_dense_until`, the
        # high-sigma phase where it varies fastest -- and held constant in between. Each
        # recompute costs `trace_M` Hutchinson VJP probes (operator.jacobian_trace).
        self.guidance_mode = guidance_mode
        self.trace_M = trace_M
        self.trace_every = max(1, int(trace_every))
        self.trace_dense_until = int(trace_dense_until)
        # CG iterations for guidance_mode='full' (anisotropic PiGDM measurement-space solve).
        # The sigma_y^2 I + sigma_t^2 J J^T system is well-conditioned (regularized), so a
        # handful of iterations suffices; raise if the logged CG rel_residual is large.
        self.cg_iters = int(cg_iters)
        # Single global magnitude calibration for guidance_mode='full'. The PiGDM gradient is
        # the matrix-inverse solution, whose scale differs from the fixed-mode 1/r_t^2 gradient
        # the SDE drift coefficient was tuned for; pigdm_scale rescales it (one constant, not
        # per-cell). Sweep once to find the value that keeps the SDE stable.
        self.pigdm_scale = float(pigdm_scale)
        # Feynman-Kac potential refinement (TMLR ensemble sampler). When on, the actual
        # negative-log-likelihood value mu_y(x) is added to the FK log-weight so particles
        # are scored by their real data-misfit level, not only by the ||grad||^2 - Tr(Hess)
        # curvature terms. Requires operator.likelihood_value(x, y, sigma). Default off to
        # keep existing runs byte-identical.
        self.use_value = use_value
        # ESS-based resampling (AFDPS Algorithm 1). The released AFDPS skips this step
        # "to save computational cost ... in a parallel way" (paper App. E), letting the
        # Feynman-Kac weights accumulate over all steps until one particle dominates.
        # Restoring it culls degenerate (near-zero-weight) particles and duplicates the
        # high-weight ones, keeping the effective sample size up so the returned
        # highest-weight reconstruction is drawn from a healthier ensemble. Default off
        # to keep existing runs byte-identical; enable via sampler_kwargs.resample=true.
        self.resample = resample
        self.resample_threshold = resample_threshold  # c in (0,1): normalized-ESS trigger
        # Where to evaluate the likelihood gradient/Laplacian each step:
        #   'denoised' : at the Tweedie estimate D_x (DPS-style). REQUIRED for a
        #                nonlinear PDE forward -- the solver run from a noise-
        #                dominated x_track violates CFL and diverges, and the
        #                guidance ~1/sigma^2 explodes. D_x is smooth/in-distribution
        #                and shrinks to ~0 at high noise, so guidance anneals in.
        #   'noisy'    : at x_track (the original reference behaviour, valid only
        #                for linear-Gaussian image operators).
        self.likelihood_at = likelihood_at
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver
        self.discretization = discretization
        self.schedule = schedule
        self.scaling = scaling
        self.epsilon_s = epsilon_s
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.alpha = alpha
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.mode = mode

        assert solver in ['euler'], "Only Euler solver is supported."
        assert discretization in ['vp', 've', 'iddpm', 'edm']
        assert schedule in ['vp', 've', 'linear']
        assert scaling in ['vp', 'none']
        assert mode in ['sde', 'pfode'], "Only SDE and PFODE modes are supported."

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
        vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (self.sigma(t) + 1 / self.sigma(t))
        vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma ** 2

        # Select default noise level range based on the specified time step discretization.
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
            sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 10, 'edm': 8}[discretization]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
        vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        if discretization == 'vp':
            orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif discretization == 've':
            orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
            sigma_steps = ve_sigma(orig_t_steps)
        elif discretization == 'iddpm':
            u = torch.zeros(M + 1, dtype=torch.float64, device=device)
            alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
            for j in torch.arange(M, 0, -1, device=device):  # M, ..., 1
                u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
        else:
            assert discretization == 'edm'
            sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Define noise level schedule.
        if schedule == 'vp':
            self.sigma = vp_sigma(vp_beta_d, vp_beta_min)
            self.sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            self.sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif schedule == 've':
            self.sigma = ve_sigma
            self.sigma_deriv = ve_sigma_deriv
            self.sigma_inv = ve_sigma_inv
        else:
            assert schedule == 'linear'
            self.sigma = lambda t: t
            self.sigma_deriv = lambda t: 1
            self.sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if scaling == 'vp':
            self.s = lambda t: 1 / (1 + self.sigma(t) ** 2).sqrt()
            self.s_deriv = lambda t: -self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)
        else:
            assert scaling == 'none'
            self.s = lambda t: 1
            self.s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = self.sigma_inv(net.round_sigma(sigma_steps))
        self.t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    @torch.no_grad()
    def __call__(self, gt, x_noisy, num_particles, operator, noiser, return_trajectory=False):
        """Run the ensemble sampler.

        Returns a dict with:
            'ensemble'       : (num_particles, C, H, W) final particle ensemble
            'log_weights'    : (num_particles,) final Feynman-Kac log-weights
            'best_traj'      : (num_steps, C, H, W) per-step argmax-weight particle, only
                               when return_trajectory=True (else None; off the hot path).
        """
        t0 = self.t_steps[0]
        scale = self.s(t0) * self.sigma(t0)

        x_ref = operator.initialize_ensemble(gt, num_particles)
        x_track_list = operator.proximal_generator(x_ref, x_noisy, noiser.sigma, scale)
        x_return_list = torch.empty(len(self.t_steps) - 1, *x_ref[0].shape, device=self.device) if return_trajectory else None
        log_weight_list = torch.zeros(num_particles, device=self.device)
        self._lambda_bar = None   # cached tr(J J^T)/n_meas for guidance_mode='auto' (amortized)
        self._pigdm_logged = False  # one-time CG-convergence print for guidance_mode='full'

        steps = list(zip(self.t_steps[:-1], self.t_steps[1:]))
        for i, (t_cur, t_next) in enumerate(tqdm(steps, desc='AFDPS sampling', disable=not self.progress)):  # 0..N-1
            denoised = self.net(x_track_list / self.s(t_cur), self.sigma(t_cur)).to(torch.float32)
            x_eval = denoised if self.likelihood_at == 'denoised' else x_track_list
            sigma_t = float(self.sigma(t_cur))
            if self.guidance_mode == 'full':
                # Full anisotropic PiGDM: g = J^T (sigma_y^2 I + sigma_t^2 J J^T)^{-1}(P L(x_hat) - y),
                # solved matrix-free by CG in measurement space. NO guidance_gamma -- the guidance is
                # fully determined by the known measurement noise sigma_y and the schedule sigma_t
                # (gamma == 1, the principled Bayesian limit). This is the anisotropic generalization
                # of the 'fixed'/'auto' scalar guidance and recovers them as J J^T -> mean-eigval I.
                grad_x_i, _relres = operator.likelihood_gradient_pigdm(
                    x_eval, x_noisy, float(noiser.sigma), sigma_t,
                    cg_iters=self.cg_iters, pigdm_scale=self.pigdm_scale, return_diag=True)
                if not self._pigdm_logged:
                    print(f"[AFDPS full-PiGDM] CG iters={self.cg_iters} rel_residual={_relres:.3g} "
                          f"scale={self.pigdm_scale:g} (sigma_y={float(noiser.sigma):.3g}, sigma_t0={sigma_t:.3g})", flush=True)
                    self._pigdm_logged = True
                # FK reweight Laplacian: keep the isotropic curvature term at the matched r_t.
                # PiGDM changes the guidance DIRECTION; the FK weight is a particle-selection
                # heuristic (||grad||^2 - Tr(Hess)) left in its stable isotropic form. Its
                # grad_fn is the plain misfit gradient, so the forward-scheme Hutchinson needs
                # g0 = the ISOTROPIC gradient at r_t (NOT the preconditioned PiGDM gradient).
                r_t = (float(noiser.sigma) ** 2 + sigma_t ** 2) ** 0.5
                g_iso = operator.likelihood_gradient(x_eval, x_noisy, r_t)
                laplacian_x_i = operator.likelihood_laplacian(x_eval, r_t, g0=g_iso)
            else:
                if self.guidance_mode == 'auto':
                    # J-aware guidance: r_t^2 = sigma_y^2 + sigma(t)^2 * lambda_bar, with the
                    # measured mean Jacobian eigenvalue lambda_bar replacing the tuned gamma^2.
                    # Amortized recompute (dense while sigma is large, then every trace_every).
                    if (self._lambda_bar is None) or (i < self.trace_dense_until) or (i % self.trace_every == 0):
                        first = self._lambda_bar is None
                        self._lambda_bar = float(operator.jacobian_trace(x_eval, trace_M=self.trace_M))
                        if first:
                            # the effective gamma the operator 'chooses'; compare to the gamma~10/ds law.
                            print(f"[AFDPS auto-guidance] lambda_bar={self._lambda_bar:.4g} "
                                  f"-> gamma_eff=sqrt(lambda_bar)={self._lambda_bar ** 0.5:.4g}", flush=True)
                    r_t = (float(noiser.sigma) ** 2 + (sigma_t ** 2) * self._lambda_bar) ** 0.5
                else:
                    r_t = (float(noiser.sigma) ** 2 + (self.guidance_gamma * sigma_t) ** 2) ** 0.5
                grad_x_i = operator.likelihood_gradient(x_eval, x_noisy, r_t)
                # forward-difference Hutchinson can reuse this drift gradient as the
                # unperturbed point (g0); central-scheme operators ignore the kwarg.
                laplacian_x_i = operator.likelihood_laplacian(x_eval, r_t, g0=grad_x_i)

            # Robustness: a diverged particle (e.g. CFL blow-up from too-large guidance/dt)
            # must not poison the whole ensemble. Zero its guidance contribution this step.
            grad_x_i = torch.nan_to_num(grad_x_i)
            laplacian_x_i = torch.nan_to_num(laplacian_x_i)

            d_cur = (2 * (self.sigma_deriv(t_cur) / self.sigma(t_cur)) + self.s_deriv(t_cur) / self.s(t_cur)) * x_track_list \
                - self.sigma_deriv(t_cur) * self.s(t_cur) / self.sigma(t_cur) * (2 * denoised)
            grad_cur = ((2 * (self.s(t_cur) ** 2)) * (self.sigma_deriv(t_cur) * self.sigma(t_cur))) * grad_x_i
            n_cur = self.s(t_cur) * torch.sqrt(2 * self.sigma_deriv(t_cur) * self.sigma(t_cur)) * torch.randn_like(x_track_list)
            x_track_list = x_track_list + (t_next - t_cur) * d_cur + (t_next - t_cur) * grad_cur + torch.sqrt(t_cur - t_next) * n_cur

            reweight_func = ((self.s(t_cur) ** 2) * (self.sigma_deriv(t_cur) * self.sigma(t_cur))) \
                * ((grad_x_i ** 2).sum(dim=(1, 2, 3)) - laplacian_x_i) + (d_cur * grad_x_i).sum(dim=(1, 2, 3))
            if self.use_value:
                # add the actual misfit value mu_y(x) to the FK potential (TMLR refinement);
                # t0 normalizes it onto the same per-step scale as the curvature terms.
                value_x_i = operator.likelihood_value(x_eval, x_noisy, r_t)
                value_x_i = torch.nan_to_num(value_x_i)
                reweight_func = reweight_func - (1.0 / float(self.t_steps[0])) * value_x_i
            log_weight_list = log_weight_list + (t_cur - t_next) * reweight_func

            # Quarantine non-finite particles so they are never selected and do not
            # contaminate the softmax/argmax reduction in the wrapper.
            bad = ~torch.isfinite(x_track_list).flatten(1).all(dim=1)
            if bad.any():
                x_track_list = torch.nan_to_num(x_track_list)
                log_weight_list = log_weight_list.masked_fill(bad, float('-inf'))
            # Stabilize: weights are only used up to an additive constant (argmax/softmax).
            mx = log_weight_list.max()
            if torch.isfinite(mx):
                log_weight_list = log_weight_list - mx

            # AFDPS Algorithm 1: resample when the normalized Effective Sample Size drops
            # below the threshold c. Normalized ESS = (sum w)^2 / (N * sum w^2) in [1/N, 1]
            # with w the (unnormalized) Feynman-Kac weights; here w = softmax(log_weight)
            # is already sum-normalized, so ESS_norm = 1 / (N * sum w^2). Bad (-inf) particles
            # get w=0 and are never resampled in. Skip the final step so the terminal weights
            # stay discriminative for the argmax-best reduction in the wrapper.
            if self.resample and i < len(steps) - 1:
                w = torch.softmax(log_weight_list, dim=0)
                ess_norm = 1.0 / (num_particles * (w ** 2).sum().clamp_min(1e-30))
                n_finite = int(torch.isfinite(log_weight_list).sum())
                if n_finite > 1 and torch.isfinite(ess_norm) and float(ess_norm) < self.resample_threshold:
                    idx = torch.multinomial(w, num_particles, replacement=True)
                    x_track_list = x_track_list[idx]
                    log_weight_list = torch.zeros_like(log_weight_list)  # reset beta <- 1

            if return_trajectory:
                x_return_list[i] = x_track_list[torch.argmax(log_weight_list)]

        return {
            'ensemble': x_track_list,
            'log_weights': log_weight_list,
            'best_traj': x_return_list,
        }
