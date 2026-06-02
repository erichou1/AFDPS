"""Annealed-SDE diffusion posterior sampler with Feynman-Kac particle reweighting.

This is the core of AFDPS ("Approximation-Free Diffusion Posterior Sampling").
An ensemble of `num_particles` particles is evolved through an annealed (EDM) SDE.
At every step the update combines

    * the prior score, supplied by a denoiser  net(x/s, sigma)  ->  D_x   (Tweedie);
    * the likelihood drift, supplied by  operator.likelihood_gradient(x, y, sigma);

and the particles carry Feynman-Kac importance weights whose log-increment uses

    reweight = (s^2 * sigma' * sigma) * (||grad||^2 - laplacian) + (d_cur . grad)

where `laplacian = operator.likelihood_laplacian(x, sigma)` is Tr(Hessian of the
negative log-likelihood). For a *nonlinear* forward operator (Navier-Stokes) this
Laplacian has no closed form and must be Hutchinson-estimated by the operator.

`operator` must expose:
    initialize_ensemble(gt, num_particles) -> (num_particles, C, H, W)
    proximal_generator(x_ref, y, sigma_noise, scale) -> perturbed particles
    likelihood_gradient(x, y, sigma_noise) -> (num_particles, C, H, W)
    likelihood_laplacian(x, sigma_noise) -> (num_particles,)
`noiser` must expose `.sigma` (the measurement-noise std).
`net` must expose `.sigma_min`, `.sigma_max`, `.round_sigma(sigma)` and be callable
as `net(x, sigma) -> denoised`.
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

        steps = list(zip(self.t_steps[:-1], self.t_steps[1:]))
        for i, (t_cur, t_next) in enumerate(tqdm(steps, desc='AFDPS sampling', disable=not self.progress)):  # 0..N-1
            denoised = self.net(x_track_list / self.s(t_cur), self.sigma(t_cur)).to(torch.float32)
            x_eval = denoised if self.likelihood_at == 'denoised' else x_track_list
            r_t = (float(noiser.sigma) ** 2 + (self.guidance_gamma * float(self.sigma(t_cur))) ** 2) ** 0.5
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

            if return_trajectory:
                x_return_list[i] = x_track_list[torch.argmax(log_weight_list)]

        return {
            'ensemble': x_track_list,
            'log_weights': log_weight_list,
            'best_traj': x_return_list,
        }
