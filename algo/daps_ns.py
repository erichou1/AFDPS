"""
DAPS for the 2D Navier-Stokes inverse problem (improvement #3).

Decoupled Annealing Posterior Sampling (Zhang et al., 2024) ported to InverseBench NS.
Unlike the FK-SDE AFDPS sampler, DAPS DECOUPLES the diffusion trajectory from the
data-consistency optimization, which is structurally more robust to the chaotic Re=200
gradient:

  per annealing level sigma_i (geometric sigma_max -> sigma_min):
    1. reverse diffusion :  x0hat = PF-ODE(x_t, sigma_i -> 0)          (prior, via the net)
    2. MCMC data step    :  x0y   = Langevin on  p(x0 | x_t, y)
                            ∝ N(x0; x0hat, sigma_i^2 I) · exp(-mu_y(x0))   (data, via adjoint)
    3. forward diffusion :  x_t   = x0y + sigma_{i+1} * eps

Why it suits NS: the data-consistency MCMC runs on the CLEAN estimate x0 (smooth,
in-distribution), so the NS forward/adjoint never sees a noise-dominated field and does
not violate CFL -- the failure mode that forces the FK-SDE sampler to use
`likelihood_at='denoised'`. Here that compatibility is automatic.

The likelihood gradient grad mu_y = lambda(0) is the exact discrete adjoint supplied by
`operator.likelihood_gradient(x0, y, tau)`; the prior is the pretrained diffusion denoiser
`net`. Runs an ensemble of `num_particles` and reduces to the MMSE posterior mean.
"""
import torch
from tqdm import tqdm

from .base import Algo
from .enkg import ode_sampler          # deterministic PF-ODE  x_t -> x_0


class DAPS_NS(Algo):
    def __init__(self, net, forward_op,
                 num_particles=16,
                 num_annealing_steps=100,
                 sigma_max=80.0,
                 sigma_min=0.002,
                 ode_steps=5,                # PF-ODE substeps per reverse-diffusion call
                 mcmc_steps=20,              # Langevin steps in the data-consistency stage
                 mcmc_lr=1.0e-3,             # Langevin step size (normalized units)
                 mcmc_lr_min_ratio=0.1,      # anneal the Langevin lr down over the schedule
                 tau=None,                   # data-fidelity std in mu_y; default forward_op.sigma_noise
                 langevin_noise=True,        # True = sampling (Langevin); False = MAP descent
                 reduce='mean',              # 'mean' (MMSE) | 'best' | 'samples'
                 progress=True):
        super().__init__(net, forward_op)
        self.num_particles = num_particles
        self.T = num_annealing_steps
        self.sigma_max = min(float(sigma_max), float(net.sigma_max))
        self.sigma_min = max(float(sigma_min), float(net.sigma_min))
        self.ode_steps = ode_steps
        self.mcmc_steps = mcmc_steps
        self.mcmc_lr = mcmc_lr
        self.mcmc_lr_min_ratio = mcmc_lr_min_ratio
        sig = tau if tau is not None else getattr(forward_op, 'sigma_noise', 0.0)
        self.tau = max(float(sig), 1e-3)
        self.langevin_noise = langevin_noise
        self.reduce = reduce
        self.progress = progress
        self.device = forward_op.device

        # geometric annealing schedule sigma_max -> sigma_min
        steps = torch.arange(self.T, dtype=torch.float64)
        self.sigma_steps = (self.sigma_max * (self.sigma_min / self.sigma_max)
                            ** (steps / max(self.T - 1, 1))).to(torch.float32)

    # ---- data-consistency Langevin on the clean estimate (eq. p(x0 | x_t, y)) ----
    def _mcmc(self, x0hat, y, sigma_i, ratio):
        op = self.forward_op
        x0 = x0hat.clone()
        inv_s2 = 1.0 / (sigma_i ** 2)
        lr = self.mcmc_lr * (1.0 - (1.0 - self.mcmc_lr_min_ratio) * ratio)   # annealed step size
        for _ in range(self.mcmc_steps):
            grad_data = op.likelihood_gradient(x0, y, self.tau)              # +grad mu_y (adjoint)
            grad_prior = (x0 - x0hat) * inv_s2                              # tie to the diffusion estimate
            grad_data = torch.nan_to_num(grad_data)
            x0 = x0 - 0.5 * lr * (grad_prior + grad_data)
            if self.langevin_noise:
                x0 = x0 + (lr ** 0.5) * torch.randn_like(x0)
            x0 = torch.nan_to_num(x0)
        return x0

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        op = self.forward_op
        P = self.num_particles
        res = self.net.img_resolution
        ch = self.net.img_channels

        # initialize particles from the prior at sigma_max
        x_t = torch.randn(P, ch, res, res, device=self.device) * float(self.sigma_steps[0])

        pbar = tqdm(range(self.T), desc='DAPS-NS', disable=not self.progress)
        for i in pbar:
            sigma_i = float(self.sigma_steps[i])
            ratio = i / max(self.T - 1, 1)
            # 1. reverse diffusion: clean estimate via PF-ODE from sigma_i
            x0hat = ode_sampler(self.net, x_t, num_steps=self.ode_steps, sigma_start=sigma_i)
            # 2. data-consistency MCMC on the clean estimate (NS adjoint; CFL-safe)
            x0y = self._mcmc(x0hat, observation, sigma_i, ratio)
            # 3. forward diffusion back up to the next level
            if i < self.T - 1:
                x_t = x0y + torch.randn_like(x0y) * float(self.sigma_steps[i + 1])
            else:
                x_t = x0y
            if self.progress:
                with torch.no_grad():
                    mis = (op.forward(x0y) - observation).square().flatten(1).sum(1).min().item()
                pbar.set_postfix(sigma=f'{sigma_i:.2f}', misfit=f'{mis:.3e}')

        # reduce the ensemble to the requested reconstruction(s)
        if self.reduce == 'mean':
            recon = x_t.mean(dim=0, keepdim=True).repeat(num_samples, 1, 1, 1)
        elif self.reduce == 'best':
            mis = (op.forward(x_t) - observation).square().flatten(1).sum(1)
            recon = x_t[torch.argmin(mis)].unsqueeze(0).repeat(num_samples, 1, 1, 1)
        else:  # 'samples'
            idx = torch.randint(0, P, (num_samples,), device=self.device)
            recon = x_t[idx]
        return recon
