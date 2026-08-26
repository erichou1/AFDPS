"""AFDPS as an InverseBench algorithm for the linear inverse-scattering problem.

Wraps the vendored (scattering-extended) `Ensemble_Denoiser_EDM` behind the
`Algo.inference` interface. The pipeline is identical in spirit to the
Navier-Stokes port -- an annealed-SDE ensemble with Feynman-Kac reweighting driven
by a pretrained EDM diffusion prior `net` -- but the operator it drives is the
LINEAR scattering forward, so the guidance/curvature are exact and closed-form.

Reduction default is `'mean'` (the weighted ensemble mean), since Table 3 is a
PSNR/SSIM (point-estimate) leaderboard whose leaders are effectively posterior-mean
estimators; a single highest-weight particle is a posterior *sample* and loses PSNR.
Use `reduce='best'`/`'topk'` for posterior-sample draws (uncertainty quantification).
"""
import torch

from algo.base import Algo
from algo.afdps_core_scatter import Ensemble_Denoiser_EDM


class _Noiser:
    """The sampler only reads `.sigma` (the effective measurement-noise std)."""
    def __init__(self, sigma):
        self.sigma = sigma


class AFDPSScatter(Algo):
    def __init__(self, net, forward_op,
                 num_particles=512,
                 num_steps=200,
                 sigma_min=None,
                 sigma_max=None,
                 rho=7,
                 discretization='edm',
                 schedule='linear',
                 scaling='none',
                 mode='sde',
                 likelihood_at='noisy',       # linear forward never diverges -> 'noisy' is exact & paper-faithful
                 guidance_gamma=1.0,
                 reduce='mean',               # 'mean' (Table-3 number) | 'uniform_mean' | 'best' | 'topk' (UQ draws)
                 final_projection=False,      # post-hoc hard data consistency on measured V-modes
                 projection_tau=1e-3,         # relative singular-value cutoff for the projection
                 sampler_kwargs=None):
        super().__init__(net, forward_op)
        self.num_particles = num_particles
        self.reduce = reduce
        self.final_projection = bool(final_projection)
        self.projection_tau = float(projection_tau)
        self.sampler = Ensemble_Denoiser_EDM(
            net=net, device=forward_op.device,
            num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            discretization=discretization, schedule=schedule, scaling=scaling, mode=mode,
            likelihood_at=likelihood_at, guidance_gamma=guidance_gamma,
            **(sampler_kwargs or {}))

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        op = self.forward_op
        op.set_observation(observation)               # fold the affine map + cache U^T y~
        res = self.net.img_resolution
        ch = self.net.img_channels
        gt_dummy = torch.zeros(1, ch, res, res, device=op.device)
        noiser = _Noiser(op.sigma_noise_eff)          # folded sigma~ (normalized-x scale)

        out = self.sampler(gt_dummy, observation, self.num_particles, op, noiser)
        ens = out['ensemble']                         # (J, C, H, W)
        lw = out['log_weights']                       # (J,)
        w = torch.softmax(lw, dim=0)

        if self.reduce == 'mean':
            recon = (w.view(-1, 1, 1, 1) * ens).sum(dim=0, keepdim=True)
            recon = recon.repeat(num_samples, 1, 1, 1)
        elif self.reduce == 'uniform_mean':
            # Unweighted posterior-mean estimator: (1/J) sum_j x_j.
            #
            # WHY THIS EXISTS. The Feynman-Kac log-weights degenerate to ESS=1
            # (measured top1-top2 gaps of 1e3-1e5 nats), so 'mean' silently returns
            # a single particle -- a posterior SAMPLE. PSNR is an MSE metric whose
            # optimal estimator is the posterior MEAN, and for a sample
            #     E||x_sample - x*||^2 = bias^2 + sigma_p^2,
            # versus for a J-average
            #     E||x_bar - x*||^2   = bias^2 + sigma_p^2 / J.
            # Discarding the degenerate weights recovers that variance term.
            recon = ens.mean(dim=0, keepdim=True).repeat(num_samples, 1, 1, 1)
        else:  # 'best' / 'topk' -- posterior-sample draws
            order = torch.argsort(lw, descending=True)
            if num_samples <= ens.shape[0]:
                recon = ens[order[:num_samples]]
            else:
                idx = torch.multinomial(w, num_samples, replacement=True)
                recon = ens[idx]

        if self.final_projection:
            # Post-hoc DDNM-style range-space consistency: replace well-measured
            # V-modes with their exact data values (see the operator docstring).
            # Applied AFTER the reduction so the null-space content is whatever
            # estimator was chosen above; removes the residual guidance bias on
            # measured modes exactly, at zero PDE/network cost.
            recon = op.data_consistent_projection(recon, rel_tau=self.projection_tau)

        return recon.to(torch.float32)
