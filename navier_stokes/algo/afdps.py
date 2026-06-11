"""AFDPS as an InverseBench algorithm.

Wraps the vendored `Ensemble_Denoiser_EDM` sampler behind the `Algo.inference`
interface. The same sampler runs two prior modes:
    prior='diffusion'  -> the pretrained EDM denoiser `net` (Track B benchmark);
    prior='gaussian'   -> `GRFScoreNet`, the exact Tweedie denoiser of the analytic
                          GRF prior N(0,(-Delta+9I)^{-4})  (Track A verification).
"""
import torch

from .base import Algo
from .afdps_core import Ensemble_Denoiser_EDM
from inverse_problems import ns_adjoint as A


class _Noiser:
    """The sampler only reads `.sigma` (the measurement-noise std)."""
    def __init__(self, sigma):
        self.sigma = sigma


class GRFScoreNet:
    """EDM-`net`-compatible exact denoiser for the Gaussian random field prior.

    D(x, sigma) = C (C + sigma^2 I)^{-1} x with C = (-Delta + 9I)^{-4}, diagonal in
    Fourier. Assumes the sampler state lives in the same units as the prior, i.e.
    `unnorm_scale == 1` (Track A sets this).
    """
    def __init__(self, solver, img_channels=1, amp=1.0):
        self.solver = solver
        self.img_channels = img_channels
        self.img_resolution = solver.s1
        self.amp = amp
        self.sigma_min = 1e-9
        self.sigma_max = 1e4

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma):
        sig = float(sigma)
        d = A.grf_denoiser(self.solver, x.squeeze(1).to(self.solver.G.dtype), sig, amp=self.amp)
        return d.unsqueeze(1).to(x.dtype)


class AFDPS(Algo):
    def __init__(self, net, forward_op,
                 num_particles=10,
                 num_steps=100,
                 sigma_min=None,
                 sigma_max=None,
                 rho=7,
                 discretization='edm',
                 schedule='linear',
                 scaling='none',
                 mode='sde',
                 likelihood_at='denoised',
                 guidance_gamma=1.0,
                 prior='diffusion',           # 'diffusion' | 'gaussian'
                 prior_amp=1.0,               # GRF prior amplitude (prior='gaussian')
                 reduce='best',               # 'best' | 'mean' | 'topk'
                 sampler_kwargs=None):
        super().__init__(net, forward_op)
        self.num_particles = num_particles
        self.prior = prior
        self.reduce = reduce
        if prior == 'gaussian':
            # GRFScoreNet denoises in the sampler's state domain assuming it equals the
            # physical domain (the GRF prior is defined in physical vorticity units).
            assert abs(getattr(forward_op, 'unnorm_scale', 1.0) - 1.0) < 1e-9, \
                "prior='gaussian' requires forward_op.unnorm_scale == 1 (analytic GRF prior is in physical units)."
        eff_net = net if prior == 'diffusion' else GRFScoreNet(
            forward_op.solver, img_channels=net.img_channels if net is not None else 1, amp=prior_amp)
        self.eff_net = eff_net
        self.sampler = Ensemble_Denoiser_EDM(
            net=eff_net, device=forward_op.device,
            num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            discretization=discretization, schedule=schedule, scaling=scaling, mode=mode,
            likelihood_at=likelihood_at, guidance_gamma=guidance_gamma,
            **(sampler_kwargs or {}))

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        op = self.forward_op
        op._y = observation                       # for likelihood_laplacian / init
        res = self.eff_net.img_resolution
        ch = self.eff_net.img_channels
        gt_dummy = torch.zeros(1, ch, res, res, device=op.device)
        noiser = _Noiser(op.sigma_noise)

        out = self.sampler(gt_dummy, observation, self.num_particles, op, noiser)
        ens = out['ensemble']                     # (J, C, H, W)
        lw = out['log_weights']                   # (J,)
        w = torch.softmax(lw, dim=0)

        if self.reduce == 'mean':
            recon = (w.view(-1, 1, 1, 1) * ens).sum(dim=0, keepdim=True)
            recon = recon.repeat(num_samples, 1, 1, 1)
        else:  # 'best' / 'topk'
            order = torch.argsort(lw, descending=True)
            if num_samples <= ens.shape[0]:
                recon = ens[order[:num_samples]]
            else:
                idx = torch.multinomial(w, num_samples, replacement=True)
                recon = ens[idx]
        return recon
