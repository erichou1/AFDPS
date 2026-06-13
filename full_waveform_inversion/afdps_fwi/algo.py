"""AFDPS as an InverseBench algorithm for full waveform inversion.

Wraps the vendored `Ensemble_Denoiser_EDM` sampler behind the InverseBench
`Algo.inference(observation, num_samples)` interface. This is a slimmed copy of the
Navier-Stokes `algo/afdps.py`: the FWI prior is always the pretrained EDM denoiser
(`checkpoints/fwi-5m.pt`), so the analytic Gaussian-random-field prior branch used for
the Navier-Stokes verification track is removed (it has no meaning for FWI and would
otherwise pull in the Navier-Stokes-specific `ns_adjoint` module).

`Algo` is imported from the InverseBench harness shipped in `navier_stokes/`, which is
placed on `sys.path` by `full_waveform_inversion/main.py`.
"""
import torch

from algo.base import Algo  # from the reused navier_stokes/ harness (on sys.path)

from .sampler import Ensemble_Denoiser_EDM


class _Noiser:
    """The sampler only reads `.sigma` (the measurement-noise std). For the noise-free
    FWI benchmark this carries the likelihood *temperature* sigma_y instead (see the
    operator / config docs)."""

    def __init__(self, sigma):
        self.sigma = sigma


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
                 reduce='best',               # 'best' | 'mean' | 'topk'
                 sampler_kwargs=None):
        super().__init__(net, forward_op)
        self.num_particles = num_particles
        self.reduce = reduce
        self.sampler = Ensemble_Denoiser_EDM(
            net=net, device=forward_op.device,
            num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            discretization=discretization, schedule=schedule, scaling=scaling, mode=mode,
            likelihood_at=likelihood_at, guidance_gamma=guidance_gamma,
            **(sampler_kwargs or {}))

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        op = self.forward_op
        # Stash the observation so likelihood_laplacian / initialize_ensemble can reach it.
        # For FWI `observation` is the list of Devito Receiver objects returned by the
        # operator's __call__ (the recorded shot gathers); it is passed straight through to
        # the operator's gradient/loss, which already accept that representation.
        op._y = observation
        res = self.net.img_resolution
        ch = self.net.img_channels
        gt_dummy = torch.zeros(1, ch, res, res, device=op.device)
        noiser = _Noiser(op.sigma_noise)

        out = self.sampler(gt_dummy, observation, self.num_particles, op, noiser)
        ens = out['ensemble']                     # (J, C, H, W)
        lw = out['log_weights']                   # (J,)
        w = torch.softmax(lw, dim=0)

        if self.reduce == 'mean':
            # Posterior mean under the Feynman-Kac weights.
            recon = (w.view(-1, 1, 1, 1) * ens).sum(dim=0, keepdim=True)
            recon = recon.repeat(num_samples, 1, 1, 1)
        else:  # 'best' / 'topk': return the highest-weight particle(s) (the AFDPS MAP estimator)
            order = torch.argsort(lw, descending=True)
            if num_samples <= ens.shape[0]:
                recon = ens[order[:num_samples]]
            else:
                idx = torch.multinomial(w, num_samples, replacement=True)
                recon = ens[idx]
        return recon
