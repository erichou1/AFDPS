"""CPU smoke test for the AFDPS x FWI port -- runs WITHOUT Devito / GPU / checkpoints.

It exercises the parts that are problem-independent and pure-PyTorch:
  * the vendored annealed-SDE Feynman-Kac ensemble sampler (`afdps_fwi.sampler`);
  * the AFDPS `Algo.inference` wrapper reduction (`afdps_fwi.algo`);
  * the operator API contract the sampler depends on, via a tiny LINEAR mock operator
    A(x) = W x (so mu_y, its gradient, and the EXACT Hessian trace are known in closed
    form) -- this validates the finite-difference Hutchinson trace estimator math that the
    real FWI operator uses (`AFDPSAcousticWave.likelihood_laplacian('fd_divergence')`).

The real `afdps_fwi.operator.AFDPSAcousticWave` needs Devito and the InverseBench harness,
so it is NOT imported here; this test guards the wiring/maths that CAN be checked offline.

Run:  python full_waveform_inversion/tests/test_sampler_cpu.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FWI = os.path.abspath(os.path.join(_HERE, '..'))
_NS = os.path.abspath(os.path.join(_FWI, '..', 'navier_stokes'))
for _p in (_FWI, _NS):
    if _p not in sys.path:
        sys.path.append(_p)

from afdps_fwi.sampler import Ensemble_Denoiser_EDM  # noqa: E402
from afdps_fwi.algo import _Noiser  # noqa: E402


# --------------------------------------------------------------------------- #
# Mock EDM denoiser: a mild shrink toward 0 (a proper, well-behaved "prior").   #
# --------------------------------------------------------------------------- #
class MockEDMNet:
    def __init__(self, res=8, ch=1):
        self.img_resolution = res
        self.img_channels = ch
        self.sigma_min = 1e-3
        self.sigma_max = 80.0

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma):
        # Tweedie denoiser of a N(0, 1) prior: D(x) = x / (1 + sigma^2).
        s = float(sigma)
        return x / (1.0 + s ** 2)


# --------------------------------------------------------------------------- #
# Mock LINEAR operator A(x) = W x with the SAME API the FWI operator exposes.   #
# mu_y(x) = ||Wx - y||^2 / (2 sigma^2); grad = (1/sigma^2) W^T(Wx - y);          #
# Tr(Hess) = (1/sigma^2) ||W||_F^2 (constant) -- the ground truth for the         #
# Hutchinson finite-difference estimator.                                       #
# --------------------------------------------------------------------------- #
class MockLinearOperator:
    def __init__(self, res=8, ch=1, device='cpu', sigma_noise=1.0, seed=0):
        self.device = device
        self.sigma_noise = sigma_noise
        self.res = res
        self.ch = ch
        self.n = ch * res * res
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(self.n, self.n, generator=g, device=device) / (self.n ** 0.5)
        self.sigma_floor = 1e-3
        self.hutchinson_M = 4
        self.hutchinson_eps = 1e-3
        self.hutchinson_scheme = 'central'
        self._y = None

    def _flat(self, x):
        return x.reshape(x.shape[0], -1)

    def _A(self, x):
        return (self._flat(x) @ self.W.T)

    def likelihood_gradient(self, x, y, sigma):
        se = max(float(sigma), self.sigma_floor)
        r = self._A(x) - self._flat(y)               # (B, n)
        g = (r @ self.W) / (se ** 2)                 # W^T r
        return g.reshape_as(x)

    def likelihood_value(self, x, y, sigma):
        se = max(float(sigma), self.sigma_floor)
        r = self._A(x) - self._flat(y)
        return 0.5 * r.pow(2).sum(dim=1) / (se ** 2)

    def likelihood_laplacian(self, x, sigma, g0=None):
        # finite-difference divergence Hutchinson -- the exact estimator the FWI operator's
        # 'fd_divergence' mode uses (afdps_fwi/operator.py:_hutchinson_fd).
        grad_fn = lambda xx: self.likelihood_gradient(xx, self._y, sigma)
        M, eps = self.hutchinson_M, self.hutchinson_eps
        xi = (torch.randint(0, 2, (M, *x.shape), device=x.device).to(x.dtype) * 2 - 1)
        if self.hutchinson_scheme == 'forward':
            Xp = (x.unsqueeze(0) + eps * xi).reshape(M * x.shape[0], *x.shape[1:])
            gp = grad_fn(Xp).reshape(M, *x.shape)
            hvp = (gp - g0.unsqueeze(0)) / eps
        else:
            Xp = x.unsqueeze(0) + eps * xi
            Xm = x.unsqueeze(0) - eps * xi
            X = torch.cat([Xp, Xm], dim=0).reshape(2 * M * x.shape[0], *x.shape[1:])
            gg = grad_fn(X).reshape(2, M, *x.shape)
            hvp = (gg[0] - gg[1]) / (2 * eps)
        return (hvp * xi).flatten(start_dim=2).sum(dim=2).mean(dim=0)

    def exact_laplacian(self, sigma):
        se = max(float(sigma), self.sigma_floor)
        return (self.W.pow(2).sum() / (se ** 2)).item()

    def initialize_ensemble(self, gt, num_particles):
        return torch.zeros(num_particles, self.ch, self.res, self.res, device=self.device)

    def proximal_generator(self, x_ref, y, sigma, scale):
        return x_ref + scale * torch.randn_like(x_ref)


def test_hutchinson_trace_matches_exact():
    torch.manual_seed(0)
    op = MockLinearOperator(res=6, sigma_noise=1.0)
    x = torch.randn(3, 1, 6, 6)
    op._y = torch.randn(1, 1, 6, 6)
    sigma = 1.0
    # average many estimates -> should converge to the exact (1/sigma^2)||W||_F^2
    ests = torch.stack([op.likelihood_laplacian(x, sigma) for _ in range(400)]).mean(0)
    exact = op.exact_laplacian(sigma)
    rel = (ests - exact).abs().max().item() / abs(exact)
    print(f"[trace] exact={exact:.4f} est~{ests.mean():.4f} max-rel-err={rel:.3f}")
    assert rel < 0.05, f"Hutchinson trace off by {rel:.3f} (>5%)"


def test_sampler_runs_and_fits():
    torch.manual_seed(0)
    res = 6
    net = MockEDMNet(res=res)
    op = MockLinearOperator(res=res, sigma_noise=0.5)
    # ground-truth signal & its (noise-free) measurement
    x_true = torch.randn(1, 1, res, res)
    y = (op._flat(x_true) @ op.W.T).reshape(1, 1, res, res)
    op._y = y

    sampler = Ensemble_Denoiser_EDM(
        net=net, device='cpu', num_steps=60, sigma_max=20.0, mode='sde',
        likelihood_at='denoised', guidance_gamma=1.0, progress=False,
        resample=True, resample_threshold=0.5, use_value=True)
    out = sampler(torch.zeros(1, 1, res, res), y, num_particles=8, operator=op, noiser=_Noiser(op.sigma_noise))
    ens, lw = out['ensemble'], out['log_weights']
    assert torch.isfinite(ens).all(), "non-finite particles in the ensemble"
    assert torch.isfinite(lw).any(), "all log-weights are non-finite"

    best = ens[torch.argmax(lw)][None]
    init = torch.zeros(1, 1, res, res)
    misfit_best = op.likelihood_value(best, y, op.sigma_noise).item()
    misfit_init = op.likelihood_value(init, y, op.sigma_noise).item()
    print(f"[sampler] misfit init={misfit_init:.3f} -> best particle={misfit_best:.3f}")
    assert misfit_best < misfit_init, "AFDPS did not reduce the data misfit vs the init"


def test_algo_reduction_shapes():
    import afdps_fwi.algo as A
    torch.manual_seed(0)
    res = 6
    net = MockEDMNet(res=res)
    op = MockLinearOperator(res=res, sigma_noise=0.5)
    y = torch.randn(1, 1, res, res)
    algo = A.AFDPS(net=net, forward_op=op, num_particles=6, num_steps=20,
                   sigma_max=20.0, guidance_gamma=1.0,
                   sampler_kwargs={'progress': False})
    for reduce in ('best', 'mean'):
        algo.reduce = reduce
        recon = algo.inference(y, num_samples=2)
        assert recon.shape == (2, 1, res, res), f"{reduce}: bad recon shape {recon.shape}"
        assert torch.isfinite(recon).all()
    print("[algo] reductions OK (best/mean), shapes (2,1,6,6)")


if __name__ == '__main__':
    test_hutchinson_trace_matches_exact()
    test_sampler_runs_and_fits()
    test_algo_reduction_shapes()
    print("\nAll CPU smoke tests passed.")
