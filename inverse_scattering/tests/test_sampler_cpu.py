"""End-to-end sampler checks on CPU with a stub Gaussian denoiser (no UNet/data/GPU).

  * smoke: the full AFDPSScatter.inference path runs, returns finite/correct-shaped
    output, and is deterministic for a fixed seed -- across guidance modes/steps.
  * linear-Gaussian twin (the acceptance gate): with a Gaussian prior the posterior is
    closed-form in the SVD basis; AFDPS's weighted ensemble mean must recover that
    posterior mean far better than the prior mean. This is the only check that
    exercises the state dynamics AND the Feynman-Kac weights jointly.
"""
import math
import pytest
import torch

pytest.importorskip("scipy")

from inverse_problems.inverse_scatter_afdps import AFDPSInverseScatter
from algo.afdps_scatter import AFDPSScatter


class GaussianDenoiser:
    """EDM-net-compatible exact Tweedie denoiser of an isotropic Gaussian prior
    N(0, rho^2 I):  D(x, sigma) = rho^2/(rho^2 + sigma^2) * x."""
    def __init__(self, rho, res, ch=1, sigma_min=2e-3, sigma_max=80.0):
        self.rho = float(rho)
        self.img_resolution = res
        self.img_channels = ch
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma):
        s = float(sigma)
        return (self.rho ** 2 / (self.rho ** 2 + s ** 2)) * x


def _make_op(tmp_path, sigma_noise, scale=0.5, shift=1.0, Nx=16, numTrans=3, numRec=6):
    import os
    os.chdir(tmp_path)
    return AFDPSInverseScatter(Lx=0.18, Ly=0.18, Nx=Nx, Ny=Nx, wave=6,
                               numRec=numRec, numTrans=numTrans, sensorRadius=1.6,
                               sigma_noise=sigma_noise, unnorm_shift=shift, unnorm_scale=scale,
                               device='cpu', svd=True)


@pytest.mark.parametrize("gmode,gstep", [("full", "exact_linear"),
                                          ("fixed", "euler"),
                                          ("auto", "exact_linear")])
def test_sampler_smoke_and_determinism(tmp_path, gmode, gstep):
    torch.manual_seed(0)
    op = _make_op(tmp_path, sigma_noise=1e-2)
    net = GaussianDenoiser(rho=1.0, res=op.Nx)
    obs = op.forward(2 * torch.rand(1, 1, op.Ny, op.Nx) - 1, unnormalize=True)

    kw = dict(guidance_mode=gmode, guidance_step=gstep, use_value=True,
              value_coef=('exact' if gstep == 'exact_linear' else 't0'), progress=False)
    algo = AFDPSScatter(net, op, num_particles=8, num_steps=5, sigma_max=80.0,
                        guidance_gamma=1.0, reduce='mean', sampler_kwargs=kw)

    torch.manual_seed(123)
    r1 = algo.inference(obs, num_samples=1)
    torch.manual_seed(123)
    r2 = algo.inference(obs, num_samples=1)

    assert r1.shape == (1, 1, op.Ny, op.Nx)
    assert torch.isfinite(r1).all()
    assert torch.equal(r1, r2), "same seed must give identical CPU output"


@pytest.mark.parametrize("gmode,gstep", [("full", "exact_linear"), ("auto", "exact_linear")])
def test_linear_gaussian_twin(tmp_path, gmode, gstep):
    # Gaussian prior N(0, rho^2 I) + Gaussian likelihood => closed-form posterior whose
    # mean is diagonal in the V-basis:
    #   mean_u_i = (S_i y~U_i / sig~^2) / (1/rho^2 + S_i^2 / sig~^2),  x_pm = V mean_u.
    # A posterior *sampler* explores the prior on UNMEASURED (S~0) directions, so the
    # meaningful, weight-degeneracy-robust check is that the AFDPS weighted mean recovers
    # the posterior mean on the WELL-MEASURED subspace (S^2/sig~^2 large). The noise is
    # set small enough that a sizeable subspace is informative (the tiny test operator
    # has S_max ~ 1e-2). This jointly exercises the guidance dynamics and the FK weights.
    rho = 1.0
    op = _make_op(tmp_path, sigma_noise=3e-4, Nx=16, numTrans=3, numRec=6)
    sig2 = op.sigma_noise_eff ** 2

    torch.manual_seed(7)
    x_true = rho * torch.randn(1, 1, op.Ny, op.Nx)
    obs = op.forward(x_true, unnormalize=True)            # noiseless A * unnormalize(x_true)
    op.set_observation(obs)

    mean_u = (op._S * op._yU / sig2) / (1.0 / rho ** 2 + op._S2 / sig2)   # (k,) posterior mean coeffs
    measured = (op._S2 / sig2) > 5.0                                      # well-informed directions
    assert int(measured.sum()) >= 5, "test operator should have several measured modes"

    net = GaussianDenoiser(rho=rho, res=op.Nx)
    kw = dict(guidance_mode=gmode, guidance_step=gstep, use_value=True, value_coef='exact',
              progress=False)
    algo = AFDPSScatter(net, op, num_particles=512, num_steps=300, sigma_max=20.0,
                        guidance_gamma=1.0, reduce='mean', sampler_kwargs=kw)

    torch.manual_seed(0)
    recon = algo.inference(obs, num_samples=1)
    u_rec = op._to_coeff(recon).squeeze(0)

    rel = (u_rec[measured] - mean_u[measured]).norm() / mean_u[measured].norm()
    assert torch.isfinite(recon).all()
    assert float(recon.norm()) < 3.0 * math.sqrt(op._n)   # not blown up (null-space prior scale)
    assert float(rel) < 0.25, f"[{gmode}/{gstep}] measured-subspace posterior-mean rel err = {float(rel):.3f}"

