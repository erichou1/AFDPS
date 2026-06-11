"""CPU verification ladder for AFDPSInverseScatter (tiny grid, no data/GPU needed).

Validates the exact-SVD likelihood engine and the affine-normalization fold against
the UNTOUCHED InverseBench operator, so a pass means the math is right before any
GB200 run. Run: `pytest inverse_scattering/tests/test_operator_cpu.py -q`.
"""
import math
import pytest
import torch

pytest.importorskip("scipy")  # InverseScatter imports scipy.special.hankel1 at module load

from inverse_problems.inverse_scatter import InverseScatter
from inverse_problems.inverse_scatter_afdps import AFDPSInverseScatter

SIGMA_NOISE = 1e-4
SHIFT, SCALE = 1.0, 0.5


@pytest.fixture(scope="module")
def op(tmp_path_factory):
    # cache/ is CWD-relative inside compute_svd -> run from a tmp dir.
    d = tmp_path_factory.mktemp("scatter_cache")
    cwd = __import__("os").getcwd()
    __import__("os").chdir(d)
    torch.manual_seed(0)
    o = AFDPSInverseScatter(Lx=0.18, Ly=0.18, Nx=16, Ny=16, wave=6,
                            numRec=4, numTrans=2, sensorRadius=1.6,
                            sigma_noise=SIGMA_NOISE, unnorm_shift=SHIFT, unnorm_scale=SCALE,
                            device='cpu', svd=True)
    yield o
    __import__("os").chdir(cwd)


def _rand_x(op, J=3):
    return (2 * torch.rand(J, 1, op.Ny, op.Nx, dtype=torch.float32) - 1)


def test_complex_noise_convention():
    # The benchmark adds sigma * randn_like(complex): per-real-component variance sigma^2/2.
    torch.manual_seed(1)
    z = torch.randn(2_000_000, dtype=torch.complex128)
    assert abs(z.real.var().item() - 0.5) < 1e-2
    # => folded sigma~ = sigma_y / (sqrt(2) * scale)
    assert abs(math.sqrt(0.5) / SCALE - 1.0 / (math.sqrt(2.0) * SCALE)) < 1e-12


def test_fairness_no_override(op):
    # The AFDPS subclass must NOT change observation generation / data consistency.
    assert AFDPSInverseScatter.forward is InverseScatter.forward
    assert AFDPSInverseScatter.__call__ is InverseScatter.__call__
    assert AFDPSInverseScatter.loss is InverseScatter.loss
    assert AFDPSInverseScatter.normalize is InverseScatter.normalize
    assert AFDPSInverseScatter.unnormalize is InverseScatter.unnormalize


def test_sigma_fold(op):
    assert abs(op.sigma_noise_eff - SIGMA_NOISE / (math.sqrt(2.0) * SCALE)) < 1e-12


def test_matrix_matches_forward(op):
    # A @ f_phys (real-stacked) == view_as_real(forward(f_phys)).flatten().
    f = _rand_x(op, J=1)
    y = op.forward(f, unnormalize=False)                      # (1, nT, nR) complex
    y_real = torch.view_as_real(y.flatten()).flatten().double()
    A_f = (op.A @ f.flatten().double())
    assert torch.allclose(A_f, y_real, atol=1e-8, rtol=1e-6)


def test_fold_consistency_zero_residual(op):
    # If the observation is exactly A f(x_true) (noiseless), mu_y(x_true) and grad must be ~0.
    x_true = _rand_x(op, J=1)
    obs = op.forward(x_true, unnormalize=True)                 # = A * unnormalize(x_true)
    op.set_observation(obs)
    val = op.likelihood_value(x_true, obs, op.sigma_noise_eff)
    g = op.likelihood_gradient(x_true, obs, op.sigma_noise_eff)
    assert float(val.max()) < 1e-6
    assert float(g.abs().max()) < 1e-4


def test_gradient_matches_autograd_through_benchmark(op):
    # Our exact gradient == autograd of the BENCHMARK loss / sigma_y^2 (the fold algebra).
    x = _rand_x(op, J=1)
    obs = op.forward(_rand_x(op, J=1), unnormalize=True)
    op.set_observation(obs)
    g_ours = op.likelihood_gradient(x, obs, op.sigma_noise_eff).double()

    pred = x.clone().requires_grad_(True)
    loss = op.loss(pred, obs).sum()
    g_bench = torch.autograd.grad(loss, pred)[0].double()
    expected = g_bench / (SIGMA_NOISE ** 2)
    assert torch.allclose(g_ours, expected, rtol=1e-6, atol=1e-6)


def test_laplacian_is_exact_frobenius(op):
    # Tr(Hessian) = ||A||_F^2 / sigma^2 = sum_i s_i^2 / sigma^2 (constant).
    x = _rand_x(op, J=4)
    obs = op.forward(_rand_x(op, J=1), unnormalize=True)
    op.set_observation(obs)
    s = op.sigma_noise_eff
    lap = op.likelihood_laplacian(x, s)
    fro = float((op.A ** 2).sum())
    assert torch.allclose(op._S2.sum().cpu(), torch.tensor(fro, dtype=torch.float64), rtol=1e-6)
    assert torch.allclose(lap, torch.full((4,), fro / s ** 2, dtype=torch.float64), rtol=1e-6)
    assert lap.shape == (4,)


def test_jacobian_trace_exact(op):
    lam = op.jacobian_trace(_rand_x(op, J=2))
    assert abs(lam - float(op._S2.sum()) / op._n_meas) < 1e-9


def test_pigdm_matches_dense_solve(op):
    # Closed-form PiGDM == dense measurement-space solve A^T (sy^2 I + st^2 A A^T)^-1 r.
    x = _rand_x(op, J=1)
    obs = op.forward(_rand_x(op, J=1), unnormalize=True)
    op.set_observation(obs)
    sy, st = op.sigma_noise_eff, 0.7
    g = op.likelihood_gradient_pigdm(x, obs, sy, st).double().flatten()

    A = op.A
    r = (A @ x.flatten().double()) - op._y_tilde                 # A x - y~  (2m,)
    M = (sy ** 2) * torch.eye(A.shape[0], dtype=torch.float64) + (st ** 2) * (A @ A.T)
    z = torch.linalg.solve(M, r)
    g_dense = A.T @ z
    assert torch.allclose(g, g_dense, rtol=1e-5, atol=1e-8)


def test_exact_linear_matches_fine_euler(op):
    # The closed-form guidance substep == many-step Euler on the same linear ODE (fp64).
    x = _rand_x(op, J=2)
    obs = op.forward(_rand_x(op, J=1), unnormalize=True)
    op.set_observation(obs)
    t_cur, t_next, gamma_e2 = 0.5, 0.4, 1.0
    sig2 = op.sigma_noise_eff ** 2

    # exact closed form (coefficient space)
    u0 = op._to_coeff(x)
    x_exact = op.exact_linear_substep(x.double(), t_cur, t_next, mode='iso', gamma_e2=gamma_e2)
    u_exact = op._to_coeff(x_exact)

    # fine fp64 Euler on du/dt = 2 t (S * (S u - yU)) / r_t^2
    u = u0.clone()
    N = 20000
    ts = torch.linspace(t_cur, t_next, N + 1, dtype=torch.float64)
    for j in range(N):
        tc = float(ts[j]); dt = float(ts[j + 1] - ts[j])
        r2 = sig2 + gamma_e2 * tc ** 2
        rho = op._S * u - op._yU
        u = u + dt * (2 * tc * (op._S * rho) / r2)
    rel = (u_exact - u).norm() / u.norm().clamp_min(1e-12)
    assert float(rel) < 1e-3, f"exact_linear vs fine Euler rel={float(rel):.2e}"
