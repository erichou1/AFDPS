"""Numerical verification ladder for the Navier-Stokes adjoint/gradient engine.

Run:  python verification/checks.py
All checks run in float64 on CPU at small resolution. Each rung prints a metric
and PASS/FAIL against a tolerance. This is the adversarial verification for the
numerical core -- finite-difference gradient checks, adjoint consistency, and a
brute-force trace comparison are far stronger for numerical code than code review.
"""
import os
import sys
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_problems.navier_stokes import NavierStokes2d
from inverse_problems import ns_adjoint as A

torch.manual_seed(0)
DT = torch.float64
DEV = 'cpu'
L = 2 * math.pi


def make_solver(n):
    return NavierStokes2d(n, n, L, L, device=DEV, dtype=DT)


def forcing(n):
    t = torch.linspace(0, L, n + 1, device=DEV, dtype=DT)[:-1]
    _, y = torch.meshgrid(t, t, indexing='ij')
    return -4 * torch.cos(4.0 * y)


def _ip(a, b):
    return (a * b).sum().item()


results = []


def report(name, val, tol, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.3e}  (tol {tol:.1e})")


# --------------------------------------------------------------------------- #
def rung0_primitives(n=32, factor=2):
    print("\n== Rung 0: FFT/measurement adjoint primitives ==")
    s = make_solver(n)
    x = torch.randn(1, n, n, dtype=DT)
    # irfft2 . rfft2 == I
    rt = fft_roundtrip = torch.fft.irfft2(torch.fft.rfft2(x), s=(n, n))
    e = (rt - x).abs().max().item()
    report("irfft2(rfft2(x)) == x", e, 1e-12, e < 1e-12)
    # P / P* adjoint: <P x, r> == <x, P* r>
    r = torch.randn(1, n // factor, n // factor, dtype=DT)
    lhs = _ip(A.downsample(x, factor), r)
    rhs = _ip(x, A.upsample_adjoint(r, factor, n, n))
    e = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    report("<P x, r> == <x, P* r>", e, 1e-12, e < 1e-12)


def rung1_forward(n=32):
    print("\n== Rung 1: forward solver sanity ==")
    s = make_solver(n)
    w0 = A.grf_sample(s, 1) * 5.0  # realistic amplitude
    # (a) enstrophy non-increasing with f=0
    dt, T = 2e-3, 0.2
    ens0 = w0.pow(2).sum().item()
    wT = A.forward_solve(s, w0, None, T, 200.0, dt)
    ensT = wT.pow(2).sum().item()
    report("enstrophy(T) <= enstrophy(0) [f=0]", ensT - ens0, 1e-9, ensT <= ens0 + 1e-9)
    # (b) mean-zero preserved
    mean = wT.mean().abs().item()
    report("mean(omega(T)) == 0", mean, 1e-10, mean < 1e-10)
    # (c) dt convergence order ~2 (Heun+CN), with forcing
    f = forcing(n)
    Re, T = 200.0, 0.1
    sols = {}
    for k in [1, 2, 4]:
        dt = (1e-2) / k
        sols[k] = A.forward_solve(s, w0, f, T, Re, dt)
    e1 = (sols[1] - sols[4]).norm().item()
    e2 = (sols[2] - sols[4]).norm().item()
    order = math.log(e1 / e2) / math.log(2) if e2 > 0 else float('nan')
    report("dt-convergence order (~2)", order, 0.0, 1.6 <= order <= 2.4)


def rung3_gradient_fd(n=32, factor=2):
    print("\n== Rung 3: finite-difference gradient check (decisive) ==")
    s = make_solver(n)
    f = forcing(n)
    Re, T, dt, sigma = 200.0, 0.1, 5e-3, 1.0
    w0 = A.grf_sample(s, 1) * 5.0
    # build observation from a different field so misfit != 0
    w_true = A.grf_sample(s, 1) * 5.0
    y = A.downsample(A.forward_solve(s, w_true, f, T, Re, dt), factor).detach()

    g = A.grad_misfit_autograd(s, w0, f, T, Re, dt, y, factor, sigma)
    h = torch.randn_like(w0)
    h = h / h.norm()
    gdot_h = _ip(g, h)

    def mu(w):
        return A.misfit(s, w, f, T, Re, dt, y, factor, sigma).sum().item()

    best = float('inf')
    for eps in [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        fd = (mu(w0 + eps * h) - mu(w0 - eps * h)) / (2 * eps)
        rel = abs(fd - gdot_h) / (abs(gdot_h) + 1e-30)
        best = min(best, rel)
        print(f"    eps={eps:.1e}  fd={fd:.6e}  grad.h={gdot_h:.6e}  rel={rel:.2e}")
    report("min rel err (autograd grad vs central FD)", best, 1e-5, best < 1e-5)

    # Taylor remainder: |mu(w+eps h)-mu(w)-eps<g,h>| = O(eps^2) -> slope ~2
    m0 = mu(w0)
    rs = []
    for eps in [1e-2, 5e-3, 2.5e-3]:
        rem = abs(mu(w0 + eps * h) - m0 - eps * gdot_h)
        rs.append((eps, rem))
    slope = math.log(rs[0][1] / rs[2][1]) / math.log(rs[0][0] / rs[2][0])
    report("Taylor-remainder slope (~2)", slope, 0.0, 1.6 <= slope <= 2.4)


def rung4_continuous_adjoint(n=32, factor=2):
    print("\n== Rung 4: continuous adjoint vs discrete (autograd) -- O(dt) trend ==")
    s = make_solver(n)
    f = forcing(n)
    Re, T, sigma = 200.0, 0.1, 1.0
    w0 = A.grf_sample(s, 1) * 5.0
    w_true = A.grf_sample(s, 1) * 5.0
    prev = None
    rels = []
    for dt in [4e-3, 2e-3, 1e-3]:
        y = A.downsample(A.forward_solve(s, w_true, f, T, Re, dt), factor).detach()
        g_ad = A.grad_misfit_autograd(s, w0, f, T, Re, dt, y, factor, sigma)
        g_co = A.continuous_adjoint_gradient(s, w0, f, T, Re, dt, y, factor, sigma)
        rel = (g_ad - g_co).norm().item() / (g_ad.norm().item() + 1e-30)
        rels.append((dt, rel))
        print(f"    dt={dt:.1e}  ||g_ad - g_co|| / ||g_ad|| = {rel:.3e}")
    # expect decreasing with dt; require it shrinks and is moderate
    decreasing = rels[0][1] > rels[-1][1]
    report("continuous-adjoint rel err shrinks with dt", rels[-1][1], 5e-1, decreasing and rels[-1][1] < 0.5)


def rung5_hutchinson(n=16, factor=2):
    print("\n== Rung 5: Hutchinson Laplacian vs brute-force trace (small grid) ==")
    s = make_solver(n)
    f = forcing(n)
    Re, T, dt, sigma = 200.0, 0.08, 4e-3, 1.0
    w0 = A.grf_sample(s, 1) * 5.0
    w_true = A.grf_sample(s, 1) * 5.0
    y = A.downsample(A.forward_solve(s, w_true, f, T, Re, dt), factor).detach()

    def grad_fn(w):
        return A.grad_misfit_autograd(s, w, f, T, Re, dt, y, factor, sigma)

    # brute-force Hessian trace via column-by-column FD of the gradient
    d = n * n
    eps = 1e-4
    g0 = grad_fn(w0)
    trace = 0.0
    H_cols = []
    for j in range(d):
        e = torch.zeros_like(w0).flatten()
        e[j] = 1.0
        e = e.view_as(w0)
        col = (grad_fn(w0 + eps * e) - grad_fn(w0 - eps * e)) / (2 * eps)
        H_cols.append(col.flatten())
        trace += col.flatten()[j].item()
    H = torch.stack(H_cols)  # (d, d), row j = H e_j
    sym = (H - H.t()).norm().item() / (H.norm().item() + 1e-30)
    report("Hessian symmetry ||H-H^T||/||H||", sym, 1e-3, sym < 1e-3)

    # Hutchinson estimate with large M (original loop)
    gen = torch.Generator().manual_seed(1)
    M = 200
    est = A.hutchinson_laplacian(grad_fn, w0, M, eps, generator=gen).item()
    se = abs(est - trace) / (abs(trace) + 1e-30)
    print(f"    brute trace={trace:.6e}  hutchinson(M={M})={est:.6e}  rel={se:.2e}")
    report("Hutchinson trace within 3% of brute force", se, 3e-2, se < 3e-2)

    # Batched central Hutchinson (efficiency refactor) -- same estimator, must match brute trace
    gen2 = torch.Generator().manual_seed(2)
    est_b = A.hutchinson_laplacian_batched(grad_fn, w0, M, eps, scheme='central', generator=gen2).item()
    se_b = abs(est_b - trace) / (abs(trace) + 1e-30)
    print(f"    batched-central(M={M})={est_b:.6e}  rel={se_b:.2e}")
    report("Batched-central Hutchinson within 3% of brute force", se_b, 3e-2, se_b < 3e-2)

    # Forward-difference scheme (reuses g0) -- O(eps) biased, relaxed tolerance
    gen3 = torch.Generator().manual_seed(3)
    est_f = A.hutchinson_laplacian_batched(grad_fn, w0, M, eps, scheme='forward', g0=g0, generator=gen3).item()
    se_f = abs(est_f - trace) / (abs(trace) + 1e-30)
    print(f"    batched-forward(M={M}, reuse g0)={est_f:.6e}  rel={se_f:.2e}")
    report("Forward-diff Hutchinson within 6% of brute force", se_f, 6e-2, se_f < 6e-2)


def rung6_grf_score(n=32):
    print("\n== Rung 6: GRF prior score vs autograd of -1/2 <w, C^{-1} w> ==")
    s = make_solver(n)
    w = A.grf_sample(s, 1) * 5.0
    w = w.detach().requires_grad_(True)
    # log p0 = -1/2 <w, C^{-1} w>; score = -C^{-1} w
    logp = -0.5 * (w * A.grf_apply_inv_cov(s, w)).sum()
    g_auto = torch.autograd.grad(logp, w)[0]
    g_closed = A.grf_prior_score(s, w.detach())
    e = (g_auto - g_closed).norm().item() / (g_closed.norm().item() + 1e-30)
    report("GRF score (closed form) == autograd", e, 1e-9, e < 1e-9)
    # exact Gaussian denoiser sanity: for the mean-zero prior, D(x, sigma->0) == x
    # on mean-zero inputs (the denoiser projects out the DC mode, matching grf_sample).
    x = torch.randn(1, n, n, dtype=DT)
    x = x - x.mean()
    d0 = A.grf_denoiser(s, x, 1e-10)
    e2 = (d0 - x).abs().max().item()
    report("GRF denoiser D(x, sigma->0) == x (mean-zero)", e2, 1e-6, e2 < 1e-6)
    # and it must kill the DC mode (consistency with grf_sample / dealias)
    dc = A.grf_denoiser(s, torch.ones(1, n, n, dtype=DT), 1e-10).mean().abs().item()
    report("GRF denoiser zeros the DC/mean mode", dc, 1e-10, dc < 1e-10)


if __name__ == "__main__":
    rung0_primitives()
    rung1_forward()
    rung3_gradient_fd()
    rung4_continuous_adjoint()
    rung5_hutchinson()
    rung6_grf_score()
    print("\n================ SUMMARY ================")
    n_pass = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{len(results)} checks passed.")
    sys.exit(0 if n_pass == len(results) else 1)
