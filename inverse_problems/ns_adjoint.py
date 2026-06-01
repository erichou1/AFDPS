"""Gradient/adjoint engine for the 2D Navier-Stokes inverse problem.

Everything here works in the *physical* vorticity domain on the 2pi torus and
reuses the spectral tensors of InverseBench's ``NavierStokes2d`` solver
(``k1, k2, G, inv_lap, dealias``) so that the discretization matches the forward
solver bit-for-bit.

Primary likelihood gradient: the **discrete adjoint via reverse-mode autograd**.
``NavierStokes2d.solve`` is built from differentiable torch ops and (with
``adaptive=False``) has a fixed-length, input-independent computation graph, so
``torch.autograd.grad`` through it yields the exact discrete adjoint -- including
the correct ``rfft2`` adjoint, which is *not* ``irfft2``.

Secondary (cross-check / pedagogical): a hand-coded **continuous adjoint** PDE
solver following the mentor's note (eqs 32-45). It is the optimize-then-discretize
scheme and therefore agrees with the discrete adjoint only to O(dt); it is used as
a physics cross-check, not as the production gradient.
"""
import math
import torch
import torch.fft as fft


# --------------------------------------------------------------------------- #
# Measurement operator P (strided subsampling) and its exact adjoint P*        #
# --------------------------------------------------------------------------- #
def downsample(full, factor):
    """P: strided spatial subsampling. full (..., s1, s2) -> (..., s1/f, s2/f)."""
    return full[..., ::factor, ::factor]


def upsample_adjoint(obs, factor, s1, s2):
    """P*: exact adjoint of strided subsampling = zero-insertion upsampling."""
    out = torch.zeros(*obs.shape[:-2], s1, s2, device=obs.device, dtype=obs.dtype)
    out[..., ::factor, ::factor] = obs
    return out


# --------------------------------------------------------------------------- #
# Differentiable forward solve and misfit                                      #
# --------------------------------------------------------------------------- #
def forward_solve(solver, w0, force, T, Re, dt):
    """Differentiable forward map L: omega0 -> omega(T) (full resolution).

    w0: (B, s1, s2) physical vorticity. Returns (B, s1, s2). adaptive=False so the
    trajectory length is fixed and the graph is well defined for autograd.
    """
    return solver.solve(w0, force, T, Re, adaptive=False, delta_t=dt)


def misfit(solver, w0, force, T, Re, dt, y, factor, sigma):
    """mu_y(omega0) = ||P L(omega0) - y||^2 / (2 sigma^2). Returns (B,)."""
    wT = forward_solve(solver, w0, force, T, Re, dt)
    r = downsample(wT, factor) - y
    return 0.5 * r.pow(2).flatten(start_dim=1).sum(dim=1) / (sigma ** 2)


def grad_misfit_autograd(solver, w0, force, T, Re, dt, y, factor, sigma):
    """Discrete-adjoint gradient d mu_y / d omega0 (physical domain) via autograd.

    Returns a tensor with the same shape as ``w0`` ((B, s1, s2)).
    """
    w0 = w0.detach().requires_grad_(True)
    with torch.enable_grad():
        m = misfit(solver, w0, force, T, Re, dt, y, factor, sigma).sum()
        g = torch.autograd.grad(m, w0)[0]
    return g


# --------------------------------------------------------------------------- #
# Hutchinson trace estimate of the likelihood Laplacian  Tr(Hessian mu_y)      #
# --------------------------------------------------------------------------- #
def hutchinson_laplacian(grad_fn, w0, M, eps, generator=None):
    """Estimate Tr(d^2 mu_y / d omega0^2) per sample via central differences.

    grad_fn: callable w0 -> d mu_y / d omega0 (same shape as w0).
    Returns (B,). Cost = 2*M evaluations of grad_fn.
    """
    B = w0.shape[0]
    acc = torch.zeros(B, device=w0.device, dtype=w0.dtype)
    for _ in range(M):
        xi = torch.randint(0, 2, w0.shape, device=w0.device, generator=generator).to(w0.dtype) * 2 - 1
        gp = grad_fn(w0 + eps * xi)
        gm = grad_fn(w0 - eps * xi)
        hvp = (gp - gm) / (2 * eps)               # ~ Hessian @ xi
        acc += (hvp * xi).flatten(start_dim=1).sum(dim=1)
    return acc / M


def hutchinson_laplacian_batched(grad_fn, w0, M, eps, scheme='central', g0=None, generator=None):
    """Vectorized Hutchinson trace estimate of Tr(d^2 mu_y / d omega0^2), per sample.

    All M probes are stacked onto the batch dimension and pushed through `grad_fn`
    in ONE call (grad_fn is expected to chunk internally for memory). This collapses
    the per-step Python loop / kernel-launch overhead -- the dominant lever for the
    GB200 sweep -- while being mathematically identical to `hutchinson_laplacian`
    (it differs only in the RNG draw order, so it is statistically, not bitwise, equal).

    scheme='central' : 2*M gradient evals, O(eps^2) accurate (default; what the ladder pins).
    scheme='forward' : M gradient evals, reusing the precomputed unperturbed gradient
                       g0 = grad_fn(w0) (e.g. the drift gradient already computed by the
                       sampler). O(eps) accurate; halves the solve count per step.
    Returns (B,).
    """
    B = w0.shape[0]
    xi = (torch.randint(0, 2, (M, *w0.shape), device=w0.device, generator=generator).to(w0.dtype) * 2 - 1)
    if scheme == 'forward':
        assert g0 is not None, "scheme='forward' requires g0 = grad_fn(w0)"
        Xp = (w0.unsqueeze(0) + eps * xi).reshape(M * B, *w0.shape[1:])
        gp = grad_fn(Xp).reshape(M, *w0.shape)
        hvp = (gp - g0.unsqueeze(0)) / eps                    # ~ Hessian @ xi
    else:
        assert scheme == 'central'
        Xp = (w0.unsqueeze(0) + eps * xi)
        Xm = (w0.unsqueeze(0) - eps * xi)
        X = torch.cat([Xp, Xm], dim=0).reshape(2 * M * B, *w0.shape[1:])
        g = grad_fn(X).reshape(2, M, *w0.shape)
        hvp = (g[0] - g[1]) / (2 * eps)
    return (hvp * xi).flatten(start_dim=2).sum(dim=2).mean(dim=0)  # (M,B)->(B,)


# --------------------------------------------------------------------------- #
# Gaussian random field prior  omega0 ~ N(0, C),  C = (-Delta + 9 I)^{-4}      #
# --------------------------------------------------------------------------- #
def _grf_symbol(solver):
    """C^{-1} Fourier symbol (|k|^2 + 9)^4 on the rfft2 grid. On the 2pi torus
    solver.G == |k|^2 exactly (the 4 pi^2 / L^2 factor is 1)."""
    assert abs(solver.L1 - 2 * math.pi) < 1e-9 and abs(solver.L2 - 2 * math.pi) < 1e-9, \
        "GRF symbol (|k|^2+9)^4 assumes the 2pi torus; solver.G must equal |k|^2."
    return (solver.G + 9.0) ** 4


def grf_apply_inv_cov(solver, w0, amp=1.0):
    """(amp^2 C)^{-1} w0 = (1/amp^2) (-Delta + 9 I)^4 w0, diagonal in Fourier."""
    wh = fft.rfft2(w0)
    return fft.irfft2((_grf_symbol(solver) / (amp ** 2)) * wh, s=(solver.s1, solver.s2))


def grf_prior_score(solver, w0, amp=1.0):
    """Prior score grad log p0(omega0) = -(amp^2 C)^{-1} omega0  (mentor eq 19)."""
    return -grf_apply_inv_cov(solver, w0, amp)


def grf_denoiser(solver, x, sigma, amp=1.0):
    """Exact Tweedie denoiser for the Gaussian prior N(0, amp^2 C) at noise level
    sigma: D(x, sigma) = Cs (Cs + sigma^2 I)^{-1} x with Cs = amp^2 C, diagonal in
    Fourier. (For x = omega0 + sigma * eps with omega0 ~ N(0, amp^2 C), = E[omega0|x].)"""
    xh = fft.rfft2(x)
    c = (amp ** 2) * (solver.G + 9.0) ** (-4)     # eigenvalues of amp^2 C
    Dh = (c / (c + sigma ** 2)) * xh
    Dh[..., 0, 0] = 0.0                            # mean-zero, consistent with grf_sample / dealias
    return fft.irfft2(Dh, s=(solver.s1, solver.s2))


def grf_sample(solver, n_samples, amp=1.0, generator=None, dtype=None, device=None):
    """Draw omega0 ~ N(0, amp^2 C) by applying amp (-Delta+9I)^{-2} to white noise."""
    dtype = dtype or solver.G.dtype
    device = device or solver.G.device
    n = torch.randn(n_samples, solver.s1, solver.s2, device=device, dtype=dtype, generator=generator)
    nh = fft.rfft2(n)
    half = amp * (solver.G + 9.0) ** (-2)         # (amp^2 C)^{1/2} symbol
    out_h = half * nh
    out_h[..., 0, 0] = 0.0                         # mean-zero vorticity (torus convention)
    return fft.irfft2(out_h, s=(solver.s1, solver.s2))


# --------------------------------------------------------------------------- #
# Hand-coded continuous adjoint (cross-check; agrees with autograd to O(dt))   #
# --------------------------------------------------------------------------- #
def _ddx(solver, f_h):
    """spectral d/dx of a field given its rfft2."""
    return (2 * math.pi / solver.L1) * 1j * solver.k1 * f_h


def _ddy(solver, f_h):
    """spectral d/dy of a field given its rfft2."""
    return (2 * math.pi / solver.L2) * 1j * solver.k2 * f_h


@torch.no_grad()
def continuous_adjoint_gradient(solver, w0, force, T, Re, dt, y, factor, sigma):
    """Mentor's continuous-adjoint gradient (eqs 32-45).

    Forward-solve storing {omega^n}; terminal lambda(T) = P*(P omega(T) - y)/sigma^2;
    march backward  -d_t lambda = nu Lap lambda + u.grad lambda + (-Lap)^{-1} curl(lambda grad omega);
    return lambda(0). Semi-implicit (Crank-Nicolson) on viscosity to mirror the forward.

    NOTE: optimize-then-discretize -> matches the autograd discrete adjoint only to
    O(dt). Used for cross-checking, not as the production gradient.
    """
    nu = 1.0 / Re
    s1, s2 = solver.s1, solver.s2
    # ---- forward pass, store vorticity trajectory in Fourier ----
    w_h = fft.rfft2(w0)
    f_h = fft.rfft2(force) if force is not None else None
    GG = nu * solver.G
    n_steps = int(round(T / dt))
    traj = [w_h]
    for _ in range(n_steps):
        nl1 = solver.nonlinear_term(w_h, f_h)
        w_tilde = (w_h + dt * (nl1 - 0.5 * GG * w_h)) / (1.0 + 0.5 * dt * GG)
        nl2 = solver.nonlinear_term(w_tilde, f_h)
        w_h = (w_h + dt * (0.5 * (nl1 + nl2) - 0.5 * GG * w_h)) / (1.0 + 0.5 * dt * GG)
        traj.append(w_h)

    # ---- terminal condition lambda(T) ----
    wT = fft.irfft2(traj[-1], s=(s1, s2))
    r = downsample(wT, factor) - y
    lam = upsample_adjoint(r, factor, s1, s2) / (sigma ** 2)     # physical
    lam_h = fft.rfft2(lam)

    # ---- backward march ----
    for n in range(n_steps, 0, -1):
        w_h_n = traj[n] * solver.dealias
        w_n = fft.irfft2(w_h_n, s=(s1, s2))
        # velocity from vorticity at step n
        psi_h = solver.inv_lap * w_h_n
        q, v = solver.velocity_field(psi_h, real_space=True)        # u = (q, v)
        lam_phys = fft.irfft2(lam_h * solver.dealias, s=(s1, s2))
        # advection  u . grad(lambda)
        dlx = fft.irfft2(_ddx(solver, lam_h * solver.dealias), s=(s1, s2))
        dly = fft.irfft2(_ddy(solver, lam_h * solver.dealias), s=(s1, s2))
        adv_h = fft.rfft2(q * dlx + v * dly)
        # (-Lap)^{-1} curl(lambda grad omega):  q1 = lam * w_x, q2 = lam * w_y
        wx = fft.irfft2(_ddx(solver, w_h_n), s=(s1, s2))
        wy = fft.irfft2(_ddy(solver, w_h_n), s=(s1, s2))
        q1_h = fft.rfft2(lam_phys * wx)
        q2_h = fft.rfft2(lam_phys * wy)
        g_h = -_ddy(solver, q1_h) + _ddx(solver, q2_h)             # curl-perp . q
        a_h = solver.inv_lap * g_h
        a_h[..., 0, 0] = 0.0
        B_h = adv_h + a_h
        lam_h = (lam_h + dt * B_h) / (1.0 + dt * GG)

    return fft.irfft2(lam_h, s=(s1, s2))
