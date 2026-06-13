"""Devito data-space adjoint probes for the Gauss-Newton AFDPS Laplacian (FWI).

Used ONLY when `AFDPSAcousticWave(laplacian_mode='gn_hutchinson')` or the sampler's
J-aware `guidance_mode='auto'` is selected. It estimates the Gauss-Newton Hessian trace

    Tr(grad^2_x mu_y) ~ (1/sigma^2) ||J||_F^2 ,   J = d(A(x))/d(x),

via the data-space Hutchinson identity ||J||_F^2 = Tr(J J^T) = E_w ||J^T w||^2, with w a
random probe in receiver/data space. Each J^T w is one Devito adjoint solve: the probe is
injected as a *synthetic* data residual and back-propagated, exactly like the parent
operator's true-residual adjoint -- only the residual differs.

Because the Frobenius norm is additive over output blocks, ||J||_F^2 = sum_shots ||J_s||_F^2,
so each shot's contribution is estimated independently (independent probes per shot, lower
variance, and the shot's forward wavefield u0 is reused across that shot's M probes:
1 forward + M adjoints per shot).

This file mirrors the parent's `gradient_single_shot`
(`navier_stokes/inverse_problems/acoustic.py`) line-for-line in how it builds the Devito
objects and applies the slowness->velocity->normalized chain rule. It is BEST-EFFORT and
deliberately not on the default path: it must be smoke-tested on a machine with Devito
before being trusted at scale. The default `laplacian_mode='fd_divergence'` needs none of
this (it reuses the already-validated parent gradient).
"""
import numpy as np

from devito import Function
from examples.seismic import Receiver
from examples.seismic.acoustic import AcousticWaveSolver


def _slowness_adjoint(model, geometry, u0, residual_np, fs):
    """Back-propagate one data-space probe `residual_np` (shape (T, nreceivers)) through the
    adjoint wave equation and return the cropped gradient w.r.t. squared slowness, (X, Z).
    `u0` is the saved forward wavefield for this shot (reused across probes)."""
    grad = Function(name="grad", grid=model.grid)
    residual = Receiver(name='rec', grid=model.grid,
                        time_range=geometry.time_axis,
                        coordinates=geometry.rec_positions)
    solver = AcousticWaveSolver(model, geometry, space_order=4)
    T = residual.data.shape[0]
    residual.data[:] = residual_np[:T, :]
    solver.gradient(rec=residual, u=u0, vp=model.vp, grad=grad)
    nbl = model.nbl
    z_start = 0 if fs else nbl
    return np.array(grad.data[:])[nbl:-nbl, z_start:-nbl]  # (X, Z) slowness-gradient


def jtw_norm_sq(operator, x_norm_single, observation, M):
    """mean_w ||J^T w||^2 for a single normalized particle x (shape (1,1,H,W)).

    Returns a Python float: an unbiased estimate of ||J||_F^2 for J = d(A(x))/d(x) in
    normalized-x units (the slowness->velocity->normalized chain rule is applied per cell,
    identical to the parent operator's true-residual gradient)."""
    model = operator.model
    fs = operator.fs
    nbl = model.nbl
    z_start = 0 if fs else nbl

    # Set the model velocity (km/s) exactly as the parent gradient does: (X, Z) layout.
    vel_xz = operator.unnormalize(x_norm_single).detach().transpose(-2, -1).cpu().numpy()[0, 0]
    model.vp.data[nbl:-nbl, z_start:-nbl] = vel_xz
    # Per-cell chain-rule factor d(m)/d(x) = (-2 / v^3) * unnorm_scale, m = 1/v^2.
    chain = (-2.0 / vel_xz ** 3) * operator.unnorm_scale  # (X, Z)

    fro2 = 0.0
    for geometry in operator.geometry_list:
        solver = AcousticWaveSolver(model, geometry, space_order=4)
        # One forward solve per shot, saved, reused across this shot's M probes.
        d_pred, u0 = solver.forward(vp=model.vp, save=True)[0:2]
        shape = d_pred.data.shape  # (T, nreceivers)
        shot_acc = 0.0
        for _ in range(max(1, int(M))):
            w = np.random.randn(*shape).astype(np.float32)
            gs = _slowness_adjoint(model, geometry, u0, w, fs)  # (X, Z) slowness-grad
            g_x = gs * chain                                    # J_s^T w in normalized-x units
            shot_acc += float(np.sum(g_x ** 2))
        fro2 += shot_acc / max(1, int(M))
        del u0, d_pred
    return fro2
