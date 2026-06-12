"""EKI (Ensemble Kalman Inversion) for the inverse scattering problem.

Port of ``navier_stokes/algo/eki.py`` for complex-valued scattering measurements.
The forward operator (``BatchInverseScatter``) returns complex ``(B, T, R)``
tensors.  The Kalman update requires *real*-valued linear algebra (the
permittivity is real), so we convert complex measurements to a real Re/Im-stacked
representation ``(B, 2·T·R)`` before every update step.

This is the ONLY change relative to the NS version; the EKI algorithm (GRF
initialization, Kalman covariance estimate, adaptive learning rate) is identical.
"""
import math
import torch

from algo.base import Algo
from algo.eki import GaussianRF2d

import wandb


class EKIScatter(Algo):
    """Ensemble Kalman Inversion for complex-measurement inverse problems.

    Parameters
    ----------
    net : nn.Module
        Pretrained diffusion model.  EKI does not use the denoiser during
        inference — it is only queried for ``img_channels`` and
        ``img_resolution`` (ensemble shape).
    forward_op : BatchInverseScatter
        Batched forward operator returning complex ``(B, T, R)``.
    guidance_scale : float
        Step-size multiplier for the Kalman update (normalised by the
        operator-norm of the cross-covariance).
    num_updates : int
        Number of Kalman iterations.
    num_samples : int
        Ensemble size (number of particles).
    resolution : int
        Spatial resolution (default 128 for the InverseBench scattering grid).
    L : float
        GRF domain size (default ``2π``; controls the prior correlation length).
    device : torch.device
        Device for the GRF samples and the Kalman update.
    """

    def __init__(self, net, forward_op,
                 guidance_scale, num_updates,
                 num_samples=512,
                 resolution=128,
                 L=2 * math.pi,
                 init_std=0.5,
                 device=torch.device('cuda')):
        super().__init__(net, forward_op)
        self.guidance_scale = guidance_scale
        self.num_updates = num_updates
        self.num_samples = num_samples
        self.init_std = init_std
        self.device = device

        self.grf = GaussianRF2d(s1=resolution, s2=resolution, L1=L, L2=L,
                                alpha=4.0, tau=3.0, device=device)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_real(y):
        """Complex ``(B, …)`` → real ``(B, -1)`` float32.

        ``view_as_real`` appends a trailing dim-2 ``[Re, Im]``; ``flatten``
        merges it with the measurement dims.  The ``float32`` cast matches the
        NS version (float32 throughout the Kalman update).
        """
        return torch.view_as_real(y).flatten(start_dim=1).to(torch.float32)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=False):
        # Initialize particles: small noise in the normalized [-1, 1] domain.
        # GRF is inappropriate here (periodic, wrong scale for bounded
        # permittivity). Use iid Gaussian with controllable std instead.
        x_next = torch.randn(
            self.num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=self.device) * self.init_std

        # Convert the complex observation to real ONCE
        obs_real = self._to_real(observation)           # (1, 2·T·R)

        for i in range(self.num_updates):
            # Batched forward (complex) → real
            ys_complex = self.forward_op.forward(x_next)  # (N, T, R) complex
            ys = self._to_real(ys_complex)                 # (N, 2·T·R) real

            xs_diff = x_next - x_next.mean(dim=0, keepdim=True)
            ys_diff = ys - ys.mean(dim=0, keepdim=True)
            ys_err = ys - obs_real

            coef = (
                torch.matmul(
                    ys_err.reshape(ys_err.shape[0], -1),
                    ys_diff.reshape(ys_diff.shape[0], -1).T,
                )
                / self.num_samples
            )

            dxs = coef @ xs_diff.reshape(self.num_samples, -1)
            lr = self.guidance_scale / torch.linalg.matrix_norm(coef)

            x_next = x_next - lr * dxs.reshape(x_next.shape)

            if wandb.run is not None:
                abs_err = torch.abs(ys_err)
                avg_err = torch.mean(abs_err)
                max_err = torch.max(abs_err)
                std = torch.std(x_next, dim=0, keepdim=True)
                avg_std = torch.mean(std)
                wandb.log({
                    "EKI/abs error": avg_err.item(),
                    'EKI/max error': max_err.item(),
                    "EKI/std": avg_std.item(),
                })

        return x_next
