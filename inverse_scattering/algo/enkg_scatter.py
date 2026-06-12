"""EnKG (Ensemble Kalman Diffusion Guidance) for the inverse scattering problem.

Port of ``navier_stokes/algo/enkg.py`` for complex-valued scattering measurements.
Two adaptations from the NS version:

1. **Complex → real conversion**: the forward operator returns complex
   ``(B, T, R)`` tensors; the Kalman update works in a real Re/Im-stacked
   representation ``(B, 2·T·R)`` so that the permittivity update stays real.

2. **Bypass** ``gradient_m``: the base ``BaseOperator.gradient_m`` uses
   ``.square()`` which is *complex* squaring (``z²``), not ``|z|²``.  For the L2
   loss with real measurements, ``0.5 * gradient_m(ys, obs)`` simplifies to
   ``ys - obs``, so we compute the residual directly.

The ODE denoiser and diffusion time-schedule are unchanged — the EDM prior works
identically on the 128×128×1 permittivity field.
"""
import torch
from tqdm import tqdm

from algo.base import Algo
from algo.enkg import ode_sampler

import wandb


class EnKGScatter(Algo):
    """Ensemble Kalman Diffusion Guidance for complex-measurement inverse problems.

    Parameters match the NS ``EnKG`` constructor.  The denoiser ``net`` is used
    inside ``ode_sampler`` to map noisy particles ``x_t`` → clean estimates
    ``x_0``; the forward operator ``forward_op`` must support batched
    ``forward(x)`` returning complex ``(B, T, R)``.
    """

    def __init__(self,
                 net,
                 forward_op,
                 guidance_scale,
                 num_steps,
                 num_updates,
                 sigma_max,
                 sigma_min,
                 num_samples=1024,
                 threshold=0.1,
                 batch_size=128,
                 lr_min_ratio=0.0,
                 rho: int = 7,
                 factor: int = 4):
        super().__init__(net, forward_op)
        self.rho = rho
        self.num_steps = num_steps
        self.num_updates = num_updates
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.threshold = threshold
        self.num_samples = num_samples
        self.lr_min_ratio = lr_min_ratio
        self.factor = factor

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_real(y):
        """Complex ``(B, …)`` → real ``(B, -1)`` float32."""
        return torch.view_as_real(y).flatten(start_dim=1).to(torch.float32)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def inference(self, observation, num_samples=1):
        device = self.forward_op.device
        x_initial = (
            torch.randn(
                self.num_samples, self.net.img_channels,
                self.net.img_resolution, self.net.img_resolution,
                device=device)
            * self.sigma_max
        )

        # Time-step discretisation (EDM schedule)
        step_indices = torch.arange(self.num_steps, dtype=torch.float32,
                                    device=device)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho)
               - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = torch.cat(
            [self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Convert complex observation to real ONCE
        obs_real = self._to_real(observation)             # (1, 2·T·R)

        num_batches = x_initial.shape[0] // self.batch_size
        x_next = x_initial
        denoised = torch.zeros_like(x_initial)

        for i, (t_cur, t_next) in tqdm(
            enumerate(zip(t_steps[:-1], t_steps[1:]))
        ):
            x_cur = x_next

            # Ensemble Kalman guidance
            if (i < (self.num_steps - int(0.5 * self.threshold))
                    and i > self.threshold):
                x_hat, t_hat = self.update_particles(
                    x_cur, obs_real,
                    num_steps=min(
                        1 + (self.num_steps - i) // self.factor, 20),
                    sigma_start=t_cur,
                    guidance_scale=self.get_lr(i),
                )
            else:
                t_hat = t_cur
                x_hat = x_cur

            # Batched denoiser forward
            for j in range(num_batches):
                start = j * self.batch_size
                end = (j + 1) * self.batch_size
                denoised[start:end] = self.net(x_hat[start:end], t_hat)

            # Euler step
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

        return x_next

    # ------------------------------------------------------------------ #
    def get_lr(self, i):
        if self.lr_min_ratio > 0.0:
            return (self.guidance_scale
                    * (1 - self.lr_min_ratio)
                    * (self.num_steps - i) / self.num_steps
                    + self.lr_min_ratio)
        return self.guidance_scale

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_particles(self, particles, obs_real, num_steps,
                         sigma_start, guidance_scale=1.0):
        """Kalman update in real measurement space.

        ``obs_real`` is already converted to real ``(1, 2·T·R)`` float32.
        This replaces the NS version's ``0.5 * gradient_m(ys, obs)`` with the
        equivalent but complex-safe ``ys_real - obs_real``.
        """
        x0s = torch.zeros_like(particles)
        num_batchs = particles.shape[0] // self.batch_size
        N, *spatial = particles.shape
        t_hat = sigma_start

        for j in range(self.num_updates):
            # ODE denoise: x_t → x_0 estimate (batched)
            for k in range(num_batchs):
                start = k * self.batch_size
                end = (k + 1) * self.batch_size
                x0s[start:end] = ode_sampler(
                    self.net, particles[start:end],
                    num_steps=num_steps,
                    sigma_start=sigma_start,
                )

            # Batched forward model on denoised estimates
            ys_complex = self.forward_op.forward(x0s)       # (N, T, R) complex
            ys = self._to_real(ys_complex)                   # (N, 2·T·R) real

            # Kalman update — all real-valued
            xs_diff = particles - particles.mean(dim=0, keepdim=True)
            ys_diff = ys - ys.mean(dim=0, keepdim=True)
            # Equivalent to 0.5 * gradient_m(ys, obs) for L2 loss on real data
            ys_err = ys - obs_real

            coef = (
                torch.matmul(
                    ys_err.reshape(N, -1),
                    ys_diff.reshape(N, -1).T,
                )
                / N
            )
            dxs = coef @ xs_diff.reshape(N, -1)              # (N, C·H·W)
            lr = guidance_scale / torch.linalg.matrix_norm(coef)
            particles = particles - lr * dxs.reshape(N, *spatial)

            if wandb.run is not None:
                abs_ys = torch.abs(ys_err)
                abs_err = torch.mean(abs_ys)
                max_err = torch.max(abs_ys)
                std = torch.std(particles, dim=0, keepdim=True)
                avg_std = torch.mean(std)
                wandb.log({
                    "EnKG/abs error": abs_err.item(),
                    'EnKG/max error': max_err.item(),
                    "EnKG/averaged norm of updates": torch.mean(
                        torch.linalg.vector_norm(dxs, dim=1)).item(),
                    "EnKG/lr": lr,
                    "EnKG/std": avg_std.item(),
                })

        return particles, t_hat
