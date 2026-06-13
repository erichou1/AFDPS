"""DDNM port for the linear inverse-scattering problem.

Paper: Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model
Reference: navier_stokes/algo/ddnm.py (verbatim logic, adapted imports)

DDNM requires SVD of the forward operator.  InverseScatter already provides
.U, .S (property), .Vt() (method), .V() (method), .M (property), .forward().
The SVD is precomputed and cached per (numTrans, numRec) pair.
"""
import torch
import tqdm
import numpy as np
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.helper import has_svd


class DDNM(Algo):
    def __init__(self, net, forward_op, scheduler_config, eta, L):
        super(DDNM, self).__init__(net, forward_op)
        assert has_svd(forward_op), \
            "DDNM only works with linear forward operators with SVD decomposition"

        self.scheduler = Scheduler(**scheduler_config)
        self.eta = eta
        self.L = L

    def score(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (d - x) / sigma ** 2

    def pseudo_inverse(self, op, y):
        return op.V(op.M * op.Ut(y) / op.S)

    def projection(self, op, x):
        return x - self.pseudo_inverse(op, op.forward(x))

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x = torch.randn(num_samples, self.net.img_channels,
                         self.net.img_resolution, self.net.img_resolution,
                         device=device) * self.scheduler.sigma_max
        pbar = tqdm.trange(self.scheduler.num_steps)
        sigma_y = max(self.forward_op.sigma_noise, 1e-4)
        for step in pbar:
            L = min(self.L, step)
            sigma = self.scheduler.sigma_steps[step]
            sigma_L = self.scheduler.sigma_steps[step - L]
            x = ((x / self.scheduler.scaling_steps[step])
                 + np.sqrt(sigma_L ** 2 - sigma ** 2) * torch.randn_like(x)) \
                * self.scheduler.scaling_steps[step - L]
            for j in range(L + 1):
                sigma = self.scheduler.sigma_steps[step - L + j]
                denoised = self.net(
                    x / self.scheduler.scaling_steps[step - L + j],
                    torch.as_tensor(sigma).to(x.device))

                x0hat = self.pseudo_inverse(self.forward_op, observation) \
                    + self.projection(self.forward_op, denoised)
                sigma_next = self.scheduler.sigma_steps[step - L + j + 1]
                # DDNM+
                lamb = min(1, sigma_next / sigma_y)
                gamma = np.sqrt(max(0, sigma_next ** 2 - (lamb * sigma_y) ** 2))
                x0hat = lamb * x0hat + (1 - lamb) * denoised
                x = x0hat + np.sqrt(1 - self.eta ** 2) * sigma_next / sigma \
                    * (x - x0hat) + self.eta * gamma * torch.randn_like(x)
                x = x * self.scheduler.scaling_steps[step - L + j + 1]
        return x
