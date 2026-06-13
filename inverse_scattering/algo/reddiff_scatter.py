"""RED-diff port for the linear inverse-scattering problem.

Paper: A Variational Perspective on Solving Inverse Problems with Diffusion Models
Reference: navier_stokes/algo/reddiff.py (verbatim logic, adapted imports)

RED-diff is an optimization-based method: it optimizes a latent variable mu by
combining the diffusion prior (via epsilon-prediction) with the data-fitting
gradient.  No SVD required — uses only forward_op.gradient() (autograd through
the forward map), which InverseScatter already supports.
"""
import torch
import tqdm
from algo.base import Algo
import numpy as np
from utils.scheduler import Scheduler


class REDDiff(Algo):
    def __init__(self, net, forward_op, num_steps=1000, observation_weight=1.0,
                 base_lambda=0.25, base_lr=0.5, lambda_scheduling_type='constant'):
        super(REDDiff, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op

        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp',
                                   timestep='vp', scaling='vp')
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = torch.zeros(num_samples, self.net.img_channels, self.net.img_resolution,
                         self.net.img_resolution, device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))
        for step in pbar:
            with torch.no_grad():
                sigma = self.scheduler.sigma_steps[step]
                scaling = self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation,
                                                           return_loss=True)
            gradient = gradient * self.observation_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}')
        return mu
