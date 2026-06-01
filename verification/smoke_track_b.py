"""Track B smoke test: validate the full Hydra-instantiated code path WITHOUT the
real checkpoint or a GPU.

Composes the actual `problem=navier-stokes-afdps` + `algorithm=afdps` configs (with
small-resolution overrides), instantiates the AFDPS forward operator and algorithm
exactly as `main.py` does, swaps in a mock EDM denoiser, and runs one inference on a
synthetic observation. Confirms config wiring + the diffusion-prior path execute and
produce a finite reconstruction. The real run uses `main.py` with `ns-5m.pt`.

Usage: python verification/smoke_track_b.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydra import initialize, compose
from hydra.utils import instantiate


class MockEDMNet:
    """Minimal EDM-`net`-compatible denoiser (Tweedie for an N(0,I) prior):
    D(x, sigma) = x / (1 + sigma^2). Satisfies the sampler's net contract."""
    def __init__(self, img_resolution, img_channels=1):
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sigma_min = 0.0
        self.sigma_max = float('inf')

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def __call__(self, x, sigma):
        sig = float(sigma)
        return x / (1.0 + sig ** 2)


def main():
    res, steps, particles = 32, 30, 4
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config", overrides=[
            "problem=navier-stokes-afdps",
            "algorithm=afdps",
            f"problem.model.resolution={res}",
            "problem.model.forward_time=0.06",
            "problem.model.delta_t=0.002",
            "problem.model.sigma_noise=0.05",
            f"problem.model.grad_chunk={particles}",
            f"algorithm.method.num_particles={particles}",
            f"algorithm.method.num_steps={steps}",
            "algorithm.method.sigma_max=2.0",
            "algorithm.method.sigma_min=0.01",
            "algorithm.method.guidance_gamma=5.0",  # tuned to this (small) scale
        ])
    device = 'cpu'
    torch.manual_seed(0)

    # exactly as main.py: instantiate forward op, then algo with (net, forward_op)
    forward_op = instantiate(cfg.problem.model, device=device)
    net = MockEDMNet(img_resolution=res, img_channels=1)
    algo = instantiate(cfg.algorithm.method, forward_op=forward_op, net=net)

    # synthetic observation in the normalized domain (physical std ~ 4 after x10)
    x_true = 0.4 * torch.randn(1, 1, res, res)
    observation = forward_op({'target': x_true})
    recon = algo.inference(observation, num_samples=1)

    rel_l2 = (recon - x_true).norm().item() / x_true.norm().item()
    finite = torch.isfinite(recon).all().item()
    print(f"config composed OK | forward_op={type(forward_op).__name__} | algo={type(algo).__name__}")
    print(f"observation shape {tuple(observation.shape)} | recon shape {tuple(recon.shape)}")
    print(f"recon finite={finite} | rel-L2(recon, true)={rel_l2:.4f}")
    print("(note: a trivial mock prior is used here; recovery quality needs the real ns-5m.pt)")
    ok = finite and recon.shape == (1, 1, res, res)
    print(f"\n[{'PASS' if ok else 'FAIL'}] Track B code path executes end-to-end")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
