"""Vendored, cleaned AFDPS ensemble sampler.

Source: AFDPS-TMLR (Chen, Ren, Min, Ying, Izzo, TMLR 2026),
`pnpdm/samplers/pnp_edm/ensemble_denoiser_edm.py`.

Only the core annealed-SDE + Feynman-Kac particle-reweighting sampler is kept.
All image-specific diagnostics (monai PSNR, torchvision transforms, lpips,
[-1,1] clamping) and debug prints have been removed so the sampler is
domain-agnostic and safe for the Navier-Stokes vorticity field.
"""
from .ensemble_denoiser_edm import Ensemble_Denoiser_EDM

__all__ = ["Ensemble_Denoiser_EDM"]
