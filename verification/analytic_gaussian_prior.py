"""Analytic Gaussian random field (GRF) prior for the synthetic Track A problem.

omega0 ~ N(0, amp^2 * C),  C = (-Delta + 9 I)^{-4}  (InverseBench's data-generation
prior; mentor's note eqs 8-11). All operators are diagonal in Fourier and live in
`inverse_problems.ns_adjoint`; this module is a thin, self-documenting re-export so
Track A code reads against an "analytic_gaussian_prior" surface.
"""
from inverse_problems.ns_adjoint import (
    grf_apply_inv_cov,   # (amp^2 C)^{-1} omega0 = (1/amp^2)(-Delta+9I)^4 omega0
    grf_prior_score,     # grad log p0 = -(amp^2 C)^{-1} omega0
    grf_denoiser,        # exact Tweedie denoiser D(x,sigma) = Cs (Cs+sigma^2 I)^{-1} x
    grf_sample,          # draw omega0 ~ N(0, amp^2 C)
)

__all__ = ["grf_apply_inv_cov", "grf_prior_score", "grf_denoiser", "grf_sample"]
