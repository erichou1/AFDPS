"""AFDPS for InverseBench Full Waveform Inversion (FWI).

This package applies AFDPS ("Approximation-Free Diffusion Posterior Sampling",
Chen, Ren, Min, Ying, Izzo, TMLR 2026) to the InverseBench full waveform inversion
benchmark. It reuses the InverseBench harness shipped in ``navier_stokes/`` as a
library (the FWI Devito forward operator ``inverse_problems.acoustic.AcousticWave``,
the ``eval.AcousticWave`` evaluator, the ``training.dataset.LMDBData`` loader and the
pretrained EDM prior), and adds only the FWI-specific pieces:

    * ``sampler.Ensemble_Denoiser_EDM`` -- the annealed-SDE Feynman-Kac ensemble
      sampler (vendored verbatim from the Navier-Stokes AFDPS port; problem-agnostic).
    * ``algo.AFDPS`` -- the ``Algo.inference`` wrapper.
    * ``operator.AFDPSAcousticWave`` -- the FWI operator API the sampler drives
      (``initialize_ensemble`` / ``proximal_generator`` / ``likelihood_gradient`` /
      ``likelihood_laplacian`` / ``likelihood_value``), built on the adjoint-state
      gradient already provided by the Devito ``AcousticWave`` operator.

The package is deliberately named ``afdps_fwi`` (rather than ``algo`` /
``inverse_problems``) so it never shadows the identically named packages it reuses
from ``navier_stokes/`` when both directories are on ``sys.path``.

``operator.AFDPSAcousticWave`` is imported lazily (via module ``__getattr__``) because it
pulls in Devito through the parent ``AcousticWave``; this keeps ``afdps_fwi.sampler`` and
``afdps_fwi.algo`` importable (and unit-testable) on machines without Devito installed.
"""

from .sampler import Ensemble_Denoiser_EDM
from .algo import AFDPS

__all__ = ["Ensemble_Denoiser_EDM", "AFDPS", "AFDPSAcousticWave"]


def __getattr__(name):
    # PEP 562 lazy import: only pull in the Devito-backed operator when actually requested.
    if name == "AFDPSAcousticWave":
        from .operator import AFDPSAcousticWave
        return AFDPSAcousticWave
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
