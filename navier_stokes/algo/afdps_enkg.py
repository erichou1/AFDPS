"""
EnKG warm-start -> AFDPS refinement hybrid (improvement #2).

Combines the complementary strengths of the two methods:

  * Stage 1 -- EnKG (derivative-free ensemble Kalman) gets cheaply and robustly CLOSE.
               Its strength is exactly where AFDPS is weakest: it needs no gradient, so the
               chaotic Re=200 adjoint never destabilizes it, and a large cheap ensemble
               gives a low-variance estimate.
  * Stage 2 -- AFDPS refines from that warm start. Instead of annealing from pure noise at
               sigma_max=80, it starts the diffusion at a REDUCED sigma (warm_sigma_max) so
               the EnKG estimate survives, then uses the learned diffusion prior + the exact
               adjoint likelihood to add the high-frequency detail EnKG misses.

The hope is to beat BOTH parents in the sparse + high-noise regime: EnKG's robustness to
seed the basin, AFDPS's prior to sharpen it. The warm start is passed to the AFDPS sampler
through the operator's `init_mode='warm'` + `_warm_start` hook; the per-particle diversity
noise is the truncated top-level sigma (`warm_sigma_max`), added by `proximal_generator`.
"""
import torch

from .base import Algo
from .enkg import EnKG
from .afdps import AFDPS


class AFDPS_EnKG(Algo):
    def __init__(self, net, forward_op,
                 enkg=None,               # kwargs dict for the EnKG warm-start stage
                 afdps=None,              # kwargs dict for the AFDPS refinement stage
                 ):
        super().__init__(net, forward_op)
        self.enkg = EnKG(net, forward_op, **dict(enkg or {}))
        self.afdps = AFDPS(net, forward_op, **dict(afdps or {}))

    @torch.no_grad()
    def inference(self, observation, num_samples=1, **kwargs):
        op = self.forward_op

        # ---- Stage 1: EnKG warm start ----
        enkg_ensemble = self.enkg.inference(observation)        # (N_enkg, C, H, W), normalized
        warm = enkg_ensemble.mean(dim=0, keepdim=True)          # low-variance warm mean

        # ---- Stage 2: AFDPS refinement seeded at the warm start ----
        # initialize_ensemble returns `warm` (J copies); proximal_generator adds
        # warm_sigma_max * N(0, I) for ensemble diversity, then the (truncated) anneal runs.
        prev_init, prev_warm = op.init_mode, getattr(op, '_warm_start', None)
        op._warm_start = warm
        op.init_mode = 'warm'
        try:
            recon = self.afdps.inference(observation, num_samples=num_samples)
        finally:
            op.init_mode, op._warm_start = prev_init, prev_warm   # restore operator state
        return recon
