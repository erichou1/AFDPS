"""Batch-capable forward operator for EKI/EnKG on the inverse scattering problem.

Subclasses InverseScatter and overrides ``forward()`` to handle batched input
``(B, 1, Ny, Nx)`` using the cached real SVD matrix ``A`` for a single GEMM.
Returns complex ``(B, numTrans, numRec)`` matching the original interface exactly.

Observation generation (``__call__``), ``loss()``, ``unnormalize()``, and every
metric API are **inherited unchanged** — the forward map is mathematically
identical, just vectorized over the batch dimension.

Use with EKI/EnKG, which pass N (512-2048) particles through ``forward()`` in one
call.  The base ``InverseScatter.forward()`` only handles ``batch_size=1``
(``f[0, 0]``), so calling it with ``N > 1`` silently computes the forward for the
first sample only — this class fixes that.
"""
import torch

from .inverse_scatter import InverseScatter


class BatchInverseScatter(InverseScatter):
    """InverseScatter with a batched forward via the cached SVD matrix ``A``.

    The real matrix ``A`` (shape ``(2m, n)``, ``m = numTrans * numRec``,
    ``n = Ny * Nx``) is computed once by ``InverseScatter.compute_svd()`` and
    cached on disk.  The batched forward is a single GEMM::

        y_real = x_flat @ A^T          (B, n) × (n, 2m) → (B, 2m)

    followed by a reshape to complex ``(B, numTrans, numRec)``.

    The row ordering of ``A`` interleaves ``[Re_0, Im_0, Re_1, Im_1, …]`` for
    each of the ``m = numTrans × numRec`` measurements, which is exactly
    ``torch.view_as_real(y_complex.flatten()).flatten()``.  The reshape and
    ``view_as_complex`` roundtrip is therefore exact.
    """

    def __init__(self, svd=True, **kwargs):
        assert svd, "BatchInverseScatter requires the cached SVD (svd=True)."
        super().__init__(svd=True, **kwargs)
        # Pin A to operator device in fp64 for the batched GEMM.
        self.A = self.A.to(self.device, torch.float64)

    def forward(self, f, unnormalize=True):
        """Batched forward using the cached real SVD matrix ``A``.

        Parameters
        ----------
        f : Tensor, shape ``(B, 1, Ny, Nx)``
            Normalized permittivity contrast in ``[-1, 1]``.
        unnormalize : bool
            If True (default), map from model domain to physical domain before
            applying the forward operator.

        Returns
        -------
        y : Tensor, shape ``(B, numTrans, numRec)``, complex128
            Complex scattered fields — mathematically identical to the base
            ``InverseScatter.forward()`` but supporting arbitrary batch size.
        """
        f = f.to(torch.float64)
        if unnormalize:
            f = self.unnormalize(f)
        B = f.shape[0]
        f_flat = f.reshape(B, -1)                                # (B, n)
        y_real = f_flat @ self.A.T                                # (B, 2m)
        m = self.numTrans * self.numRec
        y_complex = torch.view_as_complex(
            y_real.reshape(B, m, 2).contiguous()
        )                                                         # (B, m)
        return y_complex.reshape(B, self.numTrans, self.numRec)
