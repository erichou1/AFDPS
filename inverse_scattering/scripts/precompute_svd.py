"""Pre-build the forward-operator SVD caches for all receiver counts BEFORE any
parallel/sharded run.

The base `InverseScatter.compute_svd()` builds the cache lazily at operator __init__
(10-20 min the first time) and writes it CWD-relative under
`cache/inv-scatter_numT_<T>_numR_<R>/`. If several sharded inference processes start
at once they would race on that build (and torch.load without map_location can land
factors on the wrong device). Run this once, up front, to materialize every cache
serially; sharded jobs then only ever read.

Race guard: a cache is considered complete only when `matrix_inv.pt` exists (the LAST
file written by compute_svd). Incomplete dirs are rebuilt.

Usage (from inverse_scattering/):
  python scripts/precompute_svd.py --numTrans 20 --numRec 360 180 60
"""
import argparse
import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # inverse_scattering/
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'navier_stokes'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numTrans", type=int, default=20)
    ap.add_argument("--numRec", type=int, nargs="+", default=[360, 180, 60])
    ap.add_argument("--Nx", type=int, default=128)
    args = ap.parse_args()

    # Always operate from inverse_scattering/ so cache/ lands where the runs expect it.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)
    import torch  # noqa
    from inverse_problems.inverse_scatter import InverseScatter

    for R in args.numRec:
        done = os.path.join("cache", f"inv-scatter_numT_{args.numTrans}_numR_{R}", "matrix_inv.pt")
        if os.path.exists(done):
            print(f"[skip] R={R}: cache already complete ({done})")
            continue
        print(f"[build] R={R} (numTrans={args.numTrans}, Nx={args.Nx}) -> this can take 10-20 min ...",
              flush=True)
        # Constructing the operator with svd=True triggers compute_svd() and writes the cache.
        InverseScatter(Nx=args.Nx, Ny=args.Nx, numRec=R, numTrans=args.numTrans,
                       sigma_noise=1e-4, unnorm_shift=1.0, unnorm_scale=0.5,
                       device='cuda' if torch.cuda.is_available() else 'cpu', svd=True)
        assert os.path.exists(done), f"cache build for R={R} did not produce {done}"
        print(f"[done]  R={R}: {done}", flush=True)
    print("All SVD caches ready.")


if __name__ == "__main__":
    main()
