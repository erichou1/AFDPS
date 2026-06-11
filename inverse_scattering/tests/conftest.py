"""Pytest bootstrap: merge this tree with the sibling navier_stokes/ tree on sys.path
so `inverse_problems` / `algo` namespace-merge and the benchmark modules resolve."""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_IS = os.path.dirname(_TESTS)                                   # inverse_scattering/
_NS = os.path.join(os.path.dirname(_IS), 'navier_stokes')       # ../navier_stokes/

for _p in (_NS, _IS):
    if _p not in sys.path:
        sys.path.insert(1, _p)
