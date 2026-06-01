#!/usr/bin/env bash
# Download the assets needed for Track B (the real InverseBench Navier-Stokes benchmark).
#   1) the pretrained diffusion prior  ns-5m.pt
#   2) the NS test/validation datasets (LMDB)
#
# Sources (from README_InverseBench.md):
#   - Pretrained weights: https://github.com/devzhk/InverseBench/releases/tag/diffusion-prior
#   - Datasets:           https://data.caltech.edu/records/jfdr4-6ws87
#
# These are large and are NOT downloaded automatically. Run on the GPU box.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/checkpoints"

echo "== 1) Pretrained NS diffusion prior =="
echo "Download 'ns-5m.pt' from:"
echo "  https://github.com/devzhk/InverseBench/releases/tag/diffusion-prior"
echo "and place it at: $REPO_ROOT/checkpoints/ns-5m.pt"
echo
echo "  e.g.:  curl -L -o $REPO_ROOT/checkpoints/ns-5m.pt \\"
echo "           https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/ns-5m.pt"
echo
echo "== 2) NS datasets (LMDB) =="
echo "Download the Navier-Stokes test/val data from the Caltech data page:"
echo "  https://data.caltech.edu/records/jfdr4-6ws87"
echo "and extract so the following paths exist (matching the configs):"
echo "  ../data/navier-stokes-test/Re200.0-t5.0   (test, 10 samples)"
echo "  ../data/navier-stokes-val/Re200.0-t5.0    (val,  1 sample)"
echo
echo "(Paths are relative to the repo root, i.e. a sibling 'data/' directory.)"
