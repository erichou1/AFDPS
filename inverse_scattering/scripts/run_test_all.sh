#!/usr/bin/env bash
# Run the full test set for all three receiver counts (360 -> 180 -> 60) and aggregate
# each into the Table-3 format. Thin wrapper over scripts/run_test.sh.
#
# Usage (from inverse_scattering/):  bash scripts/run_test_all.sh [NSHARD] [J] [STEPS]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
NSHARD="${1:-2}"; J="${2:-1024}"; STEPS="${3:-200}"
for R in 360 180 60; do
  echo "############################  R=$R  ############################"
  bash scripts/run_test.sh "$R" "$NSHARD" "$J" "$STEPS"
done
echo "All receiver counts complete. Compare the AFDPS rows above against InverseBench Table 3."
