#!/usr/bin/env bash
# Download the assets for the AFDPS linear inverse-scattering benchmark:
#   1) the pretrained diffusion prior            -> checkpoints/inv-scatter-5m.pt
#   2) the inverse-scattering test (+val) LMDBs  -> ../data/inv-scatter-{test,val}
#
# Run this on the box where you will infer (e.g. the GB200). Needs curl + unzip.
# Always invoked from the inverse_scattering/ directory.
#
# Usage:
#   bash scripts/download_assets.sh           # checkpoint + test/val data
#   bash scripts/download_assets.sh data      # data only
#   bash scripts/download_assets.sh ckpt      # checkpoint only
#
# Asset-name caveats (see plan "Risks"): the GitHub release asset has been seen as both
# `inv-scatter-5m.pt` and `in-scatter-5m.pt`; the CaltechDATA share token can expire.
# This script tries both checkpoint names and reports a clear error if neither resolves.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"        # inverse_scattering/
DATA_DIR="$(cd "$HERE/.." && pwd)/data"          # ../data (shared with the NS configs)
WHAT="${1:-all}"

REL_BASE="https://github.com/devzhk/InverseBench/releases/download/diffusion-prior"
CKPT_CANDIDATES=("inv-scatter-5m.pt" "in-scatter-5m.pt")
CKPT_OUT="$HERE/checkpoints/inv-scatter-5m.pt"

CALTECH_REC="https://data.caltech.edu/api/records/zg89b-mpv16/files"
# Public share token from upstream InverseBench (README_InverseBench.md). May expire ->
# if data downloads 401/403, refresh from https://data.caltech.edu/records/jfdr4-6ws87.
CALTECH_TOKEN="eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdiNDk4OGU3LWQ0NTgtNGYwNy04NDc4LWE5YWE3OWIzOTU0MSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYTk1ZjU0YTdmZjcwZTQ1OTYzZTNiZTRkNTBhYmJmMiJ9.NFEYlpOyrepCIFkR6EBrVaQcGGfVam5gileyMjbnrjBCZFemXLsGyGY-qlxlPf9tGE_L1qH3lCpUJz_RTeOfiQ"

dl_ckpt () {
  echo "== Pretrained inverse-scattering diffusion prior -> $CKPT_OUT =="
  mkdir -p "$HERE/checkpoints"
  for name in "${CKPT_CANDIDATES[@]}"; do
    echo "  trying release asset: $name ..."
    if curl -fSL --retry 3 -o "$CKPT_OUT" "$REL_BASE/$name"; then
      echo "  done: $(ls -la "$CKPT_OUT" | awk '{print $5" bytes"}') (from $name)"
      return 0
    fi
  done
  echo "  ERROR: could not download the checkpoint under any known name (${CKPT_CANDIDATES[*]})."
  echo "         Update CKPT_CANDIDATES / REL_BASE in this script."
  return 1
}

dl_data () {
  echo "== Inverse-scattering datasets -> $DATA_DIR =="
  mkdir -p "$DATA_DIR"; local tmp; tmp="$(mktemp -d)"
  for z in inv-scatter-test inv-scatter-val; do
    echo "  downloading $z.zip ..."
    if curl -fSL --retry 3 -o "$tmp/$z.zip" "$CALTECH_REC/$z.zip/content?token=$CALTECH_TOKEN"; then
      unzip -oq "$tmp/$z.zip" -d "$DATA_DIR" && echo "    unpacked $z"
    else
      echo "    WARNING: $z.zip not available (val split may not be published separately)."
    fi
  done
  rm -rf "$tmp"
  echo "  layout:"; find "$DATA_DIR" -maxdepth 1 -type d -name 'inv-scatter*' | sed 's/^/    /'
}

case "$WHAT" in
  data) dl_data ;;
  ckpt|checkpoint) dl_ckpt ;;
  all) dl_ckpt; dl_data ;;
  *) echo "usage: bash scripts/download_assets.sh [all|data|ckpt]"; exit 1 ;;
esac
echo "Requested assets processed."
