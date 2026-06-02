#!/usr/bin/env bash
# Download the assets for Track B (the real InverseBench Navier-Stokes benchmark):
#   1) the pretrained diffusion prior  ns-5m.pt   (~102 MB)  -> checkpoints/ns-5m.pt
#   2) the NS test + validation LMDB datasets     (~7 MB)    -> ../data/navier-stokes-{test,val}/Re200.0-t5.0
#
# These are intentionally NOT in git (.gitignore excludes checkpoints/ and data/).
# Run this on the box where you will train/infer (e.g. the GB200). Needs curl + unzip.
#
# Usage:
#   bash scripts/download_assets.sh            # download checkpoint + data
#   bash scripts/download_assets.sh data       # data only
#   bash scripts/download_assets.sh ckpt       # checkpoint only
#
# The CaltechDATA record has PUBLIC metadata but RESTRICTED files, so a share token
# is required. The token below is the public one from README_InverseBench.md (upstream
# InverseBench). Share tokens can expire -- if the data download 401/403s, get a fresh
# link from https://data.caltech.edu/records/jfdr4-6ws87 (or check if the record's files
# became fully public) and replace CALTECH_TOKEN.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(cd "$REPO_ROOT/.." && pwd)/data"     # sibling of the repo == the configs' ../data
WHAT="${1:-all}"

CKPT_URL="https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/ns-5m.pt"
CALTECH_REC="https://data.caltech.edu/api/records/jfdr4-6ws87/files"
CALTECH_TOKEN="eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdiNDk4OGU3LWQ0NTgtNGYwNy04NDc4LWE5YWE3OWIzOTU0MSIsImRhdGEiOnt9LCJyYW5kb20iOiJlYTk1ZjU0YTdmZjcwZTQ1OTYzZTNiZTRkNTBhYmJmMiJ9.NFEYlpOyrepCIFkR6EBrVaQcGGfVam5gileyMjbnrjBCZFemXLsGyGY-qlxlPf9tGE_L1qH3lCpUJz_RTeOfiQ"

dl_data () {
  echo "== Navier-Stokes datasets -> $DATA_DIR =="
  mkdir -p "$DATA_DIR"
  local tmp; tmp="$(mktemp -d)"
  for z in navier-stokes-test navier-stokes-val; do
    echo "  downloading $z.zip ..."
    curl -fSL --retry 3 -o "$tmp/$z.zip" "$CALTECH_REC/$z.zip/content?token=$CALTECH_TOKEN"
    echo "  unzipping $z.zip ..."
    unzip -oq "$tmp/$z.zip" -d "$DATA_DIR"
  done
  rm -rf "$tmp"
  echo "  done. Layout:"
  find "$DATA_DIR" -maxdepth 2 -type d | sed 's/^/    /'
  echo "  (matches the configs' root: ../data/navier-stokes-{test,val}/Re200.0-t5.0)"
}

dl_ckpt () {
  echo "== Pretrained NS diffusion prior -> $REPO_ROOT/checkpoints/ns-5m.pt (~102 MB) =="
  mkdir -p "$REPO_ROOT/checkpoints"
  curl -fSL --retry 3 -o "$REPO_ROOT/checkpoints/ns-5m.pt" "$CKPT_URL"
  echo "  done: $(ls -la "$REPO_ROOT/checkpoints/ns-5m.pt" | awk '{print $5" bytes"}')"
}

case "$WHAT" in
  data) dl_data ;;
  ckpt|checkpoint) dl_ckpt ;;
  all) dl_ckpt; dl_data ;;
  *) echo "usage: bash scripts/download_assets.sh [all|data|ckpt]"; exit 1 ;;
esac
echo "All requested assets downloaded."
