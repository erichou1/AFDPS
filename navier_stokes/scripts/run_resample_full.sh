#!/bin/bash
###############################################################################
# Full n=10 validation of ESS resampling at the two tuned cells.
# Compares against the locked no-resample baselines:
#     ds=8 sigma=1 gamma=0.7 : 0.760 +/- 0.110
#     ds=4 sigma=1 gamma=2.0 : 0.580 +/- 0.129
# 4-way sharded (0-2, 3-4, 5-7, 8-9); ~2.9 h per cell.
###############################################################################
set -e
cd ~/test/AFDPS
mkdir -p logs/resample_full
SHARDS=("0-2" "3-4" "5-7" "8-9")

run_cell () {  # $1=ds  $2=gamma  $3=tag
  local ds=$1 gamma=$2 tag=$3 gpu=0
  echo "[$(date)] === resample n=10: ds=$ds gamma=$gamma ==="
  for sh in "${SHARDS[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
      algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
      problem.model.adaptive=False \
      problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
      problem.model.sigma_noise=1 problem.model.downsample_factor=$ds \
      problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
      algorithm.method.guidance_gamma=$gamma \
      algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
      algorithm.method.sigma_max=80 \
      algorithm.method.sampler_kwargs.resample=true \
      algorithm.method.sampler_kwargs.resample_threshold=0.5 \
      problem.data.id_list=$sh \
      exp_name=resample_${tag}_shard${gpu} \
      > logs/resample_full/${tag}_${sh}.log 2>&1 &
    gpu=$((gpu+1))
  done
  wait
  echo "[$(date)] ${tag} done"
}

collect () {  # $1=tag
  local tag=$1
  python3 - "$tag" <<'PY'
import sys, glob, re, statistics as st
tag=sys.argv[1]
vals=[]
for f in sorted(glob.glob(f"logs/resample_full/{tag}_*.log")):
    raw=re.findall(r"relative l2': ([0-9.eE+-]+)", open(f, errors="ignore").read())
    vals += [float(x) for x in raw[::2]]   # dedup print+logging
if vals:
    print(f"  {tag}: n={len(vals)} mean={st.mean(vals):.3f} "
          f"std={st.pstdev(vals):.3f}  per-sample={[round(v,3) for v in vals]}")
else:
    print(f"  {tag}: no values parsed")
PY
}

run_cell 8 0.7 ds8
run_cell 4 2.0 ds4
echo ""
echo "=================  RESAMPLE n=10 RESULTS  ================="
collect ds8
collect ds4
echo "Baselines (no resample): ds8=0.760+/-0.110  ds4=0.580+/-0.129"
echo "Targets to beat:         DPG ds8=0.591  EnKG ds8=0.546 | DPG ds4=0.361  EnKG ds4=0.271"
