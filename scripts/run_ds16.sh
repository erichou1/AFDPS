#!/bin/bash
###############################################################################
# ds=16 campaign — the extreme-downsampling regime NOT covered by InverseBench
# Table 8. AFDPS degrades slowest with ds (ds4->ds8 ratio 1.31x vs EnKG 2.01x,
# DPG 1.64x), so the curves are predicted to cross here.
#
#   Phase 0: gamma probe (scaling law gamma*ds~5-6 => gamma~0.3-0.4), sample 0
#   Phase A: AFDPS n=10 at auto-selected best gamma, ESS resampling ON
#   Phase B: EnKG  n=10 baseline
#   Phase C: DPG   n=10 baseline
# 128x128 grid, ds=16 => 8x8 = 64 observations (sparse but valid).
###############################################################################
set -e
cd ~/test/AFDPS
mkdir -p logs/ds16
DS=16
SIG=1
ROOT=../data/navier-stokes-test/Re200.0-t5.0
SHARDS=("0-2" "3-4" "5-7" "8-9")

#######################  PHASE 0: gamma probe  ################################
echo "[$(date)] === ds=16 gamma probe (sample 0) ==="
PROBE=(0.25 0.30 0.35 0.40); gpu=0
for gm in "${PROBE[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
    algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
    problem.model.adaptive=False problem.data.root=$ROOT \
    problem.model.sigma_noise=$SIG problem.model.downsample_factor=$DS \
    problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
    algorithm.method.guidance_gamma=$gm \
    algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
    algorithm.method.sigma_max=80 \
    algorithm.method.sampler_kwargs.resample=true \
    problem.data.id_list=0 \
    exp_name=ds16_probe_g$gm \
    > logs/ds16/probe_g${gm}.log 2>&1 &
  gpu=$((gpu+1))
done
wait
echo "[$(date)] probe done:"
for gm in "${PROBE[@]}"; do
  val=$(grep "Metric results" logs/ds16/probe_g${gm}.log | tail -1 | grep -oP "(?<=relative l2': )[0-9.eE+-]+")
  echo "  gamma=$gm rel-L2=$val"
done
# auto-select smallest finite rel-L2 (sort -g handles scientific notation)
GAMMA=$(for gm in "${PROBE[@]}"; do
  val=$(grep "Metric results" logs/ds16/probe_g${gm}.log | tail -1 | grep -oP "(?<=relative l2': )[0-9.eE+-]+")
  [ -n "$val" ] && echo "$val $gm"
done | sort -g | head -1 | awk '{print $2}')
echo "[$(date)] selected best gamma = $GAMMA"

#######################  PHASE A: AFDPS n=10  ################################
echo "[$(date)] === AFDPS ds=16 n=10 (gamma=$GAMMA, resample ON) ==="
gpu=0
for sh in "${SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
    algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
    problem.model.adaptive=False problem.data.root=$ROOT \
    problem.model.sigma_noise=$SIG problem.model.downsample_factor=$DS \
    problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
    algorithm.method.guidance_gamma=$GAMMA \
    algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
    algorithm.method.sigma_max=80 \
    algorithm.method.sampler_kwargs.resample=true \
    algorithm.method.sampler_kwargs.resample_threshold=0.5 \
    problem.data.id_list=$sh \
    exp_name=ds16_afdps_shard${gpu} \
    > logs/ds16/afdps_${sh}.log 2>&1 &
  gpu=$((gpu+1))
done
wait
echo "[$(date)] AFDPS ds=16 done"

#######################  PHASE B: EnKG n=10  ################################
echo "[$(date)] === EnKG ds=16 n=10 ==="
gpu=0
for sh in "${SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
    algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
    problem.model.downsample_factor=$DS problem.model.sigma_noise=$SIG \
    problem.data.root=$ROOT problem.data.id_list=$sh \
    exp_name=ds16_enkg_shard${gpu} \
    > logs/ds16/enkg_${sh}.log 2>&1 &
  gpu=$((gpu+1))
done
wait
echo "[$(date)] EnKG ds=16 done"

#######################  PHASE C: DPG n=10  ################################
echo "[$(date)] === DPG ds=16 n=10 ==="
gpu=0
for sh in "${SHARDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
    algorithm=dpg problem=navier-stokes pretrain=navier-stokes \
    problem.model.downsample_factor=$DS problem.model.sigma_noise=$SIG \
    problem.data.root=$ROOT problem.data.id_list=$sh \
    exp_name=ds16_dpg_shard${gpu} \
    > logs/ds16/dpg_${sh}.log 2>&1 &
  gpu=$((gpu+1))
done
wait
echo "[$(date)] DPG ds=16 done"

#######################  COLLECT  ###########################################
echo ""
echo "=====================  ds=16 sigma=1 RESULTS  ====================="
python3 - <<'PY'
import glob, re, statistics as st
def mean_std(prefix):
    vals=[]
    for f in sorted(glob.glob(f"logs/ds16/{prefix}_*.log")):
        raw=re.findall(r"relative l2': ([0-9.eE+-]+)", open(f, errors="ignore").read())
        vals += [float(x) for x in raw[::2]]
    return vals
for name in ["afdps","enkg","dpg"]:
    v=mean_std(name)
    if v:
        print(f"  {name.upper():6s} n={len(v)} rel-L2 = {st.mean(v):.3f} +/- {st.pstdev(v):.3f}")
    else:
        print(f"  {name.upper():6s} no values parsed")
print("\n  Lower is better. AFDPS wins if its mean < EnKG and < DPG at ds=16.")
PY
