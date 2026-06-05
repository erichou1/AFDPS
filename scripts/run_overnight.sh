#!/bin/bash
set -e
cd ~/test/AFDPS

###############################################################################
# PHASE A: AFDPS tuned γ at σ=2 (ds=4 γ=2.0 + ds=8 γ=0.7), n=10
###############################################################################
echo "[$(date)] === PHASE A: AFDPS σ=2 with tuned γ ==="
mkdir -p logs/overnight

CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
  problem.model.adaptive=False \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.model.sigma_noise=2 problem.model.downsample_factor=4 \
  problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
  algorithm.method.guidance_gamma=2.0 \
  algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
  algorithm.method.sigma_max=80 \
  problem.data.id_list=0-4 \
  exp_name=overnight_afdps_ds4s2_shard0 \
  > logs/overnight/afdps_ds4_sig2_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
  problem.model.adaptive=False \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.model.sigma_noise=2 problem.model.downsample_factor=4 \
  problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
  algorithm.method.guidance_gamma=2.0 \
  algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
  algorithm.method.sigma_max=80 \
  problem.data.id_list=5-9 \
  exp_name=overnight_afdps_ds4s2_shard1 \
  > logs/overnight/afdps_ds4_sig2_shard1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
  algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
  problem.model.adaptive=False \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.model.sigma_noise=2 problem.model.downsample_factor=8 \
  problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
  algorithm.method.guidance_gamma=0.7 \
  algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
  algorithm.method.sigma_max=80 \
  problem.data.id_list=0-4 \
  exp_name=overnight_afdps_ds8s2_shard0 \
  > logs/overnight/afdps_ds8_sig2_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
  algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
  problem.model.adaptive=False \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.model.sigma_noise=2 problem.model.downsample_factor=8 \
  problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
  algorithm.method.guidance_gamma=0.7 \
  algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
  algorithm.method.sigma_max=80 \
  problem.data.id_list=5-9 \
  exp_name=overnight_afdps_ds8s2_shard1 \
  > logs/overnight/afdps_ds8_sig2_shard1.log 2>&1 &

wait
echo "[$(date)] Phase A done"

###############################################################################
# PHASE B: EnKG — all 6 σ>0 cells × 10 samples
# EnKG is fast (~5 min/sample), so run 4 cells at a time
###############################################################################
echo "[$(date)] === PHASE B: EnKG baselines ==="

# Round B1: ds=2 σ=1, ds=2 σ=2, ds=4 σ=1, ds=4 σ=2  (4 GPUs, 1 cell each)
gpu=0
for ds in 2 4; do
  for sig in 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
      algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
      problem.model.downsample_factor=$ds \
      problem.model.sigma_noise=$sig \
      problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
      problem.data.id_list=0-9 \
      exp_name=overnight_enkg_ds${ds}_sig${sig} \
      > logs/overnight/enkg_ds${ds}_sig${sig}.log 2>&1 &
    gpu=$((gpu+1))
  done
done
wait
echo "[$(date)] EnKG round B1 done (ds=2,4)"

# Round B2: ds=8 σ=1, ds=8 σ=2 — sharded across 4 GPUs (2 GPUs per cell)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=1 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=0-4 \
  exp_name=overnight_enkg_ds8_sig1_shard0 \
  > logs/overnight/enkg_ds8_sig1_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=1 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=5-9 \
  exp_name=overnight_enkg_ds8_sig1_shard1 \
  > logs/overnight/enkg_ds8_sig1_shard1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
  algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=2 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=0-4 \
  exp_name=overnight_enkg_ds8_sig2_shard0 \
  > logs/overnight/enkg_ds8_sig2_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
  algorithm=enkg problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=2 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=5-9 \
  exp_name=overnight_enkg_ds8_sig2_shard1 \
  > logs/overnight/enkg_ds8_sig2_shard1.log 2>&1 &

wait
echo "[$(date)] EnKG round B2 done (ds=8, sharded)"
echo "[$(date)] Phase B done"

###############################################################################
# PHASE C: DPS — all 6 σ>0 cells × 10 samples
# DPS is slower (~15 min/sample), run 4 cells at a time
###############################################################################
echo "[$(date)] === PHASE C: DPS baselines ==="

# Round C1: ds=2 σ=1, ds=2 σ=2, ds=4 σ=1, ds=4 σ=2  (4 GPUs, 1 cell each)
gpu=0
for ds in 2 4; do
  for sig in 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
      algorithm=dps problem=navier-stokes pretrain=navier-stokes \
      problem.model.downsample_factor=$ds \
      problem.model.sigma_noise=$sig \
      problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
      problem.data.id_list=0-9 \
      exp_name=overnight_dps_ds${ds}_sig${sig} \
      > logs/overnight/dps_ds${ds}_sig${sig}.log 2>&1 &
    gpu=$((gpu+1))
  done
done
wait
echo "[$(date)] DPS round C1 done (ds=2,4)"

# Round C2: ds=8 σ=1, ds=8 σ=2 — sharded across 4 GPUs (2 GPUs per cell)
CUDA_VISIBLE_DEVICES=0 nohup python main.py \
  algorithm=dps problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=1 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=0-4 \
  exp_name=overnight_dps_ds8_sig1_shard0 \
  > logs/overnight/dps_ds8_sig1_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py \
  algorithm=dps problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=1 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=5-9 \
  exp_name=overnight_dps_ds8_sig1_shard1 \
  > logs/overnight/dps_ds8_sig1_shard1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python main.py \
  algorithm=dps problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=2 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=0-4 \
  exp_name=overnight_dps_ds8_sig2_shard0 \
  > logs/overnight/dps_ds8_sig2_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python main.py \
  algorithm=dps problem=navier-stokes pretrain=navier-stokes \
  problem.model.downsample_factor=8 \
  problem.model.sigma_noise=2 \
  problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
  problem.data.id_list=5-9 \
  exp_name=overnight_dps_ds8_sig2_shard1 \
  > logs/overnight/dps_ds8_sig2_shard1.log 2>&1 &

wait
echo "[$(date)] DPS round C2 done (ds=8, sharded)"
echo "[$(date)] Phase C done"

###############################################################################
# PHASE D: Quick γ probe for ds=2 σ=2 (check if γ=5.844 is optimal)
###############################################################################
echo "[$(date)] === PHASE D: ds=2 σ=2 γ probe ==="
mkdir -p logs/overnight/ds2_sig2_probe

for i in 0 1 2 3; do
  gamma=(3.0 4.0 5.0 8.0)
  g=${gamma[$i]}
  CUDA_VISIBLE_DEVICES=$i nohup python main.py \
    algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
    problem.model.adaptive=False \
    problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
    problem.model.sigma_noise=2 problem.model.downsample_factor=2 \
    problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
    algorithm.method.guidance_gamma=$g \
    algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
    algorithm.method.sigma_max=80 \
    problem.data.id_list=0-0 \
    exp_name=overnight_ds2s2_probe_g${g} \
    > logs/overnight/ds2_sig2_probe/g${g}.log 2>&1 &
done
wait
echo "[$(date)] Phase D done"

###############################################################################
# COLLECT ALL RESULTS
###############################################################################
echo ""
echo "================================================================"
echo "  OVERNIGHT RUN COMPLETE — $(date)"
echo "================================================================"

python3 - <<'PYEOF'
import re, glob, statistics

def extract(pattern):
    vals=[]
    for f in sorted(glob.glob(pattern)):
        with open(f) as fp:
            for line in fp:
                m=re.search(r"Metric results: \{'relative l2': ([0-9.e+-]+)\}", line)
                if m and 'Final' not in line: vals.append(float(m.group(1)))
    vals=vals[::2] if len(vals)>10 else vals
    return vals

print("\n=== AFDPS (tuned γ, σ=2) ===")
for tag, label in [('ds4_sig2', 'ds=4 γ=2.0 σ=2'), ('ds8_sig2', 'ds=8 γ=0.7 σ=2')]:
    vals = extract(f'logs/overnight/afdps_{tag}_shard*.log')
    if len(vals)>=2:
        print(f'  {label}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} (n={len(vals)})')
    else:
        print(f'  {label}: {len(vals)} samples')

print("\n=== EnKG baselines ===")
for ds in [2,4,8]:
    for sig in [1,2]:
        # Collect from both unsharded and sharded log files
        vals = extract(f'logs/overnight/enkg_ds{ds}_sig{sig}.log')
        vals += extract(f'logs/overnight/enkg_ds{ds}_sig{sig}_shard*.log')
        if len(vals)>=2:
            print(f'  ds={ds} σ={sig}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} (n={len(vals)})')
        else:
            print(f'  ds={ds} σ={sig}: {len(vals)} samples')

print("\n=== DPS baselines ===")
for ds in [2,4,8]:
    for sig in [1,2]:
        # Collect from both unsharded and sharded log files
        vals = extract(f'logs/overnight/dps_ds{ds}_sig{sig}.log')
        vals += extract(f'logs/overnight/dps_ds{ds}_sig{sig}_shard*.log')
        if len(vals)>=2:
            print(f'  ds={ds} σ={sig}: {statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f} (n={len(vals)})')
        else:
            print(f'  ds={ds} σ={sig}: {len(vals)} samples')

print("\n=== ds=2 σ=2 γ probe ===")
for g in ['3.0', '4.0', '5.0', '8.0']:
    vals = extract(f'logs/overnight/ds2_sig2_probe/g{g}.log')
    if vals:
        print(f'  γ={g}: rel-L2 = {vals[0]:.4f}')
    else:
        print(f'  γ={g}: no result')
PYEOF
