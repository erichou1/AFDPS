#!/bin/bash
###############################################################################
# Resampling pilot — does restoring AFDPS Algorithm-1 ESS resampling improve the
# reported best-particle reconstruction at ds=8?
#
# The released AFDPS skips its own resampling step "to save computational cost
# ... in a parallel way" (6054 TMLR'26, App. E). We restore it (config-gated) and
# A/B the SAME 4 samples (ids 0-3) against the recorded no-resample baseline:
#     baseline best-particle rel-L2 (ds=8, sigma=1, gamma=0.7):
#         s0=0.578  s1=0.738  s2=0.901  s3=0.790   (mean 0.752)
# One sample per GPU, ~57 min wall-clock.
###############################################################################
set -e
cd ~/test/AFDPS
mkdir -p logs/resample_pilot

GAMMA=0.7
THR=0.5
echo "[$(date)] === Resampling pilot: ds=8 sigma=1 gamma=$GAMMA threshold=$THR, samples 0-3 ==="

for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s nohup python main.py \
    algorithm=afdps problem=navier-stokes-afdps pretrain=navier-stokes \
    problem.model.adaptive=False \
    problem.data.root=../data/navier-stokes-test/Re200.0-t5.0 \
    problem.model.sigma_noise=1 problem.model.downsample_factor=8 \
    problem.model.hutchinson_M=2 problem.model.hutchinson_scheme=forward \
    algorithm.method.guidance_gamma=$GAMMA \
    algorithm.method.num_particles=32 algorithm.method.num_steps=400 \
    algorithm.method.sigma_max=80 \
    algorithm.method.sampler_kwargs.resample=true \
    algorithm.method.sampler_kwargs.resample_threshold=$THR \
    problem.data.id_list=$s \
    exp_name=resample_pilot_ds8_s$s \
    > logs/resample_pilot/ds8_g${GAMMA}_s${s}.log 2>&1 &
done
wait
echo "[$(date)] pilot runs done; collecting..."

base=(0.578 0.738 0.901 0.790)
echo ""
echo "sample | baseline(best) | resample(best) | delta"
echo "-------+----------------+----------------+---------"
sum_b=0; sum_r=0
for s in 0 1 2 3; do
  val=$(grep "Metric results" logs/resample_pilot/ds8_g${GAMMA}_s${s}.log | tail -1 | grep -oP "(?<=relative l2': )[0-9.eE+-]+")
  if [ -z "$val" ]; then val="NaN"; fi
  d=$(awk -v b="${base[$s]}" -v r="$val" 'BEGIN{ if (r=="NaN"){print "  -  "} else {printf "%+.3f", r-b} }')
  printf "   %s   |     %.3f      |     %-8s   | %s\n" "$s" "${base[$s]}" "$val" "$d"
  sum_b=$(awk -v a="$sum_b" -v b="${base[$s]}" 'BEGIN{printf "%.4f", a+b}')
  if [ "$val" != "NaN" ]; then sum_r=$(awk -v a="$sum_r" -v r="$val" 'BEGIN{printf "%.4f", a+r}'); fi
done
echo "-------+----------------+----------------+---------"
awk -v b="$sum_b" -v r="$sum_r" 'BEGIN{printf "  mean |     %.3f      |     %.3f      | %+.3f\n", b/4, r/4, (r-b)/4}'
echo ""
echo "If the resample mean is clearly below 0.752 (and ideally below DPG's 0.591),"
echo "run the full n=10 validation:  bash scripts/run_resample_full.sh"
