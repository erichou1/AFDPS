#!/usr/bin/env bash
# GB200 smoke + step benchmark. Run this BEFORE any sweep. Order matters:
#   0) assert the real checkpoint exposes the EDM-net interface the sampler needs;
#   1) materialize the R=360 SVD cache (serial, no race);
#   2) one real inference case (J=64, 50 steps) end-to-end;
#   3) timed per-step benchmark at J in {512,1024,2048} to pick the batch that fills
#      the device (watch GPU util alongside via scripts/gpu_log.sh).
#
# Usage (from inverse_scattering/):  bash scripts/smoke_gb200.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY="${PYTHON:-python3}"
CKPT="checkpoints/inv-scatter-5m.pt"

echo "== [0/3] checkpoint interface assertion =="
"$PY" - "$CKPT" <<'PY'
import sys, os, pickle, torch
sys.path.insert(1, os.path.abspath('.')); sys.path.insert(1, os.path.abspath('../navier_stokes'))
path = sys.argv[1]
try:
    with open(path, 'rb') as f:
        net = pickle.load(f)['ema']
except Exception:
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base='1.3'):
        cfg = compose(config_name='config')
    net = instantiate(cfg.pretrain.model)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net.load_state_dict(ck['ema'] if 'ema' in ck else ck['net'])
for attr in ('sigma_min', 'sigma_max', 'round_sigma', 'img_channels', 'img_resolution'):
    assert hasattr(net, attr), f"net missing {attr}"
assert net.img_channels == 1 and net.img_resolution == 128, (net.img_channels, net.img_resolution)
x = torch.zeros(2, 1, 128, 128)
d = net(x, torch.tensor(1.0))
assert d.shape == x.shape, d.shape
print(f"  OK: img={net.img_resolution}x{net.img_channels}, sigma in [{net.sigma_min:.3g},{net.sigma_max:.3g}], net(x,sigma)->D_x OK")
PY

echo "== [1/3] precompute R=360 SVD cache =="
"$PY" scripts/precompute_svd.py --numTrans 20 --numRec 360

echo "== [2/3] one real inference case (J=64, 50 steps) =="
"$PY" main.py problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter \
  num_samples=1 wandb=false exp_name=smoke \
  problem.data.id_list=0 problem.model.numRec=360 \
  algorithm.method.num_particles=64 algorithm.method.num_steps=50 \
  algorithm.method.sampler_kwargs.progress=true

echo "== [3/3] per-step throughput benchmark =="
"$PY" - <<'PY'
import os, sys, time, torch, pickle
sys.path.insert(1, os.path.abspath('.')); sys.path.insert(1, os.path.abspath('../navier_stokes'))
from inverse_problems.inverse_scatter_afdps import AFDPSInverseScatter
from algo.afdps_scatter import AFDPSScatter
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
op = AFDPSInverseScatter(Nx=128, Ny=128, numRec=360, numTrans=20, sigma_noise=1e-4,
                         unnorm_shift=1.0, unnorm_scale=0.5, device=dev, svd=True)
try:
    with open('checkpoints/inv-scatter-5m.pt', 'rb') as f:
        net = pickle.load(f)['ema'].to(dev).eval()
except Exception:
    from hydra.utils import instantiate
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base='1.3'):
        cfg = compose(config_name='config')
    net = instantiate(cfg.pretrain.model)
    ck = torch.load('checkpoints/inv-scatter-5m.pt', map_location=dev, weights_only=False)
    net.load_state_dict(ck['ema'] if 'ema' in ck else ck['net'])
    net = net.to(dev).eval()
torch.manual_seed(0)
obs = op.forward(2*torch.rand(1,1,128,128,device=dev)-1, unnormalize=True)
for J in (512, 1024, 2048):
    algo = AFDPSScatter(net, op, num_particles=J, num_steps=20, sigma_max=80.0,
                        reduce='mean', sampler_kwargs=dict(guidance_mode='full',
                        guidance_step='exact_linear', use_value=True, value_coef='exact', progress=False))
    if dev == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); algo.inference(obs, num_samples=1)
    if dev == 'cuda': torch.cuda.synchronize()
    dt = (time.time()-t0)/20
    mem = torch.cuda.max_memory_allocated()/1024**3 if dev=='cuda' else 0.0
    print(f"  J={J:5d}: {dt*1000:7.1f} ms/step  peak_mem={mem:5.2f} GB")
print("Pick the largest J that fits and keeps util high (see gpu_log.sh).")
PY
echo "Smoke complete."
