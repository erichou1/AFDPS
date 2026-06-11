"""Generate SETUP_GUIDE.pdf -- a step-by-step setup & run guide for the AFDPS
linear-inverse-scattering port.

Reproducible: `python scripts/make_setup_pdf.py` writes ../SETUP_GUIDE.pdf.
Text is ASCII-only so it renders with PDF core fonts (no font embedding needed).
The renderer is the same lightweight FPDF engine used by the Navier-Stokes guide.
"""
import os
from fpdf import FPDF

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SETUP_GUIDE.pdf")

# ----------------------------------------------------------------------------- content model
# Each item: ("h1"|"h2"|"body"|"bullet"|"num"|"code"|"space"|"rule", text)
C = []
def h1(t): C.append(("h1", t))
def h2(t): C.append(("h2", t))
def body(t): C.append(("body", t))
def bullet(t): C.append(("bullet", t))
def num(t): C.append(("num", t))
def code(t): C.append(("code", t))
def space(): C.append(("space", ""))
def rule(): C.append(("rule", ""))

h1("AFDPS for Linear Inverse Scattering -- Setup & Run Guide")
body("This guide takes you from a fresh checkout to the full InverseBench Table-3 comparison for "
     "the AFDPS linear-inverse-scattering port: applying AFDPS (Approximation-Free Diffusion "
     "Posterior Sampling, TMLR 2026) to recover the permittivity contrast from a complex scattered "
     "light field. The numerical core is already built and VERIFIED on CPU (24/24 tests); the "
     "remaining work is asset download and the GPU (GB200) runs.")
space()
body("All AFDPS-scattering code lives under the inverse_scattering/ directory. The InverseBench "
     "harness, forward operator, dataset loader, EDM prior, and evaluator are REUSED by import from "
     "the sibling navier_stokes/ tree, untouched, so observation generation and metrics are exactly "
     "the benchmark's and the comparison to Table 3 is fair.")
space()
body("Goal: place the AFDPS PSNR / SSIM / Meas-err next to the InverseBench Table 3 baselines at "
     "360 / 180 / 60 receivers (noise sigma_y = 1e-4). Best PnP-diffusion baselines @360 receivers: "
     "RED-diff 36.56, DDNM 36.38 PSNR; DiffPIR / PnP-DM 0.988 SSIM.")

rule()
h1("Why scattering is different from (and easier than) Navier-Stokes")
body("The Navier-Stokes port had a NONLINEAR PDE forward, so its likelihood gradient needed a "
     "discrete-adjoint solve and its log-likelihood Laplacian needed a stochastic Hutchinson "
     "estimate. Linear inverse scattering (first Born approximation) has a LINEAR forward operator "
     "A, and the benchmark operator already caches a real SVD  A = U S V^T. So the entire AFDPS "
     "likelihood engine here is exact and closed-form:")
bullet("gradient:   grad mu_y = A^T (A x - y~) / sigma^2     (one V/U projection, no autograd)")
bullet("Laplacian:  Tr(Hessian) = sum_i s_i^2 / sigma^2      (an exact constant, no Hutchinson)")
bullet("PiGDM:      A^T (sigma_y^2 I + sigma_t^2 A A^T)^-1 r  (diagonal in the U-basis, no CG)")
bullet("guidance:   the linear guidance ODE is integrated EXACTLY each step in the V-basis "
       "(unconditionally stable; a DDNM-like data projection at the final step).")
space()
body("Two consequences shape the defaults. (1) sigma_y = 1e-4 is near-noiseless, so the problem is "
     "conditioning-dominated: enforce data on well-measured singular directions, keep the prior on "
     "the null space. (2) Table 3 ranks POINT estimates (PSNR/SSIM), whose leaders are effectively "
     "posterior-mean estimators -- so the reported AFDPS estimate is the weighted ENSEMBLE MEAN, "
     "not a single posterior sample.")

rule()
h1("Where things stand (what is already done)")
bullet("Operator, sampler, algorithm, configs, Hydra harness, GB200 scripts, and an aggregator are "
       "all implemented under inverse_scattering/.")
bullet("Numerical core VERIFIED on CPU (24/24 tests, ~3 s): the analytic gradient matches autograd "
       "through the UNTOUCHED benchmark loss; the exact Laplacian and Jacobian trace match; "
       "closed-form PiGDM matches a dense solve; the exact-linear guidance substep matches a "
       "20k-step Euler integration; sampler determinism; fairness guards (no operator overrides); "
       "Hydra composition + searchpath; and the linear-Gaussian TWIN -- the AFDPS weighted mean "
       "recovers the closed-form posterior mean on the measured subspace.")
bullet("The forward operator, dataset, EDM-prior architecture, and evaluator are the benchmark's, "
       "reused by import (fairness).")
space()
body("So the next steps are purely: download assets, then run on the GPU box (smoke -> validation "
     "sweep -> full test -> aggregate).")

rule()
h1("Prerequisites")
bullet("A Linux GPU box (the GB200). NVIDIA driver + CUDA runtime matching your torch build.")
bullet("Python 3.9+ (3.10 / 3.11 recommended).")
bullet("~5-10 GB disk for the checkpoint + datasets + SVD caches. Internet to GitHub + Caltech data.")
bullet("Both trees side by side: inverse_scattering/ and its sibling navier_stokes/ (the harness is "
       "imported from the latter).")

rule()
h1("Step 1 -- Get the repo and enter the scattering directory")
body("Clone the repository and cd into the inverse_scattering directory. Every command below is run "
     "from there (the entry points pin the working directory so ../data, checkpoints/, cache/, "
     "exps/ and the Hydra searchpath ../navier_stokes/configs all resolve consistently).")
code("git clone https://github.com/erichou1/AFDPS.git\n"
     "cd AFDPS/inverse_scattering")
body("Confirm the sibling harness is present (the AFDPS-scattering code imports it):")
code("ls ../navier_stokes/main.py ../navier_stokes/eval.py   # should exist")

rule()
h1("Step 2 -- Environment")
body("Create a virtual environment and install the dependencies. On a box that already has a "
     "system torch, --system-site-packages lets you reuse it:")
code("python -m venv --system-site-packages .venv\n"
     "source .venv/bin/activate\n"
     "pip install torch numpy scipy pyyaml tqdm omegaconf hydra-core lmdb piq requests fpdf")
body("Inference and verification need neither wandb nor an account.")

rule()
h1("Step 3 -- CPU verification (no GPU, no checkpoint, no data)")
body("The whole value of the SVD likelihood engine is that it is correct. Run the ladder; it "
     "finishes in seconds:")
code("python -m pytest tests/ -q        # expect: 24 passed")
body("This proves the gradient / Laplacian / PiGDM / exact-linear integrator / affine fold / "
     "Feynman-Kac weighting are all correct against the untouched benchmark, BEFORE any GPU time.")

rule()
h1("Step 4 -- Download assets (checkpoint + data)")
body("Download the pretrained diffusion prior and the inverse-scattering test (and validation) "
     "datasets. Run on the box where you will infer:")
code("bash scripts/download_assets.sh            # checkpoint + test/val data\n"
     "# bash scripts/download_assets.sh ckpt     # checkpoint only\n"
     "# bash scripts/download_assets.sh data     # data only")
body("This writes checkpoints/inv-scatter-5m.pt and ../data/inv-scatter-{test,val}. The script "
     "tries both known release-asset names (inv-scatter-5m.pt and in-scatter-5m.pt). If the data "
     "download 401/403s, the Caltech share token has expired -- refresh it in the script header "
     "(see the Troubleshooting section).")

rule()
h1("Step 5 -- Precompute the SVD caches (once, up front)")
body("The forward operator builds a cached SVD the first time it is constructed (10-20 min) and "
     "writes it under cache/. Build all three receiver counts serially BEFORE any parallel run so "
     "sharded jobs never race on the build:")
code("python scripts/precompute_svd.py --numTrans 20 --numRec 360 180 60")
body("A cache is treated as complete only when matrix_inv.pt exists (the last file written), so a "
     "re-run safely skips finished caches and rebuilds partial ones.")

rule()
h1("Step 6 -- GB200 smoke + throughput benchmark")
body("Validate the GPU path and pick the batch size that fills the device. This asserts the real "
     "checkpoint exposes the EDM-net interface FIRST, runs one real inference case, then benchmarks "
     "per-step time and peak memory at J in {512, 1024, 2048}:")
code("bash scripts/smoke_gb200.sh")
body("Pick the largest particle count J that fits and keeps utilization high. Watch utilization "
     "live in another shell with the GPU logger:")
code("bash scripts/gpu_log.sh start exps/util.csv\n"
     "#  ... run something ...\n"
     "bash scripts/gpu_log.sh stop exps/util.csv\n"
     "bash scripts/gpu_log.sh summarize exps/util.csv   # mean / median / p10 util, peak mem")

rule()
h1("Step 7 -- Validation sweep (find the best AFDPS config per receiver count)")
body("Run a small coordinate-descent grid on the 10 validation cases and rank by PSNR. The grid "
     "covers the guidance family (full anisotropic PiGDM / auto isotropic / fixed-gamma), the "
     "guidance step (exact_linear vs the paper-faithful euler), step count, reduction (mean vs "
     "best), and resampling:")
code("bash scripts/run_val_sweep.sh 360 512      # receiver-count J\n"
     "bash scripts/run_val_sweep.sh 180 512\n"
     "bash scripts/run_val_sweep.sh 60  512")
body("It prints a ranked table (scripts/val_table.py). Use the winning config for that receiver "
     "count in the full test run. The promoted primary is full PiGDM + exact_linear + ensemble "
     "mean; expect it to lead at 360 / 180, with more particles / resampling helping most at 60.")

rule()
h1("Step 8 -- Full test set + Table-3 comparison")
body("Run all 100 test cases for each receiver count, sharded across co-tenant GPU processes with "
     "utilization logging, then aggregate into the Table-3 format:")
code("bash scripts/run_test_all.sh 2 1024 200     # NSHARD J STEPS  (360 -> 180 -> 60)")
body("Or one receiver count at a time, then aggregate manually:")
code("bash scripts/run_test.sh 360 2 1024 200\n"
     "python scripts/aggregate_table3.py --numRec 360 \\\n"
     "    \"exps/inference/inverse-scatter-afdps/AFDPS/final_R360_shard*/result_*.pt\"")
body("The aggregator prints the full 12-method InverseBench baseline table for the chosen receiver "
     "count with the AFDPS row appended (markdown + LaTeX), and flags whether AFDPS leads on PSNR.")
space()
body("A single reference inference run (no sharding):")
code("python main.py problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter \\\n"
     "    num_samples=1 wandb=false problem.model.numRec=360")

rule()
h1("Tuning knobs (override on the CLI)")
bullet("algorithm.method.num_particles -- ensemble size; raise to fill the GPU (batched).")
bullet("algorithm.method.num_steps -- annealed-SDE steps (100-400).")
bullet("algorithm.method.sampler_kwargs.guidance_mode -- full (anisotropic PiGDM, primary) | "
       "auto (isotropic, gamma_e^2 = lambda_bar) | fixed (hand-set guidance_gamma).")
bullet("algorithm.method.sampler_kwargs.guidance_step -- exact_linear (primary) | euler "
       "(paper-faithful AFDPS-SDE ablation).")
bullet("algorithm.method.reduce -- mean (Table-3 number) | best / topk (posterior-sample draws).")
bullet("algorithm.method.guidance_gamma -- only used by guidance_mode=fixed.")
space()
body("Paper-faithful AFDPS-SDE ablation example:")
code("python main.py problem=inv-scatter-afdps algorithm=afdps pretrain=inv-scatter wandb=false \\\n"
     "    algorithm.method.sampler_kwargs.guidance_mode=fixed \\\n"
     "    algorithm.method.sampler_kwargs.guidance_step=euler \\\n"
     "    algorithm.method.guidance_gamma=10 algorithm.method.reduce=best")

rule()
h1("GB200 efficiency")
bullet("The ensemble is batched across particles (one UNet forward over J particles per step) -- "
       "raise num_particles to saturate the device; smoke_gb200.sh benchmarks J to pick the largest "
       "that fits.")
bullet("Cases are seeded per global id before observation generation, so the 100-case test set can "
       "be sharded across co-tenant processes (run_test.sh NSHARD) with the aggregated mean PSNR "
       "independent of shard count.")
bullet("gpu_log.sh reports mean / median / p10 utilization so you can confirm the device stays near "
       "100%. Benchmark single-shard utilization first; only raise NSHARD if a real gap remains "
       "(each shard reloads the multi-GB SVD factors and contends for SMs).")
bullet("Per-case seeding reproduces INPUTS, not bitwise outputs (CUDA GEMM/conv nondeterminism), "
       "unless torch.use_deterministic_algorithms(True) -- the realistic guarantee is statistically "
       "identical mean PSNR.")

rule()
h1("Troubleshooting")
bullet("Data download 401/403: the Caltech share token expired -> refresh CALTECH_TOKEN in "
       "scripts/download_assets.sh (get a fresh link from the InverseBench data record).")
bullet("Checkpoint name: the release asset has been seen as both inv-scatter-5m.pt and "
       "in-scatter-5m.pt; the downloader tries both. If neither resolves, update CKPT_CANDIDATES.")
bullet("Parallel shards race / load errors on cache: run scripts/precompute_svd.py FIRST so every "
       "SVD cache is complete before launching shards.")
bullet("CUDA out of memory: lower algorithm.method.num_particles (and/or NSHARD). The SVD factors "
       "at R=360 add a few GB on top of the ensemble.")
bullet("PSNR below the baselines: confirm reduce=mean (not best), prefer guidance_mode=full + "
       "guidance_step=exact_linear, and re-check the per-receiver winner from the validation sweep.")

rule()
h1("Open items to confirm before publishing")
bullet("A separate inv-scatter-val LMDB (10 images) is downloadable; otherwise split the test set.")
bullet("The release checkpoint name and that the Caltech share token is still valid.")
bullet("numTrans = 20 matches the InverseBench Table-3 setup.")
bullet("The Meas-err (%) definition: the aggregator uses 100*||A xhat - y|| / ||y||; pin this to "
       "the InverseBench appendix before reporting the comparison column.")

rule()
h1("Key files & reference")
bullet("README.md -- overview + the same run commands.")
bullet("plan.md -- full design, derivations (affine fold, sqrt(2) noise, exact-linear), risks.")
bullet("inverse_problems/inverse_scatter_afdps.py -- AFDPSInverseScatter (exact SVD engine).")
bullet("algo/afdps_scatter.py + algo/afdps_core_scatter/ensemble_denoiser_edm.py -- algorithm + sampler.")
bullet("configs/{config.yaml, problem/inv-scatter-afdps.yaml, algorithm/afdps.yaml}.")
bullet("scripts/ -- download_assets.sh, precompute_svd.py, smoke_gb200.sh, run_val_sweep.sh, "
       "run_test.sh, run_test_all.sh, gpu_log.sh, aggregate_table3.py, val_table.py, metrics_lib.py.")
bullet("tests/ -- the 24-check CPU verification ladder (operator, configs, sampler twin, aggregator).")


# ----------------------------------------------------------------------------- renderer
class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 8, f"AFDPS inverse-scattering setup guide   -   page {self.page_no()}", align="C")

pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(18, 16, 18)
pdf.add_page()
EPW = pdf.w - pdf.l_margin - pdf.r_margin

def clean(t):
    return (t.replace("\u2192", "->").replace("\u2248", "~").replace("\u00d7", "x")
             .replace("\u2014", "--").replace("\u2013", "-").replace("\u2019", "'")
             .replace("\u201c", '"').replace("\u201d", '"').encode("latin-1", "replace").decode("latin-1"))

for kind, text in C:
    text = clean(text)
    pdf.set_x(pdf.l_margin)   # reset horizontal position for every block (no drift)
    if kind == "h1":
        pdf.ln(2); pdf.set_font("Helvetica", "B", 14); pdf.set_text_color(20, 40, 90)
        pdf.set_x(pdf.l_margin); pdf.multi_cell(EPW, 7, text, align="L"); pdf.set_text_color(0); pdf.ln(1)
    elif kind == "h2":
        pdf.ln(1); pdf.set_font("Helvetica", "B", 11); pdf.set_text_color(40, 40, 40)
        pdf.set_x(pdf.l_margin); pdf.multi_cell(EPW, 6, text, align="L"); pdf.set_text_color(0)
    elif kind == "body":
        pdf.set_font("Helvetica", "", 10); pdf.set_x(pdf.l_margin)
        pdf.multi_cell(EPW, 5.2, text, align="L"); pdf.ln(0.5)
    elif kind == "bullet":
        pdf.set_font("Helvetica", "", 10); pdf.set_x(pdf.l_margin)
        pdf.cell(5, 5.2, "-", new_x="RIGHT", new_y="TOP")
        pdf.multi_cell(EPW - 5, 5.2, text, align="L")
    elif kind == "num":
        pdf.set_font("Helvetica", "", 10); pdf.set_x(pdf.l_margin)
        pdf.multi_cell(EPW, 5.2, text, align="L")
    elif kind == "code":
        pdf.ln(1); pdf.set_font("Courier", "", 8.5); pdf.set_fill_color(244, 244, 246)
        pdf.set_text_color(20, 20, 20); pdf.set_x(pdf.l_margin)
        pdf.multi_cell(EPW, 4.4, text, border=0, fill=True, align="L")
        pdf.set_text_color(0); pdf.ln(1.5)
    elif kind == "space":
        pdf.ln(2.5)
    elif kind == "rule":
        pdf.ln(2); pdf.set_draw_color(200); pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y()); pdf.ln(3)

pdf.output(OUT)
print("wrote", OUT)
