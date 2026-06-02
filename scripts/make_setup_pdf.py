"""Generate SETUP_GUIDE.pdf -- a step-by-step setup & run guide for afdps-ns.

Reproducible: `python scripts/make_setup_pdf.py` writes ../SETUP_GUIDE.pdf.
Text is ASCII-only so it renders with PDF core fonts (no font embedding needed).
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

h1("AFDPS for Navier-Stokes -- Setup & Run Guide")
body("This guide takes you from a fresh GPU box to a first benchmark number and a full "
     "hyperparameter sweep for the afdps-ns project (AFDPS applied to the InverseBench 2D "
     "Navier-Stokes inverse problem). The numerical core is already built and verified on CPU "
     "(15/15 checks); the remaining work is GPU setup, asset download, and running Track B.")
space()
body("GitHub: https://github.com/erichou1/AFDPS   (public -- clone it directly on the GPU box; "
     "no copying from a laptop). NOTE: on GitHub the repo is named AFDPS and the files are at the "
     "ROOT -- there is no 'afdps-ns' subfolder (that is only the local clone-directory name).")
body("Goal: beat the InverseBench baselines on relative-L2 -- EnKG ~ 0.12, DPG ~ 0.32 "
     "(x2 subsampling, sigma=0).")

rule()
h1("Where things stand (what is already done)")
bullet("Repo built: InverseBench fork + vendored AFDPS sampler + the new adjoint gradient engine.")
bullet("Numerical core VERIFIED on CPU (float64): discrete-adjoint gradient matches finite "
       "differences to ~1e-10; Hutchinson Laplacian matches the brute-force trace; "
       "adjoint/FFT primitives exact. Run: python verification/checks.py  -> 15/15.")
bullet("Track A (synthetic, analytic Gaussian prior) recovers the field end-to-end.")
bullet("Track B (real benchmark) code path is smoke-tested; it needs the checkpoint + data + GPU.")
space()
body("So the next steps are purely: get it on the GPU box, download assets, verify, then run.")

rule()
h1("Prerequisites")
bullet("A Linux GPU box (the GB200s). NVIDIA driver + CUDA runtime that matches your torch build.")
bullet("Python 3.9+ (3.10/3.11 recommended).")
bullet("~5-10 GB disk for the checkpoint + datasets. Internet access to GitHub + Caltech data.")
bullet("(Optional) a Weights & Biases account for the sweep (wandb login).")

rule()
h1("Step 1 -- Get the repo onto the GPU box (git clone)")
body("The project is on GitHub (public), so you do NOT copy anything from your laptop. On the "
     "GPU box, just clone it:")
code("git clone https://github.com/erichou1/AFDPS.git\n"
     "cd AFDPS")
body("This gives you a folder 'AFDPS' containing the whole project (main.py, algo/, "
     "inverse_problems/, configs/, verification/, SETUP_GUIDE.pdf, ...). If the repo is later made "
     "private, authenticate first (gh auth login, or an SSH key / personal access token on the box).")
body("Large assets are intentionally NOT in git (.gitignore excludes checkpoints/ and data/) -- "
     "you download those in Step 3. To pull later updates: git pull.")

rule()
h1("Step 2 -- Python environment")
body("Two options. Use whichever your cluster prefers. Option A (uv) matches InverseBench.")
h2("Option A: uv (recommended)")
code("curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed\n"
     "cd AFDPS                                          # the cloned repo\n"
     "uv sync          # installs the NS-focused deps from pyproject.toml\n"
     "source .venv/bin/activate")
h2("Option B: pip / conda")
code("python -m venv .venv && source .venv/bin/activate\n"
     "pip install --upgrade pip\n"
     "# install the CUDA build of torch that matches your driver, then the rest:\n"
     "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
     "pip install hydra-core omegaconf lmdb numpy tqdm piq pyyaml\n"
     "# wandb is OPTIONAL (only for the Step 8 sweep): pip install wandb")
body("Sanity check the GPU is visible:")
code('python -c "import torch; print(torch.__version__, torch.cuda.is_available(), '
     'torch.cuda.get_device_name(0))"')
body("NOTE: the trimmed pyproject.toml omits the FWI/MRI/black-hole solvers (devito, ehtim, "
     "fastmri, sigpy). That is intentional -- they are not needed for Navier-Stokes. If you ever "
     "run those other InverseBench problems, restore them from env.yaml / README_InverseBench.md.")

rule()
h1("Step 3 -- Download the assets (checkpoint + data)")
body("One command downloads everything -- the ~102 MB diffusion prior AND the ~7 MB NS datasets -- "
     "and unzips the data to the exact paths the configs expect (needs curl + unzip):")
code("bash scripts/download_assets.sh          # or: ... data   |   ... ckpt")
body("Result: checkpoints/ns-5m.pt  and  ../data/navier-stokes-{test,val}/Re200.0-t5.0 "
     "(an LMDB dir with data.mdb + lock.mdb; 'data' is a SIBLING of the repo folder).")
h2("About the data access (CaltechDATA)")
body("The NS data lives in a CaltechDATA record whose METADATA is public but whose FILES are "
     "RESTRICTED, so the bare record URL will not let you download. Access needs a share token -- "
     "the script uses the public one already in README_InverseBench.md. If the data step ever "
     "returns 401/403, the token expired: open the tokenized link in README_InverseBench.md (or get "
     "a fresh share link from the record page) and replace CALTECH_TOKEN in scripts/download_assets.sh.")
body("The record has 4 versions. The NS test/val files are byte-identical across all of them, so the "
     "version does not matter for the benchmark; the downloader uses the LATEST (zg89b-mpv16), which "
     "also contains navier-stokes-train.zip (~1.24 GB) -- only needed if you retrain the prior "
     "(bash scripts/download_assets.sh train).")
h2("Manual equivalent (if you prefer)")
code("# checkpoint:\n"
     "curl -L -o checkpoints/ns-5m.pt \\\n"
     "  https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/ns-5m.pt\n"
     "# data (TOK = the token from README_InverseBench.md):\n"
     "B=https://data.caltech.edu/api/records/jfdr4-6ws87/files\n"
     "curl -L -o ns-test.zip \"$B/navier-stokes-test.zip/content?token=$TOK\"\n"
     "curl -L -o ns-val.zip  \"$B/navier-stokes-val.zip/content?token=$TOK\"\n"
     "mkdir -p ../data && unzip -o ns-test.zip -d ../data && unzip -o ns-val.zip -d ../data")

rule()
h1("Step 4 -- Verify the install")
body("Run all three checks. The first two need no GPU and no checkpoint; the third needs neither "
     "the real checkpoint nor a GPU (it uses a mock prior to test the code path).")
code("python verification/checks.py          # expect: 15/15 checks passed.\n"
     "python verification/run_track_a.py --res 32 --steps 150 --particles 8\n"
     "python verification/smoke_track_b.py   # expect: PASS (Track B code path executes)")
bullet("checks.py: the numerical verification ladder (gradient vs finite-diff, Hutchinson vs "
       "brute-force trace, adjoint/FFT primitives, GRF prior).")
bullet("run_track_a.py: end-to-end AFDPS on a synthetic Gaussian-prior problem; prints rel-L2 "
       "(should be well below the prior-mean baseline of 1.0).")
bullet("smoke_track_b.py: composes the real Hydra configs and runs one inference with a mock net.")

rule()
h1("Step 5 -- Validate the forward solver at full 128 resolution (open item)")
body("The dataset was generated with ADAPTIVE time-stepping, but the adjoint needs a FIXED dt "
     "(adaptive=False, delta_t=0.002). Confirm delta_t=0.002 is sub-CFL at 128x128 BEFORE the "
     "full run -- if the forward diverges (NaN), lower delta_t (e.g. 0.001).")
code('python -c "import torch,sys; sys.path.insert(0,\'.\');\\\n'
     'from inverse_problems.navier_stokes_afdps import AFDPSNavierStokes2d;\\\n'
     'op=AFDPSNavierStokes2d(resolution=128, forward_time=1.0, Re=200.0,\\\n'
     '  downsample_factor=2, delta_t=0.002, adaptive=False, unnorm_scale=10.0,\\\n'
     '  device=\'cuda\');\\\n'
     'x=0.5*torch.randn(2,1,128,128,device=\'cuda\');\\\n'
     'y=op.forward(x); print(\'finite:\', torch.isfinite(y).all().item(), \'shape:\', tuple(y.shape))"')
body("Expect finite: True. If False, set problem.model.delta_t=0.001 everywhere below.")

rule()
h1("Step 6 -- First Track B run (one setting)")
body("Run AFDPS on the real benchmark at x2 subsampling, sigma=0. This prints relative-L2 per "
     "test case and an aggregate at the end.")
code("python main.py problem=navier-stokes-afdps algorithm=afdps pretrain=navier-stokes \\\n"
     "    num_samples=1 wandb=false")
body("Results are saved under exps/inference/navier-stokes-afdps-ds2/AFDPS/. If you see NaN or "
     "rel-L2 >> 1, it is almost always the guidance strength -- go to Step 7.")

rule()
h1("Step 7 -- Tune the guidance (the single most important knob)")
body("guidance_gamma controls the annealed data-guidance strength and MUST be scaled to the data. "
     "Too small -> unstable / NaN; too large -> ignores the data (rel-L2 ~ 1). Sweep it by hand "
     "first to find the stable, useful range, e.g.:")
code("for g in 1 3 10 30; do \\\n"
     "  python main.py problem=navier-stokes-afdps algorithm=afdps pretrain=navier-stokes \\\n"
     "    algorithm.method.guidance_gamma=$g num_samples=1 wandb=false \\\n"
     "    exp_name=gamma_$g ; done")
body("Other useful knobs (all overridable on the command line):")
bullet("algorithm.method.num_steps (e.g. 100/200/400) -- more steps = more stable, slower.")
bullet("algorithm.method.num_particles (8/16/32/64) -- ensemble size; batched on GPU.")
bullet("algorithm.method.sigma_max (40/80) -- top of the noise schedule.")
bullet("problem.model.hutchinson_M (1/2/4) -- # Laplacian probes; raise for less noisy weights.")
bullet("problem.model.hutchinson_scheme=forward -- halves the Laplacian solves (faster).")
bullet("problem.model.grad_chunk -- raise to fill the GPU (memory permitting).")

rule()
h1("Step 8 -- Hyperparameter sweep (OPTIONAL -- needs wandb)")
body("This is the ONLY step that needs Weights & Biases (a free account + 'pip install wandb'). "
     "It is optional: everything else runs with wandb=false (the default) and no account. If you "
     "skip wandb, just tune by hand as in Step 7 (a shell loop over configs) -- results print to "
     "stdout and save under exps/.")
body("With wandb: run the bayes sweep over the validation set.")
code("wandb login                                   # once\n"
     "wandb sweep configs/sweep/navier-stokes/afdps.yaml\n"
     "# copy the printed sweep ID, then launch one or more agents (one per GPU):\n"
     "wandb agent <ENTITY/PROJECT/SWEEP_ID>")
body("The sweep optimizes 'relative l2' (the metric main.py actually logs on the AFDPS path) over "
     "num_particles, num_steps, guidance_gamma, sigma_max, hutchinson_M, hutchinson_scheme.")

rule()
h1("Step 9 -- Full benchmark grid + compare to baselines")
body("Run the best config across the 3x3 grid: subsampling in {2,4,8} x noise in {0,1,2}. Each "
     "is a separate run via problem.model overrides. Example (x4, sigma=1):")
code("python main.py problem=navier-stokes-afdps algorithm=afdps pretrain=navier-stokes \\\n"
     "    problem.model.downsample_factor=4 problem.model.sigma_noise=1.0 \\\n"
     "    algorithm.method.guidance_gamma=<tuned> num_samples=1 wandb=false \\\n"
     "    exp_name=ds4_sig1")
body("Collect the aggregate relative-L2 for each cell and tabulate against the published baselines "
     "(EnKG, DPG, EKI, DPS-GSG). You can also run those baselines in this same repo for an "
     "apples-to-apples comparison, e.g.:")
code("python main.py problem=navier-stokes algorithm=enkg pretrain=navier-stokes wandb=false\n"
     "python main.py problem=navier-stokes algorithm=dpg  pretrain=navier-stokes wandb=false")
body("(Baselines use problem=navier-stokes, the black-box forward op; AFDPS uses "
     "problem=navier-stokes-afdps, the adjoint-enabled op.)")

rule()
h1("Optional / later")
bullet("Continuous-adjoint cross-check: set problem.model.adjoint_mode=continuous to use the "
       "hand-coded backward PDE solver instead of autograd (agrees to O(dt); a good sanity check "
       "and a pedagogical artifact for the paper).")
bullet("Ablations: diffusion prior vs analytic GRF prior; hutchinson_M and particle-count scaling; "
       "central vs forward Hutchinson; guidance schedules.")
bullet("Extend to the inverse-scattering problem (InverseBench Appendix B.4) reusing the adapter "
       "pattern, if you want a second benchmark for the paper.")

rule()
h1("Troubleshooting")
bullet("NaN / rel-L2 >> 1: lower guidance_gamma (most common), or raise num_steps, or lower "
       "delta_t. The sampler now quarantines diverged particles, but a bad config still scores poorly.")
bullet("Forward diverges at 128^2: delta_t too large for CFL -> set problem.model.delta_t=0.001.")
bullet("CUDA out of memory: lower problem.model.grad_chunk and/or algorithm.method.num_particles; "
       "consider gradient checkpointing of the forward trajectory (noted in the plan).")
bullet("Sweep shows no objective: ensure the metric is 'relative l2' (already fixed in the sweep "
       "yaml) -- the AFDPS path does NOT log 'data_fitting_loss' (only the baselines do).")
bullet("Slow: use hutchinson_scheme=forward (halves Laplacian solves), raise grad_chunk to fill "
       "the GPU, and run the 9 grid cells in parallel across GPUs.")

rule()
h1("Key files & reference")
bullet("README.md -- overview + the same run commands.")
bullet("configs/{problem/navier-stokes-afdps.yaml, algorithm/afdps.yaml, sweep/navier-stokes/afdps.yaml}")
bullet("inverse_problems/{ns_adjoint.py, navier_stokes_afdps.py} -- gradient engine + operator.")
bullet("algo/{afdps.py, afdps_core/ensemble_denoiser_edm.py} -- the algorithm + sampler.")
bullet("verification/ -- checks.py, run_track_a.py, smoke_track_b.py.")
bullet("Plan: /Users/eric/.claude/plans/plan-out-the-entire-virtual-stearns.md   "
       "Math recipe: AFDPS_PDE_Inverse_Problem (3).pdf.")


# ----------------------------------------------------------------------------- renderer
class PDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150)
        self.cell(0, 8, f"afdps-ns setup guide   -   page {self.page_no()}", align="C")

pdf = PDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=16)
pdf.set_margins(18, 16, 18)
pdf.add_page()
EPW = pdf.w - pdf.l_margin - pdf.r_margin

def clean(t):
    return (t.replace("→", "->").replace("≈", "~").replace("×", "x")
             .replace("—", "--").replace("–", "-").replace("’", "'")
             .replace("“", '"').replace("”", '"').encode("latin-1", "replace").decode("latin-1"))

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
