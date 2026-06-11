'''
Inference + evaluation entry point for the AFDPS linear inverse-scattering port.

This is the upstream InverseBench harness (navier_stokes/main.py) reused verbatim
in spirit, with three deliberate differences:
  1. a sys.path bootstrap so the benchmark modules (utils, training, models, eval)
     resolve from the sibling navier_stokes/ tree, while `inverse_problems` and
     `algo` namespace-merge across both trees (the AFDPS operator/algorithm live
     here, the benchmark forward/dataset/evaluator live there);
  2. the CWD is pinned to this directory before Hydra composition so the config
     searchpath (file://../navier_stokes/configs), ../data, checkpoints/, cache/
     and exps/ all resolve consistently;
  3. torch.load(..., weights_only=False) for the pickled-`ema` checkpoint path
     (torch >= 2.6 changed the default), matching how the prior was saved.

Inference steps: (1) instantiate the forward model, (2) the test dataloader,
(3) load the pretrained diffusion model, (4) run the inference algorithm.
Evaluation steps: (1) instantiate the metric(s), (2) evaluate the results.
'''
import os
import sys

# --- sys.path bootstrap: merge this tree with the sibling navier_stokes/ tree ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_NS = os.path.join(os.path.dirname(_HERE), 'navier_stokes')
sys.path.insert(1, _NS)     # utils / training / models / eval resolve from here
sys.path.insert(1, _HERE)   # inverse_problems / algo namespace-merge (this tree first)

import pickle
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader

from utils.helper import open_url, create_logger


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)

    if config.wandb:
        import wandb
        problem_name = config.get('problem')['name']
        wandb.init(project=problem_name, group=config.algorithm.name,
                   config=OmegaConf.to_container(config),
                   reinit=True, settings=wandb.Settings(start_method="fork"))
        config = OmegaConf.create(dict(wandb.config))

    exp_dir = os.path.join(config.problem.exp_dir, config.algorithm.name, config.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(exp_dir)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    forward_op = instantiate(config.problem.model, device=device)
    testset = instantiate(config.problem.data)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    logger.info(f"Loaded {len(testset)} test samples...")
    # load pre-trained diffusion prior
    ckpt_path = config.problem.prior
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except Exception:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(config.problem.prior, map_location=device, weights_only=False)
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        else:
            net.load_state_dict(ckpt['net'])
        net = net.to(device)

    del ckpt
    net.eval()
    if config.compile:
        net = torch.compile(net)
    logger.info(f"Loaded pre-trained model from {config.problem.prior}...")

    algo = instantiate(config.algorithm.method, forward_op=forward_op, net=net)
    evaluator = instantiate(config.problem.evaluator, forward_op=forward_op)

    for i, data in enumerate(testloader):
        if isinstance(data, torch.Tensor):
            data = data.to(device)
        elif isinstance(data, dict):
            assert 'target' in data.keys(), "'target' must be in the data dict"
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    data[key] = val.to(device)
        data_id = testset.id_list[i]
        # Seed per GLOBAL case id (not the loop index) so each case is independent of
        # run order -> results are reproducible whether cases run sequentially on one
        # GPU or split across shards (each process reseeds per case).
        try:
            torch.manual_seed(config.seed + int(data_id))
        except (ValueError, TypeError):
            torch.manual_seed(config.seed + i)
        save_path = os.path.join(exp_dir, f'result_{data_id}.pt')
        if config.inference:
            observation = forward_op(data)
            target = data['target']
            logger.info(f'Running inference on test sample {data_id}...')
            recon = algo.inference(observation, num_samples=config.num_samples)
            if torch.cuda.is_available():
                logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB')

            result_dict = {
                'observation': observation.cpu(),
                'recon': forward_op.unnormalize(recon).cpu(),
                'target': forward_op.unnormalize(target).cpu(),
            }
            torch.save(result_dict, save_path)
            logger.info(f"Saved results to {save_path}.")
        else:
            result_dict = torch.load(save_path)
            logger.info(f"Loaded results from {save_path}.")

        metric_dict = evaluator(pred=result_dict['recon'], target=result_dict['target'],
                                observation=result_dict['observation'])
        logger.info(f"Metric results: {metric_dict}...")

    logger.info("Evaluation completed...")
    metric_state = evaluator.compute()
    logger.info(f"Final metric results: {metric_state}...")
    if config.wandb:
        import wandb
        wandb.log(metric_state)
        wandb.finish()


if __name__ == "__main__":
    # Pin CWD so the Hydra searchpath (file://../navier_stokes/configs), ../data,
    # checkpoints/, cache/ and exps/ all resolve from this directory.
    os.chdir(_HERE)
    main()
