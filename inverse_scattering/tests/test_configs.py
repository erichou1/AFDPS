"""Hydra composition test: verifies the searchpath reuse of ../navier_stokes/configs
and that the AFDPS _target_s resolve. Catches searchpath / CWD breakage early."""
import importlib
import os
import pytest


def test_compose_targets_and_searchpath():
    pytest.importorskip("hydra")
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    _IS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(_IS, "configs")
    cwd = os.getcwd()
    os.chdir(_IS)  # the relative searchpath file://../navier_stokes/configs resolves from here
    try:
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            cfg = compose(config_name="config",
                          overrides=["problem.model.numRec=4", "problem.model.numTrans=2"])
        # AFDPS targets
        assert cfg.problem.model._target_ == "inverse_problems.inverse_scatter_afdps.AFDPSInverseScatter"
        assert cfg.algorithm.method._target_ == "algo.afdps_scatter.AFDPSScatter"
        # searchpath worked iff the reused upstream pretrain/evaluator are present
        assert cfg.pretrain.model._target_ == "models.precond.EDMPrecond"
        assert cfg.problem.evaluator._target_ == "eval.InverseScatter"
        # physics single-sourced from the inherited inv-scatter problem
        assert float(cfg.problem.model.unnorm_scale) == 0.5
        assert float(cfg.problem.model.sigma_noise) == pytest.approx(1e-4)
        assert int(cfg.problem.model.numRec) == 4
        # primary sampler wiring
        assert cfg.algorithm.method.sampler_kwargs.guidance_step == "exact_linear"
        assert cfg.algorithm.method.sampler_kwargs.guidance_mode == "full"
        assert cfg.algorithm.method.reduce == "mean"
    finally:
        os.chdir(cwd)
        GlobalHydra.instance().clear()


def test_algo_target_importable():
    # The algorithm target imports no heavy deps (no scipy) -> always importable.
    mod, cls = "algo.afdps_scatter", "AFDPSScatter"
    assert hasattr(importlib.import_module(mod), cls)


def test_operator_target_importable():
    pytest.importorskip("scipy")
    mod, cls = "inverse_problems.inverse_scatter_afdps", "AFDPSInverseScatter"
    assert hasattr(importlib.import_module(mod), cls)
