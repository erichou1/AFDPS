"""Unit tests for the aggregator helpers (no GPU/data/piq needed)."""
import os
import sys

import torch

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import metrics_lib as M


def test_case_id_and_dedup(tmp_path):
    # Two shard dirs each holding some result_<id>.pt; collect() dedups by case id.
    for shard in ("shard0", "shard1"):
        d = tmp_path / shard
        d.mkdir()
        for cid in (0, 1, 2):
            torch.save({"x": cid}, d / f"result_{cid}.pt")
    by_id = M.collect([str(tmp_path / "shard*" / "result_*.pt")])
    assert set(by_id) == {"0", "1", "2"}            # 6 files -> 3 unique cases
    assert M.sort_case_ids(by_id) == ["0", "1", "2"]


def test_relative_meas_err_pct_real():
    obs = torch.tensor([3.0, 4.0])                  # ||obs|| = 5
    pred = torch.tensor([3.0, 4.0 - 0.5])           # diff norm = 0.5 -> 10%
    assert abs(M.relative_meas_err_pct(pred, obs) - 10.0) < 1e-5


def test_relative_meas_err_pct_complex():
    obs = torch.tensor([3 + 0j, 0 + 4j], dtype=torch.complex128)   # ||obs|| = 5
    pred = obs.clone(); pred[0] += 0.5 + 0j                         # diff norm 0.5 -> 10%
    assert abs(M.relative_meas_err_pct(pred, obs) - 10.0) < 1e-5


def test_mean_std():
    m, s = M.mean_std([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-9 and abs(s - (2.0 / 3.0) ** 0.5) < 1e-9


def test_apply_forward_loops_samples():
    class FakeOp:
        def forward(self, f, unnormalize=True):
            return f.flatten(1).sum(dim=1, keepdim=True)   # (1,1) per sample
    recon = torch.arange(2 * 4, dtype=torch.float64).reshape(2, 1, 2, 2)
    out = M.apply_forward(FakeOp(), recon)
    assert out.shape == (2, 1)
    assert torch.allclose(out.flatten(), torch.tensor([recon[0].sum(), recon[1].sum()]))


def test_baselines_table_complete():
    # The hardcoded Table-3 baselines must have all 12 methods for every receiver count.
    sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import aggregate_table3 as A
    assert set(A.BASELINES) == {360, 180, 60}
    for R, d in A.BASELINES.items():
        assert len(d) == 12, (R, len(d))
        for name, tup in d.items():
            assert len(tup) == 6, (R, name)
