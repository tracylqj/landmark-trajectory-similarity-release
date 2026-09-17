import numpy as np
import torch

from ltsl.metrics import full_pool_metrics
from ltsl.model import TrajectoryEncoder
from ltsl.targets import netdtw, netdtw_energy, tp_distance


def test_parameter_count():
    model = TrajectoryEncoder(np.zeros((20, 32), np.float32))
    assert sum(parameter.numel() for parameter in model.parameters()) == 202368


def test_self_retrieval_is_excluded():
    values = np.arange(60, dtype=np.float64)
    target = np.abs(values[:, None] - values[None, :])
    result = full_pool_metrics(target, target)
    assert result["HR@1"] == 1.0


def test_target_functions():
    shortest = np.array([[0.0, 1.0], [1.0, 0.0]])
    left = np.array([0], dtype=np.int64)
    right = np.array([1], dtype=np.int64)
    assert netdtw(left, right, shortest) == 1.0
    assert netdtw_energy(left, right, shortest)[0, 0] == 0.0
    np.testing.assert_allclose(tp_distance(left, right, shortest, 1.0), 1.0 - np.exp(-1.0))


def test_batch_independence():
    torch.manual_seed(43)
    model = TrajectoryEncoder(np.random.default_rng(1).normal(size=(10, 32)).astype(np.float32))
    model.eval()
    sequences = torch.tensor([[1, 2, 3], [4, 5, 0]])
    lengths = torch.tensor([3, 2])
    with torch.no_grad():
        single = model(sequences[:1], lengths[:1])[0]
        batch = model(sequences, lengths)[0][:1]
    torch.testing.assert_close(single, batch)
