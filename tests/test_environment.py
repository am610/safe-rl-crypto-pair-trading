import numpy as np

from crypto_pair_rl.environment import PairExecutionEnv, deterministic_shield


def test_shield_blocks_divergence_and_limits_concentration():
    result = deterministic_shield(
        np.array([1.0, 1.0, 1.0]),
        np.array([4.5, 2.0, 3.0]),
        maximum_active_pairs=1,
    )
    assert np.array_equal(result, np.array([0.0, 0.0, 1.0]))


def test_environment_reports_cost_and_shielding():
    observations = np.zeros((4, 6))
    returns = np.ones((4, 2)) * 0.001
    directions = np.ones((4, 2))
    zscores = np.array([[5.0, 2.0]] * 4)
    environment = PairExecutionEnv(observations, returns, directions, zscores, cost_bps=10)
    environment.reset()
    _, _, _, _, info = environment.step(np.array([2, 2]))
    assert info["shield_changed_action"]
    assert info["active_pairs"] == 1
    assert info["turnover"] == 0.5
