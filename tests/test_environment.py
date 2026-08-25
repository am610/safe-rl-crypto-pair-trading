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


def test_activity_penalty_reduces_reward_without_changing_return():
    observations = np.zeros((4, 2))
    returns = np.ones((4, 1)) * 0.001
    directions = np.ones((4, 1))
    zscores = np.ones((4, 1)) * 2.0
    plain = PairExecutionEnv(observations, returns, directions, zscores)
    selective = PairExecutionEnv(observations, returns, directions, zscores, activity_penalty_bps=1.0)
    plain.reset()
    selective.reset()
    _, plain_reward, _, _, plain_info = plain.step(np.array([2]))
    _, selective_reward, _, _, selective_info = selective.step(np.array([2]))
    assert selective_reward < plain_reward
    assert selective_info["net_return"] == plain_info["net_return"]
