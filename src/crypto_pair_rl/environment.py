"""Shielded execution environment for cryptocurrency pair allocations."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def deterministic_shield(
    requested: np.ndarray,
    zscores: np.ndarray,
    maximum_active_pairs: int = 2,
    divergence_limit: float = 4.0,
) -> np.ndarray:
    """Enforce divergence and portfolio concentration limits."""
    safe = np.asarray(requested, dtype=np.float32).copy()
    safe[np.abs(zscores) >= divergence_limit] = 0.0
    active = np.flatnonzero(np.abs(safe) > 0)
    if len(active) > maximum_active_pairs:
        keep = active[np.argsort(np.abs(zscores[active]))[-maximum_active_pairs:]]
        remove = np.setdiff1d(active, keep)
        safe[remove] = 0.0
    return safe


class PairExecutionEnv(gym.Env):
    """Choose how much of each deterministic pair signal to execute."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        observations: np.ndarray,
        next_returns: np.ndarray,
        heuristic_directions: np.ndarray,
        zscores: np.ndarray,
        cost_bps: float = 10.0,
        maximum_active_pairs: int = 2,
        activity_penalty_bps: float = 0.0,
    ) -> None:
        super().__init__()
        self.observations = np.asarray(observations, dtype=np.float32)
        self.next_returns = np.asarray(next_returns, dtype=np.float32)
        self.heuristic_directions = np.asarray(heuristic_directions, dtype=np.float32)
        self.zscores = np.asarray(zscores, dtype=np.float32)
        if not (
            len(self.observations)
            == len(self.next_returns)
            == len(self.heuristic_directions)
            == len(self.zscores)
        ):
            raise ValueError("Environment arrays must have equal length")
        self.pair_count = self.next_returns.shape[1]
        self.cost_rate = cost_bps / 10_000
        self.activity_penalty_bps = activity_penalty_bps
        self.maximum_active_pairs = maximum_active_pairs
        self.action_space = spaces.MultiDiscrete([3] * self.pair_count)
        feature_count = self.observations.shape[1] + self.pair_count
        self.observation_space = spaces.Box(low=-10, high=10, shape=(feature_count,), dtype=np.float32)
        self.location = 0
        self.current_positions = np.zeros(self.pair_count, dtype=np.float32)
        self.wealth = 1.0
        self.peak = 1.0

    def _observation(self) -> np.ndarray:
        values = np.concatenate([self.observations[self.location], self.current_positions])
        return np.clip(values, -10, 10).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.location = 0
        self.current_positions = np.zeros(self.pair_count, dtype=np.float32)
        self.wealth = 1.0
        self.peak = 1.0
        return self._observation(), {}

    def step(self, action: np.ndarray):
        scales = np.asarray(action, dtype=np.float32) / 2.0
        requested = scales * self.heuristic_directions[self.location]
        positions = deterministic_shield(
            requested,
            self.zscores[self.location],
            maximum_active_pairs=self.maximum_active_pairs,
        )
        turnover = np.abs(positions - self.current_positions).sum() / self.pair_count
        gross_return = float(np.mean(positions * self.next_returns[self.location]))
        net_return = gross_return - self.cost_rate * turnover
        self.wealth *= max(1.0 + net_return, 1e-6)
        self.peak = max(self.peak, self.wealth)
        drawdown = 1.0 - self.wealth / self.peak
        active_fraction = np.count_nonzero(positions) / self.pair_count
        reward = net_return * 10_000 - self.activity_penalty_bps * active_fraction - 0.05 * drawdown
        self.current_positions = positions
        self.location += 1
        terminated = self.location >= len(self.observations) - 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._observation()
        info = {
            "net_return": net_return,
            "turnover": float(turnover),
            "active_pairs": int(np.count_nonzero(positions)),
            "shield_changed_action": bool(np.any(positions != requested)),
        }
        return observation, float(reward), terminated, False, info


class PermissionGateEnv(PairExecutionEnv):
    """Authorize or reject deterministic signals under the same risk shield."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = spaces.MultiBinary(self.pair_count)

    def step(self, action: np.ndarray):
        requested = np.asarray(action, dtype=np.float32) * self.heuristic_directions[self.location]
        positions = deterministic_shield(
            requested,
            self.zscores[self.location],
            maximum_active_pairs=self.maximum_active_pairs,
        )
        turnover = np.abs(positions - self.current_positions).sum() / self.pair_count
        gross_return = float(np.mean(positions * self.next_returns[self.location]))
        net_return = gross_return - self.cost_rate * turnover
        self.wealth *= max(1.0 + net_return, 1e-6)
        self.peak = max(self.peak, self.wealth)
        drawdown = 1.0 - self.wealth / self.peak
        active_fraction = np.count_nonzero(positions) / self.pair_count
        reward = net_return * 10_000 - self.activity_penalty_bps * active_fraction - 0.05 * drawdown
        self.current_positions = positions
        self.location += 1
        terminated = self.location >= len(self.observations) - 1
        observation = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._observation()
        info = {
            "net_return": net_return,
            "turnover": float(turnover),
            "active_pairs": int(np.count_nonzero(positions)),
            "shield_changed_action": bool(np.any(positions != requested)),
        }
        return observation, float(reward), terminated, False, info
