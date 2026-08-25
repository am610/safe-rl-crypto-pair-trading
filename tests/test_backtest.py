import numpy as np
import pandas as pd

from crypto_pair_rl.backtest import causal_pair_positions, pair_returns


def test_positions_are_delayed_and_bounded():
    spread = pd.Series([0.0] * 30 + [4.0, 0.0, 0.0])
    positions = causal_pair_positions(spread, lookback=20, entry_z=2.0, exit_z=0.5, stop_z=8.0)
    assert positions.iloc[30] == 0
    assert set(positions.unique()).issubset({-1.0, 0.0, 1.0})


def test_costs_never_improve_return():
    first = pd.Series([100.0, 101.0, 102.0, 101.0])
    second = pd.Series([100.0, 100.5, 100.0, 100.5])
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])
    result = pair_returns(first, second, 1.0, positions, cost_bps=10)
    assert (result["net_return"] <= result["gross_return"] + 1e-15).all()
