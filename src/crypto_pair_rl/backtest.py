"""Causal heuristic baseline for pair execution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spread_from_hedge(first: pd.Series, second: pd.Series, hedge_ratio: float) -> pd.Series:
    """Construct a log price spread with a fixed hedge ratio."""
    aligned = pd.concat([first, second], axis=1).dropna()
    return np.log(aligned.iloc[:, 0]) - hedge_ratio * np.log(aligned.iloc[:, 1])


def causal_pair_positions(
    spread: pd.Series,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> pd.Series:
    """Apply a causal stateful threshold rule and delay execution by one hour."""
    mean = spread.rolling(lookback, min_periods=lookback).mean()
    scale = spread.rolling(lookback, min_periods=lookback).std(ddof=1)
    zscore = (spread - mean) / scale.replace(0, np.nan)
    raw = pd.Series(0.0, index=spread.index)
    state = 0.0
    for timestamp, value in zscore.items():
        if np.isnan(value):
            state = 0.0
        elif state == 0 and abs(value) >= entry_z and abs(value) < stop_z:
            state = -float(np.sign(value))
        elif state != 0 and (abs(value) <= exit_z or abs(value) >= stop_z):
            state = 0.0
        raw.loc[timestamp] = state
    return raw.shift(1).fillna(0.0)


def pair_returns(
    first: pd.Series,
    second: pd.Series,
    hedge_ratio: float,
    positions: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    """Calculate gross and net returns using normalized dollar neutral legs."""
    aligned = pd.concat([first, second, positions.rename("position")], axis=1).dropna()
    first_return = aligned.iloc[:, 0].pct_change().fillna(0.0)
    second_return = aligned.iloc[:, 1].pct_change().fillna(0.0)
    hedge_weight = abs(hedge_ratio)
    gross = aligned["position"] * (first_return - hedge_ratio * second_return) / (1.0 + hedge_weight)
    turnover = aligned["position"].diff().abs().fillna(aligned["position"].abs())
    net = gross - turnover * cost_bps / 10_000
    return pd.DataFrame({"gross_return": gross, "turnover": turnover, "net_return": net})


def annualized_metrics(returns: pd.Series, periods_per_year: int = 24 * 365) -> dict[str, float]:
    """Summarize an hourly return series."""
    clean = returns.dropna()
    wealth = (1 + clean).cumprod()
    drawdown = 1 - wealth / wealth.cummax()
    volatility = clean.std(ddof=1) * np.sqrt(periods_per_year)
    annual_return = clean.mean() * periods_per_year
    sharpe = annual_return / volatility if volatility > 0 else np.nan
    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "maximum_drawdown": float(drawdown.max()),
        "ending_wealth": float(wealth.iloc[-1]),
    }
