"""Leakage aware diagnostics for cryptocurrency pair candidates."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def pair_diagnostics(first: pd.Series, second: pd.Series) -> dict[str, float]:
    """Estimate a log price hedge relation and spread diagnostics."""
    aligned = pd.concat([first, second], axis=1).dropna()
    if len(aligned) < 80:
        raise ValueError("At least 80 aligned observations are required")
    log_first = np.log(aligned.iloc[:, 0])
    log_second = np.log(aligned.iloc[:, 1])
    design = np.column_stack([np.ones(len(log_second)), log_second.to_numpy()])
    intercept, hedge_ratio = np.linalg.lstsq(design, log_first.to_numpy(), rcond=None)[0]
    spread = log_first - intercept - hedge_ratio * log_second
    adf_statistic, adf_pvalue, *_ = adfuller(spread, autolag="AIC")
    spread_lag = spread.shift(1)
    delta = spread.diff()
    regression = pd.concat([delta, spread_lag], axis=1).dropna()
    slope = np.linalg.lstsq(
        np.column_stack([np.ones(len(regression)), regression.iloc[:, 1]]),
        regression.iloc[:, 0],
        rcond=None,
    )[0][1]
    half_life = np.inf if slope >= 0 else float(-np.log(2) / slope)
    return {
        "hedge_ratio": float(hedge_ratio),
        "adf_statistic": float(adf_statistic),
        "adf_pvalue": float(adf_pvalue),
        "half_life_hours": half_life,
        "spread_volatility": float(spread.std(ddof=1)),
    }


def screen_pairs(closes: pd.DataFrame) -> pd.DataFrame:
    """Rank every available pair using only the supplied research window."""
    rows = []
    for first, second in combinations(closes.columns, 2):
        diagnostics = pair_diagnostics(closes[first], closes[second])
        rows.append({"first": first, "second": second, **diagnostics})
    return pd.DataFrame(rows).sort_values(["adf_pvalue", "half_life_hours"]).reset_index(drop=True)
