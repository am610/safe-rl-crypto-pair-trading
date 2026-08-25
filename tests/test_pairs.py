import numpy as np
import pandas as pd

from crypto_pair_rl.pairs import pair_diagnostics, screen_pairs


def test_cointegrated_pair_has_finite_diagnostics():
    rng = np.random.default_rng(7)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    first = pd.Series(base * np.exp(rng.normal(0, 0.005, 400)))
    second = pd.Series(base)
    result = pair_diagnostics(first, second)
    assert np.isfinite(result["hedge_ratio"])
    assert 0 <= result["adf_pvalue"] <= 1


def test_screen_returns_every_pair():
    rng = np.random.default_rng(11)
    closes = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.01, (240, 4)), axis=0)),
        columns=list("ABCD"),
    )
    result = screen_pairs(closes)
    assert len(result) == 6
    assert result["adf_pvalue"].is_monotonic_increasing
