"""Create a compact research animation from the real BTC and DOGE history."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd

from crypto_pair_rl.backtest import causal_pair_positions, spread_from_hedge
from crypto_pair_rl.pairs import pair_diagnostics


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    btc = pd.read_csv(ROOT / "data" / "BTC-USD_hourly.csv", index_col=0, parse_dates=True)["close"]
    doge = pd.read_csv(ROOT / "data" / "DOGE-USD_hourly.csv", index_col=0, parse_dates=True)["close"]
    prices = pd.concat([btc.rename("BTC"), doge.rename("DOGE")], axis=1).dropna()
    training = prices.loc[:"2024-12-31 23:00:00+00:00"]
    hedge_ratio = pair_diagnostics(training["BTC"], training["DOGE"])["hedge_ratio"]
    spread = spread_from_hedge(prices["BTC"], prices["DOGE"], hedge_ratio)
    mean = spread.rolling(336).mean()
    scale = spread.rolling(336).std(ddof=1)
    zscore = ((spread - mean) / scale).dropna()
    positions = causal_pair_positions(spread, 336, 1.5, 0.5, 4.0).loc[zscore.index]
    view_index = zscore.index[-2200:]
    normalized = prices.loc[view_index] / prices.loc[view_index].iloc[0]
    locations = np.linspace(200, len(view_index) - 1, 55, dtype=int)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor="white", sharex=True)
    figure.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.9, hspace=0.18)

    def draw(frame_number: int) -> None:
        location = locations[frame_number]
        visible_index = view_index[: location + 1]
        for axis in axes:
            axis.clear()
            axis.grid(color="#edf1f5", linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].plot(visible_index, normalized.loc[visible_index, "BTC"], color="#f7931a", linewidth=2, label="BTC")
        axes[0].plot(visible_index, normalized.loc[visible_index, "DOGE"], color="#6657d9", linewidth=2, label="DOGE")
        axes[0].set_ylabel("Normalized price")
        axes[0].legend(frameon=False, ncol=2, loc="upper left")
        axes[1].plot(visible_index, zscore.loc[visible_index], color="#18a999", linewidth=1.5, label="Spread z score")
        axes[1].axhline(1.5, color="#d1495b", linewidth=1, linestyle=":")
        axes[1].axhline(-1.5, color="#d1495b", linewidth=1, linestyle=":")
        active = positions.loc[visible_index] != 0
        axes[1].fill_between(visible_index, -4, 4, where=active, color="#f4d35e", alpha=0.15, label="Heuristic active")
        axes[1].set_ylim(-4.2, 4.2)
        axes[1].set_ylabel("Spread z score")
        axes[1].set_xlabel("UTC date")
        axes[1].legend(frameon=False, ncol=2, loc="upper left")
        timestamp = visible_index[-1].strftime("%Y %m %d")
        figure.suptitle(f"BTC and DOGE pair evolution through {timestamp}", fontsize=15)

    animation = FuncAnimation(figure, draw, frames=len(locations), interval=110)
    destination = ROOT / "docs" / "assets" / "btc_doge_pair_evolution.gif"
    animation.save(destination, writer=PillowWriter(fps=9), dpi=90)
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
