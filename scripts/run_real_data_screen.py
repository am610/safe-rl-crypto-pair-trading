"""Download real hourly cryptocurrency data and produce the first pair screen."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crypto_pair_rl.data import fetch_coinbase_candles
from crypto_pair_rl.pairs import screen_pairs


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "LTC-USD"]


def main() -> None:
    frames = {}
    metadata = {}
    for product in PRODUCTS:
        candles = fetch_coinbase_candles(product)
        frames[product] = candles["close"]
        metadata[product] = {
            "rows": len(candles),
            "start": candles.index.min().isoformat(),
            "end": candles.index.max().isoformat(),
            "provider": candles.attrs["provider"],
            "retrieved_at_utc": candles.attrs["retrieved_at_utc"],
        }
    closes = pd.concat(frames, axis=1).dropna()
    results = screen_pairs(closes)
    ROOT.joinpath("outputs").mkdir(exist_ok=True)
    ROOT.joinpath("docs", "assets").mkdir(parents=True, exist_ok=True)
    results.to_csv(ROOT / "outputs" / "initial_pair_screen.csv", index=False)
    with open(ROOT / "outputs" / "data_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    view = results.head(10).sort_values("adf_pvalue", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 5.8), facecolor="white")
    labels = view["first"].str.replace("-USD", "") + " with " + view["second"].str.replace("-USD", "")
    colors = ["#18a999" if value < 0.05 else "#9fb3c8" for value in view["adf_pvalue"]]
    axis.barh(labels, view["adf_pvalue"], color=colors)
    axis.axvline(0.05, color="#d1495b", linewidth=1.5, label="Five percent reference")
    axis.set_title("Initial hourly cryptocurrency pair screen")
    axis.set_xlabel("Augmented Dickey Fuller p value")
    axis.grid(axis="x", color="#edf1f5")
    axis.legend(frameon=False)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(ROOT / "docs" / "assets" / "initial_pair_screen.png", dpi=160, facecolor="white")
    plt.close(figure)
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
