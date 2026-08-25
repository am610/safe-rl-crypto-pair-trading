"""Build the locked real data heuristic baseline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crypto_pair_rl.backtest import annualized_metrics, causal_pair_positions, pair_returns, spread_from_hedge
from crypto_pair_rl.data import fetch_coinbase_history
from crypto_pair_rl.pairs import screen_pairs


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "LTC-USD"]
START = "2024-01-01"
END = "2026-08-24"
TRAIN_END = pd.Timestamp("2024-12-31 23:00", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-12-31 23:00", tz="UTC")


def load_closes() -> pd.DataFrame:
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    series = {}
    for product in PRODUCTS:
        path = data_dir / f"{product}_hourly.csv"
        if path.exists():
            candles = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            candles = fetch_coinbase_history(product, START, END)
            candles.to_csv(path)
        series[product] = candles["close"]
    return pd.concat(series, axis=1).dropna()


def main() -> None:
    closes = load_closes()
    train = closes.loc[:TRAIN_END]
    validation = closes.loc[TRAIN_END + pd.Timedelta(hours=1) : VALIDATION_END]
    test = closes.loc[VALIDATION_END + pd.Timedelta(hours=1) :]
    screen = screen_pairs(train)
    selected = screen.loc[screen["adf_pvalue"] < 0.05].head(3)
    if selected.empty:
        selected = screen.head(1)

    parameter_grid = [(168, 1.5, 0.5), (168, 2.0, 0.5), (336, 1.5, 0.5), (336, 2.0, 0.5)]
    validation_rows = []
    for lookback, entry_z, exit_z in parameter_grid:
        returns = []
        for row in selected.itertuples():
            full_spread = spread_from_hedge(closes[row.first], closes[row.second], row.hedge_ratio)
            positions = causal_pair_positions(full_spread, lookback, entry_z, exit_z, stop_z=4.0)
            pair_result = pair_returns(closes[row.first], closes[row.second], row.hedge_ratio, positions, cost_bps=10)
            returns.append(pair_result["net_return"].rename(f"{row.first}_{row.second}"))
        portfolio = pd.concat(returns, axis=1).mean(axis=1)
        metrics = annualized_metrics(portfolio.loc[validation.index.intersection(portfolio.index)])
        validation_rows.append({"lookback": lookback, "entry_z": entry_z, "exit_z": exit_z, **metrics})
    validation_table = pd.DataFrame(validation_rows).sort_values("sharpe", ascending=False)
    chosen = validation_table.iloc[0]

    pair_results = []
    for row in selected.itertuples():
        spread = spread_from_hedge(closes[row.first], closes[row.second], row.hedge_ratio)
        positions = causal_pair_positions(spread, int(chosen.lookback), chosen.entry_z, chosen.exit_z, stop_z=4.0)
        result = pair_returns(closes[row.first], closes[row.second], row.hedge_ratio, positions, cost_bps=10)
        pair_results.append(result["net_return"].rename(f"{row.first}_{row.second}"))
    portfolio = pd.concat(pair_results, axis=1).mean(axis=1)
    test_returns = portfolio.loc[test.index.intersection(portfolio.index)]
    metrics = annualized_metrics(test_returns)
    metrics.update(
        rows=int(len(closes)),
        selected_pairs=[f"{row.first} with {row.second}" for row in selected.itertuples()],
        lookback=int(chosen.lookback),
        entry_z=float(chosen.entry_z),
        exit_z=float(chosen.exit_z),
        cost_bps=10.0,
        train_end=TRAIN_END.isoformat(),
        validation_end=VALIDATION_END.isoformat(),
    )

    outputs = ROOT / "outputs"
    assets = ROOT / "docs" / "assets"
    outputs.mkdir(exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    screen.to_csv(outputs / "training_pair_screen.csv", index=False)
    validation_table.to_csv(outputs / "heuristic_validation_grid.csv", index=False)
    with open(outputs / "heuristic_results.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    wealth = (1 + test_returns).cumprod()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), facecolor="white")
    axes[0].plot(wealth.index, wealth, color="#18a999", linewidth=2)
    axes[0].set_title("Locked heuristic test wealth")
    axes[0].set_ylabel("Growth of one dollar")
    axes[1].barh(
        ["Maximum drawdown", "Annual volatility", "Annual return"],
        [metrics["maximum_drawdown"], metrics["annual_volatility"], metrics["annual_return"]],
        color=["#d1495b", "#9fb3c8", "#18a999"],
    )
    axes[1].set_title(f"Test Sharpe {metrics['sharpe']:.2f}")
    axes[1].set_xlabel("Annualized decimal value")
    for axis in axes:
        axis.grid(color="#edf1f5")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Real data cryptocurrency heuristic baseline", fontsize=16)
    figure.tight_layout()
    figure.savefig(assets / "heuristic_baseline.png", dpi=160, facecolor="white")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
