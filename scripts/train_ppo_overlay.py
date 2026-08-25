"""Train and evaluate a shielded PPO execution overlay."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from crypto_pair_rl.backtest import annualized_metrics, causal_pair_positions, spread_from_hedge
from crypto_pair_rl.environment import PairExecutionEnv
from crypto_pair_rl.pairs import screen_pairs
from run_heuristic_baseline import TRAIN_END, VALIDATION_END, load_closes


ROOT = Path(__file__).resolve().parents[1]


def build_arrays(closes: pd.DataFrame, selected: pd.DataFrame, lookback: int = 336):
    feature_parts = []
    return_parts = []
    direction_parts = []
    zscore_parts = []
    for row in selected.itertuples():
        spread = spread_from_hedge(closes[row.first], closes[row.second], row.hedge_ratio)
        rolling_mean = spread.rolling(lookback).mean()
        rolling_scale = spread.rolling(lookback).std(ddof=1).replace(0, np.nan)
        zscore = (spread - rolling_mean) / rolling_scale
        direction = causal_pair_positions(spread, lookback, 1.5, 0.5, 4.0)
        first_return = closes[row.first].pct_change().shift(-1)
        second_return = closes[row.second].pct_change().shift(-1)
        pair_return = (first_return - row.hedge_ratio * second_return) / (1 + abs(row.hedge_ratio))
        feature_parts.extend(
            [
                zscore.rename(f"z_{row.Index}"),
                zscore.diff().rename(f"dz_{row.Index}"),
                spread.diff().rolling(24).std().rename(f"vol_{row.Index}"),
                direction.rename(f"signal_{row.Index}"),
            ]
        )
        return_parts.append(pair_return.rename(f"return_{row.Index}"))
        direction_parts.append(direction.rename(f"direction_{row.Index}"))
        zscore_parts.append(zscore.rename(f"zscore_{row.Index}"))
    joined = pd.concat(feature_parts + return_parts + direction_parts + zscore_parts, axis=1).dropna()
    feature_count = len(feature_parts)
    pair_count = len(selected)
    features = joined.iloc[:, :feature_count]
    pair_returns = joined.iloc[:, feature_count : feature_count + pair_count]
    directions = joined.iloc[:, feature_count + pair_count : feature_count + 2 * pair_count]
    zscores = joined.iloc[:, feature_count + 2 * pair_count :]
    scale = features.loc[:TRAIN_END].std(ddof=1).replace(0, 1)
    features = (features / scale).clip(-10, 10)
    return joined.index, features, pair_returns, directions, zscores


def make_environment(features, returns, directions, zscores, index, activity_penalty_bps=0.0):
    return PairExecutionEnv(
        features.loc[index].to_numpy(),
        returns.loc[index].to_numpy(),
        directions.loc[index].to_numpy(),
        zscores.loc[index].to_numpy(),
        cost_bps=10,
        maximum_active_pairs=2,
        activity_penalty_bps=activity_penalty_bps,
    )


def evaluate(model: PPO, environment: PairExecutionEnv):
    observation, _ = environment.reset()
    rows = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        rows.append(info)
        done = terminated or truncated
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(17)
    closes = load_closes()
    selected = screen_pairs(closes.loc[:TRAIN_END])
    selected = selected.loc[selected["adf_pvalue"] < 0.05].head(3)
    index, features, returns, directions, zscores = build_arrays(closes, selected)
    train_index = index[index <= TRAIN_END]
    validation_index = index[(index > TRAIN_END) & (index <= VALIDATION_END)]
    test_index = index[index > VALIDATION_END]
    candidate_configs = [
        {"seed": 17, "activity_penalty_bps": 0.25},
        {"seed": 29, "activity_penalty_bps": 0.50},
        {"seed": 41, "activity_penalty_bps": 1.00},
    ]
    candidates = []
    models = []
    for config in candidate_configs:
        train_environment = make_environment(
            features,
            returns,
            directions,
            zscores,
            train_index,
            config["activity_penalty_bps"],
        )
        check_env(train_environment)
        candidate = PPO(
            "MlpPolicy",
            train_environment,
            seed=config["seed"],
            learning_rate=2e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.995,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=0,
        )
        candidate.learn(total_timesteps=80_000, progress_bar=False)
        validation = evaluate(
            candidate,
            make_environment(features, returns, directions, zscores, validation_index),
        )
        metrics = annualized_metrics(validation["net_return"])
        candidates.append(
            {
                **config,
                "validation_sharpe": metrics["sharpe"],
                "validation_return": metrics["annual_return"],
                "validation_drawdown": metrics["maximum_drawdown"],
                "validation_activity": float((validation["active_pairs"] > 0).mean()),
            }
        )
        models.append(candidate)
    candidate_table = pd.DataFrame(candidates)
    scores = candidate_table["validation_sharpe"].fillna(-1e9)
    selected_location = int(scores.to_numpy().argmax())
    model = models[selected_location]
    selected_config = candidate_configs[selected_location]
    test = evaluate(model, make_environment(features, returns, directions, zscores, test_index))
    test_metrics = annualized_metrics(test["net_return"])
    test_metrics.update(
        active_pair_hours=int((test["active_pairs"] > 0).sum()),
        activity_fraction=float((test["active_pairs"] > 0).mean()),
        shield_interventions=int(test["shield_changed_action"].sum()),
        validation_sharpe=float(candidate_table.iloc[selected_location]["validation_sharpe"]),
        validation_activity=float(candidate_table.iloc[selected_location]["validation_activity"]),
        training_steps=80_000,
        seed=int(selected_config["seed"]),
        activity_penalty_bps=float(selected_config["activity_penalty_bps"]),
        candidate_count=len(candidate_configs),
        evaluation_status="exploratory because the 2026 period was viewed after the first policy",
    )

    outputs = ROOT / "outputs"
    assets = ROOT / "docs" / "assets"
    models = ROOT / "models"
    for directory in [outputs, assets, models]:
        directory.mkdir(parents=True, exist_ok=True)
    model.save(models / "ppo_execution_overlay")
    candidate_table.to_csv(outputs / "ppo_validation_candidates.csv", index=False)
    test.to_csv(outputs / "ppo_test_path.csv", index=False)
    with open(outputs / "ppo_results.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    heuristic_metrics = json.loads((outputs / "heuristic_results.json").read_text())
    wealth = (1 + test["net_return"]).cumprod()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), facecolor="white")
    axes[0].plot(wealth.index, wealth, color="#6657d9", linewidth=2)
    axes[0].set_title("Shielded PPO test wealth")
    axes[0].set_ylabel("Growth of one dollar")
    axes[1].bar(
        ["Heuristic", "Shielded PPO"],
        [heuristic_metrics["sharpe"], test_metrics["sharpe"]],
        color=["#9fb3c8", "#6657d9"],
    )
    axes[1].axhline(0, color="#d6dee8", linewidth=1)
    axes[1].set_title("Untouched test Sharpe comparison")
    for axis in axes:
        axis.grid(color="#edf1f5")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Safe reinforcement learning execution overlay", fontsize=16)
    figure.tight_layout()
    figure.savefig(assets / "ppo_overlay.png", dpi=160, facecolor="white")
    plt.close(figure)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
