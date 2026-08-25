"""Train a safe PPO permission gate for deterministic pair signals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from crypto_pair_rl.backtest import annualized_metrics
from crypto_pair_rl.environment import PermissionGateEnv
from crypto_pair_rl.pairs import screen_pairs
from run_heuristic_baseline import TRAIN_END, VALIDATION_END, load_closes
from train_ppo_overlay import build_arrays, evaluate


ROOT = Path(__file__).resolve().parents[1]


def make_gate(features, returns, directions, zscores, index, penalty):
    return PermissionGateEnv(
        features.loc[index].to_numpy(),
        returns.loc[index].to_numpy(),
        directions.loc[index].to_numpy(),
        zscores.loc[index].to_numpy(),
        cost_bps=10,
        maximum_active_pairs=2,
        activity_penalty_bps=penalty,
    )


def main() -> None:
    closes = load_closes()
    selected = screen_pairs(closes.loc[:TRAIN_END])
    selected = selected.loc[selected["adf_pvalue"] < 0.05].head(3)
    index, features, returns, directions, zscores = build_arrays(closes, selected)
    train_index = index[index <= TRAIN_END]
    validation_index = index[(index > TRAIN_END) & (index <= VALIDATION_END)]
    test_index = index[index > VALIDATION_END]
    configurations = [
        {"seed": 17, "penalty": 0.5},
        {"seed": 29, "penalty": 1.0},
        {"seed": 41, "penalty": 2.0},
    ]
    rows = []
    trained = []
    for config in configurations:
        environment = make_gate(features, returns, directions, zscores, train_index, config["penalty"])
        check_env(environment)
        model = PPO(
            "MlpPolicy",
            environment,
            seed=config["seed"],
            learning_rate=2e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.995,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [128, 128]},
            verbose=0,
        )
        model.learn(total_timesteps=80_000, progress_bar=False)
        validation = evaluate(model, make_gate(features, returns, directions, zscores, validation_index, 0.0))
        metrics = annualized_metrics(validation["net_return"])
        rows.append(
            {
                **config,
                "validation_sharpe": metrics["sharpe"],
                "validation_return": metrics["annual_return"],
                "validation_drawdown": metrics["maximum_drawdown"],
                "validation_activity": float((validation["active_pairs"] > 0).mean()),
            }
        )
        trained.append(model)
    candidates = pd.DataFrame(rows)
    selection_score = candidates["validation_sharpe"].replace([np.inf, -np.inf], np.nan)
    selection_score = selection_score.fillna(candidates["validation_return"] * 100)
    winner = int(selection_score.to_numpy().argmax())
    model = trained[winner]
    test = evaluate(model, make_gate(features, returns, directions, zscores, test_index, 0.0))
    metrics = annualized_metrics(test["net_return"])
    metrics.update(
        seed=int(configurations[winner]["seed"]),
        training_activity_penalty_bps=float(configurations[winner]["penalty"]),
        validation_sharpe=float(candidates.iloc[winner]["validation_sharpe"]),
        validation_activity=float(candidates.iloc[winner]["validation_activity"]),
        test_activity=float((test["active_pairs"] > 0).mean()),
        shield_interventions=int(test["shield_changed_action"].sum()),
        evaluation_status="exploratory because the 2026 period was previously viewed",
    )

    outputs = ROOT / "outputs"
    assets = ROOT / "docs" / "assets"
    models = ROOT / "models"
    for directory in [outputs, assets, models]:
        directory.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(outputs / "permission_gate_candidates.csv", index=False)
    test.to_csv(outputs / "permission_gate_test_path.csv", index=False)
    with open(outputs / "permission_gate_results.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    model.save(models / "permission_gate")

    heuristic = json.loads((outputs / "heuristic_results.json").read_text())
    prior_ppo = json.loads((outputs / "ppo_results.json").read_text())
    wealth = (1 + test["net_return"]).cumprod()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), facecolor="white")
    axes[0].plot(wealth.index, wealth, color="#e07a5f", linewidth=2)
    axes[0].set_title("Permission gate exploratory wealth")
    axes[0].set_ylabel("Growth of one dollar")
    axes[1].bar(
        ["Heuristic", "Scaled PPO", "Permission gate"],
        [heuristic["sharpe"], prior_ppo["sharpe"], metrics["sharpe"]],
        color=["#9fb3c8", "#6657d9", "#e07a5f"],
    )
    axes[1].axhline(0, color="#d6dee8", linewidth=1)
    axes[1].set_title("Exploratory Sharpe comparison")
    for axis in axes:
        axis.grid(color="#edf1f5")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Safe permission based execution", fontsize=16)
    figure.tight_layout()
    figure.savefig(assets / "permission_gate.png", dpi=160, facecolor="white")
    plt.close(figure)
    print(candidates.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
