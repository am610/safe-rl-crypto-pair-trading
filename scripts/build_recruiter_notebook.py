"""Build the compact executed notebook used for GitHub and Colab."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    heuristic = json.loads((ROOT / "outputs" / "heuristic_results.json").read_text())
    scaled = json.loads((ROOT / "outputs" / "ppo_results.json").read_text())
    gate = json.loads((ROOT / "outputs" / "permission_gate_results.json").read_text())
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": "Safe RL Crypto Pair Trading.ipynb", "provenance": []},
    }
    cells = []
    cells.append(nbf.v4.new_markdown_cell("""# Safe Reinforcement Learning for Crypto Pair Trading

This notebook studies whether a shielded reinforcement learning execution policy can improve a dynamic cryptocurrency pair trading baseline. It uses real hourly Coinbase data, chronological evaluation, explicit trading costs, and deterministic risk limits.

The central finding is cautious. An unrestricted scaling policy fails badly. A permission gate reduces damage and drawdown, but the strategy remains unprofitable. This is a research result rather than a trading recommendation."""))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import os, sys, subprocess

if 'google.colab' in sys.modules:
    if not Path('/content/safe-rl-crypto-pair-trading').exists():
        subprocess.run(['git', 'clone', 'https://github.com/am610/safe-rl-crypto-pair-trading.git'], check=True)
    os.chdir('/content/safe-rl-crypto-pair-trading')
elif Path.cwd().name == 'notebooks':
    os.chdir(Path.cwd().parent)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[rl]'], check=True)
print('Project root:', Path.cwd())"""))
    cells.append(nbf.v4.new_markdown_cell("""## Research design

The experiment uses 2024 for training, 2025 for validation, and 2026 for exploratory evaluation. The word exploratory matters because the 2026 period was viewed during early policy diagnosis. Every strategy includes a one hour execution delay and ten basis points of turnover cost."""))
    cells.append(nbf.v4.new_code_cell("""import json
import pandas as pd
from IPython.display import display, Image

products = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD', 'LTC-USD']
summary = pd.DataFrame({
    'Period': ['Training', 'Validation', 'Exploratory evaluation'],
    'Dates': ['2024', '2025', '2026 through August 24'],
    'Purpose': ['Pair selection and policy fitting', 'Parameter and candidate selection', 'Locked comparison with disclosure'],
})
display(summary)"""))
    cells.append(nbf.v4.new_markdown_cell("""## Real data and pair evolution

The pipeline downloads public hourly candles from Coinbase Exchange. Binance futures access is restricted in the development location, so this project does not make futures specific claims. The animation shows normalized BTC and DOGE prices, the fixed hedge spread z score, entry thresholds, and periods when the heuristic is active."""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename='docs/assets/btc_doge_pair_evolution.gif'))"""))
    cells.append(nbf.v4.new_markdown_cell("""## Training only pair selection

The training screen selects BTC with DOGE, SOL with DOGE, and BTC with SOL. Hedge ratios and stationarity diagnostics are estimated without validation or evaluation observations."""))
    cells.append(nbf.v4.new_code_cell(f"""heuristic = {json.dumps(heuristic, indent=2)}
pd.DataFrame({{
    'Selected pair': heuristic['selected_pairs'],
    'Selection source': ['Training only'] * len(heuristic['selected_pairs'])
}})"""))
    cells.append(nbf.v4.new_markdown_cell("""## Locked heuristic baseline

The deterministic baseline uses a 336 hour rolling window, a 1.5 standard deviation entry threshold, and a 0.5 standard deviation exit threshold. It fails on the exploratory evaluation period, which establishes a demanding but honest benchmark."""))
    cells.append(nbf.v4.new_code_cell(f"""scaled = {json.dumps(scaled, indent=2)}
gate = {json.dumps(gate, indent=2)}

comparison = pd.DataFrame([
    {{'Model': 'Locked heuristic', 'Sharpe': heuristic['sharpe'], 'Maximum drawdown': heuristic['maximum_drawdown'], 'Ending wealth': heuristic['ending_wealth']}},
    {{'Model': 'Scaled PPO', 'Sharpe': scaled['sharpe'], 'Maximum drawdown': scaled['maximum_drawdown'], 'Ending wealth': scaled['ending_wealth']}},
    {{'Model': 'Permission gate PPO', 'Sharpe': gate['sharpe'], 'Maximum drawdown': gate['maximum_drawdown'], 'Ending wealth': gate['ending_wealth']}},
])
display(comparison.style.format({{'Sharpe': '{{:.2f}}', 'Maximum drawdown': '{{:.1%}}', 'Ending wealth': '{{:.2f}}'}}))"""))
    cells.append(nbf.v4.new_code_cell("""display(Image(filename='docs/assets/permission_gate.png'))"""))
    cells.append(nbf.v4.new_markdown_cell("""## What the reinforcement learning experiments show

The scaled PPO policy participates too often and performs far worse than the heuristic. The permission gate can only authorize or reject deterministic signals. It cannot choose direction, add leverage, or override the shield. This safer formulation improves drawdown and ending wealth, but it does not produce a profitable strategy.

The conclusion is therefore narrow and defensible: explicit abstention and deterministic shielding reduce damage in this experiment. They do not establish a durable cryptocurrency trading edge."""))
    cells.append(nbf.v4.new_code_cell("""risk_controls = pd.DataFrame({
    'Control': ['Direction anchoring', 'Divergence shield', 'Concentration limit', 'Execution delay', 'Turnover cost'],
    'Implementation': ['Deterministic signal only', 'Forced flat beyond absolute z score 4', 'At most two active pairs', 'One hour', 'Ten basis points'],
})
display(risk_controls)"""))
    cells.append(nbf.v4.new_markdown_cell("""## Limitations and next evidence

1. Coinbase spot candles are not Binance futures observations.

2. The 2026 period is exploratory after early inspection.

3. Only three PPO candidates are compared in each controlled search.

4. Funding, order book impact, and exchange specific fill mechanics are not modeled.

5. Bootstrap uncertainty and cost sensitivity remain necessary before the research can be considered complete."""))
    notebook["cells"] = cells
    destination = ROOT / "notebooks" / "01_safe_rl_crypto_pair_trading.ipynb"
    destination.parent.mkdir(exist_ok=True)
    nbf.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
