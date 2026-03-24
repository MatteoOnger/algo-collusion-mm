# Algorithmic Collusion in Market Making

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/)

This repository contains code and experiments used for research on algorithmic collusion in market-making environments. It includes simulation environments, several agent families (informed and uninformed), utilities for analysis and plotting, and example scripts and notebooks to reproduce experiments.

## Implemented algorithms
- Makers (informed): CRM, Hedge, Q-Learning (QL)
- Makers (uninformed): Exp3, MLQL
- Trader baselines: NoPassTrader and simple/basic traders

## Repository layout

- `algo_collusion_mm/` — Main Python package containing environments, agents, utilities, and experiment helpers.
  - `envs.py` — environment definitions and wrappers.
  - `agents/` — agent implementations and factories:
    - `agents/agent.py` — base agent abstractions.
    - `agents/makers/` — market maker agents and strategies (informed/uninformed).
    - `agents/traders/` — trader implementations and simple baselines.
  - `utils/` — helper modules for plotting, statistics, storage, and common utilities.
- `experiments/` — recommended place to store produced outputs, compressed results and archives.
- `notebooks/` — Jupyter notebooks for experimentation and analysis (e.g., `dev.ipynb`, `notebook_hedge.ipynb`).
- `scripts/` — example experiment entrypoints and helpers:
  - `runner.py` — library functions that orchestrate experiment grids, parallel runs, statistics, result saving and plotting (intended for import, not direct execution).
  - Executables: `script_hedge.py`, `script_exp3.py`, `script_crm.py`, etc.
- Packaging & metadata: `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.py`.

## Quick start

Recommended: Conda (uses `environment.yml`)

```bash
conda env create -f environment.yml
conda activate <env-name>   # or use name in environment.yml
pip install -e .
```

Lightweight alternative: `venv` + `pip`

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Run examples and notebooks

```bash
# Run an example script
python scripts/script_hedge.py
python scripts/script_exp3.py

# Start Jupyter
jupyter lab   # or: jupyter notebook
# Open notebooks in `notebooks/` (e.g. notebooks/dev.ipynb)
```

## Programmatic usage

Import and use the runner or compose environments and agents directly for custom experiments.

Example: run programmatically using `scripts.runner` helpers

```python
from scripts.runner import run_experiment_suite

# prepare `fixed_params` and `variable_params` (see runner docstring)
run_experiment_suite(fixed_params, variable_params, base_path='experiments/myrun', max_workers=4)
```

Advanced example: build a custom environment and agent mix

```python
from algo_collusion_mm.agents.makers.uninformed.exp3 import MakerExp3
from algo_collusion_mm.agents.traders.nopass import NoPassTrader
from algo_collusion_mm.envs import CGMEnv

makers = [MakerExp3(epsilon=0.01, name='maker_0'), MakerExp3(epsilon=0.01, name='maker_1')]
traders = [NoPassTrader(tie_breaker='rand', name='trader_0')]

env = CGMEnv(generate_vt=lambda: 0.5, n_rounds=100, makers=makers, traders=traders)

obs, infos = env.reset()
for agent in env.possible_agents:
    agent.reset()

for next_agent in env.agent_iter():
    action = next_agent.act(env.observe(next_agent))
    obs, rewards, terminations, truncations, infos = env.step(action)

# inspect agent histories, rewards, or save results after the episode
```

## Notebooks

- The `notebooks/` folder contains interactive analyses and experiment explorations. Open them with Jupyter Lab/Notebook after installing dependencies.

## Development notes

- Install in editable mode for development: `pip install -e .`.
- Use the Conda environment in `environment.yml` for consistent dependency management.

## Tips

- To specify an explicit conda environment name: `conda env create -f environment.yml -n myenv`.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Contact

For questions about the codebase or experiments, open an issue or contact the repository owner.
