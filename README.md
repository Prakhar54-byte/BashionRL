# BastionRL

## Project contributors

| Name | Roll number |
|---|---|
| Prasangeet Dongre | B23CH1033 |
| Prakhar Rashmikant Chauhan | B23BB1032 |


**BastionRL** is a 2D tower-defense simulation used to study sequential decisions under limited resources. A reinforcement-learning agent learns when and where to build defenses, invest in resource production, and reconstruct destroyed King Towers while enemy waves evolve.

The project is an RL testbed inspired by real-world resource allocation and infrastructure protection. It is not a validated real-world deployment.

## Screenshots

The following screenshots show the project as captured in the repository.

| Screenshot 1 | Screenshot 2 |
|---|---|
| ![BastionRL screenshot 1](docs/screenshots/1.png) | ![BastionRL screenshot 2](docs/screenshots/2.png) |

## Overview

- **Environment:** Pygame simulation with map geometry and routes loaded from Tiled.
- **RL algorithm:** Maskable Proximal Policy Optimization (Maskable PPO) from `sb3-contrib`.
- **Policy:** A custom hierarchical policy that learns an action category and then a target.
- **Observations:** 1,462 float features describing the economy, wave progression, available slots, structures, King Towers, and up to 64 enemies.
- **Actions:** 102 discrete actions, with action masks used to exclude illegal choices.
- **Training:** Gymnasium environment, optional parallel workers, curriculum learning, normalized observations/rewards, checkpointing, and outcome-based evaluation.

The submitted decision-making agent is trained with PPO. A scripted diagnostic, where used, is separate from the learned policy and is not the submitted agent.

## The RL formulation

The underlying simulator is modelled as an episodic Markov decision process, `M = (S, A, P, R, γ)`. Enemy movement, attacks, projectiles, wave progression, resource income, structure damage, repairs, and terminal outcomes are simulated by the environment. The agent receives a compact partial observation of the simulator state.

### State space (`S`)

The observation is a 1,462-dimensional `float32` vector bounded to `[0, 1]`. It includes:

- Global health, resources, wave progress, timers, enemy counts, and economy features.
- Configured features for each tower and enemy type.
- Occupancy and route-proximity features for defense and resource slots.
- Health, status, and repair information for the three King Towers.
- A padded/truncated representation of up to 64 active enemies.

Enemy truncation and summarized combat information omit parts of the full simulator state. The observation is therefore a partially observed representation and is not guaranteed to be Markov on its own.

### Action space (`A`)

The discrete action space contains 102 actions:

| Action group | Count | Meaning |
|---|---:|---|
| Wait | 1 | Take no construction or repair action |
| Defense | 92 | Place a Cannon or Double Cannon in one of 46 defense slots |
| Resource | 6 | Place a Resource Tower in one of 6 resource slots |
| Repair | 3 | Begin reconstruction of an eligible destroyed King Tower |

An action mask identifies legal actions based on affordability, slot occupancy, game state, and repair rules. The hierarchical policy learns the action kind and target; it does not use a hard-coded placement strategy.

### Reward function (`R`)

The reward combines progress toward the objective with costs and damage. Let `ΔD` be the sum of actual damage dealt to each enemy divided by that enemy's maximum HP during a transition: `ΔD = Σᵢ (damageᵢ / max_HPᵢ)`. Let `ΔK` be enemies killed, `ΔB` King Tower damage received, `H₀` the maximum combined King health, `ΔW` waves completed, and `ΔC` resources spent during that transition. The implemented reward is equivalent to:

```text
r = 0.05 × max(0, ΔD)
  + 0.025 × ΔK
  - ΔB / H₀
  + 0.1 × ΔW
  + terminal_reward
  - 0.005 × invalid_action
  - 0.001 × ΔC / cheapest_defense_cost
```

`terminal_reward` is `+2.5` for a win, `-1.0` for a loss or time-limit truncation, and `0` otherwise. The implementation applies the terms when the agent takes a decision step. Action masks are intended to prevent invalid choices during normal policy inference.

### Algorithm and training configuration

The main policy uses Maskable PPO with a custom hierarchical action distribution. The released checkpoint metadata records a baseline training configuration with a requested budget of 2,000,000 environment transitions, 8 parallel environments, 512 rollout steps per environment, a batch size of 256, 5 PPO epochs, `γ = 0.999`, and GAE `λ = 0.98`. It also records curriculum training and VecNormalize statistics.

Training transitions are **agent–environment interactions**, not complete games or raw physics frames. The best checkpoint can be saved before the requested training budget is reached. Check `submission/model/best_model.json` for the selected checkpoint's recorded step count and parameters.

## Requirements

- Python 3.11 or newer
- `uv` for environment and dependency management
- Pygame for the visual simulation
- Gymnasium, NumPy, PyTorch, and `sb3-contrib` for RL
- pytest and Ruff for development checks

The repository's `pyproject.toml` specifies the project dependencies. On Linux x86-64, it configures the CPU PyTorch package source by default.

## Installation

Install `uv` using the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and install the RL and development dependencies:

```sh
git clone https://github.com/prasangeet/BashionRL.git
cd BashionRL
uv sync --extra rl --dev
```

Run commands from the repository root so that the map and asset paths resolve correctly.

## Run the game

### Play manually

```sh
uv run python main.py --difficulty baseline --seed 42
```

Controls:

- `Q`: select Cannon
- `W`: select Double Cannon
- `E`: select Resource Tower
- Left-click an eligible slot: build the selected structure
- Left-click an eligible destroyed King Tower: begin reconstruction
- `R`: restart the episode
- `Esc`: quit

### Watch the trained agent

The trained checkpoint is stored in `submission/model/`. Keep the ZIP, JSON metadata, and normalization statistics together under the same basename.

```sh
uv run --extra rl python main.py \
  --difficulty baseline \
  --model submission/model/best_model.zip \
  --max-steps 2400 \
  --seed 42
```

The model loader checks that the checkpoint's environment schema/map configuration matches the current environment. Do not move or rename just one of these files:

```text
submission/model/best_model.zip
submission/model/best_model.json
submission/model/best_model.vecnormalize.pkl
```

## Evaluate and reproduce results

### Evaluate the trained policy

To reproduce the report's evaluation, use the released checkpoint for 100 baseline episodes with seeds 90000–90099 (`--seed 90000 --episodes 100`). This range is separate from the checkpoint-selection seeds 10000–10029. For a new held-out test, choose another seed range that was not used for checkpoint selection.

```sh
uv run --extra rl python -m src.game.rl evaluate \
  --difficulty baseline \
  --model submission/model/best_model.zip \
  --episodes 100 \
  --seed 90000 \
  --max-steps 2400 \
  --output runs/evaluation-baseline.json
```

The command writes episode-level results and aggregate metrics to the specified JSON file, including win rate, waves completed, remaining King health, reward, and action diagnostics. Report the episode count, difficulty, seed range, and checkpoint step count alongside any result. This README does not claim a particular win rate.

### Run a random legal-action baseline

A random-policy baseline is provided as a simple reference. It samples from legal actions; it is not an optimal strategy.

```sh
uv run --extra rl python -m src.game.rl evaluate \
  --difficulty baseline \
  --policy random \
  --episodes 100 \
  --seed 90000 \
  --max-steps 2400 \
  --output runs/evaluation-random.json
```

For a fair comparison, use the same difficulty, episode count, seed range, and episode horizon for the learned policy and random baseline. Ensure the seed range has not been used to select the checkpoint before labelling it a final held-out test.

### Train from scratch

The following command reproduces the principal settings recorded for the released model. Training may take substantial time depending on CPU speed and parallel-worker overhead.

```sh
uv run --extra rl python -m src.game.rl train \
  --difficulty baseline \
  --strategy hierarchical \
  --steps 2000000 \
  --seed 0 \
  --n-envs 8 \
  --n-steps 512 \
  --batch-size 256 \
  --n-epochs 5 \
  --device cpu \
  --learning-rate 0.0003 \
  --lr-final 0.00005 \
  --ent-coef 0.02 \
  --ent-final 0.005 \
  --curriculum mixed \
  --promotion-successes 30 \
  --normalize \
  --eval-freq 65536 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --max-steps 2400 \
  --output runs/reproduced/seed-0
```

The output directory contains checkpoints, metadata, evaluation history, and training telemetry. The `.zip`, `.json`, and `.vecnormalize.pkl` files for a checkpoint must be kept together. If resuming, use the documented `--resume` option and check the saved metadata so the intended total budget and training settings are preserved.

### Optional difficulty tests

`baseline`, `medium`, and `hard` configurations are available. A model tested on a different difficulty should be reported as a separate evaluation; do not infer generalization from baseline results alone.

```sh
uv run --extra rl python -m src.game.rl evaluate \
  --difficulty medium \
  --model submission/model/best_model.zip \
  --episodes 20 \
  --seed 234567 \
  --max-steps 2400 \
  --output runs/evaluation-medium.json
```

Choose a seed range not used during training-time model selection before describing an evaluation as held out.

## Training and diagnostic plots

The repository includes plots for inspecting training behavior and PPO optimization. These support analysis and transparency; they are **not, by themselves, evidence of a successful or generalizable policy**. Win-rate, goal-success-rate, and held-out-outcome plots are intentionally omitted here.

### Training behavior

![Training return over the run](submission/plots/01_training_return.png)

*Training return is a learning signal; interpret it alongside the reward definition and independent evaluation.*

### Environment progress metrics

| Validation waves completed | Remaining King health |
|---|---|
| ![Validation waves](submission/plots/03_validation_waves.png) | ![Validation King health](submission/plots/04_validation_king_health.png) |

*These show metrics beyond win rate. Interpret them with the evaluation configuration and episode seeds.*

### PPO optimization diagnostics

| Critic explained variance | Policy KL divergence |
|---|---|
| ![Critic explained variance](submission/plots/05_critic_explained_variance.png) | ![Policy KL divergence](submission/plots/06_policy_kl.png) |

| Learning-rate schedule | Entropy-coefficient schedule |
|---|---|
| ![Learning rate](submission/plots/07_learning_rate.png) | ![Entropy coefficient](submission/plots/08_entropy_coefficient.png) |

### Curriculum and action-selection diagnostics

| Curriculum goal during training | Learned action distribution |
|---|---|
| ![Curriculum goal](submission/plots/09_curriculum_goal.png) | ![Learned action diagnostics](submission/plots/14_learned_actions.png) |

Open the [full plots directory](submission/plots/) to inspect additional files. PDF versions are available alongside the PNGs.

## Tests and code quality

```sh
uv run --extra rl pytest
uv run --extra rl ruff check .
```

## Repository layout

```text
Assets/                  Map, sprites, and other game assets
configs/game.toml        Game and difficulty configuration
src/game/simulation.py   Shared game simulation
src/game/rl/             Gymnasium environment, PPO policy, training, evaluation
tests/                   Simulation and RL tests
submission/model/        Trained checkpoint and required sidecar files
submission/plots/        Exported diagnostic plots
docs/                    Engineering notes and detailed experiment history
main.py                  Pygame manual-play and trained-agent viewer
```

## Limitations and reporting notes

- This is a single-map simulation and does not demonstrate deployment in a physical or operational environment.
- Results depend on the checkpoint, scenario configuration, random seeds, and evaluation protocol.
- Do not report training-budget settings as completed steps. Use the checkpoint metadata and training summary to report actual progress.
- Separate validation/model-selection episodes from final testing. Include seed ranges and publish the generated evaluation JSON when reporting performance.
- Training plots are useful diagnostics, but should be interpreted with the corresponding run metadata and evaluation protocol.

See [`docs/rl-training-and-combat-report.md`](docs/rl-training-and-combat-report.md) for detailed engineering notes and experiment history.
