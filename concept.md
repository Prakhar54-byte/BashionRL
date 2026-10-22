# BastionRL: RL Technique and States

## 1. RL technique

The project uses **Maskable Proximal Policy Optimization (Maskable PPO)** from
Stable-Baselines3 Contrib, running a PyTorch neural network on the CPU. The agent
learns tower placement from game-state numbers and rewards.

### Methods used

| Method | How it is used here |
|---|---|
| Model-free reinforcement learning | Learn from playing the simulation; the agent does not learn a separate model to predict the game. |
| On-policy learning | Collect experience with the current policy, train on that rollout, then collect fresh experience. An episode may span several rollouts. |
| Actor–critic | The actor assigns probabilities to legal actions. The critic estimates future discounted reward from an observation. Each has two hidden layers of 64 units. |
| Clipped PPO objective | Compare new and previous probabilities for sampled actions. Clip the probability ratio around 1 with a range of 0.2 to discourage excessively large policy changes. This is an objective constraint, not a hard guarantee on policy change. |
| Generalized Advantage Estimation (GAE) | Combine rewards and critic estimates to estimate whether an action performed better than expected. Lambda defaults to 0.95. |
| Discounting | Gamma is 0.995 per decision, so future rewards matter while more distant rewards receive less weight. |
| Advantage normalization | Standardize advantages within each training minibatch before calculating the policy loss. |
| Invalid-action masking | Exclude occupied slots and unaffordable tower purchases from the action distribution. Waiting remains legal. Apply masks during training and model prediction. |
| Minibatch optimization | Collect 256 decisions, divide them into minibatches of 64, and train for 10 epochs over the rollout. That normally gives 40 optimizer steps per rollout with one environment. |
| Gradient clipping | The library limits gradient norm to 0.5 to reduce excessively large parameter updates. |
| Reward shaping | Kill/wave rewards and damage, spending, and invalid-action penalties provide feedback before the final outcome. |

The policy uses `MlpPolicy` with separate actor and critic MLPs, Tanh hidden
activations, and Adam optimization. The learning rate defaults to 0.0003. The
value-loss coefficient defaults to 0.5. The entropy coefficient defaults to 0.0:
training samples actions stochastically, but adds no explicit entropy bonus.

### Decisions and rewards

There are **93 discrete actions** for the supplied map:

- Action `0`: wait.
- Actions `1` through `92`: choose one of 46 slots and one of two towers.
- Encoding: `1 + slot_index * 2 + tower_index`.
- Tower index `0` is Cannon; index `1` is Double Cannon in the current TOML order.
- Slots use ascending Tiled object IDs, rather than the IDs themselves as indices.

Each action advances up to 30 simulation ticks at 1/60 second per tick: normally
**0.5 simulated seconds per decision**. Advancement stops early if the game ends.
Spectator mode renders between individual physics ticks; headless training consumes
the same ticks immediately, without drawing or real-time delays.

Step reward:

```text
0.1 × new kills
+ 1.0 × newly completed waves
- combined King Tower HP lost / 10
+ 20 on victory OR -20 on defeat
- 0.1 for an invalid action
- resources actually spent / 1000
```

A Cannon purchase costs 0.1 reward and a Double Cannon costs 0.15 reward. Failed
purchases spend nothing and receive only the invalid-action penalty. Invalid
actions, including out-of-range indices, advance time as a no-op. Maskable PPO
normally avoids these through its mask. Waiting receives no passive reward;
completing a wave provides the survival/progression reward. Resource income from
kills affects purchasing power separately from the RL reward.

The training cycle is: observe → mask illegal actions → sample an action → advance
combat → record reward and next observation → repeat for 256 decisions → estimate
advantages/returns → update actor and critic → collect the next rollout.

Clear all 10 waves with a surviving King Tower to win. Destroying the final
configured King Tower is immediate defeat. The base-health fields now report
combined King health, rather than a separate leak-based objective.
Either is episode termination. Reaching the configured `max_steps` (default 2400)
without either outcome is time-limit truncation, without an extra win/loss reward.
Both require a reset. Training, evaluation, and model viewing expose `--max-steps`.
The default `baseline` uses original wave settings with no added enemy HP growth;
`medium` and `hard` are optional balance experiments, not baseline requirements.
Combat/objective schema 3 requires retraining previous checkpoints.
The larger map changes slot counts and checkpoint metadata, so old-map models
also require retraining; incompatible checkpoints are rejected. Light and Heavy
Tanks use separate chassis and gun sprites, with projectile attacks.
Enemies follow seeded routes from the soldier or tank route family.
King repairs are simulation/manual controls, not additional RL actions.

Implementation: `src/game/rl/__main__.py` and `src/game/rl/environment.py`.
Algorithm reference: [Maskable PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## 2. States / observations

The simulator holds the complete internal game state. The policy receives a
fixed **622-element float32 observation vector** for the supplied map:

```text
8 global values + 46 slots × 5 values + 64 padded enemy rows × 6 values
= 8 + 230 + 384
= 622
```

### Global values: indices 0–7

| Index | Value sent to the agent | Why it matters |
|---|---|---|
| 0 | Combined King HP / maximum combined King HP | Remaining objective health. |
| 1 | Current resources / 200 | Money available for tower purchases. |
| 2 | Current wave number / maximum waves | Current stage and increasing difficulty. |
| 3 | Completed waves / maximum waves | Progress through the episode. |
| 4 | Enemies still waiting to spawn / 50 | Incoming pressure in the current wave. |
| 5 | Spawn countdown, in seconds | Time until another scheduled spawn. |
| 6 | Intermission countdown, in seconds | Preparation time recorded by the wave director. |
| 7 | Decisions elapsed / configured max_steps | Proximity to the episode time limit. |

The denominators 200 and 50 are fixed scaling constants. These values are not
clipped, and money or other scaled quantities can exceed 1. Timers are raw
seconds. Wave-director phase rules determine when each countdown is active;
the stored intermission timer does not always mean preparation is underway.

### Slot values: indices 8–237

Each of the 46 slots contributes these five values in ascending slot-ID order:

| Position within each slot | Value sent to the agent |
|---|---|
| 0 | Slot center X / map width. |
| 1 | Slot center Y / map height. |
| 2 | Occupancy: 0 for empty, 1 for occupied. |
| 3 | Tower type: 0 for empty, 0.5 for Cannon, 1 for Double Cannon. |
| 4 | Tower firing cooldown in seconds, or 0 for an empty slot. |

These values describe where a tower can be placed, existing defenses, and their
firing readiness. The five-value block for slot index `i` starts at `8 + 5*i`.
Tower costs, ranges, and damage are fixed game configuration, not extra inputs.

### Padded enemy values: indices 238–621

Every active enemy has an individual row; unused rows are entirely zero.
There are 64 rows, with six float32 values per row:

| Column | Encoding |
|---|---|
| 0 | Presence: 1 for an enemy, 0 for padding. |
| 1 | Current X position / map width. |
| 2 | Current Y position / map height. |
| 3 | Current individual health / 100; values can exceed 1. |
| 4 | Enemy type: 1/4 for Walking Soldier, 2/4 for Advanced Soldier, 3/4 for Light Tank, 1 for Heavy Tank. |
| 5 | Distance travelled along the enemy’s assigned route / that route’s total length. |

Sort enemies by descending relative progress along their assigned routes. Equal
progress retains spawn order. Keep the first 64 rows if capacity is exceeded,
report the excess in `info["enemies_omitted"]`, and continue simulating all enemies.
The supplied presets spawn at most 34 enemies in a wave, below this capacity.
Raise capacity and retrain before using substantially larger waves.

### Information limits

The observation provides individual enemy positions and HP, but omits projectile
states, random-generator state, an explicit difficulty label, and wave-phase flag.
Individual structure HP, repair progress, and enemy attack cooldowns are also
hidden; the global health input reports combined King HP.
Capacity overflow can also omit enemies. Consequently, it remains partially
observed: different internal situations can produce the same observation.
The current MLP has no recurrent memory.

The separate 93-element Boolean action mask identifies currently legal actions;
it is not part of the 622-element observation. Diagnostic `info` fields such as
success, total kills, leaks, spending, and reward totals are not additional
policy inputs.

Implementation: `BastionEnv.observation()` and `BastionEnv.action_masks()` in
`src/game/rl/environment.py`.
