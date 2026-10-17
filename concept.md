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
| Reward shaping | Small kill/wave rewards and base-damage penalties provide feedback before the final win/loss reward. |

The policy uses `MlpPolicy` with separate actor and critic MLPs, Tanh hidden
activations, and Adam optimization. The learning rate defaults to 0.0003. The
value-loss coefficient defaults to 0.5. The entropy coefficient defaults to 0.0:
training samples actions stochastically, but adds no explicit entropy bonus.

### Decisions and rewards

There are **63 discrete actions** for the supplied map:

- Action `0`: wait.
- Actions `1` through `62`: choose one of 31 slots and one of two towers.
- Encoding: `1 + slot_index * 2 + tower_index`.
- Tower index `0` is Cannon; index `1` is Double Cannon in the current TOML order.
- Slots use ascending Tiled object IDs, rather than the IDs themselves as indices.

Each action advances up to 30 simulation ticks at 1/60 second per tick: normally
**0.5 simulated seconds per decision**. Advancement stops early if the game ends.

Step reward:

```text
0.1 × new kills
+ 1.0 × newly completed waves
- base HP lost / 10
+ 20 on victory OR -20 on defeat
```

Buying a tower, spending resources, waiting, and simply remaining alive do not
earn rewards. Resource income from kills affects purchasing power separately
from the RL reward.

The training cycle is: observe → mask illegal actions → sample an action → advance
combat → record reward and next observation → repeat for 256 decisions → estimate
advantages/returns → update actor and critic → collect the next rollout.

Clear all 10 waves with base HP remaining to win. Reaching zero base HP is defeat.
Either is episode termination. Reaching 2400 decisions without either outcome is
time-limit truncation, without an extra win/loss reward. Both require a reset.

Implementation: `src/game/rl/__main__.py` and `src/game/rl/environment.py`.
Algorithm reference: [Maskable PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## 2. States / observations

The simulator holds the complete internal game state. The policy receives a
compact **211-element float32 observation vector**:

```text
8 global values + 31 slots × 5 values + 2 enemy types × 12 route bins × 2 values
= 8 + 155 + 48
= 211
```

### Global values: indices 0–7

| Index | Value sent to the agent | Why it matters |
|---|---|---|
| 0 | Current base HP / maximum base HP | Remaining margin for enemy leaks. |
| 1 | Current resources / 200 | Money available for tower purchases. |
| 2 | Current wave number / maximum waves | Current stage and increasing difficulty. |
| 3 | Completed waves / maximum waves | Progress through the episode. |
| 4 | Enemies still waiting to spawn / 50 | Incoming pressure in the current wave. |
| 5 | Spawn countdown, in seconds | Time until another scheduled spawn. |
| 6 | Intermission countdown, in seconds | Preparation time recorded by the wave director. |
| 7 | Decisions elapsed / 2400 | Proximity to the episode time limit. |

The denominators 200 and 50 are fixed scaling constants. These values are not
clipped, and money or other scaled quantities can exceed 1. Timers are raw
seconds. Wave-director phase rules determine when each countdown is active;
the stored intermission timer does not always mean preparation is underway.

### Slot values: indices 8–162

Each of the 31 slots contributes these five values in ascending slot-ID order:

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

### Enemy-route values: indices 163–210

Divide the route into **12 equal-distance bins along the actual polyline**.
For each enemy type and bin, record:

| Value | Encoding | Purpose |
|---|---|---|
| Enemy count | Number of enemies in the bin / 10 | Where enemies are concentrated. |
| Remaining health | Sum of current enemy HP in the bin / 100 | How much damage is needed there. |

The current enemy types are Walking Soldier (`enemy1`) and Advanced Soldier
(`enemy2`). Their blocks follow the enemy order in the TOML configuration. Each
type has 12 bins, each containing count followed by health. Bins progress from the
spawn point toward the base. Three enemies with 230 total remaining HP produce
`[0.3, 2.3]` for that type/bin.

### Information limits

The observation summarizes enemies instead of providing exact positions,
individual HP, projectile states, or random-generator state. It also omits an
explicit difficulty label and wave-phase flag. Consequently, it is partially
observed: different internal situations can produce the same observation.
The current MLP has no recurrent memory.

The separate 63-element Boolean action mask identifies currently legal actions;
it is not part of the 211-element observation. Diagnostic `info` fields such as
success, total kills, leaks, spending, and reward totals are not additional
policy inputs.

Implementation: `BastionEnv.observation()` and `BastionEnv.action_masks()` in
`src/game/rl/environment.py`.
