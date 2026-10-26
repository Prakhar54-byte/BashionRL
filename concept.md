# BastionRL: RL Technique and States

## 1. RL technique

The upstream sync introduces schema **7**. Earlier schema-5/6 checkpoints and their
normalization statistics are preserved in `runs/`, but require the old code for
inference or resuming; start fresh for the updated game. Historical experiments
below describe the earlier game and are not performance results for main.


The agent uses **hierarchical Maskable PPO**, implemented with Gymnasium, SB3
Contrib, and PyTorch. PPO remains the optimizer; the action strategy now has a
learned kind decision followed by a learned target decision. Every defense,
resource purchase, repair, and deliberate wait is chosen
by the learned policy. Training uses no demonstrations, placement rules, heuristic
fallback, or automatic repair strategy. The current game uses upstream main (`129b809`), including its updated map,
combat rules, economy, and wave growth. A training-only curriculum gradually increases the episode wave goal.

The actor and critic use separate two-layer MLPs with 128 units per layer.
The actor outputs four kind logits (wait, defense, resource, repair) and a logit
for every concrete action. A kind is legal when it has at least one legal target.
Within each kind, its targets receive their own masked softmax. The joint
probability is `P(kind | state) × P(target | kind, state)`. PPO optimizes joint
log probabilities and joint entropy, so both decisions learn from game outcomes.
Sampling this joint distribution is equivalent to sampling kind then target.
Deterministic play chooses the most probable legal kind, then its most probable
legal target; it does not compare wait against each individual defense slot.
Evaluation and the Pygame viewer use that same rule. This prevents a diffuse
defense preference from losing to wait solely because defenses have 92 targets.

The policy uses normalized game state;
legal-action masks enforce occupancy, affordability, repair eligibility, and
terminal status. Masks do not reject locations because of tactical quality.
No slot coordinates, repair thresholds, or preferred build sequences are coded
into action selection. Map geometry is supplied as state measurements.

PPO collects 256 meaningful transitions per environment, trains in minibatches of 64 for ten
epochs, and uses learning rate 0.0003, gamma 0.999, GAE lambda 0.98, entropy
coefficient 0.02, clipping range 0.2, and target KL 0.03. Advantages are
normalized. The entropy bonus supports exploration among legal choices. These
settings are an initial configuration, not evidence of optimal tuning.
`--ent-coef` and `--learning-rate` allow equal-budget comparisons. No static
per-target bias correction is used: the learned kind softmax separates action
frequency from the number of slots. All legal kinds and targets remain learnable.

Optional `--lr-final` and `--ent-final` linearly anneal their initial coefficients
to the requested endpoints using absolute completed training steps. The plan is
saved in the model; resume continues it instead of restarting the decay. Extending
a run beyond its original schedule horizon holds the final values. Explicit new
coefficient flags start a new schedule for the additional budget and must be
identified as a changed experiment. The default controls remain constant.

`--n-envs N` runs N independently seeded headless CPU environments. N greater
than one uses SB3 SubprocVecEnv with spawn; curriculum promotions are broadcast
to every worker and apply on reset. `--steps` and `--eval-freq` count aggregate
transitions across workers, not transitions per worker. A rollout contains
`256 × N` samples, so equal transition budgets with different worker counts are
not otherwise identical experiments. `--device cpu` is the default; `cuda`
requires a CUDA-enabled PyTorch and an available GPU. The project's Linux torch
source is CPU-only. GPU speedup is not established for this small MLP and Python
simulation. Independent training seeds are still required to assess consistency.

Training-only curriculum goals are derived from the configured wave count.
For the supplied game they are 1 → 2 → 3 → … → 10 waves. A goal advances after
at least 40 successes among the latest 50 episodes at that goal. Training never
forces an unmastered jump based on elapsed budget. Changes apply on episode reset;
observations include the current episode's goal, while wave/progression values
retain full-task normalization. Training logs include `wave_goal`: success at an
earlier curriculum stage is not a full-game win. Validation and CLI evaluation
always use all configured waves.
`--curriculum mixed` instead samples an episode goal: 60% current, 20% uniformly
chosen easier goal, and 20% full task. At goal 1 the unavailable easier branch
falls back to current; at the final goal, current/full branches coincide. These
are episode proportions, not proportions of transitions. Every worker samples
independently using its seeded environment RNG. Promotion counts only episodes
at the current goal. `--promotion-successes 30` tests a 30/50 threshold; the staged
control defaults to 40/50. These flags do not change enemy strength or costs.

Every training validation replays the same explicit seed range (`--eval-seed`,
default 10000), independently of training. `best_model.zip` is selected by win
rate, then mean waves cleared, then mean remaining King health. Shaped reward
does not break ties. `evaluations.json` records each scenario and reward components;
`evaluations.npz` retains rewards, lengths, timesteps, and successes. Evaluation
also reports action-kind probability and actual selection frequency conditioned
on that kind having a legal action. These expose ignored repairs/economy without
forcing either action. Final held-out test seeds should be separate from validation.
`latest.zip` and its metadata are saved at every validation for recovery;
`best_model.zip` follows the outcome ranking and `final.zip` saves the final
updated policy. New checkpoints restore weights, Adam optimizer, curriculum goal
and promotion window, schedule progress, and the original seed by default.
Worker RNG streams and in-flight episodes restart; resume is not an exact
interrupted-episode continuation. Old checkpoints without curriculum state still
start at goal 1 and print that limitation explicitly.
`nohup` only keeps a process running after logout; it does not change learning.

The supplied map has **102 actions**:

- `0`: wait.
- `1–92`: build either defense type in any of 46 defense slots.
- `93–98`: build a resource tower in any of six resource slots.
- `99–101`: reconstruct a destroyed (zero-health) map-defined King; living damaged Kings cannot be repaired. Another King must still be alive and the episode must be running.

One decision advances at least 30 fixed 1/60-second ticks, unless the episode
ends earlier. After a valid action, if only waiting remains legal, the environment
continues advancing until another choice becomes legal, checking every 30 ticks.
Every physics tick still runs; spectator mode renders between ticks. Deliberate
waits with other legal actions remain ordinary half-second decisions. Condensing
unavoidable waits gives PPO transitions that connect choices with their combat
consequences. Discounting applies per recorded transition.

`--max-steps 2400` remains a budget of 2400 half-second units: 20 simulated
minutes, including condensed waits. Logged `steps` counts actual policy
transitions; `simulation_ticks` and `simulated_seconds` report elapsed game time.
All configured waves must clear with a surviving King to win. Final King
loss terminates immediately; reaching the time budget truncates the episode.

Reward components per transition:

| Component | Reward |
|---|---:|
| Enemy damage | +0.05 per full enemy-health fraction removed |
| Enemy killed | +0.025 |
| Completed wave | +0.1 |
| King damage | −actual damage / initial combined maximum King HP |
| Victory | +2.5 |
| Defeat or time-limit truncation | −1 |
| Invalid action | −0.005 |
| Spending | −0.001 × actual spending / cheapest defense cost |

The reward scale keeps critic targets smaller than the previous −300 damage
penalty. Enemy damage is accumulated by the shared damage API as the fraction of each
enemy's maximum health actually removed. Spawning, suicide attacks, and route
exits give no damage reward; repeated hits on dead enemies add nothing.
King damage uses the cumulative damage counter, so healing gives no reward and
cannot hide incoming damage. Repeated damage after repairs can accumulate more
than one initial health pool. Spending includes defenses, resources, and repairs.
Income, waiting, and surviving without progression provide no passive reward.

The reward audit keeps the existing formula for this experiment: no purchase,
income, or healing bonus is introduced. Repeated damage penalties may discourage
repairs; action diagnostics and component totals allow testing that hypothesis
separately rather than changing several incentives together.

Schema **5** records the observation/action/reward contract and forced-wait
semantics. Earlier checkpoints require fresh training; their files are preserved.
Flat and hierarchical schema-7 policies remain evaluable and viewable. Switching their action head
to the hierarchical policy requires fresh training; their optimizer state and
action head cannot be resumed into the new architecture. Hierarchical checkpoints
can resume with a different worker count. Metadata records policy class, workers,
device, and validation selection alongside the environment contract.

For controlled experiments, `--strategy flat|hierarchical` selects the existing
architectures without changing combat, curriculum, observations, or rewards.
The flat control restores the previous per-kind initial bias; hierarchical PPO
uses its learned kind/target factorization. `--n-steps`, `--batch-size`, and
`--n-epochs` expose the native PPO controls. Match `n_envs × n_steps`, minibatch
size, epoch cap, sample budget, and validation schedule across cases. A rollout
contains more minibatches when its size grows: fewer rollout iterations do not
by themselves establish fewer optimizer steps. `training-summary.json` records
actual transitions, completed rollouts, completed PPO epochs, and optimizer steps.
KL early stopping can still make actual optimizer counts differ.
Scheduled validation runs after a completed PPO update, with requested frequency
rounded up to the next rollout boundary; final updated weights are also evaluated.
Metadata records the effective frequency and `checkpoint_steps` so recovery can
request only the remaining transitions. Resumed runs must still be identified
when interpreting experiment results because worker RNG/episode state restarts.

Validation defaults to 20 games. `wait_rate_when_other_action_legal` counts only
decisions with an available non-wait action. `forced_wait_decisions` counts
decision points where only wait was legal; `condensed_wait_ticks` tracks simulated
time beyond the normal decision interval. They measure different things.
`training-goals.json` and `goals/*` metrics report success separately by objective,
and `training-episodes.jsonl` stores per-episode goal/success, structure-death
events, and wave-end resource balances. Evaluation JSON includes the same timelines.
These additions are telemetry only; the current schema-7 environment retains these telemetry features.
Wave snapshots now include counts built and alive by tower type; destruction
events include tower type and tick, allowing failures after the heavy-tank unlock
to be compared without assuming Cannon selection is the cause.

`--normalize` opts into SB3 VecNormalize for both observations and rewards.
Each checkpoint saves matching `.vecnormalize.pkl` statistics beside its `.zip`
and `.json`. Evaluation and the viewer apply those observation statistics once,
without updating them; reported evaluation rewards remain in raw game units.
Resumed training restores the statistics and updates them again. Missing statistics
fail explicitly. Changing raw/normalized input contracts requires a fresh run,
and value losses from the two contracts are not directly comparable. Normalization
is opt-in so the previous controlled experiment remains reproducible.

The separate `diagnose` command runs a deterministic scripted probe with the
normal resource budget and objective. It scores map-route coverage and current
threats, saves for the defense with the best configured damage per cost, and
places defenses. Its optional economy variant tries an early resource tower
after establishing two defenses. Neither diagnostic requests King reconstruction or is
available to PPO training or saved-model gameplay. A successful diagnostic can
demonstrate winnability for its tested seed; failures do not prove impossibility.

The first hierarchical pilot used seed 0, four CPU workers, and 50,176 actual
transitions under `nohup`. On 20 test episodes (seeds 60000–60019), its selected
checkpoint cleared 2.25 waves on average versus 4.20 for the previous flat-policy
checkpoint; neither won. Repairs increased from 0 to 9.3 per game, but resource
construction remained 0. This is an experimental strategy, not a measured
performance improvement. Worker counts and validation schedules also differed,
so this pilot does not isolate the architecture's effect. Results and commands
are saved in `runs/hierarchical-v1-20261024-144723/`. Larger budgets and multiple
training seeds are needed before making performance claims; hardware alone
does not establish why a policy fails.
Saved metadata contains training parameters, map/config compatibility, and seed.
Evaluation uses the learned policy with masks; an explicit random policy remains
available as a sanity baseline. Training validation and final test seeds are
separate. Compare win rate, waves, health, and kills across reward versions;
raw returns from different reward definitions are not directly comparable.

References: [Maskable PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html),
[SB3 training and evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html).

## 2. States / observations

The supplied map produces **1462 bounded float32 values**, all in `[0, 1]`:

| Block | Size | Information |
|---|---:|---|
| Global | 15 | Combined King health, resources, wave/progression, pending spawns, timers, elapsed time, wave phase, surviving King fraction, enemy count, resource production rate, current episode wave goal / full task wave count, passive-income payout progress, next-wave enemy count from the configured scheduler. |
| Available tower types | 2 × 6 | Cost, range, damage, fire rate, burst size, maximum health. |
| Configured enemy types | 4 × 7 | Maximum health, speed, attack damage, configured range, attack interval, kill income, soldier/tank family. Available before enemies spawn. |
| Defense slots | 46 × 12 | Position, occupancy, tower type, cooldown, health, range, DPS, distance to soldier routes, distance to tank routes, nearest enemy distance, nearby enemy health. |
| Resource slots | 6 × 10 | Position, occupancy, health, production progress/rate, cost, distances to each route family, nearest enemy distance. |
| Kings | 3 × 9 | Position, health, alive/repairing flags, repair progress, attacking enemy DPS, incoming projectile damage, repair cost. |
| Padded enemies | 64 × 12 | Presence, position, health fraction, type, assigned-route progress, maximum health, speed, attack damage/effective range, attacking flag, attack cooldown. |

Positive unbounded quantities use `value / (value + scale)`. Health fractions use
the entity's own maximum health. Positions use map dimensions and ranges/distances
use the map diagonal. Final values are clipped to `[0, 1]`, including off-map
coordinates. Scaling constants are derived from configuration where appropriate.

Route distances measure geometry from Tiled's polylines. Nearby enemy health
summarizes enemies within the largest configured defense range. These are state
measurements; no rule turns them into a placement decision. Different route
families remain visible even before their enemies appear.

Enemy rows are ordered by relative progress along each enemy's assigned route.
Tanks exit their routes; soldiers are removed after their one hit at a King slot.
Unused enemy rows are zero.
If more than 64 enemies are active, retain the most advanced 64, report the
omitted count, and continue simulating everyone. Increase capacity before
training larger waves.

The separate Boolean action mask identifies legal choices and is not concatenated
to the observation. Random-generator state, complete future wave composition,
and individual projectile positions remain hidden; King-targeted projectile
damage is summarized. The next-wave count comes from the shared wave scheduler without sampling enemy
types or consuming RNG. Enemy capabilities and the payout clock help PPO learn
economy and defense timing; they do not select actions or prescribe a build order.
The policy is not recurrent and the representation is
partially observed. Fixed observation/action sizes and checkpoint guards still
require retraining for incompatible maps or entity sets; this is not a claim of
universal unseen-map performance.
