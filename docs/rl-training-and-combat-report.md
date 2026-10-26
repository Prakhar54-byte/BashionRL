> Sections 1–7 report historical experiments before upstream main `129b809`.
> The synced game now uses soldier suicide hits, moving tanks, passive income,
> cheaper Resource Towers, exponential wave growth, and RL schema 7 (1462 observation values).
> King repairs now start only at zero health while another King survives.
> Previous checkpoints remain on disk but cannot resume against this game.
> Current mechanics are documented in `structure-combat.md` and `concept.md`.
> Section 8 reports the schema-7 pilot against the updated game.

# BastionRL: training results, strategy, combat, and possible failure paths

Original report date: 2026-10-24; schema-7 pilot added 2026-10-25.
The earlier experiments in sections 1–7 did not demonstrate a winning policy.

## 1. Training and evaluation results

### Latest paired evaluation

Both runs used training seed 0 and finished at 50,176 actual transitions.
Their selected checkpoints were evaluated on the same 20 baseline episodes,
seeds 60000–60019, with `--max-steps 2400` and the full ten-wave objective.

| Metric | Previous flat Maskable PPO | Hierarchical Maskable PPO |
|---|---:|---:|
| Selected checkpoint step | 50,000 | 30,000 |
| Training environments | 1 | 4 |
| Validation episodes per checkpoint | 10 | 5 |
| Full-game wins | 0 / 20 | 0 / 20 |
| Mean waves completed | 4.20 | 2.25 |
| Mean enemies killed | 41.30 | 20.60 |
| Mean endpoint arrivals / leaks | 38.60 | 13.90 |
| Mean final combined King HP | 0 | 0 |
| Mean defenses built | 15.30 | 3.80 |
| Mean resource towers built | 0 | 0 |
| Mean repairs started | 0 | 9.30 |
| Mean resources spent | 1,995 | 895 |
| Mean cumulative King damage | 3,000 | 6,657.67 |
| Mean shaped episode reward | 1.576 | -1.412 |
| Wait selection across all decisions | 97.72% | 94.04% |
| Defense selection when a defense was legal | 3.11% | 97.44% |

**Finding:** the hierarchical policy changed behavior and learned repairs, but
cleared fewer waves in this pilot. Its lower leak count is not evidence of better
defense: it also reached fewer waves. A high defense selection rate when affordable
does not mean it built enough defenses; it had far fewer affordable opportunities.
Different worker counts and validation schedules mean this is not an isolated
architecture comparison. One training seed and 20 test games are limited evidence.

Artifacts, relative to the project root:

- Previous checkpoint: `runs/rl-v5-ppo/best_model.zip`.
- New checkpoint: `runs/hierarchical-v1-20261024-144723/best_model.zip`.
- Reports: `runs/hierarchical-v1-20261024-144723/comparison-flat.json`,
  `comparison-hierarchical.json`, and `comparison.json`.
- New run commands and logs: `command.json`, `training.log`, and `evaluations.json`
  in the same hierarchical run directory. Training took approximately 548 seconds.

### Earlier baseline results

| Run | Evaluation games | Starting test seed | Win rate | Mean waves completed | Mean kills |
|---|---:|---:|---:|---:|---:|
| `baseline-combat-v3` | 100 | 40000 | 0% | 1.22 | 12.01 |
| `rl-v4-fresh-20261024-0951` | 100 | 40000 | 0% | 2.95 | 29.20 |

These are historical results, not controlled comparisons with the latest pair:
observation, reward, curriculum, and checkpoint contracts changed between versions.
The previously reported pre-structure-combat medium/hard results used older rules
and do not establish that the current objective is solved.

Your latest flat-policy console validation reported 3.60 waves and zero wins.
That used validation scenarios, whereas the 4.20 figure above comes from the
separate paired test set. They are different measurements.

## 2. RL strategy currently implemented

- **Hierarchical Maskable PPO:** PPO remains the optimizer; the action policy changed.
  First choose `wait`, `defense`, `resource`, or `repair`; then choose a legal target
  within that kind. Both choices are learned from normalized game state.
- **Joint learning:** `P(action) = P(kind | state) × P(target | kind, state)`.
  PPO trains joint log probabilities and joint entropy. Deterministic play selects
  the highest-probability legal kind, then its highest-probability legal target.
  Evaluation and the Pygame viewer use this same selection rule.
- **Why change the flat policy:** a total defense probability of 0.68 may be spread
  over many slots. One wait action can exceed every individual slot probability,
  making flat argmax repeatedly wait. The earlier bias-only change did not remove
  this structural issue.
- **No scripted gameplay:** no preferred placement coordinates, automatic repairs,
  demonstrations, army generation, or tactical fallback. Masks enforce legality.
- **Network and defaults:** separate actor/critic MLPs, two 128-unit layers each;
  256 rollout steps per worker; batch size 64; ten PPO epochs; learning rate 0.0003;
  gamma 0.999; GAE lambda 0.98; entropy coefficient 0.02; clip range 0.2; target KL 0.03.
- **State:** schema 5, 1,432 bounded float32 values: global economy/progression and
  current wave goal, tower configurations, defense/resource slots, individual King
  health and threats, and 64 padded enemy rows. There are 102 concrete actions:
  one wait, 92 defense placements, six resource placements, and three King repairs.
- **Curriculum:** goals progress 1 → 2 → … → 10 after at least 40 successes in the
  latest 50 episodes at the current goal. Changes apply on reset. No forced jump
  occurs based on elapsed budget. The pilot finished at the three-wave stage;
  all validation and test episodes still required ten waves.
- **Timing:** simulation runs at 60 fixed ticks/second. Decisions normally span
  30 ticks. When only wait is legal, unavoidable waits are condensed until another
  choice is available. Discounting is per recorded transition. The configured
  2,400-unit horizon is 1,200 simulated seconds, including condensed waits.
- **Training operations:** independently seeded CPU workers via `--n-envs`, fixed
  validation seeds, and best-model ranking by wins, then waves, then King health.
  `latest.zip` supports recovery; `final.zip` saves the final updated weights.
  `nohup` keeps a job running after logout; it does not improve the learning rule.
  GPU use requires a compatible PyTorch installation; the project defaults to CPU.

### Reward currently used

| Event | Reward |
|---|---:|
| Remove one full enemy-health fraction | +0.05 |
| Kill an enemy | +0.025 |
| Complete a wave | +0.1 |
| King damage | −actual damage / initial combined maximum King HP |
| Win | +2.5 |
| Lose or hit the time limit | −1 |
| Invalid action | −0.005 |
| Spend resources | −0.001 × spending / cheapest defense cost |

There is no passive income, waiting, survival, or healing reward. Actual damage
after healing is penalized again. Currency earned from kills is separate from
the numerical RL kill reward.

## 3. Attack and defense logic

### Enemy attacks

- The wave director automatically spawns enemies; RL does not generate an army.
  Baseline starts with 200 resources and five enemies in wave one, adding two
  enemies per wave. Spawns are one second apart; light tanks unlock at wave two
  and heavy tanks at wave three. Stronger-type probability grows with progression.
  Baseline enemy HP does not grow between waves; medium/hard configure HP growth.
- Soldiers use one of two map routes; tanks use one of six. Routes are selected
  by seeded randomness. Enemies only acquire structures already within attack range.
- Acquisition priority is **Defense Tower → Resource Tower → King Tower**, then
  nearest distance, then Tiled structure ID. Enemies keep their current target
  while it remains alive, registered, and in range, even if another candidate appears.
- Attacking pauses route movement, including during cooldown. Soldiers apply direct
  damage; tanks aim a separate gun and launch projectiles. Movement resumes when
  the target becomes invalid.
- At a route endpoint, enemies remain alive and targetable. An arrival increments
  leak telemetry once; it does not remove the enemy or directly cause defeat.
  Endpoint enemies approach the nearest surviving King, retaining that objective
  while valid. This final approach is straight-line movement without terrain collision.

| Enemy | HP | Attack damage | Attack interval | Attack range | Kill income |
|---|---:|---:|---:|---:|---:|
| Walking Soldier | 100 | 10 | 1.0 s | 85 | 25 |
| Advanced Soldier | 180 | 15 | 0.8 s | 85 | 40 |
| Light Tank | 250 | 25 | 2.0 s | 180 | 50 |
| Heavy Tank | 500 | 40 | 2.5 s | 180 | 75 |

### Defenses and economy

- RL chooses what and where to build; towers automatically select the nearest
  living enemy in range. RL does not control aiming, target priority, or firing.
- Cannon: cost 100, HP 250, range 180, damage 25 per projectile, one-second cooldown.
- Double Cannon: cost 150, HP 250, range 220, damage 20 per projectile; two-shot
  bursts separated by 0.12 seconds, then a 0.5-second cooldown after the burst.
- Resource Tower: cost 125, HP 250, produces 10 resources every 15 seconds while
  alive. It does not attack. With no other benefit and no destruction, production
  first covers its purchase cost after 13 payouts, approximately 195 seconds.
- Destroyed defenses stop firing; destroyed resource towers stop producing.
  Their slots become available for rebuilding. Current controls do not repair or
  upgrade those towers. Already-fired shots can survive the shooter's destruction;
  shots with dead or removed targets are discarded safely.

### King objectives and repairs

- The map defines three Kings, each with 1,000 HP. Kings are objectives and repairable
  structures; they do not fire. Combined King HP is the reported base health.
- A repair costs 50 once and restores missing HP gradually over eight seconds.
  Incoming damage still applies and lethal damage cancels repair. A destroyed King
  can begin repair only while another King remains alive and the episode is running.
- Reject full-health, already-repairing, unaffordable, unknown-ID, and terminal repairs.
- Destruction of the final surviving King immediately loses the episode. Repair
  cannot reverse terminal defeat. Clearing all ten waves with a surviving King wins.
- Simulation owns spending, shared damage callbacks, kill/destruction accounting,
  and terminal decisions. Entities handle their local combat or repair behavior.

## 4. Possible paths causing failure

The observations below are measured; their causal explanations still need tests.

| Possible cause | Evidence or mechanism | Next verification path |
|---|---|---|
| Limited full-game training exposure | The new pilot stopped at the three-wave curriculum stage, but tests require ten. All training workers use the current curriculum goal. | Compare the current curriculum with training that also includes full-task episodes; track performance by wave goal. |
| Repairs consume the defense budget | 9.3 repairs cost 465 resources/game, about 52% of the new policy's average spending. It built 3.8 defenses versus 15.3 previously. | Inspect purchase timelines and compare defensive value gained per repair; do not force repairs or reward spending itself. |
| Economy is not learned | Both selected policies built zero resource towers. Production needs about 195 seconds to repay the purchase and can be interrupted by destruction. | Check survival and return of early resource investments in longer training episodes, using controlled reward/credit-assignment experiments. |
| Repair credit is difficult | New policy sustained 6,657.67 cumulative damage versus 3,000 without repairs; average King-damage reward was −2.219 versus −1.0. | Compare reward variants that preserve a clear win objective while avoiding incentives to farm healing; measure waves and wins, not repair count. |
| Sparse winning experience | Neither selected policy won a full test game; many training successes concern shorter objectives. | Measure when full-task successes first appear across longer runs and multiple seeds. |
| Placement and automatic targeting may be weak | Defenses face multiple routes and attack the nearest enemy, not necessarily the most dangerous. Tanks can attack Cannons from the same configured range. | Replay deaths with slot, route, target, and tower-lifetime telemetry; confirm whether coverage or combat trades cause the losses. |
| Early tower destruction can collapse income | Enemies prioritize in-range defenses. Losing towers reduces kills, which reduces the currency available to rebuild. | Trace tower deaths → kill income → purchase affordability across failed waves. |
| Long-term credit uses uneven transition durations | Forced waits can span more time than ordinary decisions, but gamma applies once per recorded transition. | Compare against fixed-duration decisions or explicitly time-aware discounting at equal simulated-time and sample budgets. |
| Partial state and limited memory | The MLP is not recurrent; future wave composition and most projectile detail are hidden, and enemy rows are capped at 64. | Check omitted-enemy counts and failure replays before testing memory or a richer representation. |
| Balance or terrain may be limiting | Current tests do not establish that all ten waves are winnable with the available slots and economy. | Verify a successful manual playthrough or independent diagnostic strategy. Keep that diagnostic separate from RL gameplay. |
| Insufficient samples or unstable learning | This is one 50k-step pilot; changing workers also changes rollout batch size and update timing. | Compare several seeds with matched budgets and validation schedules. A CPU server can accelerate these experiments; more hardware alone is not an established fix. |

The previous flat policy's wait-selection problem is confirmed, and the hierarchical
decision rule addresses that specific mechanism. It does not establish that the
new policy's placements, repair timing, or economy are strategically good.

## 5. Verification and limits

- Latest full automated verification: **141 tests passed; Ruff clean**.
- Tested hierarchical probabilities, masks, entropy and gradients, PPO updates,
  save/load, parallel workers, checkpoint recovery/resume, and game rules.
- A saved hierarchical model rendered 600 dummy-display frames with 600 physics ticks.
- No current full-game winning policy, successful human playthrough, GPU benchmark,
  or long-budget multi-seed performance study has been established.
- This report is documentation only; it does not change training or combat behavior.

Source files: `src/game/rl/{policy,environment,curriculum,training,__main__}.py`,
`src/game/simulation.py`, `src/game/entities/{enemy,defense_tower,resource_tower,king_tower,projectile}.py`,
`src/game/systems/wave_director.py`, and `configs/game.toml`.

## 6. Experiment corrections before further architecture changes

### Matched comparison controls

- CLI training now accepts `--strategy flat|hierarchical`, `--n-steps`,
  `--batch-size`, and `--n-epochs`. The flat control retains its previous initial
  action-kind bias; hierarchical PPO retains the existing kind/target head.
- Default training budget is now 2,000,000 transitions. Validation defaults to
  20 games; the matched server commands explicitly use 30 fixed scenarios.
- The server comparison uses both policies, seeds 0/1/2, 16 workers × 256 steps
  = 4,096 samples per rollout, batch size 64, and at most ten epochs. Every case
  has the same 2M requested budget, validation seeds 10000–10029, and validation
  frequency of 65,536 transitions. Final testing uses 100 episodes from seed 90000.
  Run cases sequentially, rather than starting six sets of 16 workers together.
- A second matched layout is 1 worker × 1024 versus 4 workers × 256: both collect
  1,024 samples per rollout. Smaller worker counts are available for local hardware.
  For an architecture-only comparison, keep both worker count and per-worker
  rollout length identical, as in the server comparison; matching only their
  product still changes trajectory length and rollout collection.
- `training-summary.json` records actual transitions, completed rollouts, completed
  PPO epochs, and actual optimizer-step invocations. KL early stopping can change
  optimizer counts even when the maximum training schedule is matched.
- Validation/checkpointing now happens after completed PPO updates. Non-aligned
  validation frequencies round up to a rollout boundary. Saved metadata includes
  actual checkpoint steps; recovery commands subtract those from the original
  target instead of adding a new full budget. Summary files identify resumed runs;
  resetting curriculum/RNG state still makes them different from uninterrupted runs.
- The original 49 versus 196 rollout iterations were confounded, but do not alone
  establish four times fewer gradient steps. With batch size 64 and ten full
  epochs, their theoretical minibatch-step counts are both 7,840:
  `49 × 10 × 16` versus `196 × 10 × 4`. Actual counts were not previously recorded.
- Commands are in README. The six long 2M-step runs have not been executed;
  server execution details are still needed. The short automated matched-rollout
  test verifies configuration/counting and makes no performance claim.

### New metrics and archived-model audit

- `wait_rate_when_other_action_legal` excludes decisions where only waiting was
  legal. `forced_wait_decisions` counts those decision points separately.
  `condensed_wait_ticks` measures additional simulation time inside a transition.
- Re-evaluating the same archived models on the same 20 test seeds found **zero
  forced-only decision points** for both policies. Corrected conditional wait rates
  remain **97.72% flat** and **94.04% hierarchical**. Existing forced-wait condensation
  explains why this correction does not change their rates in these episodes.
- Condensed time was still substantial: 83,198 ticks for flat and 378,748 ticks for
  hierarchical, across those 20 games. Elapsed transition ticks are now recorded
  for a future time-aware-discount experiment; discounting itself is unchanged.
- Audit files: `runs/wait-metric-audit-20261024-173836/{flat,hierarchical}.json`.
- `training-goals.json`, `goals/*` logs, and `training-episodes.jsonl` separate
  success by episode wave goal and store structure-death timelines and unspent
  resources at each wave end. `waves/*_unspent_resources` provides scalar logs too.

### Winnability probe results

The isolated scripted diagnostic uses ordinary placement costs, damage, King
repairs, map routes, and the full ten-wave objective. It has no extra resources,
prebuilt towers, changed enemies, or connection to PPO action selection.

| Diagnostic, seeds 80000–80009 | Wins | Mean waves cleared | Mean resource towers built |
|---|---:|---:|---:|
| Route/threat coverage, defense only | 0 / 10 | 1.70 | 0 |
| Same policy with an early economy option | 0 / 10 | 2.00 | 0.60 |

Artifacts: `runs/winnability-v2-20261024-174113/`. An earlier weaker diagnostic
overweighted tank routes before their unlock; its failed results are preserved
in `runs/winnability-20261024-173710/`. The current probe weights unlocked route
families and saves for its preferred damage-per-cost tower.

**Winnability remains unverified.** These failures do not establish that the game
is impossible, nor does the small economy comparison establish resource towers'
value. A demonstrated winning manual strategy or stronger independent diagnostic
is still needed before attributing ten-wave failures solely to RL.

### Subsequent ablations, not applied in this change

1. Establish matched long-budget, multi-seed results and a successful winnability
   reference before drawing an architecture verdict.
2. Test a 60% current / 20% easier / 20% full-goal curriculum, then a separate
   randomized mid-game-start experiment; compare a 30/50 promotion threshold.
3. Separately test entity pooling, shared slot scoring, and type-then-slot choices.
4. Separately test repair masking, reward/observation normalization, and discounting
   by `gamma ** (elapsed_ticks / 30)`. Normalization requires saved statistics and
   identical preprocessing in evaluation and the viewer.
5. Potential-based health shaping must use the same effective transition discount
   as the return calculation, with consistent terminal/truncation treatment. Its
   policy-invariance claim applies to adding a valid shaping term to a fixed base
   reward; replacing the current damage penalty changes that base objective.
   See [Ng, Harada, and Russell's shaping theorem](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

Combat, curriculum thresholds, observations, reward weights, and the hierarchical
architecture remain unchanged in this experiment-control change.

## 7. Optional mixed-goal training preset (2026-10-25)

This is a new combined-settings experiment, separate from the matched architecture
control above. It does not establish that any individual setting improves wins.

- Fresh command: 8 SubprocVecEnv workers × 512 steps = 4,096 samples per rollout;
  minibatch 256, at most ten epochs, 2M requested transitions, 30 validation games.
- `--lr-final 0.00005`: linear learning-rate decay from 0.0003.
- `--ent-final 0.005`: linear entropy-coefficient decay from 0.02.
- `--curriculum mixed`: 60% current / 20% easier / 20% full-goal episodes, sampled
  independently at reset. Endpoint categories merge when no easier goal exists or
  the current goal is already the full task. Promotion uses current-goal results.
- `--promotion-successes 30`: configurable 30 successes out of 50; staged controls
  still default to 40/50.
- `--normalize`: observation and reward VecNormalize statistics are trained and
  saved with every checkpoint. Evaluation and the viewer use frozen observation
  statistics and raw game rewards. An older raw-observation model cannot be resumed
  directly into this preprocessing contract; use a fresh run.
- New checkpoints persist curriculum goal/window and absolute annealing plan.
  Resume restores these and the saved seed, while worker RNG/episode state restarts.
  Legacy checkpoints lacking curriculum state restart at goal 1 with a message.
- Native SB3 loading already restores Adam state. A regression test verifies the
  optimizer's step counters and moving moments survive save/load; the earlier
  suggestion that curriculum reset also resets Adam was incorrect.
- Tower-death events now identify Cannon/Double Cannon, and wave-end snapshots
  include each type's built/alive counts. This enables testing the wave-three
  tank hypothesis instead of assuming it from the unlock thresholds.

Verification: a fresh eight-worker 4,096-transition smoke run saved and evaluated
successfully, including normalization sidecars and curriculum state, at
`runs/mixed-normalized-smoke-20261025-053037/`. This exercises the pipeline, not
long-run learning performance. The full automated suite passed 141 tests.
No server process was stopped or restarted from this workspace; deployment and
the multi-seed 2M-step runs still require execution on the server.


## 8. Upstream sync and destroyed-only King reconstruction (2026-10-25)

The existing `rl-implementation` branch includes original upstream main `129b809`
via merge `fde0222`. The local integration retains its new entity packages,
extracted UI, map, moving tanks, soldier suicide hits, passive economy, cheaper
Resource Towers, and exponential waves. Earlier results above describe a
substantially different game; they are not a controlled comparison with this run.

King reconstruction now starts only at zero health while another King survives.
Living damaged Kings are rejected by the entity, simulation, manual clicks, and
RL mask. Reconstruction still costs 50 once and restores health gradually over
8 seconds. Defeat cannot be reversed. This is the user's new constraint,
superseding the earlier damaged-King repair rule in the historical report.

The learner remains hierarchical Maskable PPO with no scripted gameplay. Schema
7 has 1462 bounded observation values and 102 actions. It additionally exposes
the passive payout clock, next-wave count from the shared scheduler, and configured
enemy capabilities before spawning. No RNG is consumed to build these features.
Actual enemy damage is accumulated through the damage API so route exits and
suicide removals cannot create positive damage reward.

A fresh CPU pilot used seed 0, 10240 additional/fresh steps, 2 subprocess workers,
512 steps per worker (1024 samples/update), minibatches of 256, 5 PPO epochs,
mixed goals with 30/50 promotion, and observation/reward normalization. Learning
rate decayed from 3e-4 to 5e-5 and entropy from 0.02 to 0.005 over this short pilot.
This compressed schedule is for the pilot; the server command uses a 2M-step plan.

| Checkpoint | Episodes | Win rate | Mean waves cleared | Mean remaining King HP |
|---|---:|---:|---:|---:|
| Best at 5120 steps; validation seeds 10000–10002 | 3 | 33.3% | 7.67 | 381.67 |
| Final at 10240 steps; same validation seeds | 3 | 0% | 3.00 | 0 |
| Best checkpoint; held-out seeds 90000–90009 | 10 | 60% | 8.50 | 739.58 |

Artifacts: `runs/main-rebuild-pilot/{best_model,latest,final}.zip`, matching JSON
and VecNormalize sidecars, training log/summary/episode telemetry, and
`evaluation-heldout.json`. Use the best checkpoint for viewing; the final model
is not the strongest observed policy. Six held-out wins establish that the
updated game is winnable by the learned agent. A short single-seed pilot with
three validation games cannot establish reliable performance or attribute gains
to architecture rather than the upstream gameplay changes. Run longer training
with several seeds and separate held-out evaluation before judging robustness.

Validation: 158 automated tests pass; Ruff and whitespace checks pass. Tests
cover the zero-health-only rule, immediate terminal loss, gradual restoration,
one-time spending, observation/action validity, deterministic state, actual-damage
reward accounting, parallel PPO training, and normalized checkpoint restoration.
CUDA and server execution were not tested locally.
