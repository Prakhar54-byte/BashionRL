# Structure combat, repairs, and objectives

All combat/economy progression uses `Simulation.tick()` at 1/60 second, in both
manual play and the RL environment. Structure health lives in `Structure`;
resource spending, production credits, slot cleanup, and win/loss decisions live
in the simulation.

## Targeting and attacks

- Living enemies retain a target while it is alive, registered in the world, and
  in attack range. They pause route movement throughout the engagement, including
  attack cooldowns. A destroyed, removed, or out-of-range target is released on
  the next update; the enemy reacquires or resumes movement.
- New targets are ordered by `[combat].target_priority` in `configs/game.toml`:
  defense, resource, then King. Within a type, choose minimum squared distance,
  then lexicographic structure ID. Input list order does not break ties.
- Soldiers apply immediate damage when their attack cooldown expires. Tanks aim
  their gun independently and launch shared `Projectile` instances. Damage,
  interval, range, and projectile speed are validated configuration fields.
- Projectiles hit once. Dead or removed targets invalidate a shot before impact.
  Destroyed defenses clear their shots, and dead enemies' pending shots leave
  play with their owner. The simulation stops processing immediately on defeat.
- Tanks currently reuse the Advanced Soldier sprite, with a drawn gun/projectile
  cue; no new artwork is required.

## Structures and resources

- Every structure uses `take_damage()`, `health`, `max_health`, `alive`, and
  `destroyed`. Damage clamps at zero; repeated hits on a destroyed structure do
  nothing. Health changes notify the simulation immediately.
- Destroyed defenses are removed from the active tower list, release their slot,
  and stop firing. Rebuilding starts a new destruction lifecycle for the slot.
- Configured resource towers produce the configured amount after each interval.
  They return production to the simulation for credit, and stop immediately on
  destruction, including when damage occurs on the same tick as production.
- King/resource towers are configured at fixed positions and reset each episode.
  Health bars show damage; ruined King/resource structures remain visible in gray.
  Removed defenses expose the empty slot. All structure drawing APIs also handle
  direct drawing of a destroyed structure.

## King repairs

Call `Simulation.repair_king_tower(structure_id)` for gameplay. The simulation
rejects unknown IDs, terminal episodes, insufficient funds, full-health towers,
and repairs already in progress. A destroyed King additionally requires another
living King. Validation precedes charging; an accepted repair charges once.

`KingTower.start_repair()` and `update()` only manage local progress. Over the
configured duration, healing adds the health missing at repair start. Damage
received during repair is not erased; lethal damage cancels repair. A destroyed
King becomes alive after the first positive healing tick. Final defeat cancels
every pending repair, so none can resurrect the episode afterward.

In manual play, number-row keys 1–9 repair Kings in configuration order. The
default three Kings each cost 100 resources and take five seconds to repair.
The RL action space remains defense placement plus waiting; no repair action
has been added to the policy.

## Authoritative objective

Defeat occurs in the destruction callback when **all configured King Towers**
are destroyed. The count comes from configuration, not a hardcoded value. Loss
does not wait for the next tick or wave. Victory requires clearing all configured
waves with a living King. Neither terminal result can overwrite the other.

The legacy leak counter records a route-end arrival once per enemy, including its
historical `leak_damage` statistic; it no longer damages a separate base or causes
defeat. Arriving enemies remain active, approach surviving Kings if necessary,
and keep fighting. They remain valid defense/projectile targets and can be
credited as kills exactly once. Thus waves cannot clear while endpoint attackers
are still alive.

`GameState.base_health` and `max_base_health` now report the sum of current/max
King health for the HUD and RL metrics. The legacy difficulty `base_health` TOML
field remains accepted for configuration compatibility but does not control the
King objective. King HP comes from each `[[king_towers]]` entry.

## Compatibility and checks

The new combat rules, tank type encoding, and objective require checkpoint schema
3 and retraining. Existing checkpoint files are preserved; new run commands use
`runs/baseline-v3` and `runs/smoke-v3`.

Tests cover damage on all structure types, sticky priority/ties, cooldowns,
projectile travel and invalidation, production shutdown, rebuilding, repairs and
their cost/progress/rejections, configurable King counts, immediate loss,
terminal-result protection, deterministic combat, and manual rendering/repair
controls. The PPO/Gymnasium smoke checks remain part of the test suite.
