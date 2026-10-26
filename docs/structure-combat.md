# Structure combat and objectives

The Tiled map remains authoritative for routes, slot positions, and structure IDs.
The supplied map has two soldier routes, six tank routes, 46 defense slots,
six resource slots, and three King slots. Movement and combat use map coordinates
and the same seeded, 1/60-second simulation in manual play and headless RL.

## Attacks and targeting

Upstream main (`129b809`) splits enemies into `entities/enemies/` and towers
into `entities/towers/`. The Pygame UI lives in `rendering/ui.py`.

Soldiers keep following their Tiled route. Crossing any King Tower slot causes
one configured suicide hit if that King is alive, then removes the soldier even
if that King was already destroyed. Soldiers do not repeatedly attack nearby
structures and do not generate kill income when sacrificing themselves.

Tanks keep moving along their assigned routes while their independent turret
fires at the nearest living defense, resource, or King Tower. Equal-distance ties
follow the supplied order: defenses, Kings, resources, then insertion order.
Targets are recalculated each tick, preserving upstream's targeting policy.
Their effective range preserves upstream's minimum of 300 map units; larger
configured ranges are respected. Damage, cooldown, and projectile speed come
from enemy config. Reaching the endpoint removes a tank without collision damage.

Suicides and route exits count once as leak telemetry, with no legacy base-health
penalty or kill income. Damage rewards only count health actually removed by
combat; disappearing enemies cannot create positive damage reward.

## Structure damage and projectiles

All damage, including direct destruction, uses `Structure`'s shared health API.
Health changes notify Simulation immediately. Only Simulation handles resources,
slot release, destruction accounting, and win/loss decisions.

Destroyed defenses stop firing and cancel pending bursts; destroyed resource
towers stop producing, including on a tick when production would have paid out.
Their slots become available immediately. Ruins remain visible until rebuilding
replaces their entry with a new structure. Each destruction and kill counts once.

Enemy/defense attacks queue shots. Simulation collects them and advances each
projectile once per tick, then renders its shared projectile list through the
frontend. Already-fired shots survive owner destruction. Dead or removed targets
invalidate shots; hits resolve once and cannot affect a rebuilt replacement.

## Repairs and terminal outcomes

`Simulation.start_king_tower_repair(id)` validates the episode, ID, zero health,
repair progress, and affordability. Only destroyed Kings can start reconstruction; damaged living Kings are rejected.
Another King must remain alive. Rejection returns False without spending. Successful repair charges
once. The entity manages gradual healing only; incoming damage is not erased,
and lethal damage cancels its repair.

`[king_tower]` uses upstream balance: 500 HP, 50 resources, eight seconds, scale 1.
King positions and count still come from Tiled. Click a King to repair it in
manual play; Q/W select defenses, E selects resource construction, R resets.

Destroying the final King immediately loses the episode and cancels all repairs.
No remaining combat, production, or healing runs after that event. One or two
King destructions leave the default three-King episode running. Clear every
configured wave with a surviving King to win. Terminal outcomes are immutable.

Legacy difficulty `base_health` remains accepted but does not determine defeat.
`GameState.base_health` is combined King HP. `total_base_damage` records actual
King damage independently of healing, so repair cannot create damage rewards.
Leaks and their historical damage are telemetry only.

## RL compatibility and checks

RL schema 7 includes learned resource construction and repairs, normalized state,
enemy-capability and next-wave observations, a hierarchical PPO policy with
learned kind/target choices, mastery-based
curriculum with an observed goal, outcome-based validation, and condensed forced waits. The current map has 1462
observations and 102 actions. Gameplay rules and combat balance remain as above.
Earlier checkpoints require fresh training. See `concept.md` for the complete RL
state/reward contract and the commands-only README for training and evaluation.

Tests cover all structure types, both tanks, nearest-target selection, cooldowns,
projectile invalidation and owner death, slot rebuilding, repair eligibility,
cost and progress, map-defined King counts, immediate defeat, terminal priority,
route exits, passive income, reset determinism, rendering, Gymnasium, and PPO save/evaluation.

Upstream economy is retained: Resource Towers cost 50 and pay 50 every ten
seconds; passive income pays 50 every 15 seconds. Waves use upstream exponential
enemy-count growth and 15/10-second preparation/inter-wave delays.
