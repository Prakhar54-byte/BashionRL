import json

import numpy as np
import pytest

from src.game.rl.checkpoint import load_model, save_metadata
from src.game.rl.environment import BastionEnv
from src.game.simulation import Simulation


@pytest.mark.parametrize("kind", ["soldier", "tank"])
def test_departing_enemies_never_earn_combat_damage_or_kill_reward(kind):
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim._spawn_enemy("enemy1" if kind == "soldier" else "light_tank")
    enemy = sim.enemies[0]
    if kind == "soldier":
        enemy.position = sim.king_towers[0].position.copy()
    else:
        enemy.position = enemy.pathway.points[-1].copy()
        enemy.waypoint_index = len(enemy.pathway.points)
        enemy.reached_end = True
    _, _, _, _, info = env.step(0)
    assert enemy not in sim.enemies
    assert info["kills"] == 0 and info["leaks"] == 1
    assert info["reward_terms"]["enemy_damage"] == info["reward_terms"]["kills"] == 0
    assert sim.total_enemy_damage == 0


def test_passive_income_clock_is_observed_and_forced_wait_stops_at_affordability():
    env = BastionEnv()
    before, _ = env.reset(seed=0)
    after, _, _, _, _ = env.step(0)
    assert before[13] == 0
    assert after[13] == pytest.approx(0.5 / env.simulation.config.passive_income_interval)
    env.reset(seed=0)
    env.simulation.state.resources.spend(200)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert info["elapsed_ticks"] == 900
    assert info["resources"] == env.simulation.config.passive_income_amount == 50
    assert env.action_masks()[env.resource_action_start:env.repair_action_start].any()
    env.reset(seed=0)
    assert env.simulation._passive_income_elapsed == 0


def test_tank_keeps_moving_while_attacking_and_observation_reports_actual_range():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    king = sim.king_towers[0]
    king.position = tank.position + (100, 0)
    tank.update_combat(0, [], king_towers=[king])
    position = tank.position.copy()
    tank.update(sim.timestep)
    assert tank.target is king and tank.position != position
    rows = env.observation()[env.enemy_observation_start:].reshape(env.max_enemies, env.enemy_features)
    assert rows[0, 9] == pytest.approx(tank.attack_range / env.map_diagonal)
    assert tank.attack_range == max(300, tank.config.attack_range)
    assert np.isfinite(rows).all()


def test_upstream_wave_growth_and_economy_are_preserved():
    sim = Simulation()
    assert [sim.director._calculate_enemy_count(i) for i in (1, 2, 3, 4)] == [5, 7, 10, 14]
    assert sim.config.window_width == 1280 and sim.config.window_height == 720
    assert sim.config.king_tower_max_health == 500
    assert sim.config.resource_tower.cost == 50
    assert sim.config.resource_slot_path.name == "resource_slot.png"


def test_previous_schema_cannot_resume_against_updated_game(tmp_path):
    env = BastionEnv()
    path = tmp_path / "old.zip"
    save_metadata(path, env, seed=0, steps=1)
    data = json.loads(path.with_suffix(".json").read_text())
    data["environment"]["schema"] = 5
    path.with_suffix(".json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="environment differs"):
        load_model(path, env)


@pytest.mark.parametrize("health", [500, 375, 1])
def test_living_kings_cannot_be_repaired_by_entity_simulation_or_rl(health):
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    king = sim.king_towers[0]
    king.take_damage(king.health - health)
    funds = sim.state.resource_balance
    assert not king.start_repair()
    assert not sim.start_king_tower_repair(king.structure_id)
    from src.game.rendering.ui import _handle_world_click

    _handle_world_click(sim, king.position.x, king.position.y, "cannon", "defense")
    assert not env.action_masks()[env.repair_action_start]
    _, _, _, _, info = env.step(env.repair_action_start)
    assert info["invalid_action"] and info["repairs_started"] == 0
    assert king.health == health and not king.repairing
    assert sim.state.resource_balance == funds


def test_destroyed_king_reconstruction_is_gradual_and_cannot_be_restarted():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    king = sim.king_towers[0]
    king.destroy()
    assert env.action_masks()[env.repair_action_start]
    _, _, _, _, info = env.step(env.repair_action_start)
    assert not info["invalid_action"] and info["repairs_started"] == 1
    assert 0 < king.health < king.max_health and king.repairing
    assert not env.action_masks()[env.repair_action_start]
    _, _, _, _, info = env.step(env.repair_action_start)
    assert info["invalid_action"]
    assert sim.state.resources.total_spent == king.repair_cost


def test_next_wave_count_and_enemy_capabilities_are_observed_before_spawn():
    env = BastionEnv()
    observation, _ = env.reset(seed=0)
    assert not env.simulation.enemies
    assert observation[14] == pytest.approx(0.5)
    start = env.global_features + len(env.simulation.tower_types) * env.tower_features
    rows = observation[start:start + len(env.simulation.enemy_types) * env.enemy_type_features]
    rows = rows.reshape(len(env.simulation.enemy_types), env.enemy_type_features)
    assert rows[:, -1].tolist() == [0, 0, 1, 1]
    assert (rows[:, :6] > 0).all()
    env.simulation.director.wave_number = 2
    rng_state = env.simulation.director.rng.getstate()
    assert env.observation()[14] == pytest.approx(10 / 15)
    assert env.simulation.director.rng.getstate() == rng_state
    env.simulation.director.wave_number = env.simulation.balance.max_waves
    assert env.observation()[14] == 0
