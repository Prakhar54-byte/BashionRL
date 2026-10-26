import json
from dataclasses import replace

import pygame
import pytest

from src.game.config import KingTowerConfig, load_config
from src.game.entities.enemies import EnemyConfig
from src.game.entities.towers.projectile import Projectile
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


@pytest.fixture
def sim():
    simulation = Simulation(seed=7)
    simulation.state.resources.earn(1000)
    return simulation


def build(sim, kind):
    if kind == "defense":
        assert sim.place_tower(0, "cannon")
        return sim.towers[-1]
    if kind == "resource":
        assert sim.place_resource_tower(0)
        return sim.resource_towers[-1]
    return sim.king_towers[0]


@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_shared_health_destruction_and_rebuilding(sim, kind):
    target = build(sim, kind)
    assert target.take_damage(25) == 25
    assert target.health == target.max_health - 25
    for amount in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            target.take_damage(amount)
    target.destroy()
    assert target.destroyed and target.health == 0
    assert target.take_damage(5000) == 0
    target.destroy()
    assert sim.structures_destroyed == 1
    if kind != "king":
        slots = sim.defense_slots if kind == "defense" else sim.resource_slots
        assert not slots.slots[0].occupied
        rebuilt = build(sim, kind)
        assert rebuilt is not target and rebuilt.alive
        rebuilt.destroy()
        assert sim.structures_destroyed == 2
        assert not slots.slots[0].occupied


@pytest.mark.parametrize("destroyed", [False, True])
def test_soldier_slot_collision_is_one_suicide_hit_without_kill_income(sim, destroyed):
    king = sim.king_towers[0]
    if destroyed:
        king.destroy()
    health = king.health
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = king.position.copy()
    balance = sim.state.resource_balance
    sim.tick()
    sim.tick()
    assert enemy.suicide_attacked and not enemy.alive and enemy not in sim.enemies
    assert king.health == max(0, health - enemy.config.attack_damage)
    assert sim.state.enemies_killed == 0 and sim.state.enemies_leaked == 1
    assert sim.state.resource_balance == balance



def test_tank_selects_nearest_living_structure_and_resumes_route(sim):
    defense = build(sim, "defense")
    resource = build(sim, "resource")
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    tank.position = pygame.Vector2(500, 500)
    defense.position = tank.position + pygame.Vector2(100, 0)
    resource.position = tank.position + pygame.Vector2(50, 0)
    tank.update_combat(0, [defense], resource_towers=[resource])
    assert tank.target is resource
    resource.destroy()
    tank.update_combat(0, [defense], resource_towers=[resource])
    assert tank.target is defense
    defense.position = tank.position + pygame.Vector2(1000, 0)
    tank.update_combat(0, [defense])
    assert tank.target is None
    position = tank.position.copy()
    tank.update(0.1)
    assert tank.position != position



@pytest.mark.parametrize("enemy_type", ["light_tank", "heavy_tank"])
@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_tank_gun_projectile_hit_and_cooldown(sim, enemy_type, kind):
    target = build(sim, kind)
    sim._spawn_enemy(enemy_type)
    tank = sim.enemies[0]
    tank.position = target.position - pygame.Vector2(100, 0)
    position = tank.position.copy()
    tank.update_combat(0, [target])
    assert tank.is_tank and tank.gun_angle == pytest.approx(0)
    assert target.health == target.max_health and len(tank.projectiles) == 1
    shot = tank.projectiles[0]
    shot.update(0.1, [target])
    assert shot.position != position and shot.alive
    shot.update(1, [target])
    assert not shot.alive
    assert target.health == target.max_health - tank.config.attack_damage
    shot.update(1, [target])
    assert target.health == target.max_health - tank.config.attack_damage
    tank.projectiles.clear()
    tank.update_combat(tank.config.attack_interval / 2, [target])
    assert not tank.projectiles and tank.position == position
    tank.update_combat(tank.config.attack_interval / 2, [target])
    assert len(tank.projectiles) == 1


@pytest.mark.parametrize("removed", [False, True])
def test_projectile_discards_invalid_target(sim, removed):
    target = build(sim, "resource")
    shot = Projectile(target.position, target, 250, 10)
    if removed:
        sim.resource_towers.remove(target)
    else:
        target.destroy()
    health = target.health
    shot.update(0, sim.structures)
    assert not shot.alive and target.health == health


def test_defense_shots_finish_after_owner_destruction(sim):
    tower = build(sim, "defense")
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = tower.position + pygame.Vector2(100, 0)
    sim.tick()
    assert sim.projectiles
    tower.destroy()
    count = len(sim.projectiles)
    for _ in range(30):
        sim.tick()
    assert enemy.health == enemy.max_health - tower.damage
    assert not sim.projectiles and count == 1
    tower.update(10, sim.enemies)
    assert not tower.projectiles


def test_resource_stops_before_same_tick_payout(sim):
    tower = build(sim, "resource")
    tower._production_elapsed = tower.production_interval - sim.timestep
    sim.projectiles.append(Projectile(tower.position, tower, 250, tower.health))
    balance = sim.state.resource_balance
    sim.tick()
    assert tower.destroyed and sim.state.resource_balance == balance
    assert tower.update(1000) == 0


def test_repair_eligibility_payment_progress_and_damage(sim):
    king = sim.king_towers[0]
    assert not sim.start_king_tower_repair(king.structure_id)
    assert not sim.start_king_tower_repair("unknown")
    king.take_damage(200)
    assert not sim.start_king_tower_repair(king.structure_id)
    assert not king.start_repair()
    king.destroy()
    balance = sim.state.resource_balance
    assert sim.start_king_tower_repair(king.structure_id)
    assert sim.state.resource_balance == balance - king.repair_cost
    assert not sim.start_king_tower_repair(king.structure_id)
    king.update(king.repair_duration / 2)
    assert king.repairing and king.health == king.max_health / 2
    king.take_damage(50)
    king.update(king.repair_duration / 2)
    assert not king.repairing and king.health == king.max_health - 50
    assert sim.state.resources.total_spent == king.repair_cost
    assert sim.state.base_health == sum(tower.health for tower in sim.king_towers)
    assert sim.state.total_base_damage == king.max_health + 50


def test_repair_rejects_insufficient_funds_and_terminal_state(sim):
    king = sim.king_towers[0]
    king.destroy()
    sim.state.resources.spend(sim.state.resource_balance)
    assert not sim.start_king_tower_repair(king.structure_id)
    sim.state.resources.earn(king.repair_cost)
    sim.state.mark_won()
    assert not sim.start_king_tower_repair(king.structure_id)
    assert sim.state.resource_balance == king.repair_cost and not king.repairing


def test_destroyed_king_repairs_only_while_another_survives(sim):
    king = sim.king_towers[0]
    king.destroy()
    assert sim.state.status == GameStatus.RUNNING
    assert sim.start_king_tower_repair(king.structure_id)
    assert king.destroyed
    king.update(sim.timestep)
    assert king.alive and 0 < king.health < king.max_health
    assert king.structure_id not in sim.state.destroyed_king_tower_ids
    king.destroy()
    assert not king.repairing
    king.update(100)
    assert king.health == 0


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_final_configured_king_destruction_is_immediate_and_terminal(tmp_path, count):
    config = load_config()
    data = json.loads(config.pathway_path.read_text())
    layer = next(layer for layer in data["layers"] if layer["name"] == "king_tower_slots")
    template = layer["objects"][0]
    layer["objects"] = [dict(template, id=10000 + index) for index in range(count)]
    path = tmp_path / "map.tmj"
    path.write_text(json.dumps(data))
    simulation = Simulation(config=replace(config, pathway_path=path), seed=0)
    assert simulation.state.king_tower_count == count
    for king in simulation.king_towers[:-1]:
        king.destroy()
        assert simulation.state.status == GameStatus.RUNNING
    last = simulation.king_towers[-1]
    ticks = simulation.ticks
    Projectile(last.position, last, 250, last.health).update(0, simulation.structures)
    assert simulation.state.status == GameStatus.LOST and simulation.ticks == ticks
    assert simulation.state.base_health == 0
    assert not simulation.start_king_tower_repair(last.structure_id)
    simulation.state.mark_won()
    assert simulation.state.status == GameStatus.LOST


def test_defeat_cancels_repairs_and_stops_projectiles_immediately(sim):
    first, second, last = sim.king_towers
    first.destroy()
    assert sim.start_king_tower_repair(first.structure_id)
    second.destroy()
    sim.projectiles.extend([
        Projectile(last.position, last, 250, last.health),
        Projectile(first.position, first, 250, first.max_health),
    ])
    sim.tick()
    assert sim.state.status == GameStatus.LOST and not first.repairing
    assert sim.projectiles[1].alive
    first.update(100)
    sim.advance(100)
    assert first.destroyed and sim.ticks == 1


def test_tank_exit_is_not_a_kill_and_does_not_damage_kings(sim):
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    tank.reached_end = True
    tank.position = tank.pathway.points[-1].copy()
    tank.waypoint_index = len(tank.pathway.points)
    health = sim.state.base_health
    balance = sim.state.resource_balance
    sim.tick()
    sim.tick()
    assert tank.exited_route and not tank.alive and tank not in sim.enemies
    assert sim.state.enemies_killed == 0 and sim.state.enemies_leaked == 1
    assert sim.state.base_health == health and sim.state.resource_balance == balance



def test_kills_and_overlapping_shots_are_counted_once(sim):
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = pygame.Vector2(10, 10)
    balance = sim.state.resource_balance
    sim.projectiles.extend([Projectile(enemy.position, enemy, 250, enemy.health) for _ in range(2)])
    sim.tick()
    sim.tick()
    assert sim.state.enemies_killed == 1 and sim.state.enemies_leaked == 0
    assert sim.state.resource_balance == balance + enemy.config.kill_reward



def test_each_system_advances_once_per_tick(sim):
    tower = build(sim, "defense")
    resource = build(sim, "resource")
    king = sim.king_towers[0]
    tower.cooldown = 1
    king.destroy()
    assert sim.start_king_tower_repair(king.structure_id)
    health = king.health
    intermission = sim.director.intermission_timer
    sim.tick()
    assert sim.director.intermission_timer == pytest.approx(intermission - sim.timestep)
    assert tower.cooldown == pytest.approx(1 - sim.timestep)
    assert resource.production_time_remaining == pytest.approx(resource.production_interval - sim.timestep)
    assert king.health == pytest.approx(health + king.max_health * sim.timestep / king.repair_duration)


def test_reset_and_fixed_seed_reproduce_combat():
    first, second = Simulation(seed=42), Simulation(seed=42)
    for simulation in (first, second):
        simulation.place_tower(0, "cannon")
        simulation._spawn_enemy("light_tank")
        simulation.advance(20)
    assert first.state == second.state
    assert [(e.position, e.health, e.attack_cooldown) for e in first.enemies] == [
        (e.position, e.health, e.attack_cooldown) for e in second.enemies
    ]
    first.king_towers[0].destroy()
    first.reset(seed=42)
    assert first.structures_destroyed == first.state.total_base_damage == 0
    assert not first.projectiles and not first.enemies and not first.towers
    assert all(king.health == king.max_health and not king.repairing for king in first.king_towers)


@pytest.mark.parametrize("field", ["attack_damage", "attack_interval", "attack_range", "projectile_speed"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_enemy_attack_config_validation(field, value):
    with pytest.raises(ValueError):
        EnemyConfig("test", 10, 100, **{field: value})


def test_config_keeps_map_ids_and_balance_defaults():
    config = load_config()
    simulation = Simulation(config=config, seed=0)
    assert [king.structure_id for king in simulation.king_towers] == [slot.slot_id for slot in simulation.king_tower_slots.slots]
    assert all(king.position == slot.center for king, slot in zip(simulation.king_towers, simulation.king_tower_slots.slots))
    assert config.king_tower == KingTowerConfig()
    assert config.king_tower.max_health == 500 and config.king_tower.repair_cost == 50
    assert config.king_tower.repair_duration == 8
    for values in ({"max_health": 0}, {"repair_cost": -1}, {"repair_duration": float("nan")}):
        with pytest.raises(ValueError):
            KingTowerConfig(**values)


def test_victory_and_loss_cannot_overwrite_each_other(sim):
    sim.state.mark_won()
    for king in sim.king_towers:
        king.destroy()
    assert sim.state.status == GameStatus.WON
    assert not sim.start_king_tower_repair(sim.king_towers[0].structure_id)
    sim.state.mark_lost()
    assert sim.state.status == GameStatus.WON


def test_defeat_precedes_final_wave_victory(sim):
    for king in sim.king_towers[:-1]:
        king.destroy()
    last = sim.king_towers[-1]
    last.take_damage(last.health - 10)
    sim.director.wave_number = sim.balance.max_waves
    sim.director.wave_active = True
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = last.position.copy()
    sim.tick()
    assert sim.state.status == GameStatus.LOST
    sim.state.mark_won()
    assert sim.state.status == GameStatus.LOST


def test_manual_click_uses_map_ids_for_gradual_repair(sim):
    from main import _handle_world_click

    king = sim.king_towers[0]
    king.take_damage(200)
    assert not sim.start_king_tower_repair(king.structure_id)
    assert not king.start_repair()
    king.destroy()
    balance = sim.state.resource_balance
    _handle_world_click(sim, king.position.x, king.position.y, "cannon", "defense")
    assert king.repairing and king.health == 0
    assert sim.state.resource_balance == balance - king.repair_cost
    sim.tick()
    assert 0 < king.health < king.max_health


@pytest.mark.parametrize("tower_id", [None, [], True, 138.0])
def test_malformed_repair_ids_are_rejected_without_spending(sim, tower_id):
    king = sim.king_towers[0]
    king.take_damage(100)
    balance = sim.state.resource_balance
    assert not sim.start_king_tower_repair(tower_id)
    assert sim.state.resource_balance == balance and not king.repairing
