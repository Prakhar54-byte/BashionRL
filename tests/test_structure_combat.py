from dataclasses import replace

import pygame
import pytest

from src.game.config import load_config
from src.game.entities.enemy import EnemyConfig
from src.game.entities.king_tower import KingTower
from src.game.entities.projectile import Projectile
from src.game.entities.resource_tower import ResourceTower
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


@pytest.fixture
def sim():
    return Simulation(seed=7)


@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_shared_damage_and_single_destruction(sim, kind):
    sim.place_tower(0, "cannon")
    structure = {"defense": sim.towers[0], "resource": sim.resource_towers[0],
                 "king": sim.king_towers[0]}[kind]
    assert structure.take_damage(25) == 25
    assert structure.alive and structure.health == structure.max_health - 25
    for damage in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            structure.take_damage(damage)
    structure.take_damage(10000)
    assert structure.destroyed and structure.health == 0
    assert structure.take_damage(10000) == 0
    assert sim.structures_destroyed == 1
    assert sim.state.status == GameStatus.RUNNING
    if kind == "defense":
        assert structure not in sim.towers
        assert not sim.defense_slots.slots[0].occupied
        assert sim.place_tower(0, "cannon")
        rebuilt = sim.towers[0]
        rebuilt.take_damage(10000)
        assert rebuilt not in sim.towers and not sim.defense_slots.slots[0].occupied
        assert sim.structures_destroyed == 2


def test_soldier_cooldown_pause_and_resume(sim):
    target = sim.resource_towers[0]
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = target.position + pygame.Vector2(20, 0)
    position = enemy.position.copy()
    enemy.update(0, [target])
    assert target.health == 190
    enemy.update(0.5, [target])
    assert target.health == 190 and enemy.position == position
    enemy.update(0.5, [target])
    assert target.health == 180
    target.take_damage(10000)
    enemy.update(0.1, [target])
    assert enemy.target is None and enemy.position != position


def test_priority_tie_breaking_stickiness_and_removal(sim):
    sim.place_tower(0, "cannon")
    sim.place_tower(1, "cannon")
    first, second = sim.towers
    first.position = second.position = pygame.Vector2(110, 100)
    resource, king = sim.resource_towers[0], sim.king_towers[0]
    resource.position = king.position = pygame.Vector2(100, 100)
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = pygame.Vector2(100, 100)
    structures = [king, resource, second, first]
    enemy.update(0, structures, sim.config.target_priority)
    assert enemy.target is first
    second.position = enemy.position.copy()
    enemy.update(0.1, structures)
    assert enemy.target is first
    first.take_damage(10000)
    enemy.update(0.1, structures)
    assert enemy.target is second
    enemy.update(0.1, [king, resource])
    assert enemy.target is resource
    enemy.target = None
    enemy.update(0.1, [resource, king], ("king", "resource", "defense"))
    assert enemy.target is king


@pytest.mark.parametrize("kind", ["defense", "resource", "king"])
def test_tank_projectile_travels_hits_once_and_obeys_cooldown(sim, kind):
    if kind == "defense":
        sim.place_tower(0, "cannon")
    target = {"defense": sim.towers, "resource": sim.resource_towers,
              "king": sim.king_towers}[kind][0]
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    tank.position = target.position - pygame.Vector2(100, 0)
    position = tank.position.copy()
    sim.tick()
    assert target.health == target.max_health
    assert len(tank.projectiles) == 1
    shot = tank.projectiles[0]
    assert shot.position != position
    for _ in range(30):
        sim.tick()
    assert target.health == target.max_health - tank.config.attack_damage
    assert tank.position == position and not tank.projectiles
    shot.update(1, sim.structures)
    assert target.health == target.max_health - tank.config.attack_damage


@pytest.mark.parametrize("remove", [False, True])
def test_projectiles_discard_destroyed_or_removed_targets(sim, remove):
    target = sim.resource_towers[0]
    sim._spawn_enemy("light_tank")
    tank = sim.enemies[0]
    tank.position = target.position - pygame.Vector2(100, 0)
    sim.tick()
    shot = tank.projectiles[0]
    if remove:
        sim.resource_towers.remove(target)
    else:
        target.take_damage(10000)
    health = target.health
    sim.tick()
    assert not shot.alive and not tank.projectiles
    assert target.health == health


def test_destroyed_defense_stops_firing_and_clears_projectiles(sim):
    sim.place_tower(0, "cannon")
    tower = sim.towers[0]
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = tower.position + pygame.Vector2(100, 0)
    tower.update(sim.timestep, sim.enemies)
    assert tower.projectiles
    tower.take_damage(10000)
    tower.update(5, sim.enemies)
    assert not tower.projectiles and enemy.health == enemy.max_health


def test_resource_production_stops_on_destruction_before_credit(sim):
    tower = sim.resource_towers[0]
    tower._production_elapsed = tower.production_interval - sim.timestep
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = tower.position.copy()
    tower.take_damage(tower.health - enemy.config.attack_damage)
    balance = sim.state.resource_balance
    sim.tick()
    assert tower.destroyed and sim.state.resource_balance == balance
    assert tower.update(1000) == 0


def test_resource_production_accounts_for_multiple_intervals(sim):
    tower = sim.resource_towers[0]
    assert tower.update(12) == 20
    assert tower.update(3) == 10


def test_repair_is_gradual_and_charged_once(sim):
    king = sim.king_towers[0]
    assert not sim.repair_king_tower(king.structure_id)
    king.take_damage(200)
    assert sim.repair_king_tower(king.structure_id)
    assert sim.state.resource_balance == 100
    assert king.health == 300
    assert not sim.repair_king_tower(king.structure_id)
    sim.advance(king.repair_duration / 2)
    assert king.health == pytest.approx(400)
    assert king.repairing
    sim.advance(king.repair_duration / 2)
    assert king.health == king.max_health and not king.repairing
    assert sim.state.resources.total_spent == king.repair_cost
    assert not sim.repair_king_tower(king.structure_id)
    assert sim.state.base_health == sum(other.health for other in sim.king_towers)


def test_destroyed_king_can_repair_while_another_survives(sim):
    king = sim.king_towers[0]
    king.take_damage(king.health)
    assert king.destroyed and sim.state.status == GameStatus.RUNNING
    assert sim.repair_king_tower(king.structure_id)
    assert king.destroyed
    sim.tick()
    assert king.alive and 0 < king.health < king.max_health


def test_repair_rejects_unknown_ids_insufficient_funds_and_terminal_states(sim):
    king = sim.king_towers[0]
    king.take_damage(200)
    balance = sim.state.resource_balance
    assert not sim.repair_king_tower("unknown")
    assert sim.state.resource_balance == balance
    sim.state.resources.spend(balance)
    assert not sim.repair_king_tower(king.structure_id)
    assert not king.repairing
    sim.state.resources.earn(1000)
    sim.state.mark_won()
    assert not sim.repair_king_tower(king.structure_id)
    assert sim.state.resource_balance == 1000


def test_lethal_damage_cancels_repair(sim):
    king = sim.king_towers[0]
    king.take_damage(100)
    assert sim.repair_king_tower(king.structure_id)
    king.take_damage(10000)
    assert king.destroyed and not king.repairing
    king.update(100)
    assert king.health == 0


def test_damage_during_repair_is_not_erased(sim):
    king = sim.king_towers[0]
    king.take_damage(200)
    assert sim.repair_king_tower(king.structure_id)
    king.update(2.5)
    king.take_damage(50)
    king.update(2.5)
    assert king.health == 450 and not king.repairing
    assert sim.state.resources.total_spent == king.repair_cost


def test_two_projectiles_cannot_double_count_one_enemy_kill(sim):
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    first = Projectile(enemy.position, enemy, 250, enemy.health)
    second = Projectile(enemy.position, enemy, 250, enemy.health)
    first.update(0, sim.enemies)
    second.update(0, sim.enemies)
    assert not first.alive and not second.alive
    sim.tick()
    sim.tick()
    assert sim.state.enemies_killed == 1
    assert sim.state.resource_balance == sim.balance.starting_resources + enemy.config.kill_reward


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_only_last_configured_king_destruction_loses_immediately(count):
    config = load_config()
    kings = tuple(dict(config.king_towers[0], structure_id=f"king{index}") for index in range(count))
    sim = Simulation(config=replace(config, king_towers=kings), seed=0)
    for king in sim.king_towers[:-1]:
        king.take_damage(king.max_health)
        assert sim.state.status == GameStatus.RUNNING
    last = sim.king_towers[-1]
    projectile = Projectile(last.position, last, 250, last.health)
    ticks = sim.ticks
    projectile.update(0, sim.structures)
    assert sim.state.status == GameStatus.LOST and sim.ticks == ticks
    assert sim.state.base_health == 0
    sim.state.mark_won()
    assert sim.state.status == GameStatus.LOST
    assert not sim.repair_king_tower(last.structure_id)


def test_pending_repair_cannot_resurrect_completed_defeat(sim):
    first, second, third = sim.king_towers
    first.take_damage(10000)
    second.take_damage(10000)
    assert sim.repair_king_tower(first.structure_id)
    third.take_damage(10000)
    assert sim.state.status == GameStatus.LOST
    sim.advance(100)
    first.update(100)
    assert first.destroyed and not first.repairing
    assert not sim.repair_king_tower(first.structure_id)


def test_victory_cannot_be_overwritten_by_later_damage(sim):
    sim.state.mark_won()
    for king in sim.king_towers:
        king.take_damage(10000)
    assert sim.state.status == GameStatus.WON


def test_leaks_are_telemetry_and_endpoint_enemies_keep_fighting(sim):
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = sim.pathway.points[-1].copy()
    enemy.waypoint_index = len(sim.pathway.points) - 1
    enemy.reached_end = True
    health = sim.state.base_health
    sim.tick()
    assert enemy.alive and enemy.target is not None
    assert sim.state.base_health == health - enemy.config.attack_damage
    assert sim.state.enemies_leaked == 1
    sim.tick()
    assert sim.state.enemies_leaked == 1
    enemy.take_damage(10000)
    sim.tick()
    sim.tick()
    assert sim.state.enemies_killed == 1


def test_reset_clears_combat_repairs_and_destruction(sim):
    sim.king_towers[0].take_damage(100)
    sim.repair_king_tower("king1")
    sim.resource_towers[0].take_damage(10000)
    sim.reset(seed=7)
    assert all(tower.health == tower.max_health for tower in sim.structures)
    assert all(not tower.repairing for tower in sim.king_towers)
    assert sim.structures_destroyed == 0
    assert sim.state.resource_balance == sim.balance.starting_resources


@pytest.mark.parametrize(("field", "value"), [
    ("attack_damage", 0), ("attack_interval", -1), ("attack_range", float("nan")),
    ("projectile_speed", float("inf")), ("kind", "unknown"),
])
def test_enemy_combat_config_validation(field, value):
    with pytest.raises(ValueError):
        EnemyConfig("test", 10, 10, **{field: value})


def test_structure_and_objective_config_validation():
    with pytest.raises(ValueError):
        KingTower("king", (0, 0), 100, repair_cost=-1, repair_duration=1)
    with pytest.raises(ValueError):
        ResourceTower("resource", (0, 0), 100, production_amount=1, production_interval=0)
    config = load_config()
    with pytest.raises(ValueError):
        replace(config, king_towers=())
    with pytest.raises(ValueError):
        replace(config, king_towers=(config.king_towers[0], config.king_towers[0]))
    with pytest.raises(ValueError):
        replace(config, target_priority=("king", "king", "defense"))


def test_combat_is_reproducible_with_a_fixed_seed():
    first, second = Simulation(seed=23), Simulation(seed=23)
    for sim in (first, second):
        sim.place_tower(0, "cannon")
        sim._spawn_enemy("light_tank")
        sim.enemies[0].position = sim.resource_towers[0].position - pygame.Vector2(100, 0)
        sim.advance(20)
    assert first.state == second.state
    assert [(tower.structure_id, tower.health) for tower in first.structures] == [
        (tower.structure_id, tower.health) for tower in second.structures
    ]
    assert [(enemy.position, enemy.health, enemy.attack_cooldown) for enemy in first.enemies] == [
        (enemy.position, enemy.health, enemy.attack_cooldown) for enemy in second.enemies
    ]


def test_last_king_defeat_precedes_last_wave_victory(sim):
    for king in sim.king_towers[:-1]:
        king.take_damage(10000)
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


def test_resource_health_and_king_repair_hotkey_render_in_manual_play(monkeypatch):
    from types import SimpleNamespace

    import main

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr("sys.argv", ["main.py"])
    captured = []

    def create_simulation(*args, **kwargs):
        simulation = Simulation(*args, **kwargs)
        simulation.king_towers[0].take_damage(100)
        simulation.resource_towers[0].take_damage(100)
        captured.append(simulation)
        return simulation

    monkeypatch.setattr(main, "Simulation", create_simulation)
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=lambda fps: 1000 / 60))
    monkeypatch.setattr(pygame.event, "get", lambda: [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1), pygame.event.Event(pygame.QUIT),
    ])
    main.main()
    sim = captured[0]
    assert sim.state.resource_balance == 100
    assert sim.king_towers[0].repairing and 400 < sim.king_towers[0].health < 500
