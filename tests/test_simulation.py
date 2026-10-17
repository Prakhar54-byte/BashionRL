from dataclasses import replace

import pytest

from src.game.config import load_config
from src.game.simulation import Simulation
from src.game.state.game_state import GameStatus


def test_placement_is_atomic_and_reset_clears_episode():
    sim = Simulation(seed=5)
    assert len(sim.defense_slots.slots) == 31
    assert sim.place_tower(0, "double_cannon")
    assert sim.state.resource_balance == 50
    assert not sim.place_tower(0, "cannon")
    assert not sim.place_tower(1, "cannon")
    assert len(sim.towers) == 1
    with pytest.raises(ValueError):
        sim.place_tower(-1, "cannon")
    sim.reset(seed=5)
    assert sim.state.resource_balance == 200
    assert not sim.towers and not sim.enemies
    assert all(not slot.occupied for slot in sim.defense_slots.slots)
    assert sim.waves_completed == sim.ticks == 0


def test_kill_and_leak_are_accounted_once():
    sim = Simulation(seed=5)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].take_damage(100)
    sim.tick()
    sim.tick()
    assert sim.state.enemies_killed == 1
    assert sim.state.resource_balance == 225
    sim._spawn_enemy("enemy2")
    sim.enemies[0].reached_end = True
    sim.tick()
    sim.tick()
    assert sim.state.enemies_leaked == 1
    assert sim.state.base_health == 90
    assert sim.state.enemies_killed == 1


def test_final_wave_requires_all_spawns_and_enemies_to_clear():
    config = load_config()
    medium = replace(config.difficulties["medium"], max_waves=1)
    config = replace(config, difficulties={"medium": medium})
    sim = Simulation(config=config, seed=2)
    sim.director.wave_number = 1
    sim.director.wave_active = True
    sim.director.enemies_remaining_to_spawn = 1
    sim.director.spawn_timer = 10
    sim.tick()
    assert sim.state.status == GameStatus.RUNNING
    sim.director.enemies_remaining_to_spawn = 0
    sim._spawn_enemy("enemy1")
    sim.tick()
    assert sim.state.status == GameStatus.RUNNING
    sim.enemies[0].take_damage(100)
    sim.tick()
    assert sim.state.status == GameStatus.WON
    assert sim.waves_completed == 1
    previous = sim.ticks
    sim.advance(10)
    assert sim.ticks == previous
    assert not sim.place_tower(0, "cannon")


def test_no_defenses_eventually_loses_and_stops():
    sim = Simulation("hard", seed=0)
    for _ in range(20000):
        sim.tick()
        if sim.state.is_terminal:
            break
    assert sim.state.status == GameStatus.LOST
    assert sim.state.enemies_leaked == 10
    assert sim.state.enemies_killed == 0


def test_frame_grouping_does_not_change_simulation():
    first, second = Simulation(seed=42), Simulation(seed=42)
    for _ in range(600):
        first.advance(1 / 60)
    for _ in range(100):
        second.advance(0.1)
    assert first.ticks == second.ticks == 600
    assert [(e.name, e.position) for e in first.enemies] == [
        (e.name, e.position) for e in second.enemies
    ]
    with pytest.raises(ValueError):
        first.advance(float("nan"))


def test_hard_preset_has_more_pressure():
    config = load_config()
    medium, hard = config.difficulties["medium"], config.difficulties["hard"]
    assert medium.max_waves == hard.max_waves == 10
    assert hard.waves.initial_enemy_count > medium.waves.initial_enemy_count
    assert hard.waves.spawn_interval < medium.waves.spawn_interval
    assert hard.enemy_health_growth > medium.enemy_health_growth


def test_later_waves_scale_health_without_increasing_income():
    sim = Simulation("hard", seed=0)
    sim.director.wave_number = 4
    sim._spawn_enemy("enemy2")
    original = sim.config.enemies["enemy2"]
    enemy = sim.enemies[0]
    assert enemy.max_health == pytest.approx(369.0)
    assert enemy.config.kill_reward == original.kill_reward
    assert original.max_health == 180


def test_manual_game_renders_existing_assets(monkeypatch):
    import pygame

    import main

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr("sys.argv", ["main.py", "--difficulty", "hard"])
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    main.main()
