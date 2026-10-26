
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from src.game.config import load_config
from src.game.rl.__main__ import main
from src.game.rl.checkpoint import load_model, manifest, save_metadata
from src.game.rl.environment import BastionEnv


def test_gymnasium_api_and_seed_reproducibility():
    env = BastionEnv()
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=13)
    assert observation.shape == (env.enemy_observation_start + env.max_enemies * env.enemy_features,)
    assert env.action_space.n == 1 + len(env.simulation.defense_slots.slots) * 2 + len(env.simulation.resource_slots.slots) + len(env.simulation.king_towers)
    assert env.observation_space.contains(observation)
    for _ in range(50):
        first = env.step(0)
    env.reset(seed=13)
    for _ in range(50):
        second = env.step(0)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_headless_environment_does_not_initialize_display():
    import pygame

    pygame.quit()
    env = BastionEnv()
    env.reset(seed=0)
    env.step(0)
    assert not pygame.display.get_init()


def test_manual_and_rl_seed_produce_same_wave():
    from src.game.simulation import Simulation

    manual = Simulation(seed=42)
    env = BastionEnv()
    env.reset(seed=42)
    for _ in range(20):
        env.step(0)
        manual.advance(0.5)
    assert [(e.name, e.position) for e in manual.enemies] == [
        (e.name, e.position) for e in env.simulation.enemies
    ]


def test_masks_and_invalid_actions_never_spend_money():
    env = BastionEnv()
    env.reset(seed=0)
    assert env.action_masks()[:env.repair_action_start].all()
    assert not env.action_masks()[env.repair_action_start:].any()
    _, reward, _, _, info = env.step(1)
    assert reward == pytest.approx(-0.001)
    assert info["reward_terms"]["spending"] == pytest.approx(-0.001)
    assert env.simulation.state.resource_balance == 100
    env.simulation.state.resources.spend(75)
    assert env.action_masks().sum() == 1
    _, reward, _, _, info = env.step(1)
    assert info["invalid_action"]
    assert reward == pytest.approx(-0.005)
    assert info["reward_terms"]["spending"] == 0
    assert env.simulation.state.resource_balance == 25
    for action in (-1, int(env.action_space.n)):
        _, reward, _, _, info = env.step(action)
        assert info["invalid_action"] and reward == pytest.approx(-0.005)
    assert env.simulation.state.resources.total_spent == 175


def test_timeout_is_truncation_and_requires_reset():
    env = BastionEnv(max_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    assert env.steps == 0


def test_loss_reward_is_emitted_once():
    env = BastionEnv(max_steps=1)
    env.reset(seed=0)
    sim = env.simulation
    for king in sim.king_towers[:-1]:
        king.destroy()
    king = sim.king_towers[-1]
    king.take_damage(king.health - 10)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = king.position.copy()
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    damage_penalty = -10 / sim.state.max_base_health
    assert info["reward_terms"] == {"enemy_damage": 0, "kills": 0, "base_damage": damage_penalty,
                                    "waves": 0, "outcome": -1, "invalid_action": 0, "spending": 0}
    assert not info["survived"]
    assert reward == pytest.approx(-1 + damage_penalty)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_win_reward_and_fresh_observation_buffers():
    config = load_config()
    medium = replace(config.difficulties["medium"], max_waves=1)
    env = BastionEnv("medium", config=replace(config, difficulties={"medium": medium}))
    previous, _ = env.reset(seed=0)
    copy = previous.copy()
    sim = env.simulation
    sim.director.wave_number = 1
    sim.director.wave_active = True
    sim._spawn_enemy("enemy1")
    sim.enemies[0].take_damage(100)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["is_success"]
    assert reward == pytest.approx(2.625)
    np.testing.assert_array_equal(previous, copy)


def test_action_layout_covers_all_map_slots_and_kings():
    env = BastionEnv()
    sim = env.simulation
    assert env.resource_action_start == 1 + len(sim.defense_slots.slots) * len(sim.tower_types)
    assert env.repair_action_start == env.resource_action_start + len(sim.resource_slots.slots)
    assert env.action_space.n == env.repair_action_start + len(sim.king_towers)


def test_model_save_load_and_manifest_guard(tmp_path):
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    env = BastionEnv()
    model = MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1,
                        policy_kwargs={"net_arch": [16]}, device="cpu", seed=0)
    model.learn(8)
    path = tmp_path / "model.zip"
    model.save(path)
    save_metadata(path, env, seed=0, steps=8)
    loaded = load_model(path, env)
    observation, _ = env.reset(seed=0)
    action, _ = loaded.predict(observation, action_masks=env.action_masks())
    assert env.action_masks()[int(action)]
    metadata = json.loads(path.with_suffix(".json").read_text())
    metadata["environment"]["schema"] = 1
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Checkpoint environment differs"):
        load_model(path, env)
    metadata["environment"]["schema"] = manifest(env)["schema"]
    metadata["environment"]["slots"].reverse()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Checkpoint environment differs"):
        load_model(path, env)


def test_enemy_rows_preserve_positions_health_and_padding():
    env = BastionEnv()
    initial, _ = env.reset(seed=0)
    offset = env.enemy_observation_start
    np.testing.assert_array_equal(initial[offset:], 0)
    sim = env.simulation
    sim._spawn_enemy("enemy1")
    sim._spawn_enemy("enemy2")
    first, second = sim.enemies
    first.pathway = second.pathway = sim.pathway
    start, end = sim.pathway.points[:2]
    first.position = start.lerp(end, 0.25)
    second.position = start.lerp(end, 0.75)
    second.health = 80
    observation = env.observation()
    rows = observation[env.enemy_observation_start:].reshape(env.max_enemies, env.enemy_features)
    np.testing.assert_allclose(rows[0, :6], [
        1, second.position.x / sim.pathway.map_width,
        second.position.y / sim.pathway.map_height, 80 / second.max_health, 2 / len(sim.enemy_types),
        start.distance_to(second.position) / env.route_offsets[sim.pathway][-1],
    ], rtol=1e-6)
    assert rows[1, 3] == 1 and rows[1, 4] == pytest.approx(1 / len(sim.enemy_types))
    np.testing.assert_array_equal(rows[2:], 0)
    assert env.observation_space.contains(observation)


def test_enemy_overflow_is_bounded_and_reported():
    env = BastionEnv()
    env.reset(seed=0)
    for _ in range(65):
        env.simulation._spawn_enemy("enemy1")
    start, end = env.simulation.enemies[-1].pathway.points[:2]
    env.simulation.enemies[-1].position = start.lerp(end, 0.5)
    observation = env.observation()
    assert observation.shape == (env.enemy_observation_start + env.max_enemies * env.enemy_features,)
    rows = observation[env.enemy_observation_start:].reshape(env.max_enemies, env.enemy_features)
    assert rows[0, 5] > 0
    assert (rows[:, 0] == 1).all()
    assert env.info()["enemies_omitted"] == 1


def test_experimental_balance_and_horizon_do_not_change_baseline_schema():
    env = BastionEnv()
    config = load_config()
    settings = dict(config.difficulties)
    settings["hard"] = replace(settings["hard"], enemy_health_growth=0.9)
    other = BastionEnv(config=replace(config, difficulties=settings), max_steps=2)
    assert env.simulation.difficulty == "baseline"
    assert manifest(env) == manifest(other)
    assert env.simulation.balance.enemy_health_growth == 0


def test_king_repair_does_not_reward_healing_or_hide_damage():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    king = sim.king_towers[0]
    king.destroy()
    assert sim.start_king_tower_repair(king.structure_id)
    _, _, _, _, info = env.step(0)
    assert info["reward_terms"]["base_damage"] == 0
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    enemy.position = king.position.copy()
    before = king.health
    _, _, _, _, info = env.step(0)
    assert king.health > before - enemy.config.attack_damage
    assert info["reward_terms"]["base_damage"] == pytest.approx(-enemy.config.attack_damage / sim.state.max_base_health)


def test_observations_use_assigned_routes_and_bounded_endpoint_progress():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim._spawn_enemy("enemy1")
    enemy = sim.enemies[0]
    for route in (*sim.soldier_routes.values(), *sim.tank_routes.values()):
        lengths = [start.distance_to(end) for start, end in zip(route.points, route.points[1:])]
        segment = len(lengths) // 2
        enemy.pathway = route
        enemy.waypoint_index = segment + 1
        enemy.position = route.points[segment].lerp(route.points[segment + 1], 0.5)
        expected = (sum(lengths[:segment]) + lengths[segment] / 2) / sum(lengths)
        assert env.observation()[env.enemy_observation_start + 5] == pytest.approx(expected)
    enemy.reached_end = True
    enemy.waypoint_index = len(enemy.pathway.points)
    assert env.observation()[env.enemy_observation_start + 5] == 1


def test_destroyed_defense_is_empty_in_mask_and_observation():
    env = BastionEnv()
    env.reset(seed=0)
    env.step(1)
    tower = env.simulation.towers[0]
    tower.destroy()
    assert env.action_masks()[1]
    offset = (env.global_features + len(env.simulation.tower_types) * env.tower_features
              + len(env.simulation.enemy_types) * env.enemy_type_features)
    np.testing.assert_array_equal(env.observation()[offset + 2:offset + 8], 0)


@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_evaluation_reports_and_saves_all_required_metrics(tmp_path, monkeypatch, explicit_output):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "evaluation.json" if explicit_output else tmp_path / "runs/evaluation-baseline.json"
    arguments = ["rl", "evaluate", "--policy", "random", "--episodes", "2", "--max-steps", "1"]
    if explicit_output:
        arguments.extend(["--output", str(path)])
    monkeypatch.setattr("sys.argv", arguments)
    main()
    result = json.loads(path.read_text())
    assert result["difficulty"] == "baseline"
    assert result["max_steps"] == 1
    assert result["survival_rate"] == result["truncation_rate"] == 1
    assert result["win_rate"] == 0
    for name in ("mean_episode_reward", "std_episode_reward", "mean_base_health",
                 "mean_waves_reached", "mean_waves_completed", "mean_kills", "mean_leaks"):
        assert name in result
    assert len(result["results"]) == 2
    assert [item["seed"] for item in result["results"]] == [30000, 30001]


def test_cli_train_and_evaluate_smoke(tmp_path, monkeypatch):
    pytest.importorskip("sb3_contrib")
    output = tmp_path / "training"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--steps", "256", "--max-steps", "2",
                                    "--eval-freq", "128", "--eval-episodes", "1",
                                    "--output", str(output)])
    main()
    model = output / "final.zip"
    assert model.is_file() and (output / "best_model.zip").is_file()
    assert (output / "latest.zip").is_file() and (output / "latest.json").is_file()
    metadata = json.loads(model.with_suffix(".json").read_text())
    assert metadata["environment"]["schema"] == 7
    assert metadata["training_max_steps"] == 2
    assert metadata["environment"]["observation_size"] == 1462
    assert metadata["training_parameters"]["policy"] == "HierarchicalPolicy"
    validation = json.loads((output / "evaluations.json").read_text())
    assert [item["results"][0]["seed"] for item in validation] == [10000]
    assert [item["final"] for item in validation] == [True]
    assert all(item["results"][0]["wave_goal"] == 10 for item in validation)
    assert "action_diagnostics" in validation[0]
    assert "full_task_after" not in metadata["training_parameters"]["curriculum"]
    assert (output / "progress.csv").is_file()
    report = tmp_path / "model-evaluation.json"
    monkeypatch.setattr("sys.argv", ["rl", "evaluate", "--model", str(model),
                                    "--episodes", "2", "--max-steps", "3", "--output", str(report)])
    main()
    assert json.loads(report.read_text())["episodes"] == 2
    resumed = tmp_path / "resumed"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(model), "--steps", "256",
                                    "--max-steps", "2", "--eval-freq", "1000", "--eval-episodes", "1",
                                    "--ent-coef", "0.04", "--learning-rate", "0.0001",
                                    "--output", str(resumed)])
    main()
    assert (resumed / "best_model.zip").is_file()
    loaded = load_model(resumed / "final.zip", BastionEnv())
    assert loaded.num_timesteps == 512 and loaded.ent_coef == 0.04
    assert loaded.learning_rate == 0.0001
    inherited = tmp_path / "inherited"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(resumed / "final.zip"),
                                    "--steps", "256", "--max-steps", "2", "--eval-freq", "1000",
                                    "--eval-episodes", "1", "--output", str(inherited)])
    main()
    loaded = load_model(inherited / "final.zip", BastionEnv())
    assert loaded.ent_coef == 0.04 and loaded.learning_rate == 0.0001
    history = json.loads((inherited / "evaluations.json").read_text())
    assert len(history) == 1 and history[0]["final"]


def test_cli_rejects_invalid_horizon(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rl", "train", "--max-steps", "0"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


@pytest.mark.parametrize(("frame_ms", "ticks_per_frame", "prediction_count"),
                         [(1000 / 60, 1, 2), (2000, 6, 12)])
def test_spectator_frame_pacing_and_stall_recovery(monkeypatch, frame_ms, ticks_per_frame, prediction_count):
    import pygame

    import main as frontend

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    monkeypatch.setattr("sys.argv", ["main.py", "--model", "test.zip"])
    environments, frames = [], []
    predict = Mock(return_value=(0, None))

    def create_env(*args, **kwargs):
        env = BastionEnv(*args, **kwargs)
        environments.append(env)
        return env

    monkeypatch.setattr("src.game.rl.environment.BastionEnv", create_env)
    monkeypatch.setattr("src.game.rl.checkpoint.load_model",
                        lambda *args: SimpleNamespace(predict=predict, get_vec_normalize_env=lambda: None))
    monkeypatch.setattr(pygame.time, "Clock", lambda: SimpleNamespace(tick=lambda fps: frame_ms))
    monkeypatch.setattr(pygame.display, "flip", lambda: frames.append(environments[0].simulation.ticks))
    monkeypatch.setattr(pygame.event, "get", lambda: (
        [pygame.event.Event(pygame.QUIT)] if len(frames) >= 59 else []
    ))
    frontend.main()
    assert frames == [index * ticks_per_frame for index in range(1, 61)]
    assert predict.call_count == prediction_count


def test_rendered_ticks_match_headless_steps():
    headless, rendered = BastionEnv(), BastionEnv()
    headless.reset(seed=42)
    rendered.reset(seed=42)
    for action in [1] + [0] * 19:
        expected = headless.step(action)
        previous_ticks = rendered.simulation.ticks
        for index, result in enumerate(rendered.step_ticks(action), start=1):
            assert rendered.simulation.ticks == previous_ticks + index
            if index < 30:
                assert result is None
        assert index == 30
        np.testing.assert_array_equal(result[0], expected[0])
        assert result[1:] == expected[1:]


def test_rendered_ticks_stop_immediately_on_defeat():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    for king in sim.king_towers[:-1]:
        king.destroy()
    king = sim.king_towers[-1]
    king.take_damage(king.health - 10)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = king.position.copy()
    results = list(env.step_ticks(0))
    assert len(results) == env.simulation.ticks == 1
    assert results[0][2] and not results[0][3]


def test_resource_and_repair_actions_apply_only_legal_game_controls():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim.state.resources.earn(1000)
    action = env.resource_action_start
    assert env.action_masks()[action]
    _, _, _, _, info = env.step(action)
    assert info["resources_built"] == 1 and sim.resource_slots.slots[0].occupied
    assert sim.state.resources.total_spent == sim.config.resource_tower.cost
    assert not env.action_masks()[action]
    king = sim.king_towers[0]
    king.take_damage(200)
    assert not env.action_masks()[env.repair_action_start]
    king.destroy()
    repair = env.repair_action_start
    assert env.action_masks()[repair]
    _, _, _, _, info = env.step(repair)
    assert info["repairs_started"] == 1 and king.repairing
    assert 0 < king.health < king.max_health
    assert sim.state.resources.total_spent == sim.config.resource_tower.cost + king.repair_cost
    assert not env.action_masks()[repair]
    spent = sim.state.resources.total_spent
    _, _, _, _, info = env.step(repair)
    assert info["invalid_action"] and sim.state.resources.total_spent == spent


def test_repair_mask_handles_destroyed_and_terminal_kings():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim.king_towers[0].destroy()
    assert env.action_masks()[env.repair_action_start]
    for king in sim.king_towers[1:]:
        king.destroy()
    assert env.action_masks().sum() == 1


def test_enemy_damage_reward_does_not_reward_spawns_or_repeat_kills():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim.director.intermission_timer = 0
    _, _, _, _, info = env.step(0)
    assert sim.state.enemies_spawned > 0
    assert info["reward_terms"]["enemy_damage"] == 0
    enemy = sim.enemies[0]
    from src.game.entities.towers.projectile import Projectile

    sim.projectiles.append(Projectile(enemy.position, enemy, 10000, enemy.max_health / 2))
    _, _, _, _, info = env.step(0)
    assert info["reward_terms"]["enemy_damage"] == pytest.approx(0.025)
    sim.projectiles.append(Projectile(enemy.position, enemy, 10000, enemy.max_health))
    _, _, _, _, info = env.step(0)
    assert info["reward_terms"]["enemy_damage"] == pytest.approx(0.025)
    assert info["reward_terms"]["kills"] == pytest.approx(0.025)
    _, _, _, _, info = env.step(0)
    assert info["reward_terms"]["enemy_damage"] == info["reward_terms"]["kills"] == 0


def test_normalized_damage_penalty_and_horizon_do_not_reward_stalling():
    env = BastionEnv(max_steps=1)
    env.reset(seed=0)
    sim = env.simulation
    king = sim.king_towers[0]
    king.take_damage(king.health - 10)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = king.position.copy()
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert info["reward_terms"]["base_damage"] == pytest.approx(-10 / sim.state.max_base_health)
    assert info["reward_terms"]["outcome"] == -1 and reward < -1


def test_observations_expose_individual_king_health_and_route_geometry():
    env = BastionEnv()
    before, _ = env.reset(seed=0)
    sim = env.simulation
    start = (env.global_features + len(sim.tower_types) * env.tower_features
             + len(sim.enemy_types) * env.enemy_type_features
             + len(sim.defense_slots.slots) * env.slot_features
             + len(sim.resource_slots.slots) * env.resource_features)
    sim.king_towers[1].take_damage(200)
    after = env.observation()
    assert after[start + env.king_features + 2] == pytest.approx(1 - 200 / sim.king_towers[1].max_health)
    assert after[start + 2] == before[start + 2] == 1
    assert env.observation_space.contains(after)
    distances = list(env.soldier_route_distances.values())
    assert min(distances) < max(distances)
    assert env.action_masks()[1:env.resource_action_start].all()


def test_forced_waits_skip_only_when_no_choice_is_available():
    headless, rendered = BastionEnv(), BastionEnv()
    for env in (headless, rendered):
        env.reset(seed=42)
        env.simulation.state.resources.spend(50)
    expected = headless.step(2)
    results = list(rendered.step_ticks(2))
    assert headless.simulation.ticks > headless.decision_ticks
    assert len(results) == rendered.simulation.ticks
    assert all(result is None for result in results[:-1])
    np.testing.assert_array_equal(results[-1][0], expected[0])
    assert results[-1][1:] == expected[1:]
    assert headless.action_masks()[1:].any() or expected[2] or expected[3]


def test_forced_wait_honors_the_same_simulated_time_horizon():
    env = BastionEnv(max_steps=2)
    env.reset(seed=0)
    env.simulation.state.resources.spend(50)
    _, _, terminated, truncated, info = env.step(2)
    assert truncated and not terminated
    assert info["simulation_ticks"] == env.max_steps * env.decision_ticks
    assert info["simulated_seconds"] == 1.0
    assert env.steps == 1


def test_training_curriculum_changes_only_the_next_episode_wave_goal():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    full_waves = sim.balance.max_waves
    initial_config = sim.config
    env.set_wave_goal(1)
    assert sim.balance.max_waves == full_waves
    assert env.observation()[12] == 1
    env.reset(seed=0)
    assert sim.balance.max_waves == 1
    assert env.observation()[12] == pytest.approx(1 / full_waves)
    assert sim.config is initial_config and env.task_balance.max_waves == full_waves
    sim.director.wave_number = 1
    assert env.observation()[2] == pytest.approx(1 / full_waves)
    sim._spawn_enemy("enemy1")
    assert sim.enemies[0].max_health == sim.config.enemies["enemy1"].max_health
    sim.enemies[0].take_damage(sim.enemies[0].health)
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["is_success"] and info["wave_goal"] == 1
    evaluation = BastionEnv()
    _, info = evaluation.reset(seed=0)
    assert info["wave_goal"] == full_waves
    for invalid in (0, full_waves + 1, True, 1.5):
        with pytest.raises(ValueError):
            env.set_wave_goal(invalid)


def test_curriculum_promotes_one_wave_only_after_mastery_regardless_of_budget():
    pytest.importorskip("stable_baselines3")
    from src.game.rl.curriculum import CurriculumCallback

    env, evaluation = BastionEnv(), BastionEnv()
    from stable_baselines3.common.vec_env import DummyVecEnv

    vector = DummyVecEnv([lambda: env])
    callback = CurriculumCallback(vector, env.task_balance.max_waves)
    assert callback.goals == list(range(1, env.task_balance.max_waves + 1))
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=Mock()))
    callback.num_timesteps = 0
    callback._on_training_start()
    for index in range(50):
        callback.num_timesteps = index + 1
        callback.locals = {"infos": [{"episode": {}, "wave_goal": 1, "is_success": index < 40}]}
        assert callback._on_step()
    assert env.wave_goal == 2 and callback.successes == []
    for index in range(50):
        callback.num_timesteps = index + 51
        callback.locals = {"infos": [{"episode": {}, "wave_goal": callback.goals[1], "is_success": False}]}
        callback._on_step()
    assert env.wave_goal == callback.goals[1]
    callback.num_timesteps = 1000000
    callback._on_step()
    assert env.wave_goal == 2
    assert evaluation.wave_goal == env.task_balance.max_waves
    callback.locals = {"infos": [{"episode": {}, "wave_goal": 1, "is_success": True}]}
    before = callback.successes.copy()
    callback._on_step()
    assert callback.successes == before and env.wave_goal == 2


def test_hierarchical_policy_balances_kinds_and_preserves_legality():
    from sb3_contrib import MaskablePPO

    from src.game.rl.policy import HierarchicalPolicy

    env = BastionEnv()
    model = MaskablePPO(HierarchicalPolicy, env, n_steps=8, batch_size=8, seed=0,
                        policy_kwargs={"action_group_sizes": [1, 92, 6, 3]})
    import torch

    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    obs, _ = env.reset(seed=0)
    tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        probabilities = model.policy.get_distribution(tensor).distribution.probs.numpy()[0]
        np.testing.assert_allclose([probabilities[group].sum() for group in env.action_groups.values()],
                                   [0.25] * 4, atol=1e-6)
        mask = env.action_masks()
        probabilities = model.policy.get_distribution(tensor, action_masks=mask).distribution.probs.numpy()[0]
    assert (probabilities[~mask] == 0).all()
    np.testing.assert_allclose([probabilities[group].sum() for name, group in env.action_groups.items()
                               if name != "repair"], [1 / 3] * 3, atol=1e-6)


def test_evaluation_diagnostics_measure_only_legal_opportunities():
    from src.game.rl.evaluation import play_episodes, summarize_episodes

    env = BastionEnv(max_steps=1)
    episodes = play_episodes(env, None, [90000, 90001])
    summary = summarize_episodes(episodes)
    stats = summary["action_diagnostics"]
    assert stats["repair"]["legal_decisions"] == stats["repair"]["chosen"] == 0
    assert stats["repair"]["mean_probability_when_legal"] is None
    assert stats["resource"]["legal_decisions"] == 2
    assert stats["resource"]["mean_probability_when_legal"] == pytest.approx(6 / 99)
    assert sum(value for key, value in summary.items() if key.startswith("mean_reward_")) == pytest.approx(
        summary["mean_episode_reward"])
    assert sum(values["chosen"] for values in stats.values()) == 2
    assert episodes == play_episodes(env, None, [90000, 90001])


def test_validation_reuses_seeds_and_ranks_outcomes_ahead_of_reward(tmp_path, monkeypatch):
    from src.game.rl.training import OutcomeEvalCallback

    env = BastionEnv()
    env.reset(seed=0)
    base = dict(env.info(), seed=12000, reward=0, steps=1, truncated=False,
                action_diagnostics={"wait": {"legal_decisions": 1, "chosen": 1, "probability_sum": 1.0}})
    outcomes = [(False, 2, 0, 100), (False, 1, 0, 200), (False, 3, 0, -100),
                (True, 10, 5, -999), (False, 9, 1000, 10000), (True, 10, 6, -2000)]
    seeds_seen = []

    def play(env, model, seeds):
        seeds_seen.append(list(seeds))
        won, waves, health, reward = outcomes[len(seeds_seen) - 1]
        return [dict(base, seed=seed, is_success=won, waves_completed=waves,
                     base_health=health, reward=reward) for seed in seeds_seen[-1]]

    monkeypatch.setattr("src.game.rl.training.play_episodes", play)
    callback = OutcomeEvalCallback(env, tmp_path, eval_freq=10, episodes=2, seed=12000,
                                   training_seed=0, steps=60, training_parameters={})
    callback.model = SimpleNamespace(save=Mock(), num_timesteps=0, get_vec_normalize_env=lambda: None,
                                     logger=SimpleNamespace(record=Mock()),
                                     get_env=lambda: SimpleNamespace(envs=[env]))
    for index in range(len(outcomes)):
        callback.num_timesteps = (index + 1) * 10
        callback.evaluate()
    assert callback.model.save.call_count == 10
    assert sum(call.args[0].name == "best_model.zip" for call in callback.model.save.call_args_list) == 4
    assert callback.best_score == (1, 10, 6)
    assert seeds_seen == [[12000, 12001]] * len(outcomes)
    history = json.loads((tmp_path / "evaluations.json").read_text())
    assert len(history) == len(outcomes)
    assert np.load(tmp_path / "evaluations.npz")["successes"].shape == (6, 2)


@pytest.mark.parametrize(("option", "value"), [("--ent-coef", "-0.1"), ("--ent-coef", "nan"),
                                               ("--learning-rate", "0"), ("--learning-rate", "inf")])
def test_cli_rejects_invalid_exploration_parameters(monkeypatch, option, value):
    monkeypatch.setattr("sys.argv", ["rl", "train", option, value])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
