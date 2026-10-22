import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from src.game.config import load_config
from src.game.rl.__main__ import coverage_scores, main
from src.game.rl.checkpoint import load_model, manifest, save_metadata
from src.game.rl.environment import BastionEnv


def test_gymnasium_api_and_seed_reproducibility():
    env = BastionEnv()
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=13)
    assert observation.shape == (547,)
    assert env.action_space.n == 63
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
    assert env.action_masks().all()
    _, reward, _, _, info = env.step(2)
    assert reward == pytest.approx(-0.15)
    assert info["reward_terms"]["spending"] == pytest.approx(-0.15)
    assert env.simulation.state.resource_balance == 50
    assert env.action_masks().sum() == 1
    _, reward, _, _, info = env.step(1)
    assert info["invalid_action"]
    assert reward == pytest.approx(-0.1)
    assert info["reward_terms"]["spending"] == 0
    assert env.simulation.state.resource_balance == 50
    for action in (-1, 63):
        _, reward, _, _, info = env.step(action)
        assert info["invalid_action"] and reward == pytest.approx(-0.1)
    assert env.simulation.state.resources.total_spent == 150


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
        king.take_damage(king.max_health)
    king = sim.king_towers[-1]
    king.take_damage(king.health - 10)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = king.position.copy()
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["reward_terms"] == {"kills": 0, "base_damage": -1, "waves": 0,
                                    "outcome": -20, "invalid_action": 0, "spending": 0}
    assert not info["survived"]
    assert reward == -21
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
    assert reward == pytest.approx(21.1)
    np.testing.assert_array_equal(previous, copy)


def test_heuristic_scores_cover_all_actions():
    env = BastionEnv()
    scores = coverage_scores(env)
    assert scores.shape == (63,)
    assert np.isfinite(scores).all() and max(scores) > 0


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
    metadata["environment"]["schema"] = 3
    metadata["environment"]["slots"].reverse()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Checkpoint environment differs"):
        load_model(path, env)


def test_enemy_rows_preserve_positions_health_and_padding():
    env = BastionEnv()
    initial, _ = env.reset(seed=0)
    np.testing.assert_array_equal(initial[163:], 0)
    sim = env.simulation
    sim._spawn_enemy("enemy1")
    sim._spawn_enemy("enemy2")
    first, second = sim.enemies
    start, end = sim.pathway.points[:2]
    first.position = start.lerp(end, 0.25)
    second.position = start.lerp(end, 0.75)
    second.health = 80
    observation = env.observation()
    rows = observation[163:].reshape(64, 6)
    np.testing.assert_allclose(rows[0], [
        1, second.position.x / sim.pathway.map_width,
        second.position.y / sim.pathway.map_height, 0.8, 2 / 3,
        start.distance_to(second.position) / env.route_length,
    ], rtol=1e-6)
    assert rows[1, 3] == 1 and rows[1, 4] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(rows[2:], 0)
    assert env.observation_space.contains(observation)


def test_enemy_overflow_is_bounded_and_reported():
    env = BastionEnv()
    env.reset(seed=0)
    for _ in range(65):
        env.simulation._spawn_enemy("enemy1")
    start, end = env.simulation.pathway.points[:2]
    env.simulation.enemies[-1].position = start.lerp(end, 0.5)
    observation = env.observation()
    assert observation.shape == (547,)
    rows = observation[163:].reshape(64, 6)
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


@pytest.mark.parametrize("explicit_output", [False, True])
def test_cli_evaluation_reports_and_saves_all_required_metrics(tmp_path, monkeypatch, explicit_output):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "evaluation.json" if explicit_output else tmp_path / "runs/evaluation-baseline.json"
    arguments = ["rl", "evaluate", "--episodes", "2", "--max-steps", "1"]
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
                                    "--eval-freq", "256", "--eval-episodes", "1",
                                    "--output", str(output)])
    main()
    model = output / "final.zip"
    assert model.is_file() and (output / "best_model.zip").is_file()
    metadata = json.loads(model.with_suffix(".json").read_text())
    assert metadata["environment"]["schema"] == 3
    assert metadata["training_max_steps"] == 2
    assert (output / "progress.csv").is_file()
    report = tmp_path / "model-evaluation.json"
    monkeypatch.setattr("sys.argv", ["rl", "evaluate", "--model", str(model),
                                    "--episodes", "2", "--max-steps", "3", "--output", str(report)])
    main()
    assert json.loads(report.read_text())["episodes"] == 2


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
                        lambda *args: SimpleNamespace(predict=predict))
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
        king.take_damage(king.max_health)
    king = sim.king_towers[-1]
    king.take_damage(king.health - 10)
    sim._spawn_enemy("enemy1")
    sim.enemies[0].position = king.position.copy()
    results = list(env.step_ticks(0))
    assert len(results) == env.simulation.ticks == 1
    assert results[0][2] and not results[0][3]
