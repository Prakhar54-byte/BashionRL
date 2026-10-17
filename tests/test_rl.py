import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env

from src.game.config import load_config
from src.game.rl.__main__ import coverage_scores
from src.game.rl.checkpoint import load_model, save_metadata
from src.game.rl.environment import BastionEnv


def test_gymnasium_api_and_seed_reproducibility():
    env = BastionEnv()
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=13)
    assert observation.shape == (211,)
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
    env.step(2)
    assert env.simulation.state.resource_balance == 50
    assert env.action_masks().sum() == 1
    _, _, _, _, info = env.step(1)
    assert info["invalid_action"]
    assert env.simulation.state.resource_balance == 50
    with pytest.raises(ValueError):
        env.step(63)


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
    env = BastionEnv()
    env.reset(seed=0)
    env.simulation.state.base_health = 10
    env.simulation._spawn_enemy("enemy1")
    env.simulation.enemies[0].reached_end = True
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["reward_terms"] == {"kills": 0, "base_damage": -1, "waves": 0, "outcome": -20}
    assert reward == -21
    with pytest.raises(RuntimeError):
        env.step(0)


def test_win_reward_and_fresh_observation_buffers():
    config = load_config()
    medium = replace(config.difficulties["medium"], max_waves=1)
    env = BastionEnv(config=replace(config, difficulties={"medium": medium}))
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
    metadata["environment"]["slots"].reverse()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Checkpoint environment differs"):
        load_model(path, env)
