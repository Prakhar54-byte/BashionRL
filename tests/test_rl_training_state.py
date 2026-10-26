import copy
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.game.rl.__main__ import main
from src.game.rl.checkpoint import load_model, policy_observation, save_checkpoint
from src.game.rl.curriculum import CurriculumCallback
from src.game.rl.environment import BastionEnv
from src.game.rl.evaluation import play_episodes
from src.game.rl.policy import HierarchicalPolicy
from src.game.rl.training import AnnealingCallback


def test_mixed_goals_are_seeded_sampled_per_episode_and_observable():
    env = BastionEnv()
    env.set_wave_goal(4, mixed=True)
    counts = Counter()
    for seed in range(1000):
        obs, info = env.reset(seed=seed)
        goal = info["wave_goal"]
        counts[goal] += 1
        assert goal in (1, 2, 3, 4, 10)
        assert obs[12] == pytest.approx(goal / 10)
    assert 0.55 < counts[4] / 1000 < 0.65
    assert 0.15 < counts[10] / 1000 < 0.25
    assert 0.15 < sum(counts[goal] for goal in (1, 2, 3)) / 1000 < 0.25
    first = env.reset(seed=123)
    second = env.reset(seed=123)
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    evaluation = BastionEnv()
    assert evaluation.reset(seed=123)[1]["wave_goal"] == 10


def test_curriculum_progress_restores_window_and_counts_only_current_goal():
    env = BastionEnv()
    vector = DummyVecEnv([lambda: env])
    state = {"maximum": 10, "mode": "mixed", "promotion_successes": 30,
             "goal_index": 2, "successes": [True] * 29 + [False] * 20}
    callback = CurriculumCallback(vector, 10, mode="mixed", promotion_successes=30, state=state)
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=Mock()))
    callback._on_training_start()
    assert callback.state_dict() == state
    assert env.wave_goal == 3 and env.mixed_goals
    callback.locals = {"infos": [{"episode": {}, "wave_goal": 10, "is_success": True}]}
    assert callback._on_step() and callback.state_dict() == state
    callback.locals = {"infos": [{"episode": {}, "wave_goal": 3, "is_success": True}]}
    callback._on_step()
    assert callback.goal_index == 3 and env.wave_goal == 4 and callback.successes == []
    with pytest.raises(ValueError, match="does not match"):
        CurriculumCallback(vector, 10, mode="staged", state=state)


def test_absolute_schedules_interpolate_clamp_and_do_not_mutate_saved_plan():
    plan = {"start_step": 100, "end_step": 1100, "lr_start": 0.0003, "lr_end": 0.00005,
            "entropy_start": 0.02, "entropy_end": 0.005}
    model = SimpleNamespace(training_schedule=dict(plan), logger=SimpleNamespace(record=Mock()))
    callback = AnnealingCallback()
    callback.model = model
    for step, fraction in ((100, 0), (600, 0.5), (1100, 1), (1200, 1)):
        callback.num_timesteps = step
        callback._on_rollout_end()
        assert model.learning_rate == pytest.approx(0.0003 + fraction * (0.00005 - 0.0003))
        assert model.lr_schedule(0.123) == model.learning_rate
        assert model.ent_coef == pytest.approx(0.02 + fraction * (0.005 - 0.02))
    assert model.training_schedule == plan


def test_normalized_checkpoint_restores_adam_curriculum_schedules_and_raw_evaluation(tmp_path, monkeypatch):
    env = BastionEnv(max_steps=2)
    vector = VecNormalize(DummyVecEnv([lambda: env]), gamma=0.999)
    state = {"maximum": 10, "mode": "mixed", "promotion_successes": 30,
             "goal_index": 2, "successes": [True] * 15 + [False] * 10}
    curriculum = CurriculumCallback(vector, 10, mode="mixed", promotion_successes=30, state=state)
    model = MaskablePPO(HierarchicalPolicy, vector, n_steps=8, batch_size=8, n_epochs=1, seed=7,
                        policy_kwargs={"action_group_sizes": [1, 92, 6, 3], "net_arch": [16]})
    model.training_schedule = {"start_step": 0, "end_step": 32, "lr_start": 0.0003, "lr_end": 0.00005,
                               "entropy_start": 0.02, "entropy_end": 0.005}
    model.learn(16, callback=[curriculum, AnnealingCallback()])
    original_optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    assert original_optimizer["state"]
    path = tmp_path / "normalized.zip"
    save_checkpoint(path, model, env, seed=7, steps=32, training_parameters={}, curriculum=curriculum)
    assert path.with_suffix(".vecnormalize.pkl").is_file()
    raw = BastionEnv(max_steps=2)
    restored = load_model(path, raw)
    optimizer = restored.policy.optimizer.state_dict()
    assert optimizer["param_groups"] == original_optimizer["param_groups"]
    for key, values in original_optimizer["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(optimizer["state"][key][name], value)
    normalizer = restored.get_vec_normalize_env()
    assert not normalizer.training and not normalizer.norm_reward
    count, reward_count = normalizer.obs_rms.count, normalizer.ret_rms.count
    episodes = play_episodes(raw, restored, [70000, 70001])
    assert all(episode["wave_goal"] == 10 for episode in episodes)
    assert normalizer.obs_rms.count == count and normalizer.ret_rms.count == reward_count
    observation, _ = raw.reset(seed=70000)
    assert not np.array_equal(policy_observation(restored, observation), observation)
    total = 0.0
    while True:
        action, _ = restored.predict(policy_observation(restored, observation), deterministic=True,
                                      action_masks=raw.action_masks())
        observation, reward, terminated, truncated, _ = raw.step(int(action))
        total += reward
        if terminated or truncated:
            break
    assert total == pytest.approx(episodes[0]["reward"])
    resumed_dir = tmp_path / "resumed"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(path), "--steps", "16",
                                    "--n-envs", "2", "--max-steps", "2", "--eval-episodes", "1",
                                    "--output", str(resumed_dir)])
    main()
    resumed = load_model(resumed_dir / "final.zip", BastionEnv())
    metadata = json.loads((resumed_dir / "final.json").read_text())
    assert metadata["normalization"] and metadata["curriculum_state"]["goal_index"] == 2
    assert metadata["training_seed"] == 7
    assert metadata["curriculum_state"]["mode"] == "mixed"
    assert resumed.training_schedule == model.training_schedule
    assert resumed.learning_rate == pytest.approx(0.00005)
    assert resumed.ent_coef == pytest.approx(0.005)
    summary = json.loads((resumed_dir / "training-summary.json").read_text())
    assert summary["final_learning_rate"] == pytest.approx(0.00005)
    assert summary["final_entropy_coefficient"] == pytest.approx(0.005)
    previous_step = next(iter(original_optimizer["state"].values()))["step"]
    current_step = next(iter(resumed.policy.optimizer.state_dict()["state"].values()))["step"]
    assert current_step > previous_step
    assert resumed.get_vec_normalize_env().obs_rms.count > count
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(path), "--no-normalize"])
    with pytest.raises(ValueError, match="requires fresh training"):
        main()
    path.with_suffix(".vecnormalize.pkl").unlink()
    with pytest.raises(FileNotFoundError):
        load_model(path, BastionEnv())


def test_tower_type_mix_and_death_order_are_recorded():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    sim.state.resources.earn(1000)
    assert sim.place_tower(0, "cannon") and sim.place_tower(1, "double_cannon")
    sim.towers[1].destroy()
    sim.tick()
    sim.towers[0].destroy()
    deaths = [event for event in sim.events if event["event"] == "structure_destroyed"]
    assert [event["tower_type"] for event in deaths] == ["double_cannon", "cannon"]
    assert deaths[0]["tick"] < deaths[1]["tick"]
    assert env.info()["defense_types_built"] == {"cannon": 1, "double_cannon": 1}


@pytest.mark.parametrize("arguments", [["--lr-final", "0"], ["--ent-final", "nan"],
                                       ["--promotion-successes", "0"], ["--promotion-successes", "51"]])
def test_schedule_and_promotion_flags_validate(arguments, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rl", "train", *arguments])
    with pytest.raises(SystemExit):
        main()
