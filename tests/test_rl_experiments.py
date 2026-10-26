import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("sb3_contrib")

from src.game.rl.__main__ import main
from src.game.rl.checkpoint import load_model
from src.game.rl.diagnostics import ScriptedDefense
from src.game.rl.environment import BastionEnv
from src.game.rl.evaluation import play_episodes, summarize_episodes
from src.game.rl.training import TrainingMetricsCallback


def test_wait_metrics_exclude_forced_decisions_and_report_condensed_time():
    env = BastionEnv(max_steps=2)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    assert info["decisions_with_other_action_legal"] == info["wait_when_other_action_legal"] == 1
    assert info["forced_wait_decisions"] == info["condensed_wait_ticks"] == 0
    assert info["elapsed_ticks"] == 30
    env.reset(seed=0)
    env.simulation.state.resources.spend(200)
    _, reward, _, truncated, info = env.step(0)
    assert truncated and info["forced_wait_decisions"] == 1
    assert info["decisions_with_other_action_legal"] == info["wait_when_other_action_legal"] == 0
    assert info["elapsed_ticks"] == 60 and info["condensed_wait_ticks"] == 30
    episode = dict(info, reward=reward, truncated=True, steps=1,
                   action_diagnostics={"wait": {"legal_decisions": 1, "chosen": 1, "probability_sum": 1.0}})
    assert summarize_episodes([episode])["wait_rate_when_other_action_legal"] is None


def test_structure_death_and_wave_end_timeline_is_once_and_resets():
    env = BastionEnv()
    env.reset(seed=0)
    sim = env.simulation
    assert sim.place_tower(0, "cannon")
    sim.towers[0].destroy()
    sim.towers[0].destroy()
    deaths = [event for event in sim.events if event["event"] == "structure_destroyed"]
    assert len(deaths) == 1
    assert deaths[0]["kind"] == "defense" and deaths[0]["resources"] == 100
    sim.director.wave_number = 1
    sim.director.wave_active = True
    sim.advance(0.1)
    waves = [event for event in sim.events if event["event"] == "wave_completed"]
    assert len(waves) == 1 and waves[0]["wave"] == 1
    assert waves[0]["resources"] == 100 and waves[0]["defenses_alive"] == 0
    env.reset(seed=0)
    assert sim.events == []


def test_terminal_info_carries_episode_timeline():
    env = BastionEnv()
    env.set_wave_goal(1)
    env.reset(seed=0)
    env.simulation.director.wave_number = 1
    env.simulation.director.wave_active = True
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["is_success"]
    assert info["episode_events"][-1]["event"] == "wave_completed"
    assert info["episode_events"][-1]["resources"] == 200


def test_training_success_curves_are_separate_for_each_goal(tmp_path):
    callback = TrainingMetricsCallback(tmp_path)
    callback.model = SimpleNamespace(logger=SimpleNamespace(record=Mock(), record_mean=Mock()))
    callback.num_timesteps = 100
    env = BastionEnv(max_steps=1)
    env.reset(seed=0)
    _, _, _, _, base = env.step(0)
    base["episode_events"] = [{"event": "wave_completed", "wave": 1, "resources": 75}]
    callback.locals = {"infos": [
        dict(base, episode={"r": 2}, wave_goal=1, is_success=True),
        dict(base, episode={"r": -1}, wave_goal=10, is_success=False),
    ]}
    assert callback._on_step()
    callback._on_rollout_end()
    goals = json.loads((tmp_path / "training-goals.json").read_text())
    assert goals["1"]["success_rate"] == 1 and goals["10"]["success_rate"] == 0
    records = [json.loads(line) for line in (tmp_path / "training-episodes.jsonl").read_text().splitlines()]
    assert [record["worker"] for record in records] == [0, 1]
    assert all(record["wait_rate_when_other_action_legal"] == 1 for record in records)
    callback.model.logger.record_mean.assert_called_with("waves/1_unspent_resources", 75)


@pytest.mark.parametrize("economy", [False, True])
def test_scripted_diagnostic_is_legal_seeded_and_separate_from_training(economy):
    env = BastionEnv(max_steps=20)
    policy = ScriptedDefense(env, economy=economy)
    first = play_episodes(env, None, [42], diagnostic_policy=policy)
    second = play_episodes(env, None, [42], diagnostic_policy=policy)
    assert first == second
    assert first[0]["reward_invalid_action"] == 0
    assert first[0]["defenses_built"] > 0
    assert env.simulation.balance.max_waves == 10
    with pytest.raises(ValueError, match="cannot be combined"):
        play_episodes(env, object(), [42], diagnostic_policy=policy)


def test_cli_scripted_diagnostic_labels_its_report(tmp_path, monkeypatch):
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr("sys.argv", ["rl", "diagnose", "--economy", "--episodes", "1",
                                    "--max-steps", "1", "--output", str(output)])
    main()
    data = json.loads(output.read_text())
    assert data["policy"] == "scripted_economy" and data["results"][0]["wave_goal"] == 10


def test_matched_rollout_runs_record_equal_iterations_and_flat_resume(tmp_path, monkeypatch):
    summaries = []
    for strategy, workers, steps in (("flat", 1, 512), ("hierarchical", 2, 256)):
        output = tmp_path / strategy
        monkeypatch.setattr("sys.argv", ["rl", "train", "--strategy", strategy,
                                        "--n-envs", str(workers), "--n-steps", str(steps),
                                        "--batch-size", "128", "--n-epochs", "2", "--steps", "1024",
                                        "--learning-rate", "0.00001",
                                        "--max-steps", "2", "--eval-freq", "512", "--eval-episodes", "2",
                                        "--output", str(output)])
        main()
        summary = json.loads((output / "training-summary.json").read_text())
        assert summary["rollout_size"] == 512 and summary["actual_steps"] == 1024
        assert summary["completed_rollouts"] == 2
        assert summary["completed_ppo_epochs"] == 4
        assert summary["completed_optimizer_steps"] == 16
        validation = json.loads((output / "evaluations.json").read_text())
        assert [item["timesteps"] for item in validation] == [512, 1024]
        assert [item["final"] for item in validation] == [False, True]
        assert json.loads((output / "latest.json").read_text())["checkpoint_steps"] == 1024
        assert (output / "training-goals.json").is_file()
        assert (output / "training-episodes.jsonl").is_file()
        summaries.append(summary)
    assert summaries[0]["training_parameters"]["validation"] == summaries[1]["training_parameters"]["validation"]
    resumed = tmp_path / "resumed"
    monkeypatch.setattr("sys.argv", ["rl", "train", "--resume", str(tmp_path / "flat/final.zip"),
                                    "--steps", "512", "--max-steps", "2", "--eval-episodes", "1",
                                    "--output", str(resumed)])
    main()
    assert load_model(resumed / "final.zip", BastionEnv()).num_timesteps == 1536
    summary = json.loads((resumed / "training-summary.json").read_text())
    assert summary["training_parameters"]["strategy"] == "flat"
    assert summary["starting_timesteps"] == 1024 and summary["ending_timesteps"] == 1536
    assert summary["actual_steps"] == 512 and summary["resumed_from"]


@pytest.mark.parametrize("arguments", [["--n-steps", "0"], ["--batch-size", "0"],
                                       ["--n-epochs", "-1"], ["--n-envs", "0"]])
def test_cli_rejects_invalid_rollout_controls(arguments, monkeypatch):
    monkeypatch.setattr("sys.argv", ["rl", "train", *arguments])
    with pytest.raises(SystemExit):
        main()
