"""Small PPO initialization and outcome-based validation helpers."""

import json
import math
from collections import deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from src.game.rl.checkpoint import save_checkpoint
from src.game.rl.evaluation import play_episodes, summarize_episodes


def initialize_flat_action_bias(policy, env):
    """Reproduce the previous flat baseline's initial kind prior."""
    with torch.no_grad():
        for group in env.action_groups.values():
            count = group.stop - group.start
            if count:
                policy.action_net.bias[group] = -math.log(count)


class AnnealingCallback(BaseCallback):
    """Use absolute training steps so recovery does not restart the schedules."""

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        from stable_baselines3.common.utils import FloatSchedule

        schedule = self.model.training_schedule
        progress = min(1.0, max(0.0, (self.num_timesteps - schedule["start_step"])
                                     / (schedule["end_step"] - schedule["start_step"])))
        lr = schedule["lr_start"] + progress * (schedule["lr_end"] - schedule["lr_start"])
        entropy = schedule["entropy_start"] + progress * (schedule["entropy_end"] - schedule["entropy_start"])
        self.model.learning_rate = lr
        self.model.lr_schedule = FloatSchedule(lr)
        self.model.ent_coef = entropy
        self.logger.record("train/scheduled_learning_rate", lr)
        self.logger.record("train/entropy_coefficient", entropy)


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.goals = {}

    def _on_step(self):
        records = []
        for worker, info in enumerate(self.locals["infos"]):
            if "episode" not in info:
                continue
            goal = info["wave_goal"]
            stats = self.goals.setdefault(goal, {"episodes": 0, "wins": 0, "choices": 0,
                                                 "waits": 0, "recent": deque(maxlen=50)})
            stats["episodes"] += 1
            stats["wins"] += int(info["is_success"])
            stats["recent"].append(bool(info["is_success"]))
            choices = info["decisions_with_other_action_legal"]
            stats["choices"] += choices
            stats["waits"] += info["wait_when_other_action_legal"]
            for event in info["episode_events"]:
                if event["event"] == "wave_completed":
                    self.logger.record_mean(f"waves/{event['wave']}_unspent_resources", event["resources"])
                    for name, count in event.get("defense_types_alive", {}).items():
                        self.logger.record_mean(f"waves/{event['wave']}_{name}_alive", count)
                    for name, count in event.get("defense_types_built", {}).items():
                        self.logger.record_mean(f"waves/{event['wave']}_{name}_built", count)
            records.append({
                "timesteps": self.num_timesteps, "worker": worker, "wave_goal": goal,
                "success": info["is_success"], "return": info["episode"]["r"],
                "wait_rate_when_other_action_legal": info["wait_when_other_action_legal"] / choices if choices else None,
                "resources_at_end": info["resources"], "events": info["episode_events"],
                "defense_types_alive": info["defense_types_alive"],
                "defense_types_built": info["defense_types_built"],
            })
        if records:
            with (self.output / "training-episodes.jsonl").open("a") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")
        return True

    def _on_rollout_end(self):
        summary = {}
        for goal, stats in self.goals.items():
            summary[goal] = {"episodes": stats["episodes"], "wins": stats["wins"],
                             "success_rate": stats["wins"] / stats["episodes"],
                             "recent_success_rate": sum(stats["recent"]) / len(stats["recent"]),
                             "wait_rate_when_other_action_legal": stats["waits"] / stats["choices"] if stats["choices"] else None}
            for key, value in summary[goal].items():
                if value is not None:
                    self.logger.record(f"goals/{goal}_{key}", value)
        (self.output / "training-goals.json").write_text(json.dumps(summary, indent=2) + "\n")


class OutcomeEvalCallback(BaseCallback):
    def __init__(self, env, output, *, eval_freq, episodes, seed, training_seed, steps,
                 training_parameters, curriculum=None):
        super().__init__()
        self.env, self.output = env, output
        self.eval_freq, self.episodes, self.seed = eval_freq, episodes, seed
        self.training_seed, self.steps = training_seed, steps
        self.training_parameters = training_parameters
        self.curriculum = curriculum
        self.history = []
        self.best_score = None

    def _on_training_start(self):
        self.last_eval_step = self.num_timesteps

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # The preceding PPO update has finished; checkpoints contain all completed rollouts.
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.evaluate()
            self.last_eval_step = self.num_timesteps

    def _on_training_end(self):
        # The final PPO update has no subsequent rollout-start callback.
        self.evaluate(final=True)

    def evaluate(self, *, final=False):
        episodes = play_episodes(self.env, self.model, range(self.seed, self.seed + self.episodes))
        summary = summarize_episodes(episodes)
        self.history.append(dict(summary, timesteps=self.num_timesteps, final=final, results=episodes))
        score = tuple(summary[key] for key in ("win_rate", "mean_waves_completed", "mean_base_health"))
        path = self.output / "latest.zip"
        save_checkpoint(path, self.model, self.env, seed=self.training_seed, steps=self.steps,
                        training_parameters=self.training_parameters, curriculum=self.curriculum)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            path = self.output / "best_model.zip"
            save_checkpoint(path, self.model, self.env, seed=self.training_seed, steps=self.steps,
                            training_parameters=self.training_parameters, curriculum=self.curriculum)
        (self.output / "evaluations.json").write_text(json.dumps(self.history, indent=2) + "\n")
        np.savez(self.output / "evaluations.npz",
                 timesteps=[item["timesteps"] for item in self.history],
                 results=[[episode["reward"] for episode in item["results"]] for item in self.history],
                 ep_lengths=[[episode["steps"] for episode in item["results"]] for item in self.history],
                 successes=[[episode["is_success"] for episode in item["results"]] for item in self.history])
        for key, value in summary.items():
            if key != "action_diagnostics" and value is not None:
                self.logger.record(f"eval/{key}", value)
        for name, values in summary["action_diagnostics"].items():
            for key, value in values.items():
                if value is not None:
                    self.logger.record(f"eval_actions/{name}_{key}", value)
        print(f"Eval steps={self.num_timesteps}: wins={summary['win_rate']:.1%}, "
              f"waves={summary['mean_waves_completed']:.2f}, "
              f"King HP={summary['mean_base_health']:.1f}")
        if final:
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(self.num_timesteps)
