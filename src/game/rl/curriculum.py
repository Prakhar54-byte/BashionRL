"""Adjust training episode goals; validation and gameplay retain the full task."""

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, env, maximum, *, mode="staged", promotion_successes=40, state=None):
        super().__init__()
        self.env = env
        self.goals = list(range(1, maximum + 1))
        if (mode not in ("staged", "mixed") or isinstance(promotion_successes, bool)
                or not isinstance(promotion_successes, int) or not 1 <= promotion_successes <= 50):
            raise ValueError("Curriculum requires staged/mixed mode and 1–50 promotion successes.")
        self.mode, self.promotion_successes = mode, promotion_successes
        self.goal_index = 0
        self.successes = []
        if state is not None:
            index, successes = state["goal_index"], state["successes"]
            if (state.get("maximum") != maximum or state.get("mode") != mode
                    or state.get("promotion_successes") != promotion_successes
                    or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < maximum
                    or len(successes) > 50 or any(type(value) is not bool for value in successes)):
                raise ValueError("Saved curriculum state does not match this experiment.")
            self.goal_index, self.successes = index, list(successes)
        env.env_method("set_wave_goal", self.goals[self.goal_index], mixed=self.mode == "mixed")

    def state_dict(self):
        return {"maximum": self.goals[-1], "mode": self.mode,
                "promotion_successes": self.promotion_successes,
                "goal_index": self.goal_index, "successes": list(self.successes)}

    def _on_step(self):
        previous_index = self.goal_index
        for info in self.locals["infos"]:
            if "episode" in info and info["wave_goal"] == self.goals[self.goal_index]:
                self.successes.append(info["is_success"])
                self.successes = self.successes[-50:]
        if len(self.successes) == 50 and sum(self.successes) >= self.promotion_successes:
            self.goal_index = min(self.goal_index + 1, len(self.goals) - 1)
            self.successes.clear()
        goal = self.goals[self.goal_index]
        if self.goal_index != previous_index:
            self.env.env_method("set_wave_goal", goal, mixed=self.mode == "mixed")
        self.logger.record("curriculum/next_wave_goal", goal)
        return True
