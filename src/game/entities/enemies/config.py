from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EnemyConfig:
    """Movement, durability, rewards, and King Tower combat settings."""

    name: str
    speed: float
    max_health: float
    size_scale: float = 0.5
    leak_damage: float = 10.0
    kill_reward: int = 25

    attack_damage: float = 15.0
    attack_range: float = 45.0
    attack_interval: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Enemy name cannot be empty.")

        numeric_values = {
            "speed": self.speed,
            "max_health": self.max_health,
            "size_scale": self.size_scale,
            "leak_damage": self.leak_damage,
            "attack_damage": self.attack_damage,
            "attack_range": self.attack_range,
            "attack_interval": self.attack_interval,
        }

        for name, value in numeric_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Enemy {name} must be finite.")

        for name in (
            "speed",
            "max_health",
            "size_scale",
            "attack_damage",
            "attack_range",
            "attack_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Enemy {name} must be positive.")

        if self.leak_damage < 0:
            raise ValueError("Leak damage cannot be negative.")

        if (
            isinstance(self.kill_reward, bool)
            or not isinstance(self.kill_reward, int)
            or self.kill_reward < 0
        ):
            raise ValueError("Kill reward must be a non-negative integer.")
