from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WaveConfig:
    """Configuration controlling enemy wave generation."""

    initial_enemy_count: int = 5
    additional_enemies_per_wave: int = 2
    spawn_interval: float = 1.0
    initial_intermission: float = 3.0
    intermission_between_waves: float = 5.0
    difficulty_growth: float = 0.15

    # Unlock thresholds for ordered enemy tiers.
    light_tank_start_wave: int = 2
    strongest_enemy_start_wave: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.initial_enemy_count, bool)
            or not isinstance(self.initial_enemy_count, int)
            or self.initial_enemy_count < 1
        ):
            raise ValueError(
                "Initial enemy count must be a positive integer."
            )

        if (
            isinstance(self.additional_enemies_per_wave, bool)
            or not isinstance(self.additional_enemies_per_wave, int)
            or self.additional_enemies_per_wave < 0
        ):
            raise ValueError(
                "Additional enemies per wave must be a non-negative integer."
            )

        finite_values = {
            "spawn_interval": self.spawn_interval,
            "initial_intermission": self.initial_intermission,
            "intermission_between_waves": self.intermission_between_waves,
            "difficulty_growth": self.difficulty_growth,
        }

        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.spawn_interval <= 0:
            raise ValueError("Spawn interval must be positive.")

        if self.initial_intermission < 0:
            raise ValueError(
                "Initial intermission cannot be negative."
            )

        if self.intermission_between_waves < 0:
            raise ValueError(
                "Wave intermission cannot be negative."
            )

        if self.difficulty_growth < 0:
            raise ValueError(
                "Difficulty growth cannot be negative."
            )

        for name, value in (
            ("light_tank_start_wave", self.light_tank_start_wave),
            ("strongest_enemy_start_wave", self.strongest_enemy_start_wave),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if self.light_tank_start_wave > self.strongest_enemy_start_wave:
            raise ValueError(
                "Light tanks cannot unlock after the strongest enemy tier."
            )


class WaveDirector:
    """Schedules waves and selects enemy types according to wave progression.

    Enemy types must be ordered from weakest to strongest.

    With four or more enemy types, the final two entries are treated as
    the light-tank tier and the heavy/strongest tier, respectively.
    Earlier entries are the ordinary enemy pool.
    """

    def __init__(
        self,
        enemy_types: Sequence[str],
        config: WaveConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not enemy_types:
            raise ValueError("At least one enemy type is required.")

        if any(
            not isinstance(name, str) or not name.strip()
            for name in enemy_types
        ):
            raise ValueError(
                "Enemy type names must be non-empty strings."
            )

        if len(set(enemy_types)) != len(enemy_types):
            raise ValueError("Enemy type names must be unique.")

        self.enemy_types = tuple(enemy_types)
        self.config = config or WaveConfig()
        self.rng = random.Random(seed)

        self.wave_number = 0
        self.enemies_remaining_to_spawn = 0
        self.spawn_timer = 0.0
        self.intermission_timer = self.config.initial_intermission
        self.wave_active = False

        self.total_enemies_scheduled = 0
        self.total_enemies_spawned = 0

    @property
    def waiting_for_next_wave(self) -> bool:
        return (
            not self.wave_active
            and self.enemies_remaining_to_spawn == 0
        )

    def _start_next_wave(self) -> None:
        """Initialize the next wave and calculate its enemy count."""
        self.wave_number += 1

        self.enemies_remaining_to_spawn = (
            self.config.initial_enemy_count
            + (self.wave_number - 1)
            * self.config.additional_enemies_per_wave
        )

        self.total_enemies_scheduled += (
            self.enemies_remaining_to_spawn
        )

        self.spawn_timer = 0.0
        self.wave_active = True

    def _strongest_probability(self, unlock_wave: int) -> float:
        """Calculate the probability of selecting a newly unlocked tier."""
        growth_waves = max(0, self.wave_number - unlock_wave)

        return min(
            0.15 + growth_waves * self.config.difficulty_growth,
            0.85,
        )

    def _select_enemy_type(self) -> str:
        """Select an enemy while respecting its tier's unlock wave."""
        count = len(self.enemy_types)

        if count == 1:
            return self.enemy_types[0]

        # Preserve the existing behavior for a simple two-enemy setup.
        if count == 2:
            strongest_probability = min(
                0.15
                + (self.wave_number - 1)
                * self.config.difficulty_growth,
                0.85,
            )

            if self.rng.random() < strongest_probability:
                return self.enemy_types[-1]

            return self.rng.choice(self.enemy_types[:-1])

        # Three types: two standard enemies and one strongest tier.
        if count == 3:
            standard_types = self.enemy_types[:-1]
            strongest_type = self.enemy_types[-1]

            if (
                self.wave_number
                < self.config.strongest_enemy_start_wave
            ):
                return self.rng.choice(standard_types)

            if self.rng.random() < self._strongest_probability(
                self.config.strongest_enemy_start_wave
            ):
                return strongest_type

            return self.rng.choice(standard_types)

        # Four or more types:
        # ordinary soldiers..., light tank, heavy tank.
        standard_types = self.enemy_types[:-2]
        light_tank_type = self.enemy_types[-2]
        strongest_type = self.enemy_types[-1]

        # Heavy tanks are only eligible after their unlock wave.
        if (
            self.wave_number
            >= self.config.strongest_enemy_start_wave
            and self.rng.random()
            < self._strongest_probability(
                self.config.strongest_enemy_start_wave
            )
        ):
            return strongest_type

        # Light tanks are eligible from their own unlock wave.
        if (
            self.wave_number >= self.config.light_tank_start_wave
            and self.rng.random()
            < self._strongest_probability(
                self.config.light_tank_start_wave
            )
        ):
            return light_tank_type

        return self.rng.choice(standard_types)

    def update(
        self,
        delta_time: float,
        active_enemy_count: int,
        spawn_enemy: Callable[[str], None],
        *,
        allow_spawning: bool = True,
    ) -> None:
        """Advance the wave schedule and spawn enemies when due.

        Args:
            delta_time: Elapsed simulation time in seconds.
            active_enemy_count: Number of living enemies on the map.
            spawn_enemy: Callback that creates the requested enemy.
            allow_spawning: Whether wave progression is enabled.
        """
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if (
            isinstance(active_enemy_count, bool)
            or not isinstance(active_enemy_count, int)
            or active_enemy_count < 0
        ):
            raise ValueError(
                "Active enemy count must be a non-negative integer."
            )

        if not allow_spawning:
            return

        remaining_time = delta_time

        while True:
            if not self.wave_active:
                # Wait until all enemies have been cleared.
                if active_enemy_count > 0:
                    return

                if self.intermission_timer > remaining_time:
                    self.intermission_timer -= remaining_time
                    return

                remaining_time -= self.intermission_timer
                self.intermission_timer = 0.0
                self._start_next_wave()

            if self.enemies_remaining_to_spawn == 0:
                self.wave_active = False
                self.intermission_timer = (
                    self.config.intermission_between_waves
                )

                if remaining_time <= 0:
                    return

                continue

            if self.spawn_timer > remaining_time:
                self.spawn_timer -= remaining_time
                return

            remaining_time -= self.spawn_timer
            self.spawn_timer = self.config.spawn_interval

            enemy_type = self._select_enemy_type()
            spawn_enemy(enemy_type)

            self.enemies_remaining_to_spawn -= 1
            self.total_enemies_spawned += 1
            active_enemy_count += 1

            if remaining_time <= 0:
                return
