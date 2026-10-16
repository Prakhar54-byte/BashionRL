from __future__ import annotations

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

    def __post_init__(self) -> None:
        if self.initial_enemy_count < 1:
            raise ValueError("Initial enemy count must be positive.")

        if self.additional_enemies_per_wave < 0:
            raise ValueError("Additional enemies per wave cannot be negative.")

        if self.spawn_interval <= 0:
            raise ValueError("Spawn interval must be positive.")

        if self.initial_intermission < 0:
            raise ValueError("Initial intermission cannot be negative.")

        if self.intermission_between_waves < 0:
            raise ValueError("Wave intermission cannot be negative.")

        if self.difficulty_growth < 0:
            raise ValueError("Difficulty growth cannot be negative.")


class WaveDirector:
    """Schedules waves and selects which enemy types to spawn."""

    def __init__(
        self,
        enemy_types: Sequence[str],
        config: WaveConfig | None = None,
        seed: int | None = None,
    ) -> None:
        if not enemy_types:
            raise ValueError("At least one enemy type is required.")

        if any(not name.strip() for name in enemy_types):
            raise ValueError("Enemy type names cannot be empty.")

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
        return not self.wave_active and self.enemies_remaining_to_spawn == 0

    def _start_next_wave(self) -> None:
        self.wave_number += 1

        self.enemies_remaining_to_spawn = (
            self.config.initial_enemy_count
            + (self.wave_number - 1) * self.config.additional_enemies_per_wave
        )

        self.total_enemies_scheduled += self.enemies_remaining_to_spawn

        self.spawn_timer = 0.0
        self.wave_active = True

    def _select_enemy_type(self) -> str:
        """Choose an enemy, gradually favoring stronger variants."""

        if len(self.enemy_types) == 1:
            return self.enemy_types[0]

        # Enemy types are ordered from weakest to strongest.
        # Higher waves gradually increase the chance of stronger enemies.
        strongest_probability = min(
            0.15 + (self.wave_number - 1) * self.config.difficulty_growth,
            0.85,
        )

        if self.rng.random() < strongest_probability:
            return self.enemy_types[-1]

        return self.rng.choice(self.enemy_types[:-1])

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
            delta_time: Time elapsed since the previous update, in seconds.
            active_enemy_count: Number of living enemies currently on the map.
            spawn_enemy: Callback that creates an enemy of the given type.
            allow_spawning: Set False to pause wave progression and spawning.
        """
        if delta_time < 0:
            raise ValueError("Delta time cannot be negative.")

        if active_enemy_count < 0:
            raise ValueError("Active enemy count cannot be negative.")

        if not allow_spawning:
            return

        remaining_time = delta_time

        while True:
            if not self.wave_active:
                # Never start another wave while enemies remain alive.
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
                self.intermission_timer = self.config.intermission_between_waves

                # Don't start another wave in this same update if no
                # time remains. This also avoids zero-time loop issues.
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

            # Account for the new enemy immediately.
            active_enemy_count += 1

            if remaining_time <= 0:
                return
