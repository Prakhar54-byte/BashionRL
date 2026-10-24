from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WaveConfig:
    """Configuration controlling enemy wave generation."""

    initial_enemy_count: int = 5

    # Controls exponential growth: wave 2 initially preserves the
    # original intended increase, while later waves grow exponentially.
    additional_enemies_per_wave: int = 2

    spawn_interval: float = 1.0
    initial_intermission: float = 15.0
    intermission_between_waves: float = 10.0
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
                "Additional enemies per wave must be a "
                "non-negative integer."
            )

        finite_values = {
            "spawn_interval": self.spawn_interval,
            "initial_intermission": self.initial_intermission,
            "intermission_between_waves": self.intermission_between_waves,
            "difficulty_growth": self.difficulty_growth,
        }

        for name, value in finite_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and numeric.")

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

        if (
            self.light_tank_start_wave
            > self.strongest_enemy_start_wave
        ):
            raise ValueError(
                "Light tanks cannot unlock after the strongest enemy tier."
            )


class WaveDirector:
    """Schedule automatic waves with exponential enemy-count growth.

    Enemy types must be ordered from weakest to strongest.

    With four or more enemy types, the final two entries represent
    the light-tank and strongest-enemy tiers, respectively.
    Earlier entries form the standard enemy pool.
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
        """Whether no wave is currently spawning enemies."""
        return (
            not self.wave_active
            and self.enemies_remaining_to_spawn == 0
        )

    # ------------------------------------------------------------------
    # Exponential wave scaling

    def _calculate_enemy_count(self, wave_number: int) -> int:
        """Calculate the exponentially growing size of a wave."""
        if wave_number < 1:
            raise ValueError("Wave number must be at least 1.")

        initial_count = self.config.initial_enemy_count
        additional_count = self.config.additional_enemies_per_wave

        # Preserve the original wave-two increase.
        #
        # Example:
        # initial_count = 5
        # additional_count = 2
        # growth_factor = 1 + 2/5 = 1.4
        growth_factor = (
            1.0 + additional_count / initial_count
        )

        try:
            enemy_count = math.ceil(
                initial_count * growth_factor ** (wave_number - 1)
            )
        except OverflowError:
            raise OverflowError(
                f"Enemy count overflow at wave {wave_number}."
            ) from None

        return max(initial_count, enemy_count)

    def _start_next_wave(self) -> None:
        """Initialize the next wave with exponential enemy scaling."""
        self.wave_number += 1

        enemy_count = self._calculate_enemy_count(self.wave_number)

        self.enemies_remaining_to_spawn = enemy_count
        self.total_enemies_scheduled += enemy_count

        self.spawn_timer = 0.0
        self.wave_active = True

    # ------------------------------------------------------------------
    # Enemy selection

    def _strongest_probability(self, unlock_wave: int) -> float:
        """Calculate the selection probability for an unlocked tier."""
        growth_waves = max(0, self.wave_number - unlock_wave)

        return min(
            0.15 + growth_waves * self.config.difficulty_growth,
            0.85,
        )

    def _select_enemy_type(self) -> str:
        """Select an enemy while respecting tier unlock thresholds."""
        count = len(self.enemy_types)

        if count == 1:
            return self.enemy_types[0]

        # Two enemy types: the last type is the stronger enemy.
        if count == 2:
            strongest_probability = min(
                0.15
                + (self.wave_number - 1)
                * self.config.difficulty_growth,
                0.85,
            )

            if self.rng.random() < strongest_probability:
                return self.enemy_types[-1]

            return self.enemy_types[0]

        # Three enemy types: the final type is the strongest tier.
        if count == 3:
            standard_types = self.enemy_types[:-1]
            strongest_type = self.enemy_types[-1]

            if (
                self.wave_number
                < self.config.strongest_enemy_start_wave
            ):
                return self.rng.choice(standard_types)

            if (
                self.rng.random()
                < self._strongest_probability(
                    self.config.strongest_enemy_start_wave
                )
            ):
                return strongest_type

            return self.rng.choice(standard_types)

        # Four or more enemy types:
        # standard enemies, light tank, strongest enemy.
        standard_types = self.enemy_types[:-2]
        light_tank_type = self.enemy_types[-2]
        strongest_type = self.enemy_types[-1]

        # Try the strongest tier first once unlocked.
        if (
            self.wave_number
            >= self.config.strongest_enemy_start_wave
            and self.rng.random()
            < self._strongest_probability(
                self.config.strongest_enemy_start_wave
            )
        ):
            return strongest_type

        # Light tanks have their own unlock threshold.
        if (
            self.wave_number >= self.config.light_tank_start_wave
            and self.rng.random()
            < self._strongest_probability(
                self.config.light_tank_start_wave
            )
        ):
            return light_tank_type

        return self.rng.choice(standard_types)

    # ------------------------------------------------------------------
    # Automatic wave scheduling

    def update(
        self,
        delta_time: float,
        active_enemy_count: int,
        spawn_enemy: Callable[[str], None],
        *,
        allow_spawning: bool = True,
    ) -> None:
        """Advance the schedule and spawn enemies when their timers expire.

        Args:
            delta_time: Elapsed simulation time in seconds.
            active_enemy_count: Number of living enemies currently active.
            spawn_enemy: Callback used to spawn a selected enemy.
            allow_spawning: Whether wave progression is enabled.
        """
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
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

        if not callable(spawn_enemy):
            raise TypeError("spawn_enemy must be callable.")

        if not isinstance(allow_spawning, bool):
            raise TypeError("allow_spawning must be a boolean.")

        if not allow_spawning:
            return

        remaining_time = float(delta_time)

        while True:
            # ----------------------------------------------------------
            # Wait for remaining enemies, then begin the next countdown.
            # ----------------------------------------------------------
            if not self.wave_active:
                if active_enemy_count > 0:
                    return

                if self.intermission_timer > remaining_time:
                    self.intermission_timer -= remaining_time
                    return

                remaining_time -= self.intermission_timer
                self.intermission_timer = 0.0

                self._start_next_wave()

            # ----------------------------------------------------------
            # If all enemies have been spawned, schedule the next wave.
            # The next wave waits until active enemies are cleared.
            # ----------------------------------------------------------
            if self.enemies_remaining_to_spawn == 0:
                self.wave_active = False
                self.intermission_timer = (
                    self.config.intermission_between_waves
                )

                if remaining_time <= 0:
                    return

                continue

            # ----------------------------------------------------------
            # Wait until the next spawn is due.
            # ----------------------------------------------------------
            if self.spawn_timer > remaining_time:
                self.spawn_timer -= remaining_time
                return

            remaining_time -= self.spawn_timer

            enemy_type = self._select_enemy_type()
            spawn_enemy(enemy_type)

            self.enemies_remaining_to_spawn -= 1
            self.total_enemies_spawned += 1

            # Account for the newly spawned enemy during this update.
            active_enemy_count += 1

            self.spawn_timer = self.config.spawn_interval

            if remaining_time <= 0:
                return
