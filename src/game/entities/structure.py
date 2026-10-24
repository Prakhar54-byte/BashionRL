from __future__ import annotations

import math
from typing import TypeAlias

import pygame

StructureId: TypeAlias = int | str


class Structure:
    """Base class for damageable structures in BastionRL.

    Every structure shares:
        - A unique identifier.
        - A world-space position.
        - Maximum and current health.
        - Damage handling.
        - Destruction state.

    Specialized behavior belongs in subclasses:
        - DefenseTower: attacks enemies.
        - KingTower: passive objective with timed repairs.
        - ResourceTower: generates income while operational.

    This class deliberately does not implement repair timers,
    resource costs, targeting, or income generation.
    """

    def __init__(
        self,
        structure_id: StructureId,
        position: pygame.Vector2,
        max_health: float,
    ) -> None:
        self._validate_structure_id(structure_id)

        if not isinstance(position, pygame.Vector2):
            try:
                position = pygame.Vector2(position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Structure position must contain valid x and y coordinates."
                ) from exc

        if not (math.isfinite(position.x) and math.isfinite(position.y)):
            raise ValueError("Structure position coordinates must be finite.")

        if (
            isinstance(max_health, bool)
            or not isinstance(max_health, (int, float))
            or not math.isfinite(max_health)
            or max_health <= 0
        ):
            raise ValueError("Maximum health must be finite and positive.")

        self.structure_id = structure_id
        self.position = position.copy()

        self.max_health = float(max_health)
        self._health = self.max_health

    @staticmethod
    def _validate_structure_id(
        structure_id: StructureId,
    ) -> None:
        """Validate a structure's identifier."""
        if isinstance(structure_id, bool):
            raise TypeError("Structure ID must be a non-empty string or integer.")

        if isinstance(structure_id, int):
            return

        if isinstance(structure_id, str) and structure_id.strip():
            return

        raise ValueError("Structure ID must be a non-empty string or integer.")

    @property
    def health(self) -> float:
        """Current structure health."""
        return self._health

    @property
    def alive(self) -> bool:
        """Whether the structure is operational and has health remaining."""
        return self._health > 0.0

    @property
    def destroyed(self) -> bool:
        """Whether the structure has been destroyed."""
        return not self.alive

    @property
    def health_fraction(self) -> float:
        """Current health normalized to the interval [0.0, 1.0]."""
        return self._health / self.max_health

    def take_damage(self, amount: float) -> float:
        """Apply damage and return the amount of health actually lost.

        Damage is clamped so health cannot become negative.
        A destroyed structure cannot take additional damage.
        """
        self._validate_amount(amount, "Damage")

        if not self.alive or amount == 0.0:
            return 0.0

        previous_health = self._health
        self._health = max(0.0, self._health - amount)

        return previous_health - self._health

    def destroy(self) -> None:
        """Immediately destroy this structure."""
        self._health = 0.0

    def _restore_health(self, amount: float) -> float:
        """Restore health for use by a subclass's completed repair action.

        This method does not start a repair or advance a repair timer.
        KingTower should call it only after its configured repair
        duration has elapsed and the simulation has approved the repair.

        Returns:
            The amount of health actually restored.
        """
        self._validate_amount(amount, "Restoration amount")

        if amount == 0.0:
            return 0.0

        previous_health = self._health
        self._health = min(
            self.max_health,
            self._health + amount,
        )

        return self._health - previous_health

    @staticmethod
    def _validate_amount(
        amount: float,
        label: str,
    ) -> None:
        """Ensure damage and health restoration amounts are valid."""
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"{label} must be numeric.")

        if not math.isfinite(amount) or amount < 0:
            raise ValueError(f"{label} must be finite and non-negative.")

    def __repr__(self) -> str:
        """Return a concise representation useful for debugging."""
        return (
            f"{type(self).__name__}("
            f"id={self.structure_id!r}, "
            f"health={self.health:.1f}/{self.max_health:.1f}, "
            f"alive={self.alive})"
        )
