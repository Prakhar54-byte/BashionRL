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

    structure_kind = "structure"

    def __init__(
        self,
        structure_id: StructureId,
        position: pygame.Vector2,
        max_health: float,
        *,
        on_health_changed=None,
    ) -> None:
        self._validate_structure_id(structure_id)

        if not isinstance(position, pygame.Vector2):
            try:
                position = pygame.Vector2(position)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Structure position must contain valid x and y coordinates."
                ) from exc

        if not (
            math.isfinite(position.x)
            and math.isfinite(position.y)
        ):
            raise ValueError(
                "Structure position coordinates must be finite."
            )

        if (
            isinstance(max_health, bool)
            or not isinstance(max_health, (int, float))
            or not math.isfinite(max_health)
            or max_health <= 0
        ):
            raise ValueError(
                "Maximum health must be finite and positive."
            )

        self.structure_id = structure_id
        self.position = position.copy()

        self.max_health = float(max_health)
        self._health = self.max_health
        self.on_health_changed = on_health_changed

    @staticmethod
    def _validate_structure_id(
        structure_id: StructureId,
    ) -> None:
        """Validate a structure's identifier."""
        if isinstance(structure_id, bool):
            raise TypeError(
                "Structure ID must be a non-empty string or integer."
            )

        if isinstance(structure_id, int):
            return

        if isinstance(structure_id, str) and structure_id.strip():
            return

        raise ValueError(
            "Structure ID must be a non-empty string or integer."
        )

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

        self.health_changed()
        return previous_health - self._health

    def destroy(self) -> None:
        """Immediately destroy this structure."""
        self.take_damage(self.health)

    def _restore_health(self, amount: float) -> float:
        """Apply a subclass's gradual healing and notify the simulation."""
        self._validate_amount(amount, "Restoration amount")

        if amount == 0.0:
            return 0.0

        previous_health = self._health
        self._health = min(
            self.max_health,
            self._health + amount,
        )

        if self._health != previous_health:
            self.health_changed()
        return self._health - previous_health

    @staticmethod
    def _validate_amount(
        amount: float,
        label: str,
    ) -> None:
        """Ensure damage and health restoration amounts are valid."""
        if isinstance(amount, bool) or not isinstance(
            amount, (int, float)
        ):
            raise TypeError(f"{label} must be numeric.")

        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                f"{label} must be finite and non-negative."
            )

    def __repr__(self) -> str:
        """Return a concise representation useful for debugging."""
        return (
            f"{type(self).__name__}("
            f"id={self.structure_id!r}, "
            f"health={self.health:.1f}/{self.max_health:.1f}, "
            f"alive={self.alive})"
        )

    def health_changed(self):
        if self.on_health_changed is not None:
            self.on_health_changed(self)

    def draw_health(self, surface, map_width, map_height):
        sx, sy = surface.get_width() / map_width, surface.get_height() / map_height
        x, y = round(self.position.x * sx) - 20, round(self.position.y * sy) - 27
        pygame.draw.rect(surface, (80, 30, 30), (x, y, 40, 5))
        pygame.draw.rect(surface, (60, 210, 100), (x, y, round(40 * self.health / self.max_health), 5))

    def draw(self, surface, map_width, map_height, show_range=False):
        if map_width <= 0 or map_height <= 0:
            return
        center = (round(self.position.x * surface.get_width() / map_width),
                  round(self.position.y * surface.get_height() / map_height))
        color = (90, 90, 90) if self.destroyed else (
            (235, 190, 50) if self.structure_kind == "king" else (60, 175, 150)
        )
        pygame.draw.circle(surface, color, center, 20)
        if self.destroyed:
            pygame.draw.line(surface, (35, 35, 35), (center[0]-10, center[1]-10),
                             (center[0]+10, center[1]+10), 3)
        self.draw_health(surface, map_width, map_height)
