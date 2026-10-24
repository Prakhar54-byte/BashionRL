
"""Projectile used by defense towers and tanks.

Projectiles can target either an Enemy or a damageable Structure.
They travel toward their target and apply damage on impact.
"""

from __future__ import annotations

import math

import pygame

from src.game.entities.enemies.base import Enemy
from src.game.entities.structure import Structure


class Projectile:
    """A projectile that damages an enemy or a structure."""

    def __init__(
        self,
        position: pygame.Vector2,
        target: Enemy | Structure,
        speed: float,
        damage: float,
    ) -> None:
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or speed <= 0
        ):
            raise ValueError(
                "Projectile speed must be finite and positive."
            )

        if (
            isinstance(damage, bool)
            or not isinstance(damage, (int, float))
            or not math.isfinite(damage)
            or damage < 0
        ):
            raise ValueError(
                "Projectile damage must be finite and non-negative."
            )

        self.position = pygame.Vector2(position)
        self.target = target
        self.speed = float(speed)
        self.damage = float(damage)
        self.alive = True

    def update(self, delta_time: float) -> None:
        """Move toward the target and apply damage when it is reached."""
        if not self.alive:
            return

        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        # Stop if the target has been destroyed.
        if not self.target.alive:
            self.alive = False
            return

        # Enemies that reach the end of their route are no longer valid
        # targets. Structures do not have a reached_end attribute.
        if (
            isinstance(self.target, Enemy)
            and self.target.reached_end
        ):
            self.alive = False
            return

        offset = self.target.position - self.position
        distance = offset.length()
        travel_distance = self.speed * delta_time

        # Hit the target if the projectile can reach it this frame.
        if distance <= max(6.0, travel_distance):
            self.target.take_damage(self.damage)
            self.alive = False
            return

        # Otherwise, advance toward the target.
        if distance > 0.0:
            self.position += offset / distance * travel_distance

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
    ) -> None:
        """Render the projectile using map-space coordinates."""
        if not self.alive or map_width <= 0 or map_height <= 0:
            return

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        point = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        pygame.draw.circle(surface, (255, 220, 90), point, 4)
        pygame.draw.circle(surface, (255, 255, 220), point, 2)
