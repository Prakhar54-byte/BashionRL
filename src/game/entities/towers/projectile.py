"""Projectile fired by a DefenseTower toward an enemy target.

A projectile homes in on the enemy it was fired at and deals its
configured damage on impact.  It becomes inactive (``alive = False``)
either on hit or when the target dies before it arrives.
"""

from __future__ import annotations

import math

import pygame

from src.game.entities.enemies.base import Enemy


class Projectile:
    """A projectile that travels toward a selected enemy."""

    def __init__(
        self,
        position: pygame.Vector2,
        target: Enemy,
        speed: float,
        damage: float,
    ) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Projectile speed must be finite and positive.")

        if not math.isfinite(damage) or damage < 0:
            raise ValueError("Projectile damage must be finite and non-negative.")

        self.position = pygame.Vector2(position)
        self.target = target
        self.speed = speed
        self.damage = damage
        self.alive = True

    def update(self, delta_time: float) -> None:
        """Move the projectile and damage its target on impact."""
        if not self.alive:
            return

        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")

        if not self.target.alive or self.target.reached_end:
            self.alive = False
            return

        offset = self.target.position - self.position
        distance = offset.length()
        travel_distance = self.speed * delta_time

        if distance <= max(6.0, travel_distance):
            self.target.take_damage(self.damage)
            self.alive = False
            return

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
