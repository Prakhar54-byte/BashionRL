import math

import pygame


class Projectile:
    def __init__(
        self,
        position: pygame.Vector2,
        target,
        speed: float,
        damage: float,
    ) -> None:
        if not math.isfinite(speed) or speed <= 0 or not math.isfinite(damage) or damage < 0:
            raise ValueError("Projectile speed must be positive and damage non-negative, both finite.")
        self.position = position.copy()
        self.target = target
        self.speed = speed
        self.damage = damage
        self.alive = True

    def update(self, delta_time: float, valid_targets=None) -> None:
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")
        if not self.alive:
            return

        if not self.target.alive or (valid_targets is not None and self.target not in valid_targets):
            self.alive = False
            return

        offset = self.target.position - self.position
        distance = offset.length()
        travel_distance = self.speed * max(0.0, delta_time)

        if distance <= max(6.0, travel_distance):
            self.target.take_damage(self.damage)
            self.alive = False
            return

        if distance > 0:
            self.position += offset / distance * travel_distance

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
    ) -> None:
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

