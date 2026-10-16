import math
from dataclasses import dataclass

import pygame

from src.game.terrain.pathway import Pathway


@dataclass(frozen=True, slots=True)
class EnemyConfig:
    """Configuration for a specific enemy variant."""

    name: str
    speed: float
    max_health: float
    size_scale: float = 0.5
    leak_damage: float = 10.0
    kill_reward: int = 25

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Enemy name cannot be empty.")
        if self.speed <= 0:
            raise ValueError("Enemy speed must be positive.")
        if self.max_health <= 0:
            raise ValueError("Enemy health must be positive.")
        if self.size_scale <= 0:
            raise ValueError("Enemy size scale must be positive.")
        if self.leak_damage < 0:
            raise ValueError("Leak damage cannot be negative.")
        if (
            isinstance(self.kill_reward, bool)
            or not isinstance(self.kill_reward, int)
            or self.kill_reward < 0
        ):
            raise ValueError("Kill reward must be a non-negative integer.")


class Enemy:
    """Enemy that follows a pathway and can take damage."""

    def __init__(
        self,
        pathway: Pathway,
        sprite: pygame.Surface,
        config: EnemyConfig,
    ) -> None:
        if len(pathway.points) < 2:
            raise ValueError("The pathway needs at least two points.")

        self.pathway = pathway
        self.sprite = sprite
        self.config = config

        self.name = config.name
        self.speed = config.speed
        self.size_scale = config.size_scale
        self.leak_damage = config.leak_damage

        self.max_health = config.max_health
        self.health = config.max_health

        self.position = pathway.points[0].copy()
        self.waypoint_index = 1

        self.alive = True
        self.reached_end = False

        # Source sprites face East (right).
        self.direction = pygame.Vector2(1, 0)

    def update(self, delta_time: float) -> None:
        """Move along the pathway independently of frame rate."""
        if not self.alive or self.reached_end:
            return

        distance_remaining = self.speed * max(0.0, delta_time)

        while distance_remaining > 0 and not self.reached_end:
            target = self.pathway.points[self.waypoint_index]
            offset = target - self.position
            distance_to_target = offset.length()

            if distance_to_target <= 1e-6:
                self.position = target.copy()
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.pathway.points):
                    self.reached_end = True

                continue

            self.direction = offset / distance_to_target

            if distance_remaining >= distance_to_target:
                self.position = target.copy()
                distance_remaining -= distance_to_target
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.pathway.points):
                    self.reached_end = True
            else:
                self.position += self.direction * distance_remaining
                distance_remaining = 0.0

    def take_damage(self, amount: float) -> None:
        """Apply damage and mark the enemy dead when health reaches zero."""
        if amount < 0:
            raise ValueError("Damage cannot be negative.")

        if not self.alive:
            return

        self.health = max(0.0, self.health - amount)

        if self.health == 0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        """Render the enemy facing its movement direction."""
        if not self.alive or self.reached_end:
            return

        scale_x = surface.get_width() / self.pathway.map_width
        scale_y = surface.get_height() / self.pathway.map_height

        screen_x = self.position.x * scale_x
        screen_y = self.position.y * scale_y

        angle = math.degrees(math.atan2(-self.direction.y, self.direction.x))
        rotated_sprite = pygame.transform.rotate(self.sprite, angle)

        sprite_width = max(
            1,
            round(rotated_sprite.get_width() * scale_x * self.size_scale),
        )
        sprite_height = max(
            1,
            round(rotated_sprite.get_height() * scale_y * self.size_scale),
        )

        scaled_sprite = pygame.transform.scale(
            rotated_sprite,
            (sprite_width, sprite_height),
        )

        sprite_rect = scaled_sprite.get_rect(center=(round(screen_x), round(screen_y)))
        surface.blit(scaled_sprite, sprite_rect)

        self._draw_health_bar(
            surface,
            screen_x,
            sprite_rect.top,
            sprite_width,
        )

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        screen_x: float,
        sprite_top: int,
        sprite_width: int,
    ) -> None:
        """Draw a health bar whose width follows the rendered sprite."""
        bar_width = max(12, sprite_width)
        bar_height = 3

        bar_x = round(screen_x - bar_width / 2)
        bar_y = sprite_top - 5

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_x, bar_y, bar_width, bar_height),
        )

        health_fraction = self.health / self.max_health

        pygame.draw.rect(
            surface,
            (65, 220, 90),
            (
                bar_x,
                bar_y,
                round(bar_width * health_fraction),
                bar_height,
            ),
        )
