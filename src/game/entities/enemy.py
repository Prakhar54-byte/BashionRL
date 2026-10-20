
from __future__ import annotations

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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Enemy name cannot be empty.")

        for name, value in (
            ("speed", self.speed),
            ("max_health", self.max_health),
            ("size_scale", self.size_scale),
            ("leak_damage", self.leak_damage),
        ):
            if not math.isfinite(value):
                raise ValueError(f"Enemy {name} must be finite.")

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
            raise ValueError(
                "Kill reward must be a non-negative integer."
            )


class Enemy:
    """Enemy that follows a pathway and can take damage.

    Ordinary enemies use a single sprite.

    Tank enemies use separate base and gun sprites. The base follows
    the movement direction, while the gun can independently aim at
    a world-space target using aim_at().
    """

    def __init__(
        self,
        pathway: Pathway,
        sprite: pygame.Surface,
        config: EnemyConfig,
        *,
        base_sprite: pygame.Surface | None = None,
        gun_sprite: pygame.Surface | None = None,
    ) -> None:
        if len(pathway.points) < 2:
            raise ValueError(
                "The pathway needs at least two points."
            )

        if (base_sprite is None) != (gun_sprite is None):
            raise ValueError(
                "Tank enemies require both base_sprite and gun_sprite."
            )

        self.pathway = pathway
        self.sprite = sprite
        self.base_sprite = base_sprite
        self.gun_sprite = gun_sprite
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

        # Assigned by Simulation after the enemy is created.
        self.enemy_type: str | None = None

        # Enemy source sprites face right (East).
        self.direction = pygame.Vector2(1.0, 0.0)

        self.gun_angle = 0.0
        self._gun_target: pygame.Vector2 | None = None

        # Separate render caches prevent redundant scaling and rotation.
        self._draw_cache: dict[
            str,
            tuple[tuple[float, ...], pygame.Surface],
        ] = {}

    @property
    def is_tank(self) -> bool:
        """Whether this enemy uses separate chassis and gun sprites."""
        return (
            self.base_sprite is not None
            and self.gun_sprite is not None
        )

    @staticmethod
    def _direction_angle(direction: pygame.Vector2) -> float:
        """Convert a direction vector into a pygame rotation angle."""
        return math.degrees(
            math.atan2(-direction.y, direction.x)
        )

    def aim_at(self, target_position: pygame.Vector2) -> None:
        """Aim a tank's gun independently at a world-space position.

        The target position uses the same map coordinates as the
        pathway. This can be called by a future enemy-targeting system.
        """
        if not self.is_tank:
            return

        target = pygame.Vector2(target_position)

        if not math.isfinite(target.x) or not math.isfinite(target.y):
            raise ValueError("Gun target coordinates must be finite.")

        self._gun_target = target
        self._update_gun_angle()

    def clear_gun_target(self) -> None:
        """Return the tank gun to following the chassis direction."""
        self._gun_target = None
        self._update_gun_angle()

    def _update_gun_angle(self) -> None:
        """Update the turret angle independently of chassis rotation."""
        if self._gun_target is None:
            self.gun_angle = self._direction_angle(self.direction)
            return

        offset = self._gun_target - self.position

        if offset.length_squared() > 1e-12:
            self.gun_angle = self._direction_angle(offset)

    def update(self, delta_time: float) -> None:
        """Move along the pathway independently of frame rate."""
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if not self.alive or self.reached_end:
            return

        distance_remaining = self.speed * delta_time

        while distance_remaining > 0 and not self.reached_end:
            target = self.pathway.points[self.waypoint_index]
            offset = target - self.position
            distance_to_target = offset.length()

            if distance_to_target <= 1e-6:
                self.position = target.copy()
                self.waypoint_index += 1

                if self.waypoint_index >= len(self.pathway.points):
                    self.reached_end = True
                    break

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

        self._update_gun_angle()

    def take_damage(self, amount: float) -> None:
        """Apply damage and mark the enemy dead when health reaches zero."""
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                "Damage must be finite and non-negative."
            )

        if not self.alive:
            return

        self.health = max(0.0, self.health - amount)

        if self.health <= 0.0:
            self.alive = False

    def _get_rendered_sprite(
        self,
        layer: str,
        sprite: pygame.Surface,
        angle: float,
        scale_x: float,
        scale_y: float,
    ) -> pygame.Surface:
        """Return a cached, rotated and scaled sprite for a render layer."""
        # Quantize rotation to one degree to reduce unnecessary transforms.
        rotation = round(angle)

        cache_key = (
            float(rotation),
            round(scale_x, 5),
            round(scale_y, 5),
            self.size_scale,
        )

        cached = self._draw_cache.get(layer)

        if cached is not None and cached[0] == cache_key:
            return cached[1]

        if sprite.get_width() <= 0 or sprite.get_height() <= 0:
            raise ValueError(
                f"Enemy sprite for layer '{layer}' has invalid dimensions."
            )

        rotated_sprite = pygame.transform.rotate(sprite, rotation)

        target_width = max(
            1,
            round(
                rotated_sprite.get_width()
                * scale_x
                * self.size_scale
            ),
        )

        target_height = max(
            1,
            round(
                rotated_sprite.get_height()
                * scale_y
                * self.size_scale
            ),
        )

        rendered_sprite = pygame.transform.scale(
            rotated_sprite,
            (target_width, target_height),
        )

        self._draw_cache[layer] = (cache_key, rendered_sprite)

        return rendered_sprite

    def draw(self, surface: pygame.Surface) -> None:
        """Render the enemy or the separate layers of a tank."""
        if not self.alive or self.reached_end:
            return

        if self.pathway.map_width <= 0 or self.pathway.map_height <= 0:
            return

        scale_x = surface.get_width() / self.pathway.map_width
        scale_y = surface.get_height() / self.pathway.map_height

        screen_position = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        chassis_angle = self._direction_angle(self.direction)

        if self.is_tank:
            assert self.base_sprite is not None
            assert self.gun_sprite is not None

            # Chassis follows the movement direction.
            rendered_base = self._get_rendered_sprite(
                "tank_base",
                self.base_sprite,
                chassis_angle,
                scale_x,
                scale_y,
            )

            # Gun rotates independently and can aim at another target.
            rendered_gun = self._get_rendered_sprite(
                "tank_gun",
                self.gun_sprite,
                self.gun_angle,
                scale_x,
                scale_y,
            )

            base_rect = rendered_base.get_rect(
                center=screen_position
            )
            gun_rect = rendered_gun.get_rect(
                center=screen_position
            )

            # Draw the chassis first, then overlay the gun.
            surface.blit(rendered_base, base_rect)
            surface.blit(rendered_gun, gun_rect)

            sprite_top = min(base_rect.top, gun_rect.top)
            sprite_width = max(
                base_rect.width,
                gun_rect.width,
            )

        else:
            # Soldiers retain their original single-sprite behavior.
            rendered_sprite = self._get_rendered_sprite(
                "enemy",
                self.sprite,
                chassis_angle,
                scale_x,
                scale_y,
            )

            sprite_rect = rendered_sprite.get_rect(
                center=screen_position
            )
            surface.blit(rendered_sprite, sprite_rect)

            sprite_top = sprite_rect.top
            sprite_width = sprite_rect.width

        self._draw_health_bar(
            surface,
            screen_position[0],
            sprite_top,
            sprite_width,
        )

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        screen_x: float,
        sprite_top: int,
        sprite_width: int,
    ) -> None:
        """Draw the enemy health bar above its rendered sprite."""
        bar_width = max(12, sprite_width)
        bar_height = 4 if self.is_tank else 3

        bar_x = round(screen_x - bar_width / 2)
        bar_y = sprite_top - 5

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_x, bar_y, bar_width, bar_height),
        )

        health_fraction = max(
            0.0,
            min(1.0, self.health / self.max_health),
        )

        health_color = (
            (130, 195, 75)
            if self.is_tank
            else (65, 220, 90)
        )

        pygame.draw.rect(
            surface,
            health_color,
            (
                bar_x,
                bar_y,
                round(bar_width * health_fraction),
                bar_height,
            ),
        )
