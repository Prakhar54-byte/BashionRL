import math
from dataclasses import dataclass

import pygame

from src.game.entities.projectile import Projectile
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
    kind: str = "soldier"
    attack_damage: float = 10.0
    attack_interval: float = 1.0
    attack_range: float = 85.0
    projectile_speed: float = 250.0

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
        if self.kind not in ("soldier", "tank"):
            raise ValueError("Enemy kind must be soldier or tank.")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.speed, self.max_health, self.size_scale, self.attack_damage,
            self.attack_interval, self.attack_range, self.projectile_speed,
        )):
            raise ValueError("Enemy movement, health, and combat values must be finite and positive.")
        if not math.isfinite(self.leak_damage):
            raise ValueError("Leak damage must be finite.")


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
        self._draw_key = None
        self._draw_sprite = None
        self.target = None
        self.attack_cooldown = 0.0
        self.projectiles: list[Projectile] = []
        self.leak_registered = False
        self.gun_direction = self.direction.copy()

    def update(self, delta_time: float, structures=(), target_priority=("defense", "resource", "king")) -> None:
        """Move along the pathway independently of frame rate."""
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")
        if not self.alive:
            return
        self.attack_cooldown = max(0.0, self.attack_cooldown - delta_time)
        valid = [structure for structure in structures if structure.alive and
                 self.position.distance_to(structure.position) <= self.config.attack_range + 1e-6]
        if self.target not in valid:
            self.target = min(valid, key=lambda structure: (
                target_priority.index(structure.structure_kind),
                self.position.distance_squared_to(structure.position), structure.structure_id,
            ), default=None)
        if self.target is not None:
            offset = self.target.position - self.position
            if offset.length_squared() > 0:
                self.gun_direction = offset.normalize()
            if self.attack_cooldown <= 1e-9:
                if self.config.kind == "tank":
                    self.projectiles.append(Projectile(self.position, self.target,
                                                       self.config.projectile_speed, self.config.attack_damage))
                else:
                    self.target.take_damage(self.config.attack_damage)
                self.attack_cooldown = self.config.attack_interval
            return
        if self.reached_end:
            kings = [structure for structure in structures if structure.alive and structure.structure_kind == "king"]
            target = min(kings, key=lambda structure: (
                self.position.distance_squared_to(structure.position), structure.structure_id,
            ), default=None)
            if target is not None:
                offset = target.position - self.position
                if offset.length_squared() > 0:
                    self.direction = offset.normalize()
                    self.position += self.direction * min(self.speed * delta_time,
                                                          max(0.0, offset.length() - self.config.attack_range))
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
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Damage must be finite and non-negative.")

        if not self.alive:
            return

        self.health = max(0.0, self.health - amount)

        if self.health == 0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        """Render the enemy facing its movement direction."""
        if not self.alive:
            return

        scale_x = surface.get_width() / self.pathway.map_width
        scale_y = surface.get_height() / self.pathway.map_height

        screen_x = self.position.x * scale_x
        screen_y = self.position.y * scale_y

        angle = math.degrees(math.atan2(-self.direction.y, self.direction.x))
        draw_key = (angle, scale_x, scale_y, self.size_scale)
        if draw_key != self._draw_key:
            rotated_sprite = pygame.transform.rotate(self.sprite, angle)
            sprite_width = max(1, round(rotated_sprite.get_width() * scale_x * self.size_scale))
            sprite_height = max(1, round(rotated_sprite.get_height() * scale_y * self.size_scale))
            self._draw_sprite = pygame.transform.scale(rotated_sprite, (sprite_width, sprite_height))
            self._draw_key = draw_key
        scaled_sprite = self._draw_sprite
        sprite_width = scaled_sprite.get_width()

        sprite_rect = scaled_sprite.get_rect(center=(round(screen_x), round(screen_y)))
        surface.blit(scaled_sprite, sprite_rect)
        if self.config.kind == "tank":
            pygame.draw.line(surface, (30, 35, 35), sprite_rect.center,
                             (round(screen_x + self.gun_direction.x * 30 * scale_x),
                              round(screen_y + self.gun_direction.y * 30 * scale_y)), 5)
        for projectile in self.projectiles:
            projectile.draw(surface, self.pathway.map_width, self.pathway.map_height)

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
