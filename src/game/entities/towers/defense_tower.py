"""DefenseTower — a damageable, placeable combat structure.

The tower acquires the nearest living enemy within range, rotates its
barrel toward the target, and fires projectiles in configurable bursts.

Health and destruction state are inherited from Structure.
"""

from __future__ import annotations

import math

import pygame

from src.game.entities.enemies.base import Enemy
from src.game.entities.structure import Structure
from src.game.entities.towers.projectile import Projectile


class DefenseTower(Structure):
    """Damageable defense tower with a stationary base and rotating barrel."""

    def __init__(
        self,
        slot_id: int,
        position: pygame.Vector2,
        sprite: pygame.Surface,
        base_sprite: pygame.Surface | None = None,
        range_radius: float = 180.0,
        damage: float = 25.0,
        fire_rate: float = 1.0,
        projectile_speed: float = 350.0,
        size_scale: float = 3.0,
        base_size_scale: float | None = None,
        slot_size: tuple[float, float] = (64.0, 64.0),
        shots_per_burst: int = 1,
        burst_interval: float = 0.12,
        max_health: float = 250.0,
    ) -> None:
        """Initialize the tower and validate its configuration."""
        self._validate_combat_values(
            range_radius=range_radius,
            damage=damage,
            fire_rate=fire_rate,
            projectile_speed=projectile_speed,
            size_scale=size_scale,
            base_size_scale=base_size_scale,
            shots_per_burst=shots_per_burst,
            burst_interval=burst_interval,
            max_health=max_health,
        )

        if not isinstance(sprite, pygame.Surface):
            raise TypeError("Tower sprite must be a pygame Surface.")

        if base_sprite is not None and not isinstance(base_sprite, pygame.Surface):
            raise TypeError("Base sprite must be a Surface or None.")

        validated_slot_size = self._validate_slot_size(slot_size)

        super().__init__(
            structure_id=slot_id,
            position=position,
            max_health=float(max_health),
        )

        self.slot_id = slot_id
        self.sprite = sprite
        self.base_sprite = base_sprite

        self.range_radius = float(range_radius)
        self.damage = float(damage)
        self.fire_rate = float(fire_rate)
        self.projectile_speed = float(projectile_speed)

        self.size_scale = float(size_scale)
        self.base_size_scale = float(
            base_size_scale if base_size_scale is not None else size_scale
        )

        self.slot_size = validated_slot_size

        self.shots_per_burst = shots_per_burst
        self.burst_interval = float(burst_interval)

        self.cooldown = 0.0
        self.angle = 0.0

        self._burst_shots_remaining = 0
        self._burst_timer = 0.0

        self.projectiles: list[Projectile] = []
        self.tower_type: str | None = None

        # Cached sprite surfaces avoid repeated scaling and rotation work.
        self._scaled_sprite: pygame.Surface | None = None
        self._scaled_sprite_size: tuple[int, int] | None = None

        self._rotated_sprite: pygame.Surface | None = None
        self._draw_angle: int | None = None

        self._scaled_base_sprite: pygame.Surface | None = None
        self._scaled_base_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Validation

    @staticmethod
    def _validate_combat_values(
        *,
        range_radius: float,
        damage: float,
        fire_rate: float,
        projectile_speed: float,
        size_scale: float,
        base_size_scale: float | None,
        shots_per_burst: int,
        burst_interval: float,
        max_health: float,
    ) -> None:
        """Validate combat parameters, scaling factors, and tower health."""
        finite_values = {
            "range_radius": range_radius,
            "damage": damage,
            "fire_rate": fire_rate,
            "projectile_speed": projectile_speed,
            "size_scale": size_scale,
            "burst_interval": burst_interval,
            "max_health": max_health,
        }

        for name, value in finite_values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite numeric value.")

        if range_radius < 0:
            raise ValueError("Tower range cannot be negative.")

        if damage < 0:
            raise ValueError("Tower damage cannot be negative.")

        if fire_rate < 0:
            raise ValueError("Fire rate cannot be negative.")

        if projectile_speed <= 0:
            raise ValueError("Projectile speed must be positive.")

        if size_scale <= 0:
            raise ValueError("Tower size scale must be positive.")

        if max_health <= 0:
            raise ValueError("Tower maximum health must be finite and positive.")

        if burst_interval < 0:
            raise ValueError("Burst interval must be finite and non-negative.")

        if base_size_scale is not None and (
            isinstance(base_size_scale, bool)
            or not isinstance(base_size_scale, (int, float))
            or not math.isfinite(base_size_scale)
            or base_size_scale <= 0
        ):
            raise ValueError("Base size scale must be finite and positive.")

        if (
            isinstance(shots_per_burst, bool)
            or not isinstance(shots_per_burst, int)
            or shots_per_burst < 1
        ):
            raise ValueError("Shots per burst must be a positive integer.")

    @staticmethod
    def _validate_slot_size(
        slot_size: tuple[float, float],
    ) -> tuple[float, float]:
        """Validate and normalize the tower's placement-slot dimensions."""
        if not isinstance(slot_size, (tuple, list)) or len(slot_size) != 2:
            raise ValueError(
                "Slot dimensions must contain exactly two positive values."
            )

        for value in slot_size:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Slot dimensions must be finite and positive.")

        return float(slot_size[0]), float(slot_size[1])

    # ------------------------------------------------------------------
    # Target acquisition and combat

    def _find_target(
        self,
        enemies: list[Enemy],
    ) -> Enemy | None:
        """Return the nearest living enemy within the tower's range."""
        target: Enemy | None = None
        nearest_distance = self.range_radius

        for enemy in enemies:
            if not enemy.alive or enemy.reached_end:
                continue

            distance = self.position.distance_to(enemy.position)

            if distance <= nearest_distance:
                nearest_distance = distance
                target = enemy

        return target

    def _fire(self, target: Enemy) -> None:
        """Fire one projectile at a valid target."""
        if not self.alive or not target.alive or target.reached_end:
            return

        self.projectiles.append(
            Projectile(
                position=self.position,
                target=target,
                speed=self.projectile_speed,
                damage=self.damage,
            )
        )

    def _finish_burst_shot(self) -> None:
        """Schedule another burst shot or the next firing cycle."""
        self._burst_shots_remaining -= 1

        if self._burst_shots_remaining > 0:
            self._burst_timer = self.burst_interval
        else:
            self.cooldown = 1.0 / self.fire_rate if self.fire_rate > 0 else 0.0

    def update(
        self,
        delta_time: float,
        enemies: list[Enemy],
    ) -> None:
        """Track targets, fire projectiles, and update projectile movement."""
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError("Delta time must be finite and non-negative.")

        delta_time = float(delta_time)

        self.cooldown = max(0.0, self.cooldown - delta_time)
        self._burst_timer = max(0.0, self._burst_timer - delta_time)

        # Destroyed towers cannot acquire targets or start firing.
        if self.alive and self.fire_rate > 0:
            target = self._find_target(enemies)

            if target is not None:
                direction = target.position - self.position

                if direction.length_squared() > 0.0:
                    # The barrel artwork is assumed to face upward.
                    self.angle = (
                        math.degrees(math.atan2(-direction.y, direction.x)) - 90.0
                    )

                if self._burst_shots_remaining > 0:
                    if self._burst_timer <= 0.0:
                        self._fire(target)
                        self._finish_burst_shot()

                elif self.cooldown <= 0.0:
                    self._fire(target)

                    self._burst_shots_remaining = self.shots_per_burst - 1

                    if self._burst_shots_remaining > 0:
                        self._burst_timer = self.burst_interval
                    else:
                        self.cooldown = 1.0 / self.fire_rate

        # Already-fired projectiles finish their flight even if the tower
        # that fired them is subsequently destroyed.
        for projectile in self.projectiles:
            projectile.update(delta_time)

        self.projectiles[:] = [
            projectile for projectile in self.projectiles if projectile.alive
        ]

    # ------------------------------------------------------------------
    # Sprite scaling and rotation

    def _get_scaled_base(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface | None:
        """Return the cached, unrotated base sprite."""
        if self.base_sprite is None:
            return None

        if self._scaled_base_sprite is None or self._scaled_base_size != target_size:
            self._scaled_base_sprite = pygame.transform.smoothscale(
                self.base_sprite,
                target_size,
            )
            self._scaled_base_size = target_size

        return self._scaled_base_sprite

    def _get_scaled_barrel(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface:
        """Return the cached barrel before rotation."""
        if self._scaled_sprite is None or self._scaled_sprite_size != target_size:
            self._scaled_sprite = pygame.transform.smoothscale(
                self.sprite,
                target_size,
            )
            self._scaled_sprite_size = target_size

            # A newly scaled sprite invalidates the rotation cache.
            self._rotated_sprite = None
            self._draw_angle = None

        return self._scaled_sprite

    def _get_rotated_barrel(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface:
        """Rotate the barrel without rotating the stationary base."""
        scaled_sprite = self._get_scaled_barrel(target_size)
        draw_angle = round(self.angle)

        if self._rotated_sprite is None or self._draw_angle != draw_angle:
            self._rotated_sprite = pygame.transform.rotate(
                scaled_sprite,
                draw_angle,
            )
            self._draw_angle = draw_angle

        return self._rotated_sprite

    # ------------------------------------------------------------------
    # Rendering

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        center_x: int,
        top_y: int,
        sprite_width: int,
    ) -> None:
        """Draw the tower's health bar when it has sustained damage."""
        if self.health >= self.max_health:
            return

        bar_width = max(24, sprite_width)
        bar_height = 4

        bar_x = round(center_x - bar_width / 2)
        bar_y = top_y - 7

        # Background.
        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_x, bar_y, bar_width, bar_height),
        )

        # Remaining health.
        health_fraction = max(
            0.0,
            min(1.0, self.health_fraction),
        )

        health_width = round(bar_width * health_fraction)

        if health_width > 0:
            pygame.draw.rect(
                surface,
                (65, 220, 90),
                (
                    bar_x,
                    bar_y,
                    health_width,
                    bar_height,
                ),
            )

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
        show_range: bool = False,
    ) -> None:
        """Draw the tower, optional attack range, health bar, and projectiles."""
        if (
            not math.isfinite(map_width)
            or not math.isfinite(map_height)
            or map_width <= 0
            or map_height <= 0
        ):
            return

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        center = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        sprite_top = center[1]
        rendered_width = 0

        if self.alive:
            # Draw the optional attack-range indicator.
            if show_range:
                range_surface = pygame.Surface(
                    surface.get_size(),
                    pygame.SRCALPHA,
                )

                range_width = round(self.range_radius * scale_x)
                range_height = round(self.range_radius * scale_y)

                ellipse_rect = (
                    center[0] - range_width,
                    center[1] - range_height,
                    range_width * 2,
                    range_height * 2,
                )

                pygame.draw.ellipse(
                    range_surface,
                    (80, 180, 255, 35),
                    ellipse_rect,
                )

                pygame.draw.ellipse(
                    range_surface,
                    (80, 180, 255, 110),
                    ellipse_rect,
                    width=1,
                )

                surface.blit(range_surface, (0, 0))

            # Draw the stationary base first.
            if self.base_sprite is not None:
                base_size = (
                    max(
                        1,
                        round(
                            self.base_sprite.get_width()
                            * self.base_size_scale
                            * scale_x
                        ),
                    ),
                    max(
                        1,
                        round(
                            self.base_sprite.get_height()
                            * self.base_size_scale
                            * scale_y
                        ),
                    ),
                )

                base = self._get_scaled_base(base_size)

                if base is not None:
                    base_rect = base.get_rect(center=center)
                    surface.blit(base, base_rect)

                    sprite_top = min(sprite_top, base_rect.top)
                    rendered_width = max(
                        rendered_width,
                        base_rect.width,
                    )

            # Draw the rotating barrel over the base.
            barrel_size = (
                max(
                    1,
                    round(self.sprite.get_width() * self.size_scale * scale_x),
                ),
                max(
                    1,
                    round(self.sprite.get_height() * self.size_scale * scale_y),
                ),
            )

            barrel = self._get_rotated_barrel(barrel_size)
            barrel_rect = barrel.get_rect(center=center)
            surface.blit(barrel, barrel_rect)

            sprite_top = min(sprite_top, barrel_rect.top)
            rendered_width = max(
                rendered_width,
                barrel_rect.width,
            )

            self._draw_health_bar(
                surface,
                center[0],
                sprite_top,
                rendered_width,
            )

        # Projectiles already in flight can finish after tower destruction.
        for projectile in self.projectiles:
            projectile.draw(
                surface,
                map_width,
                map_height,
            )
