
from __future__ import annotations

import math

import pygame

from src.game.entities.enemy import Enemy
from src.game.entities.structure import Structure


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


class DefenseTower(Structure):
    """Damageable defense structure with a stationary base and rotating barrel.

    The tower inherits health, damage handling, and destruction state
    from Structure. Its combat behavior remains specific to this class.
    """

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
        self._validate_combat_values(
            range_radius=range_radius,
            damage=damage,
            fire_rate=fire_rate,
            projectile_speed=projectile_speed,
            size_scale=size_scale,
            base_size_scale=base_size_scale,
            shots_per_burst=shots_per_burst,
            burst_interval=burst_interval,
        )

        if not isinstance(sprite, pygame.Surface):
            raise TypeError("Tower sprite must be a pygame Surface.")

        if base_sprite is not None and not isinstance(
            base_sprite, pygame.Surface
        ):
            raise TypeError("Base sprite must be a Surface or None.")

        if (
            len(slot_size) != 2
            or any(
                not math.isfinite(value) or value <= 0
                for value in slot_size
            )
        ):
            raise ValueError("Slot dimensions must be finite and positive.")

        super().__init__(
            structure_id=slot_id,
            position=position,
            max_health=max_health,
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

        self.slot_size = (
            float(slot_size[0]),
            float(slot_size[1]),
        )

        self.shots_per_burst = shots_per_burst
        self.burst_interval = float(burst_interval)

        self.cooldown = 0.0
        self.angle = 0.0

        self._burst_shots_remaining = 0
        self._burst_timer = 0.0

        self.projectiles: list[Projectile] = []
        self.tower_type: str | None = None

        # Render caches.
        self._scaled_sprite: pygame.Surface | None = None
        self._scaled_sprite_size: tuple[int, int] | None = None

        self._rotated_sprite: pygame.Surface | None = None
        self._draw_angle: int | None = None

        self._scaled_base_sprite: pygame.Surface | None = None
        self._scaled_base_size: tuple[int, int] | None = None

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
    ) -> None:
        """Validate the values controlling tower combat."""
        for name, value in (
            ("range_radius", range_radius),
            ("damage", damage),
            ("fire_rate", fire_rate),
            ("projectile_speed", projectile_speed),
            ("size_scale", size_scale),
            ("burst_interval", burst_interval),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and numeric.")

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

        if base_size_scale is not None and (
            not math.isfinite(base_size_scale) or base_size_scale <= 0
        ):
            raise ValueError("Base size scale must be finite and positive.")

        if (
            isinstance(shots_per_burst, bool)
            or not isinstance(shots_per_burst, int)
            or shots_per_burst < 1
        ):
            raise ValueError("Shots per burst must be a positive integer.")

        if burst_interval < 0:
            raise ValueError("Burst interval cannot be negative.")

    def _find_target(
        self,
        enemies: list[Enemy],
    ) -> Enemy | None:
        """Select the nearest living enemy inside the tower's range."""
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
        """Fire one projectile at a living enemy."""
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
        """Schedule the next shot in a burst or the next firing cycle."""
        self._burst_shots_remaining -= 1

        if self._burst_shots_remaining > 0:
            self._burst_timer = self.burst_interval
        else:
            self.cooldown = (
                1.0 / self.fire_rate
                if self.fire_rate > 0
                else 0.0
            )

    def update(
        self,
        delta_time: float,
        enemies: list[Enemy],
    ) -> None:
        """Track enemies, fire bursts, and advance active projectiles."""
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")

        self.cooldown = max(0.0, self.cooldown - delta_time)
        self._burst_timer = max(0.0, self._burst_timer - delta_time)

        # A destroyed tower cannot acquire targets or fire.
        if self.alive and self.fire_rate > 0:
            target = self._find_target(enemies)

            if target is not None:
                direction = target.position - self.position

                if direction.length_squared() > 0.0:
                    # Tower barrel art is assumed to face upward.
                    self.angle = (
                        math.degrees(
                            math.atan2(-direction.y, direction.x)
                        )
                        - 90.0
                    )

                if self._burst_shots_remaining > 0:
                    if self._burst_timer <= 0.0:
                        self._fire(target)
                        self._finish_burst_shot()

                elif self.cooldown <= 0.0:
                    self._fire(target)

                    self._burst_shots_remaining = (
                        self.shots_per_burst - 1
                    )

                    if self._burst_shots_remaining > 0:
                        self._burst_timer = self.burst_interval
                    else:
                        self.cooldown = 1.0 / self.fire_rate

        # Projectiles already fired remain in flight if their tower
        # is destroyed. They still need to resolve their impacts.
        for projectile in self.projectiles:
            projectile.update(delta_time)

        self.projectiles[:] = [
            projectile
            for projectile in self.projectiles
            if projectile.alive
        ]

    def _get_scaled_base(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface | None:
        """Return a cached, unrotated base sprite."""
        if self.base_sprite is None:
            return None

        if (
            self._scaled_base_sprite is None
            or self._scaled_base_size != target_size
        ):
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
        """Return the cached barrel sprite before rotation."""
        if (
            self._scaled_sprite is None
            or self._scaled_sprite_size != target_size
        ):
            self._scaled_sprite = pygame.transform.smoothscale(
                self.sprite,
                target_size,
            )
            self._scaled_sprite_size = target_size
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

        if (
            self._rotated_sprite is None
            or self._draw_angle != draw_angle
        ):
            self._rotated_sprite = pygame.transform.rotate(
                scaled_sprite,
                draw_angle,
            )
            self._draw_angle = draw_angle

        return self._rotated_sprite

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        center_x: int,
        top_y: int,
        sprite_width: int,
    ) -> None:
        """Display damage sustained by the tower."""
        if self.health >= self.max_health:
            return

        bar_width = max(24, sprite_width)
        bar_height = 4

        bar_x = round(center_x - bar_width / 2)
        bar_y = top_y - 7

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_x, bar_y, bar_width, bar_height),
        )

        pygame.draw.rect(
            surface,
            (65, 220, 90),
            (
                bar_x,
                bar_y,
                round(bar_width * self.health_fraction),
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
        """Draw the tower, its health bar, and any active projectiles."""
        if map_width <= 0 or map_height <= 0:
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
            if show_range:
                range_surface = pygame.Surface(
                    surface.get_size(),
                    pygame.SRCALPHA,
                )

                range_width = round(self.range_radius * scale_x)
                range_height = round(self.range_radius * scale_y)

                pygame.draw.ellipse(
                    range_surface,
                    (80, 180, 255, 35),
                    (
                        center[0] - range_width,
                        center[1] - range_height,
                        range_width * 2,
                        range_height * 2,
                    ),
                )

                pygame.draw.ellipse(
                    range_surface,
                    (80, 180, 255, 110),
                    (
                        center[0] - range_width,
                        center[1] - range_height,
                        range_width * 2,
                        range_height * 2,
                    ),
                    width=1,
                )

                surface.blit(range_surface, (0, 0))

            # Draw the base first, without any turret rotation.
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

            # Draw the barrel above the base.
            barrel_size = (
                max(
                    1,
                    round(
                        self.sprite.get_width()
                        * self.size_scale
                        * scale_x
                    ),
                ),
                max(
                    1,
                    round(
                        self.sprite.get_height()
                        * self.size_scale
                        * scale_y
                    ),
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

        # Already-fired projectiles are allowed to finish their flight,
        # even when the source tower has been destroyed.
        for projectile in self.projectiles:
            projectile.draw(
                surface,
                map_width,
                map_height,
            )
