"""Base Enemy class shared by all enemy variants.

Movement, health, damage, and rendering live here.
Subclasses (Soldier, Tank) add end-of-route behaviour and
any variant-specific sprite logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pygame

from src.game.entities.enemies.config import EnemyConfig
from src.game.entities.king_tower import KingTower
from src.game.terrain.pathway import Pathway


class Enemy:
    """Base class for all enemies that follow a route on the map.

    Subclasses implement the specific behaviour that occurs when
    the enemy reaches the end of its assigned pathway:

    - ``Soldier`` performs a suicide attack on the King Tower slot
      it crosses while following its route (detected by Simulation).
    - ``Tank`` exits the route and is removed without attacking.

    The ``is_tank`` property lets callers distinguish the two families
    without an ``isinstance`` check.
    """

    def __init__(
        self,
        pathway: Pathway,
        sprite: pygame.Surface,
        config: EnemyConfig,
    ) -> None:
        if len(pathway.points) < 2:
            raise ValueError(
                "The pathway needs at least two points."
            )

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

        # Assigned by Simulation after the enemy is created.
        self.enemy_type: str | None = None
        self.route_name: str | None = None

        # Enemy source sprites face right (East).
        self.direction = pygame.Vector2(1.0, 0.0)

        # King Tower combat state.
        self.target_structure: KingTower | None = None
        self.suicide_attacked = False
        self.exited_route = False

        # Allow an attack immediately upon reaching attack range.
        self._attack_elapsed = config.attack_interval

        # Separate render caches prevent redundant scaling and rotation.
        self._draw_cache: dict[
            str,
            tuple[tuple[float, ...], pygame.Surface],
        ] = {}

    # ------------------------------------------------------------------
    # Subclass identity

    @property
    def is_tank(self) -> bool:
        """Whether this enemy is a Tank (light or heavy).

        Overridden to ``True`` by the ``Tank`` subclass so callers can
        distinguish enemy families without ``isinstance`` checks.
        """
        return False

    # ------------------------------------------------------------------
    # Shared properties

    @property
    def has_structure_target(self) -> bool:
        """Whether this enemy currently holds a King Tower target."""
        return self.target_structure is not None

    @property
    def is_attacking(self) -> bool:
        """Whether the current target is alive and within attack range."""
        target = self.target_structure

        if target is None or not self._is_living_king_tower(target):
            return False

        return (
            self.position.distance_to(target.position)
            <= self.config.attack_range
        )

    # ------------------------------------------------------------------
    # Static helpers

    @staticmethod
    def _direction_angle(direction: pygame.Vector2) -> float:
        """Convert a direction vector into a pygame rotation angle."""
        return math.degrees(
            math.atan2(-direction.y, direction.x)
        )

    @staticmethod
    def _is_living_king_tower(target: KingTower) -> bool:
        """Return whether a King Tower can still receive damage."""
        return target.alive and not target.destroyed

    @staticmethod
    def _tower_id(tower: KingTower) -> str:
        """Provide a stable identifier for target tie-breaking."""
        return str(tower.structure_id)

    # ------------------------------------------------------------------
    # Target selection

    def _select_king_tower(
        self,
        king_targets: Sequence[KingTower],
    ) -> KingTower | None:
        """Select the nearest surviving King Tower deterministically."""
        candidates: list[tuple[float, str, KingTower]] = []

        for tower in king_targets:
            if not self._is_living_king_tower(tower):
                continue

            distance_squared = self.position.distance_squared_to(
                tower.position
            )

            candidates.append(
                (
                    distance_squared,
                    self._tower_id(tower),
                    tower,
                )
            )

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    # ------------------------------------------------------------------
    # Movement

    def _follow_path(self, delta_time: float) -> None:
        """Move along the assigned route until its endpoint."""
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

    def _on_path_finished(self) -> None:
        """Called when the enemy reaches the end of its route.

        Subclasses override this to implement their end-of-route
        behaviour (soldier passthrough vs. tank route exit).
        """

    def update(
        self,
        delta_time: float,
        *,
        king_targets: Sequence[KingTower] = (),
    ) -> None:
        """Follow the assigned route and trigger end-of-route logic."""
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if not self.alive:
            return

        if not self.reached_end:
            self._follow_path(delta_time)

        if self.reached_end:
            self._on_path_finished()

    # ------------------------------------------------------------------
    # Damage

    def take_damage(self, amount: float) -> None:
        """Apply damage and mark the enemy dead when health reaches zero."""
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount < 0
        ):
            raise ValueError(
                "Damage must be finite and non-negative."
            )

        if not self.alive:
            return

        self.health = max(0.0, self.health - amount)

        if self.health <= 0.0:
            self.alive = False
            self.target_structure = None

    # ------------------------------------------------------------------
    # Rendering

    def _get_rendered_sprite(
        self,
        layer: str,
        sprite: pygame.Surface,
        angle: float,
        scale_x: float,
        scale_y: float,
    ) -> pygame.Surface:
        """Return a cached, rotated and scaled sprite for a render layer."""
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

    def _render_sprite_to_surface(
        self,
        surface: pygame.Surface,
        scale_x: float,
        scale_y: float,
        screen_position: tuple[int, int],
    ) -> tuple[int, int]:
        """Blit the body sprite; return (sprite_top, sprite_width).

        Overridden by ``Tank`` to composite chassis + gun layers.
        """
        chassis_angle = self._direction_angle(self.direction)

        rendered_sprite = self._get_rendered_sprite(
            "enemy",
            self.sprite,
            chassis_angle,
            scale_x,
            scale_y,
        )

        sprite_rect = rendered_sprite.get_rect(center=screen_position)
        surface.blit(rendered_sprite, sprite_rect)

        return sprite_rect.top, sprite_rect.width

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

    def draw(self, surface: pygame.Surface) -> None:
        """Render the enemy, including enemies pursuing King Towers."""
        if not self.alive:
            return

        if self.pathway.map_width <= 0 or self.pathway.map_height <= 0:
            return

        scale_x = surface.get_width() / self.pathway.map_width
        scale_y = surface.get_height() / self.pathway.map_height

        screen_position = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        sprite_top, sprite_width = self._render_sprite_to_surface(
            surface, scale_x, scale_y, screen_position
        )

        self._draw_health_bar(
            surface,
            screen_position[0],
            sprite_top,
            sprite_width,
        )
