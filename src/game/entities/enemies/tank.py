"""Tank enemy variants (light and heavy).

Tanks follow their assigned Tiled routes without chasing targets.
Their chassis follows the movement direction, while the turret aims
independently at nearby defense towers or King Towers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pygame

from src.game.entities.enemies.base import Enemy
from src.game.entities.enemies.config import EnemyConfig
from src.game.entities.structure import Structure
from src.game.entities.towers.projectile import Projectile
from src.game.terrain.pathway import Pathway


class Tank(Enemy):
    """Tank with independent turret aiming and structure combat."""

    MINIMUM_ATTACK_RANGE = 300.0

    def __init__(
        self,
        pathway: Pathway,
        sprite: pygame.Surface,
        config: EnemyConfig,
        *,
        base_sprite: pygame.Surface,
        gun_sprite: pygame.Surface,
    ) -> None:
        super().__init__(pathway, sprite, config)

        self.base_sprite = base_sprite
        self.gun_sprite = gun_sprite

        self.gun_angle = 0.0
        self._gun_target: pygame.Vector2 | None = None
        self._attack_target: Structure | None = None

        self._attack_cooldown = 0.0
        self.projectiles: list[Projectile] = []

    # ------------------------------------------------------------------
    # Identity

    @property
    def is_tank(self) -> bool:
        """Identify this enemy as a tank."""
        return True

    @property
    def attack_range(self):
        return max(float(self.config.attack_range), self.MINIMUM_ATTACK_RANGE)

    # ------------------------------------------------------------------
    # Turret aiming

    def aim_at(self, target_position: pygame.Vector2) -> None:
        """Aim the turret at a world-space position."""
        target = pygame.Vector2(target_position)

        if not math.isfinite(target.x) or not math.isfinite(target.y):
            raise ValueError("Gun target coordinates must be finite.")

        self._gun_target = target
        self._update_gun_angle()

    def clear_gun_target(self) -> None:
        """Make the turret follow the chassis direction."""
        self._gun_target = None
        self._attack_target = None
        self._update_gun_angle()

    def _update_gun_angle(self) -> None:
        """Update the turret angle without affecting chassis movement."""
        if self._gun_target is None:
            self.gun_angle = self._direction_angle(self.direction)
            return

        offset = self._gun_target - self.position

        if offset.length_squared() > 1e-12:
            self.gun_angle = self._direction_angle(offset)

    # ------------------------------------------------------------------
    # Combat

    def update_combat(
        self,
        delta_time: float,
        defense_towers: Sequence[Structure],
        king_towers: Sequence[Structure] = (),
        resource_towers: Sequence[Structure] = (),
    ) -> None:
        """Attack nearby defense, resource, or King Towers.

        Tanks continue following their assigned routes. Their turrets
        independently target the nearest living structure within range.
        """
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError("Delta time must be finite and non-negative.")

        if not self.alive:
            self.clear_gun_target()
            self.projectiles.clear()
            return

        # Advance weapon cooldown.
        self._attack_cooldown = max(
            0.0,
            self._attack_cooldown - delta_time,
        )

        attack_range = self.attack_range
        range_squared = attack_range * attack_range

        # All three structure types are valid targets.
        targets = (
            *defense_towers,
            *king_towers,
            *resource_towers,
        )

        candidates = [
            target
            for target in targets
            if target.alive
            and not target.destroyed
            and (target.position - self.position).length_squared() <= range_squared
        ]

        if not candidates:
            self.clear_gun_target()
            return

        # Select the nearest living structure in range.
        target = min(
            candidates,
            key=lambda structure: (structure.position - self.position).length_squared(),
        )

        self._attack_target = target
        self.aim_at(target.position)

        # Fire a projectile when the weapon is ready.
        if self._attack_cooldown <= 0.0:
            self.projectiles.append(
                Projectile(
                    position=pygame.Vector2(self.position),
                    target=target,
                    speed=self.config.projectile_speed,
                    damage=self.config.attack_damage,
                )
            )

            self._attack_cooldown = self.config.attack_interval

    # ------------------------------------------------------------------
    # Movement

    def _follow_path(self, delta_time: float) -> None:
        """Follow the assigned route and update the turret orientation."""
        super()._follow_path(delta_time)
        self._update_gun_angle()

    def _on_path_finished(self) -> None:
        """Exit the route without attacking by collision."""
        self.exited_route = True
        self.alive = False
        self.clear_gun_target()
        self.projectiles.clear()

    # ------------------------------------------------------------------
    # Damage

    def take_damage(self, amount: float) -> None:
        """Take damage and clear combat state when destroyed."""
        super().take_damage(amount)

        if not self.alive:
            self.clear_gun_target()
            self.projectiles.clear()

    # ------------------------------------------------------------------
    # Rendering

    def _render_sprite_to_surface(
        self,
        surface: pygame.Surface,
        scale_x: float,
        scale_y: float,
        screen_position: tuple[int, int],
    ) -> tuple[int, int]:
        """Rotate the chassis with movement and turret toward its target."""
        chassis_angle = self._direction_angle(self.direction)

        rendered_base = self._get_rendered_sprite(
            "tank_base",
            self.base_sprite,
            chassis_angle,
            scale_x,
            scale_y,
        )

        # The turret rotates independently of the chassis.
        rendered_gun = self._get_rendered_sprite(
            "tank_gun",
            self.gun_sprite,
            self.gun_angle,
            scale_x,
            scale_y,
        )

        base_rect = rendered_base.get_rect(center=screen_position)
        gun_rect = rendered_gun.get_rect(center=screen_position)

        surface.blit(rendered_base, base_rect)
        surface.blit(rendered_gun, gun_rect)

        sprite_top = min(base_rect.top, gun_rect.top)
        sprite_width = max(base_rect.width, gun_rect.width)

        return sprite_top, sprite_width
