"""Tank enemy variants (light and heavy).

Both variants use this single class.  Their stats (speed, health,
damage) differ via ``EnemyConfig`` — no subclassing required.

Tanks use separate chassis (base) and gun sprites.  The chassis rotates
with movement direction; the gun turret can aim independently.  When a
tank reaches the end of its route it exits without attacking King Towers.
"""

from __future__ import annotations

import math

import pygame

from src.game.entities.enemies.base import Enemy
from src.game.entities.enemies.config import EnemyConfig
from src.game.terrain.pathway import Pathway


class Tank(Enemy):
    """Armoured tank with an independently-aiming gun turret.

    The light and heavy tank variants share this class — they are
    distinguished only by the values in their ``EnemyConfig``.

    Tanks follow tank routes.  When they reach the route endpoint they
    exit the map (``exited_route = True``) instead of attacking King Towers.
    """

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

    # ------------------------------------------------------------------
    # Subclass identity

    @property
    def is_tank(self) -> bool:
        """Always ``True`` for Tank instances."""
        return True

    # ------------------------------------------------------------------
    # Gun aiming

    def aim_at(self, target_position: pygame.Vector2) -> None:
        """Aim the gun turret independently at a world-space position."""
        target = pygame.Vector2(target_position)

        if not math.isfinite(target.x) or not math.isfinite(target.y):
            raise ValueError("Gun target coordinates must be finite.")

        self._gun_target = target
        self._update_gun_angle()

    def clear_gun_target(self) -> None:
        """Return the gun turret to following the chassis direction."""
        self._gun_target = None
        self._update_gun_angle()

    def _update_gun_angle(self) -> None:
        """Recalculate the turret angle from the current target or direction."""
        if self._gun_target is None:
            self.gun_angle = self._direction_angle(self.direction)
            return

        offset = self._gun_target - self.position

        if offset.length_squared() > 1e-12:
            self.gun_angle = self._direction_angle(offset)

    # ------------------------------------------------------------------
    # Movement overrides

    def _follow_path(self, delta_time: float) -> None:
        """Move along the route and keep the gun angle synchronised."""
        super()._follow_path(delta_time)
        self._update_gun_angle()

    def _on_path_finished(self) -> None:
        """Exit the map when the route endpoint is reached."""
        self.exited_route = True
        self.alive = False

    # ------------------------------------------------------------------
    # Damage override

    def take_damage(self, amount: float) -> None:
        """Apply damage; clear the gun target when the tank is destroyed."""
        super().take_damage(amount)

        if not self.alive:
            self.clear_gun_target()

    # ------------------------------------------------------------------
    # Rendering override

    def _render_sprite_to_surface(
        self,
        surface: pygame.Surface,
        scale_x: float,
        scale_y: float,
        screen_position: tuple[int, int],
    ) -> tuple[int, int]:
        """Composite chassis and gun layers; return (sprite_top, sprite_width)."""
        chassis_angle = self._direction_angle(self.direction)

        # Chassis follows the movement direction.
        rendered_base = self._get_rendered_sprite(
            "tank_base",
            self.base_sprite,
            chassis_angle,
            scale_x,
            scale_y,
        )

        # Gun rotates independently toward the target.
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
