"""ResourceTower — a damageable structure that generates resources over time.

Resource production is controlled by the simulation:

    earned = tower.update(delta_time)
    resources.earn(earned)

The tower exposes its ID for placement-slot management, can be damaged
by enemies, and stops producing resources when destroyed.
"""

from __future__ import annotations

import math

import pygame

from src.game.entities.structure import Structure


class ResourceTower(Structure):
    """Damageable resource-producing structure."""

    def __init__(
        self,
        tower_id: int | str,
        position: pygame.Vector2 | tuple[float, float],
        *,
        sprite: pygame.Surface | None = None,
        max_health: float = 250.0,
        production_interval: float = 10.0,
        production_amount: int = 50,
        size_scale: float = 1.0,
    ) -> None:
        """Initialize the tower, its durability, and its production settings."""
        if isinstance(tower_id, bool) or not isinstance(tower_id, (int, str)):
            raise TypeError("Tower ID must be an integer or string.")

        if isinstance(tower_id, str) and not tower_id.strip():
            raise ValueError("Tower ID cannot be empty.")

        self._validate_positive_number("Maximum health", max_health)
        self._validate_positive_number(
            "Production interval",
            production_interval,
        )
        self._validate_positive_number("Size scale", size_scale)

        if (
            isinstance(production_amount, bool)
            or not isinstance(production_amount, int)
            or production_amount <= 0
        ):
            raise ValueError(
                "Production amount must be a positive integer."
            )

        if sprite is not None and not isinstance(sprite, pygame.Surface):
            raise TypeError(
                "Resource tower sprite must be a pygame Surface or None."
            )

        if sprite is not None and (
            sprite.get_width() <= 0 or sprite.get_height() <= 0
        ):
            raise ValueError(
                "Resource tower sprite must have valid dimensions."
            )

        super().__init__(
            structure_id=tower_id,
            position=position,
            max_health=float(max_health),
        )

        # Explicitly expose the placement ID for simulation slot management.
        self.tower_id: int | str = tower_id

        self.sprite = sprite
        self.max_health = float(max_health)
        self.production_interval = float(production_interval)
        self.production_amount = production_amount
        self.size_scale = float(size_scale)

        self._production_elapsed = 0.0
        self._render_cache: dict[
            tuple[int, int],
            pygame.Surface,
        ] = {}

    # ------------------------------------------------------------------
    # Validation

    @staticmethod
    def _validate_positive_number(name: str, value: float) -> None:
        """Require a finite, positive numeric configuration value."""
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")

    # ------------------------------------------------------------------
    # Identity

    @property
    def is_resource_tower(self) -> bool:
        """Identify this structure as a resource-producing tower."""
        return True

    # ------------------------------------------------------------------
    # Production state

    @property
    def production_progress(self) -> float:
        """Return production progress between 0.0 and 1.0."""
        return min(
            1.0,
            self._production_elapsed / self.production_interval,
        )

    @property
    def production_time_remaining(self) -> float:
        """Return seconds remaining until the next payout."""
        return max(
            0.0,
            self.production_interval - self._production_elapsed,
        )

    # ------------------------------------------------------------------
    # Update

    def update(self, delta_time: float) -> int:
        """Advance production and return the resources earned.

        Multiple production intervals are handled in one update if needed.
        Partial progress is retained between updates.

        Returns:
            The total resources earned during this update.
        """
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if not self.alive or delta_time == 0:
            return 0

        self._production_elapsed += float(delta_time)

        payouts = int(
            (self._production_elapsed + 1e-12)
            / self.production_interval
        )

        if payouts <= 0:
            return 0

        self._production_elapsed = max(
            0.0,
            self._production_elapsed
            - payouts * self.production_interval,
        )

        return payouts * self.production_amount

    # ------------------------------------------------------------------
    # Rendering helpers

    def _get_scaled_sprite(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface:
        """Return a cached sprite scaled to the requested dimensions."""
        if self.sprite is None:
            raise RuntimeError(
                "Cannot scale a missing resource tower sprite."
            )

        rendered = self._render_cache.get(target_size)

        if rendered is None:
            rendered = pygame.transform.smoothscale(
                self.sprite,
                target_size,
            )
            self._render_cache[target_size] = rendered

        return rendered

    def _draw_health_bar(
        self,
        surface: pygame.Surface,
        center_x: int,
        top_y: int,
        bar_width: int,
    ) -> None:
        """Draw the remaining health of the resource tower."""
        bar_height = 4
        bar_left = round(center_x - bar_width / 2)

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_left, top_y, bar_width, bar_height),
        )

        health_fraction = max(
            0.0,
            min(1.0, self.health / self.max_health),
        )

        health_width = round(bar_width * health_fraction)

        if health_width > 0:
            pygame.draw.rect(
                surface,
                (65, 210, 85) if self.alive else (100, 65, 65),
                (bar_left, top_y, health_width, bar_height),
            )

    # ------------------------------------------------------------------
    # Rendering

    def draw(
        self,
        surface: pygame.Surface,
        map_width: int,
        map_height: int,
    ) -> None:
        """Render the tower, its health bar, and its destroyed appearance."""
        if (
            isinstance(map_width, bool)
            or isinstance(map_height, bool)
            or not isinstance(map_width, (int, float))
            or not isinstance(map_height, (int, float))
            or not math.isfinite(map_width)
            or not math.isfinite(map_height)
            or map_width <= 0
            or map_height <= 0
        ):
            raise ValueError(
                "Map dimensions must be finite and positive."
            )

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        center = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        # Use the actual sprite when configured.
        if self.sprite is not None:
            scale = min(scale_x, scale_y) * self.size_scale

            target_size = (
                max(1, round(self.sprite.get_width() * scale)),
                max(1, round(self.sprite.get_height() * scale)),
            )

            rendered = self._get_scaled_sprite(target_size)
            rect = rendered.get_rect(center=center)

            if self.alive:
                surface.blit(rendered, rect)
            else:
                # Draw a dimmed copy without modifying the cached sprite.
                ruined = rendered.copy()
                ruined.fill(
                    (75, 75, 75, 180),
                    special_flags=pygame.BLEND_RGBA_MULT,
                )
                surface.blit(ruined, rect)

            bar_width = max(18, rect.width)
            bar_top = rect.top - 7

        else:
            # Fallback artwork when no sprite is configured.
            scale = min(scale_x, scale_y) * self.size_scale

            width = max(18, round(28 * scale))
            height = max(22, round(34 * scale))

            body = pygame.Rect(
                center[0] - width // 2,
                center[1] - height // 2,
                width,
                height,
            )

            body_color = (
                (115, 105, 85)
                if self.alive
                else (65, 65, 65)
            )
            roof_color = (
                (190, 145, 45)
                if self.alive
                else (80, 80, 80)
            )

            pygame.draw.rect(
                surface,
                body_color,
                body,
                border_radius=max(2, width // 8),
            )

            roof = [
                (body.left - width // 8, body.top + height // 4),
                (center[0], body.top - height // 5),
                (body.right + width // 8, body.top + height // 4),
            ]
            pygame.draw.polygon(surface, roof_color, roof)

            coin_radius = max(3, round(5 * scale))

            pygame.draw.circle(
                surface,
                (255, 205, 55) if self.alive else (90, 90, 90),
                center,
                coin_radius,
            )

            pygame.draw.circle(
                surface,
                (120, 80, 15) if self.alive else (45, 45, 45),
                center,
                coin_radius,
                width=max(1, coin_radius // 4),
            )

            bar_width = max(24, width)
            bar_top = body.top - 7

        self._draw_health_bar(
            surface,
            center[0],
            bar_top,
            bar_width,
        )
