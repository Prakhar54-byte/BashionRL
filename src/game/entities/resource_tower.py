from __future__ import annotations

import math

import pygame

from src.game.entities.structure import Structure


class ResourceTower(Structure):
    """A damageable structure that generates resources over time.

    Resource generation is handled by the simulation:
        earned = tower.update(delta_time)
        resources.earn(earned)

    A destroyed tower generates no resources. Its production timer preserves
    partial progress so a temporary pause does not discard accumulated time.
    """

    structure_kind = "resource"

    def __init__(
        self,
        structure_id: int | str,
        position: pygame.Vector2 | tuple[float, float],
        max_health: float = 250.0,
        *,
        sprite: pygame.Surface | None = None,
        on_health_changed=None,
        production_interval: float = 15.0,
        production_amount: int = 10,
        size_scale: float = 1.0,
    ) -> None:
        super().__init__(structure_id, position, max_health, on_health_changed=on_health_changed)

        if (
            not math.isfinite(production_interval)
            or production_interval <= 0
        ):
            raise ValueError(
                "Production interval must be finite and positive."
            )

        if (
            isinstance(production_amount, bool)
            or not isinstance(production_amount, int)
            or production_amount <= 0
        ):
            raise ValueError(
                "Production amount must be a positive integer."
            )

        if not math.isfinite(size_scale) or size_scale <= 0:
            raise ValueError("Size scale must be finite and positive.")

        if sprite is not None and (
            sprite.get_width() <= 0 or sprite.get_height() <= 0
        ):
            raise ValueError("Resource tower sprite must have valid dimensions.")

        self.sprite = sprite
        self.production_interval = float(production_interval)
        self.production_amount = production_amount
        self.size_scale = float(size_scale)

        self._production_elapsed = 0.0
        self._render_cache: dict[
            tuple[int, int],
            pygame.Surface,
        ] = {}

    @property
    def production_progress(self) -> float:
        """Progress toward the next resource payout, from 0.0 to 1.0."""
        return min(
            1.0,
            self._production_elapsed / self.production_interval,
        )

    @property
    def production_time_remaining(self) -> float:
        """Seconds until the next resource payout."""
        return max(
            0.0,
            self.production_interval - self._production_elapsed,
        )

    def update(self, delta_time: float) -> int:
        """Advance production and return the resources earned this update.

        The returned value may represent multiple payouts if delta_time
        spans multiple production intervals. The caller is responsible for
        adding that amount to the game's ResourceManager.
        """
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if not self.alive or delta_time == 0:
            return 0

        self._production_elapsed += delta_time

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

    def draw(
        self,
        surface: pygame.Surface,
        map_width: int,
        map_height: int,
    ) -> None:
        """Draw the resource tower in map-relative screen coordinates.

        A sprite is used when one is supplied. Otherwise, a simple gold-mine
        placeholder is drawn so the tower remains visible during development.
        """
        if map_width <= 0 or map_height <= 0:
            raise ValueError("Map dimensions must be positive.")

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        center = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        scale = min(scale_x, scale_y) * self.size_scale

        if self.sprite is not None:
            target_size = (
                max(1, round(self.sprite.get_width() * scale)),
                max(1, round(self.sprite.get_height() * scale)),
            )

            rendered = self._render_cache.get(target_size)

            if rendered is None:
                rendered = pygame.transform.smoothscale(
                    self.sprite,
                    target_size,
                )
                self._render_cache[target_size] = rendered

            rect = rendered.get_rect(center=center)

            if self.alive:
                surface.blit(rendered, rect)
            else:
                # Keep destroyed towers visible, but dim their sprite.
                ruined = rendered.copy()
                ruined.fill((75, 75, 75, 180), special_flags=pygame.BLEND_RGBA_MULT)
                surface.blit(ruined, rect)

            bar_width = max(18, rect.width)
            bar_top = rect.top - 7
        else:
            # Temporary gold-mine artwork until a resource tower sprite
            # is configured in the asset settings.
            width = max(18, round(28 * scale))
            height = max(22, round(34 * scale))

            body = pygame.Rect(
                center[0] - width // 2,
                center[1] - height // 2,
                width,
                height,
            )

            body_color = (115, 105, 85) if self.alive else (65, 65, 65)
            roof_color = (190, 145, 45) if self.alive else (80, 80, 80)

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
                (center[0], center[1]),
                coin_radius,
            )
            pygame.draw.circle(
                surface,
                (120, 80, 15) if self.alive else (45, 45, 45),
                (center[0], center[1]),
                coin_radius,
                width=max(1, coin_radius // 4),
            )

            bar_width = max(24, width)
            bar_top = body.top - 7

        # Health bar.
        bar_height = 4
        bar_left = center[0] - bar_width // 2

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_left, bar_top, bar_width, bar_height),
        )

        health_fraction = max(
            0.0,
            min(1.0, self.health / self.max_health),
        )

        pygame.draw.rect(
            surface,
            (65, 210, 85) if self.alive else (100, 65, 65),
            (
                bar_left,
                bar_top,
                round(bar_width * health_fraction),
                bar_height,
            ),
        )
