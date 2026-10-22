import math

from __future__ import annotations

import math
from functools import lru_cache

import pygame

from src.game.entities.structure import Structure


class KingTower(Structure):
    structure_kind = "king"

    def __init__(self, structure_id, position, max_health, *, repair_cost, repair_duration,
                 on_health_changed=None):
        super().__init__(structure_id, position, max_health, on_health_changed=on_health_changed)
        if isinstance(repair_cost, bool) or not isinstance(repair_cost, int) or repair_cost < 0:
            raise ValueError("Repair cost must be a non-negative integer.")
        if not math.isfinite(repair_duration) or repair_duration <= 0:
            raise ValueError("Repair duration must be finite and positive.")
        self.repair_cost = repair_cost
        self.repair_duration = repair_duration
        self.repairing = False
        self.repair_elapsed = 0.0
        self._repair_rate = 0.0

    @property
    def can_repair(self):
        return not self.repairing and self.health < self.max_health

    def start_repair(self):
        """Start local repair progress; eligibility and payment belong to Simulation."""
        if not self.can_repair:
            return False
        self.repairing = True
        self.repair_elapsed = 0.0
        self._repair_rate = (self.max_health - self.health) / self.repair_duration
        return True

    def take_damage(self, amount):
        was_alive = self.alive
        damage = super().take_damage(amount)
        if was_alive and self.destroyed:
            self.repairing = False
        return damage

    def update(self, delta_time):
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")
        if not self.repairing or delta_time == 0:
            return
        elapsed = min(delta_time, self.repair_duration - self.repair_elapsed)
        self.repair_elapsed += elapsed
        self.health = min(self.max_health, self.health + self._repair_rate * elapsed)
        if self.health >= self.max_health - 1e-9:
            self.health = self.max_health
        if self.repair_elapsed >= self.repair_duration - 1e-9:
            self.repairing = False
        self.health_changed()

    def draw(self, surface, map_width, map_height, show_range=False):
        super().draw(surface, map_width, map_height, show_range)
        if self.repairing and map_width > 0 and map_height > 0:
            center = (round(self.position.x * surface.get_width() / map_width),
                      round(self.position.y * surface.get_height() / map_height))
            pygame.draw.circle(surface, (80, 190, 255), center, 23, 2)
    """Passive, damageable King Tower with timed repairs.

    The King Tower never attacks enemies. Simulation is responsible
    for repair costs, repair eligibility, and the defeat condition.
    """

    CROWN_EMOJI = "👑"

    def __init__(
        self,
        structure_id: int | str,
        position: pygame.Vector2,
        sprite: pygame.Surface | None = None,
        *,
        max_health: float = 1000.0,
        repair_duration: float = 8.0,
        repair_cost: int = 50,
        size_scale: float = 1.0,
    ) -> None:
        if (
            isinstance(repair_duration, bool)
            or not isinstance(repair_duration, (int, float))
            or not math.isfinite(repair_duration)
            or repair_duration <= 0
        ):
            raise ValueError(
                "Repair duration must be finite and positive."
            )

        if (
            isinstance(repair_cost, bool)
            or not isinstance(repair_cost, int)
            or repair_cost < 0
        ):
            raise ValueError(
                "Repair cost must be a non-negative integer."
            )

        if (
            isinstance(size_scale, bool)
            or not isinstance(size_scale, (int, float))
            or not math.isfinite(size_scale)
            or size_scale <= 0
        ):
            raise ValueError(
                "Size scale must be finite and positive."
            )

        if sprite is not None and not isinstance(
            sprite, pygame.Surface
        ):
            raise TypeError(
                "King Tower sprite must be a pygame Surface or None."
            )

        super().__init__(
            structure_id=structure_id,
            position=position,
            max_health=max_health,
        )

        self.sprite = sprite
        self.repair_duration = float(repair_duration)
        self.repair_cost = repair_cost
        self.size_scale = float(size_scale)

        self.repairing = False
        self.repair_elapsed = 0.0

        self._repair_amount = 0.0
        self._repair_amount_restored = 0.0

        self._scaled_sprite: pygame.Surface | None = None
        self._scaled_sprite_size: tuple[int, int] | None = None

    @property
    def repair_progress(self) -> float:
        """Repair progress between 0.0 and 1.0."""
        if not self.repairing:
            return 0.0

        return min(
            1.0,
            self.repair_elapsed / self.repair_duration,
        )

    @property
    def repair_time_remaining(self) -> float:
        """Seconds remaining in the current repair."""
        if not self.repairing:
            return 0.0

        return max(
            0.0,
            self.repair_duration - self.repair_elapsed,
        )

    def start_repair(
        self,
        *,
        repair_amount: float | None = None,
    ) -> bool:
        """Start timed repair.

        Simulation must validate the game state and charge the resource
        cost before calling this method.

        A destroyed King Tower can be repaired while the game remains
        active, provided the overall defeat condition has not triggered.
        """
        if self.repairing or self.health >= self.max_health:
            return False

        if repair_amount is None:
            amount = self.max_health - self.health
        else:
            if (
                isinstance(repair_amount, bool)
                or not isinstance(repair_amount, (int, float))
                or not math.isfinite(repair_amount)
                or repair_amount <= 0
            ):
                raise ValueError(
                    "Repair amount must be finite and positive."
                )

            amount = min(
                float(repair_amount),
                self.max_health - self.health,
            )

        if amount <= 0:
            return False

        self.repairing = True
        self.repair_elapsed = 0.0
        self._repair_amount = amount
        self._repair_amount_restored = 0.0

        return True

    def cancel_repair(self) -> bool:
        """Cancel repair; refund rules are handled by Simulation."""
        if not self.repairing:
            return False

        self.repairing = False
        self.repair_elapsed = 0.0
        self._repair_amount = 0.0
        self._repair_amount_restored = 0.0

        return True

    def update(self, delta_time: float) -> float:
        """Advance repair time and restore health gradually."""
        if (
            isinstance(delta_time, bool)
            or not isinstance(delta_time, (int, float))
            or not math.isfinite(delta_time)
            or delta_time < 0
        ):
            raise ValueError(
                "Delta time must be finite and non-negative."
            )

        if not self.repairing or delta_time == 0:
            return 0.0

        previous_elapsed = self.repair_elapsed

        self.repair_elapsed = min(
            self.repair_duration,
            self.repair_elapsed + delta_time,
        )

        elapsed = self.repair_elapsed - previous_elapsed
        health_per_second = (
            self._repair_amount / self.repair_duration
        )

        remaining_to_restore = (
            self._repair_amount - self._repair_amount_restored
        )

        requested_restore = min(
            health_per_second * elapsed,
            remaining_to_restore,
        )

        restored = self._restore_health(requested_restore)
        self._repair_amount_restored += restored

        if (
            self.repair_elapsed >= self.repair_duration
            or self._repair_amount_restored
            >= self._repair_amount - 1e-9
        ):
            self.repairing = False
            self.repair_elapsed = 0.0
            self._repair_amount = 0.0
            self._repair_amount_restored = 0.0

        return restored

    @staticmethod
    @lru_cache(maxsize=32)
    def _make_crown_surface(size: int) -> pygame.Surface:
        """Render the crown emoji, with a graphic fallback if unavailable."""
        size = max(12, size)

        if not pygame.font.get_init():
            pygame.font.init()

        # Prefer fonts that commonly contain the crown emoji.
        font_families = (
            "Noto Color Emoji",
            "Twitter Color Emoji",
            "Segoe UI Emoji",
            "Apple Color Emoji",
            "DejaVu Sans",
        )

        for family in font_families:
            font_path = pygame.font.match_font(family)

            if font_path is None:
                continue

            try:
                font = pygame.font.Font(font_path, size)
                metrics = font.metrics(KingTower.CROWN_EMOJI)

                if not metrics or any(metric is None for metric in metrics):
                    continue

                image = font.render(
                    KingTower.CROWN_EMOJI,
                    True,
                    (255, 220, 60),
                )

                if image.get_width() > 0 and image.get_height() > 0:
                    return image

            except (pygame.error, ValueError):
                continue

        # Fallback: a small stylized crown if no available font supports 👑.
        width = round(size * 1.25)
        height = size

        image = pygame.Surface(
            (width, height),
            pygame.SRCALPHA,
        )

        points = [
            (round(width * 0.12), round(height * 0.76)),
            (round(width * 0.06), round(height * 0.20)),
            (round(width * 0.32), round(height * 0.43)),
            (round(width * 0.50), round(height * 0.06)),
            (round(width * 0.68), round(height * 0.43)),
            (round(width * 0.94), round(height * 0.20)),
            (round(width * 0.88), round(height * 0.76)),
        ]

        pygame.draw.polygon(
            image,
            (255, 207, 45),
            points,
        )

        pygame.draw.rect(
            image,
            (230, 159, 25),
            (
                round(width * 0.12),
                round(height * 0.74),
                round(width * 0.76),
                max(2, round(height * 0.17)),
            ),
            border_radius=2,
        )

        for fraction in (0.30, 0.50, 0.70):
            pygame.draw.circle(
                image,
                (225, 55, 65),
                (
                    round(width * fraction),
                    round(height * 0.68),
                ),
                max(1, size // 16),
            )

        return image

    def _get_scaled_sprite(
        self,
        target_size: tuple[int, int],
    ) -> pygame.Surface | None:
        """Return the cached King Tower sprite at the requested size."""
        if self.sprite is None:
            return None

        if (
            self._scaled_sprite is None
            or self._scaled_sprite_size != target_size
        ):
            self._scaled_sprite = pygame.transform.smoothscale(
                self.sprite,
                target_size,
            )
            self._scaled_sprite_size = target_size

        return self._scaled_sprite

    def draw(
        self,
        surface: pygame.Surface,
        map_width: float,
        map_height: float,
    ) -> None:
        """Draw the King Tower, crown, health bar, and repair progress."""
        if map_width <= 0 or map_height <= 0:
            return

        scale_x = surface.get_width() / map_width
        scale_y = surface.get_height() / map_height

        center = (
            round(self.position.x * scale_x),
            round(self.position.y * scale_y),
        )

        source_width = (
            self.sprite.get_width()
            if self.sprite is not None
            else 64
        )
        source_height = (
            self.sprite.get_height()
            if self.sprite is not None
            else 64
        )

        target_size = (
            max(
                1,
                round(source_width * self.size_scale * scale_x),
            ),
            max(
                1,
                round(source_height * self.size_scale * scale_y),
            ),
        )

        tower_sprite = self._get_scaled_sprite(target_size)

        if tower_sprite is not None:
            tower_rect = tower_sprite.get_rect(center=center)
            surface.blit(tower_sprite, tower_rect)

            if not self.alive:
                # Keep the destroyed tower visible, but darken it.
                destroyed_overlay = pygame.Surface(
                    tower_rect.size,
                    pygame.SRCALPHA,
                )
                destroyed_overlay.fill((20, 20, 20, 160))
                surface.blit(destroyed_overlay, tower_rect)

            sprite_top = tower_rect.top
            sprite_width = tower_rect.width

        else:
            # Placeholder until an asset is provided.
            body = pygame.Rect(
                0,
                0,
                target_size[0],
                target_size[1],
            )
            body.center = center

            pygame.draw.rect(
                surface,
                (105, 105, 120) if self.alive else (55, 55, 60),
                body,
                border_radius=6,
            )

            pygame.draw.rect(
                surface,
                (225, 205, 145),
                body,
                width=3,
                border_radius=6,
            )

            sprite_top = body.top
            sprite_width = body.width

        # Draw the crown centered over the upper part of the tower.
        if self.alive:
            crown_size = max(
                16,
                round(30 * min(scale_x, scale_y)),
            )
            crown = self._make_crown_surface(crown_size)
            crown_rect = crown.get_rect(
                center=(
                    center[0],
                    sprite_top + max(2, round(crown_size * 0.30)),
                )
            )
            surface.blit(crown, crown_rect)

        # Health bar appears above the crown.
        bar_width = max(28, sprite_width)
        bar_height = 5
        bar_x = round(center[0] - bar_width / 2)

        crown_height = max(
            16,
            round(30 * min(scale_x, scale_y)),
        )
        bar_y = sprite_top - crown_height - 8

        pygame.draw.rect(
            surface,
            (55, 25, 25),
            (bar_x, bar_y, bar_width, bar_height),
        )

        if self.alive:
            health_color = (
                (65, 220, 90)
                if self.health_fraction > 0.3
                else (235, 75, 65)
            )

            pygame.draw.rect(
                surface,
                health_color,
                (
                    bar_x,
                    bar_y,
                    round(bar_width * self.health_fraction),
                    bar_height,
                ),
            )

        # A blue bar indicates the ongoing timed repair.
        if self.repairing:
            repair_y = bar_y - 6

            pygame.draw.rect(
                surface,
                (30, 30, 35),
                (bar_x, repair_y, bar_width, 4),
            )

            pygame.draw.rect(
                surface,
                (75, 180, 255),
                (
                    bar_x,
                    repair_y,
                    round(bar_width * self.repair_progress),
                    4,
                ),
            )
