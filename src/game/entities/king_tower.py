import math

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
