import math

import pygame


class Structure:
    """Shared health API; the simulation owns consequences of health changes."""

    structure_kind = "structure"

    def __init__(self, structure_id, position, max_health, *, on_health_changed=None):
        if not math.isfinite(max_health) or max_health <= 0:
            raise ValueError("Structure health must be finite and positive.")
        self.structure_id = str(structure_id)
        self.position = pygame.Vector2(position)
        if not self.structure_id.strip() or not all(math.isfinite(value) for value in self.position):
            raise ValueError("Structures require a non-empty ID and finite position.")
        self.max_health = float(max_health)
        self.health = self.max_health
        self.on_health_changed = on_health_changed

    @property
    def destroyed(self):
        return self.health <= 0

    @property
    def alive(self):
        return not self.destroyed

    def take_damage(self, amount):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Damage must be finite and non-negative.")
        if self.destroyed or amount == 0:
            return 0.0
        damage = min(self.health, amount)
        self.health -= damage
        self.health_changed()
        return damage

    def health_changed(self):
        if self.on_health_changed is not None:
            self.on_health_changed(self)

    def draw_health(self, surface, map_width, map_height):
        sx, sy = surface.get_width() / map_width, surface.get_height() / map_height
        x, y = round(self.position.x * sx) - 20, round(self.position.y * sy) - 27
        pygame.draw.rect(surface, (80, 30, 30), (x, y, 40, 5))
        pygame.draw.rect(surface, (60, 210, 100), (x, y, round(40 * self.health / self.max_health), 5))

    def draw(self, surface, map_width, map_height, show_range=False):
        if map_width <= 0 or map_height <= 0:
            return
        center = (round(self.position.x * surface.get_width() / map_width),
                  round(self.position.y * surface.get_height() / map_height))
        color = (90, 90, 90) if self.destroyed else (
            (235, 190, 50) if self.structure_kind == "king" else (60, 175, 150)
        )
        pygame.draw.circle(surface, color, center, 20)
        if self.destroyed:
            pygame.draw.line(surface, (35, 35, 35), (center[0]-10, center[1]-10),
                             (center[0]+10, center[1]+10), 3)
        self.draw_health(surface, map_width, map_height)
