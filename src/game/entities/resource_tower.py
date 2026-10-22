import math

from src.game.entities.structure import Structure


class ResourceTower(Structure):
    structure_kind = "resource"

    def __init__(self, structure_id, position, max_health, *, production_amount,
                 production_interval, on_health_changed=None):
        super().__init__(structure_id, position, max_health, on_health_changed=on_health_changed)
        if (isinstance(production_amount, bool) or not isinstance(production_amount, int)
                or production_amount <= 0):
            raise ValueError("Production amount must be a positive integer.")
        if not math.isfinite(production_interval) or production_interval <= 0:
            raise ValueError("Production interval must be finite and positive.")
        self.production_amount = production_amount
        self.production_interval = production_interval
        self.cooldown = production_interval

    def update(self, delta_time):
        """Return earned resources; only the simulation may credit the economy."""
        if not math.isfinite(delta_time) or delta_time < 0:
            raise ValueError("Delta time must be finite and non-negative.")
        if self.destroyed:
            return 0
        self.cooldown -= delta_time
        amount = 0
        while self.cooldown <= 1e-9:
            amount += self.production_amount
            self.cooldown += self.production_interval
        return amount
