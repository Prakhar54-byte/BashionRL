"""Towers sub-package.

Public API — import from here, not from the individual modules.

    from src.game.entities.towers import (
        DefenseTower, KingTower, ResourceTower, Projectile
    )
"""

from src.game.entities.towers.projectile import Projectile
from src.game.entities.towers.defense_tower import DefenseTower
from src.game.entities.towers.king_tower import KingTower
from src.game.entities.towers.resource_tower import ResourceTower

__all__ = [
    "Projectile",
    "DefenseTower",
    "KingTower",
    "ResourceTower",
]
