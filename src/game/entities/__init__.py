"""Entities package.

All game entity types are accessible from here.

    from src.game.entities import Enemy, Soldier, Tank, EnemyConfig
    from src.game.entities import DefenseTower, KingTower, ResourceTower
    from src.game.entities import Structure
"""

from src.game.entities.enemies import Enemy, EnemyConfig, Soldier, Tank
from src.game.entities.structure import Structure
from src.game.entities.towers import DefenseTower, KingTower, Projectile, ResourceTower

__all__ = [
    "DefenseTower",
    "Enemy",
    "EnemyConfig",
    "KingTower",
    "Projectile",
    "ResourceTower",
    "Soldier",
    "Structure",
    "Tank",
]
