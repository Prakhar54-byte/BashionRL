"""Backward-compatibility shim.

DefenseTower and Projectile now live under src.game.entities.towers.*
This module re-exports them so existing imports keep working.

Prefer:
    from src.game.entities.towers import DefenseTower, Projectile
"""

from src.game.entities.towers.projectile import Projectile
from src.game.entities.towers.defense_tower import DefenseTower

__all__ = ["Projectile", "DefenseTower"]
