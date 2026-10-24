"""Backward-compatibility shim.

All enemy symbols now live under src.game.entities.enemies.*
This module re-exports them so existing imports keep working.

Prefer:
    from src.game.entities.enemies import Enemy, Soldier, Tank, EnemyConfig
"""

from src.game.entities.enemies.config import EnemyConfig
from src.game.entities.enemies.base import Enemy
from src.game.entities.enemies.soldier import Soldier
from src.game.entities.enemies.tank import Tank

__all__ = ["EnemyConfig", "Enemy", "Soldier", "Tank"]
