"""Enemy sub-package.

Public API — import from here, not from the individual modules.

    from src.game.entities.enemies import Enemy, Soldier, Tank, EnemyConfig
"""

from src.game.entities.enemies.config import EnemyConfig
from src.game.entities.enemies.base import Enemy
from src.game.entities.enemies.soldier import Soldier
from src.game.entities.enemies.tank import Tank

__all__ = [
    "EnemyConfig",
    "Enemy",
    "Soldier",
    "Tank",
]
