"""Soldier enemy variants (normal and heavy).

Both variants use this single class.  Their stats (speed, health,
damage) differ via ``EnemyConfig`` — no subclassing required.

Soldiers follow soldier routes.  When their movement segment crosses a
King Tower slot, Simulation applies one suicide hit and removes them.
This class does not implement that collision — it is handled entirely
by ``Simulation.tick``.
"""

from __future__ import annotations

import pygame

from src.game.entities.enemies.base import Enemy
from src.game.entities.enemies.config import EnemyConfig
from src.game.terrain.pathway import Pathway


class Soldier(Enemy):
    """Foot soldier that attacks the first King Tower slot it crosses.

    The normal and heavy soldier variants share this class — they are
    distinguished only by the values in their ``EnemyConfig``.

    Slot collision and the suicide attack are detected by
    ``Simulation.tick`` via a clipline check against King Tower slots.
    This class simply walks the path and does nothing special at the end.
    """

    def __init__(
        self,
        pathway: Pathway,
        sprite: pygame.Surface,
        config: EnemyConfig,
    ) -> None:
        super().__init__(pathway, sprite, config)

    # Soldiers do nothing at the route endpoint.
    # King Tower collision is handled externally by Simulation.
    def _on_path_finished(self) -> None:
        pass
