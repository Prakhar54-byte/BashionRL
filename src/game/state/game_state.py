from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from src.game.economy.resources import ResourceManager


KingTowerId = int | str


class GameStatus(str, Enum):
    RUNNING = "running"
    WON = "won"
    LOST = "lost"


@dataclass(slots=True)
class GameState:
    """Authoritative state for one tower-defense episode.

    Responsibilities:
    - Track the episode's status and resources.
    - Track enemy spawn, kill, and leak statistics.
    - Track destroyed King Towers.
    - Immediately lose the episode when every King Tower is destroyed.

    Base-health tracking is retained temporarily for compatibility with the
    existing Simulation while its objective and leak logic are being updated.
    """

    max_base_health: float = 100.0
    starting_resources: int = 200
    king_tower_count: int = 3

    base_health: float = field(init=False)

    status: GameStatus = field(
        default=GameStatus.RUNNING,
        init=False,
    )

    resources: ResourceManager = field(init=False)

    enemies_spawned: int = field(default=0, init=False)
    enemies_killed: int = field(default=0, init=False)
    enemies_leaked: int = field(default=0, init=False)
    total_leak_damage: float = field(default=0.0, init=False)

    _destroyed_king_tower_ids: set[KingTowerId] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_base_health) or self.max_base_health <= 0:
            raise ValueError("Maximum base health must be finite and positive.")

        if (
            isinstance(self.starting_resources, bool)
            or not isinstance(self.starting_resources, int)
            or self.starting_resources < 0
        ):
            raise ValueError("Starting resources must be a non-negative integer.")

        if (
            isinstance(self.king_tower_count, bool)
            or not isinstance(self.king_tower_count, int)
            or self.king_tower_count <= 0
        ):
            raise ValueError("King Tower count must be a positive integer.")

        self.resources = ResourceManager(
            starting_balance=self.starting_resources,
        )

        self.reset()

    @property
    def is_terminal(self) -> bool:
        """Whether the episode has reached a win or loss state."""
        return self.status in (GameStatus.WON, GameStatus.LOST)

    @property
    def is_running(self) -> bool:
        return self.status == GameStatus.RUNNING

    @property
    def resource_balance(self) -> int:
        """Current spendable resource balance."""
        return self.resources.balance

    @property
    def total_king_towers(self) -> int:
        """Configured number of King Towers in the episode."""
        return self.king_tower_count

    @property
    def king_towers_destroyed(self) -> int:
        """Number of distinct King Towers currently recorded as destroyed."""
        return len(self._destroyed_king_tower_ids)

    @property
    def king_towers_alive(self) -> int:
        """Number of King Towers not currently recorded as destroyed."""
        return max(
            0,
            self.king_tower_count - self.king_towers_destroyed,
        )

    @property
    def all_king_towers_destroyed(self) -> bool:
        return self.king_towers_destroyed >= self.king_tower_count

    @property
    def destroyed_king_tower_ids(self) -> frozenset[KingTowerId]:
        """Read-only snapshot of the IDs of destroyed King Towers."""
        return frozenset(self._destroyed_king_tower_ids)

    def register_spawn(self) -> None:
        """Record an enemy spawn while the episode is running."""
        if not self.is_terminal:
            self.enemies_spawned += 1

    def register_kill(self, reward: int = 25) -> None:
        """Record an enemy kill and grant its resource reward once."""
        if self.is_terminal:
            return

        if (
            isinstance(reward, bool)
            or not isinstance(reward, int)
            or reward < 0
        ):
            raise ValueError("Kill reward must be a non-negative integer.")

        self.enemies_killed += 1
        self.resources.earn(reward)

    def register_leak(self, damage: float) -> None:
        """Record a leaking enemy and apply legacy base-health damage.

        This behavior is retained for compatibility with the existing
        Simulation. The new King Tower objective logic will be integrated
        when Simulation is rewritten.
        """
        if not math.isfinite(damage) or damage < 0:
            raise ValueError("Leak damage must be finite and non-negative.")

        if self.is_terminal:
            return

        self.enemies_leaked += 1
        self.total_leak_damage += damage
        self.base_health = max(0.0, self.base_health - damage)

        if self.base_health <= 0:
            self.mark_lost()

    def register_king_tower_destroyed(
        self,
        tower_id: KingTowerId,
    ) -> None:
        """Record a destroyed King Tower.

        Repeated reports for the same tower do not count twice. If all
        configured King Towers are destroyed, the episode is lost immediately.
        """
        self._validate_king_tower_id(tower_id)

        if self.is_terminal:
            return

        self._destroyed_king_tower_ids.add(tower_id)

        if self.all_king_towers_destroyed:
            self.mark_lost()

    def register_king_tower_restored(
        self,
        tower_id: KingTowerId,
    ) -> None:
        """Remove a tower from the destroyed set if it is restored in time.

        Restoration is permitted only while the episode is still running.
        A terminal defeat can never be reversed by repairing a tower.
        """
        self._validate_king_tower_id(tower_id)

        if self.is_terminal:
            return

        self._destroyed_king_tower_ids.discard(tower_id)

    def mark_won(self) -> None:
        """Mark the episode as won unless it has already ended."""
        if not self.is_terminal:
            self.status = GameStatus.WON

    def mark_lost(self) -> None:
        """Mark the episode as lost unless it has already ended."""
        if not self.is_terminal:
            self.status = GameStatus.LOST

    def reset(self) -> None:
        """Restore all episode-level state for a new run."""
        self.base_health = self.max_base_health
        self.status = GameStatus.RUNNING
        self.resources.reset()

        self.enemies_spawned = 0
        self.enemies_killed = 0
        self.enemies_leaked = 0
        self.total_leak_damage = 0.0

        self._destroyed_king_tower_ids.clear()

    @staticmethod
    def _validate_king_tower_id(tower_id: KingTowerId) -> None:
        if isinstance(tower_id, bool) or not isinstance(tower_id, (int, str)):
            raise TypeError("King Tower ID must be an integer or string.")

        if isinstance(tower_id, str) and not tower_id.strip():
            raise ValueError("King Tower ID cannot be empty.")
