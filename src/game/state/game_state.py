from dataclasses import dataclass, field
from enum import Enum

from src.game.economy.resources import ResourceManager


class GameStatus(str, Enum):
    RUNNING = "running"
    WON = "won"
    LOST = "lost"


@dataclass(slots=True)
class GameState:
    """Authoritative state for one tower-defense episode."""

    max_base_health: float = 100.0
    starting_resources: int = 200

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

    def __post_init__(self) -> None:
        if self.max_base_health <= 0:
            raise ValueError("Maximum base health must be positive.")
        if self.starting_resources < 0:
            raise ValueError("Starting resources cannot be negative.")

        self.resources = ResourceManager(
            starting_balance=self.starting_resources,
        )
        self.reset()

    @property
    def is_terminal(self) -> bool:
        return self.status in (GameStatus.WON, GameStatus.LOST)

    @property
    def resource_balance(self) -> int:
        """Current spendable resource balance."""
        return self.resources.balance

    def register_spawn(self) -> None:
        if not self.is_terminal:
            self.enemies_spawned += 1

    def register_kill(self, reward: int = 25) -> None:
        """Record a kill and grant its resource reward exactly once."""
        if self.is_terminal:
            return

        self.enemies_killed += 1
        self.resources.earn(reward)

    def register_leak(self, damage: float) -> None:
        """Apply base damage for an enemy that reaches the endpoint."""
        if damage < 0:
            raise ValueError("Leak damage cannot be negative.")

        if self.is_terminal:
            return

        self.enemies_leaked += 1
        self.total_leak_damage += damage
        self.base_health = max(0.0, self.base_health - damage)

        if self.base_health <= 0:
            self.status = GameStatus.LOST

    def mark_won(self) -> None:
        if not self.is_terminal:
            self.status = GameStatus.WON

    def reset(self) -> None:
        self.base_health = self.max_base_health
        self.status = GameStatus.RUNNING
        self.resources.reset()

        self.enemies_spawned = 0
        self.enemies_killed = 0
        self.enemies_leaked = 0
        self.total_leak_damage = 0.0
