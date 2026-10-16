from dataclasses import dataclass


@dataclass(slots=True)
class ResourceManager:
    """Manages resource income and spending for one game episode."""

    starting_balance: int = 200
    balance: int = 200
    total_earned: int = 0
    total_spent: int = 0

    def __post_init__(self) -> None:
        if self.starting_balance < 0:
            raise ValueError("Starting balance cannot be negative.")
        self.reset()

    def can_afford(self, cost: int) -> bool:
        self._validate_amount(cost)
        return self.balance >= cost

    def spend(self, cost: int) -> bool:
        """Spend resources atomically; return False if unaffordable."""
        self._validate_amount(cost)

        if self.balance < cost:
            return False

        self.balance -= cost
        self.total_spent += cost
        return True

    def earn(self, amount: int) -> None:
        """Add earned resources."""
        self._validate_amount(amount)
        self.balance += amount
        self.total_earned += amount

    def reset(self) -> None:
        """Restore the initial balance and episode accounting."""
        self.balance = self.starting_balance
        self.total_earned = 0
        self.total_spent = 0

    @staticmethod
    def _validate_amount(amount: int) -> None:
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("Resource amounts must be integers.")
        if amount < 0:
            raise ValueError("Resource amounts cannot be negative.")
