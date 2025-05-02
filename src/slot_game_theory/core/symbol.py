"""
Defines the Symbol class, a fundamental component of a slot game.
"""

from typing import Any, Dict, Optional

class Symbol:
    """
    Represents a single symbol on a slot reel.

    Attributes:
        name (str): The unique identifier for the symbol (e.g., "A", "K", "Cherry", "Wild").
        display_name (Optional[str]): A user-friendly name for display purposes. If None, uses `name`.
        is_wild (bool): True if the symbol acts as a wild card.
        is_scatter (bool): True if the symbol acts as a scatter (pays anywhere, often triggers bonuses).
        pays (Optional[Dict[int, float]]): Payout multipliers for occurrences (e.g., {3: 10, 4: 50, 5: 200}).
                                           Typically used for scatter symbols or specific non-line wins.
        metadata (Optional[Dict[str, Any]]): Additional properties (e.g., expansion behavior, stacking).
    """
    def __init__(self,
                 name: str,
                 display_name: Optional[str] = None,
                 is_wild: bool = False,
                 is_scatter: bool = False,
                 pays: Optional[Dict[int, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        if not name:
            raise ValueError("Symbol name cannot be empty.")

        self.name = name
        self.display_name = display_name if display_name is not None else name
        self.is_wild = is_wild
        self.is_scatter = is_scatter
        self.pays = pays if pays is not None else {}
        self.metadata = metadata if metadata is not None else {}

        if self.is_wild and self.is_scatter:
            # While technically possible, it's often complex. Add warning or validation if needed.
            pass # Consider logging a warning or adjusting logic if this combination is problematic.

    def __str__(self) -> str:
        return self.display_name

    def __repr__(self) -> str:
        return (f"Symbol(name='{self.name}', display_name='{self.display_name}', "
                f"is_wild={self.is_wild}, is_scatter={self.is_scatter}, "
                f"pays={self.pays}, metadata={self.metadata})")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Safely retrieves a value from the metadata dictionary."""
        return self.metadata.get(key, default)

# Example Usage (can be removed or moved to tests)
if __name__ == '__main__':
    wild_symbol = Symbol(name="WILD", display_name="Wild", is_wild=True, metadata={"color": "gold"})
    scatter_symbol = Symbol(name="SCATTER", display_name="Bonus", is_scatter=True, pays={3: 5, 4: 20, 5: 100})
    ace_symbol = Symbol(name="A", display_name="Ace", metadata={"rank": 1})
    king_symbol = Symbol(name="K", display_name="King", metadata={"rank": 2})

    print(wild_symbol)
    print(repr(scatter_symbol))
    print(ace_symbol == Symbol("A"))
    print(ace_symbol == king_symbol)
    print(hash(ace_symbol))
    print(wild_symbol.get_metadata("color"))
    print(ace_symbol.get_metadata("rank"))
    print(king_symbol.get_metadata("nonexistent_key", default="Not Found")) 