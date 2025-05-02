# -*- coding: utf-8 -*-
"""
Defines classes and functions related to slot machine reels.

Includes:
- Reel strip representation
- Payline definitions
- Stop generation logic
"""
import logging
import random
from typing import List, Sequence, Dict, Any, Union, Tuple, Optional # Added Union, Tuple, Optional

# Import the enhanced Symbol class
from .symbol import Symbol

# Configure logging
logger = logging.getLogger(__name__)

class ReelStrip:
    """
    Represents a single reel strip containing an ordered sequence of symbols.
    """
    def __init__(self, symbols: Sequence[Union[Symbol, str]], name: str = "Reel"):
        """
        Initializes a ReelStrip.

        Args:
            symbols: A sequence of Symbol objects or string names representing the symbols on the strip.
                     If strings are provided, basic Symbol objects (only name) will be created.
                     The sequence must not be empty.
            name: An optional name for the reel strip (e.g., "Reel 1", "Bonus Reel"). Defaults to "Reel".

        Raises:
            ValueError: If the symbols sequence is empty.
            TypeError: If the symbols sequence contains invalid types or name is not a string.
        """
        if not symbols:
            raise ValueError("Reel strip symbols sequence cannot be empty.")
        if not isinstance(name, str):
             raise TypeError("ReelStrip name must be a string.")

        self.symbols: List[Symbol] = []
        for idx, s_input in enumerate(symbols):
            if isinstance(s_input, str):
                # Create a basic Symbol object if only a name string is provided.
                # Consider using a SymbolRegistry or dictionary lookup for predefined symbols
                # if more complex symbol properties are needed universally from just a name.
                logger.debug(f"Creating basic Symbol from string '{s_input}' for ReelStrip '{name}' at index {idx}")
                self.symbols.append(Symbol(name=s_input)) # Use imported Symbol
            elif isinstance(s_input, Symbol):
                self.symbols.append(s_input) # Use imported Symbol
            else:
                raise TypeError(f"Invalid type in symbols sequence at index {idx}: {type(s_input)}. Expected Symbol or str.")

        self.name: str = name
        self.length: int = len(self.symbols)
        logger.debug(f"ReelStrip created: Name='{self.name}', Length={self.length}")


    def get_symbol_at(self, index: int) -> Symbol:
        """
        Gets the symbol at a specific index on the strip, handling wrap-around.

        Args:
            index: The index to retrieve the symbol from. Can be negative or exceed strip length.

        Returns:
            The Symbol object at the calculated wrapped index.

        Raises:
            IndexError: If the reel strip is empty (should be prevented by __init__).
        """
        if self.length == 0:
             # This case should ideally be prevented by the __init__ check
             raise IndexError("Cannot get symbol from empty reel strip.")
        wrapped_index = index % self.length
        return self.symbols[wrapped_index]

    def __len__(self) -> int:
        """Returns the number of symbols on the reel strip."""
        return self.length

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the ReelStrip."""
        # Limit displayed symbols for very long strips
        max_display = 20
        # Use display_name if available, otherwise name
        symbols_repr = [s.display_name for s in self.symbols[:max_display]]
        if self.length > max_display:
            symbols_repr.append("...")
        return f"ReelStrip(name='{self.name}', length={self.length}, symbols={symbols_repr})"

# --- Payline Definitions ---
# Paylines define which positions on the visible window form winning lines.
# Can be simple horizontal lines, diagonals, zig-zags, etc.
# Represented typically as a list of (reel_index, row_index) tuples.
# Example: A 3x3 grid, middle horizontal line: [(0, 1), (1, 1), (2, 1)]

Payline = List[Tuple[int, int]] # List of (reel_idx, row_idx) coordinates

def get_standard_paylines(num_reels: int, num_rows: int) -> Dict[str, Payline]:
    """
    Generates a basic set of standard paylines for a given grid size.

    Includes horizontal lines and simple diagonals.

    Args:
        num_reels: The number of reels (width of the grid).
        num_rows: The number of visible rows (height of the grid).

    Returns:
        A dictionary mapping payline names (e.g., "line_mid") to their
        coordinate lists (List[Tuple[int, int]]). Returns an empty dict
        if num_reels or num_rows are less than 1.
    """
    if num_reels < 1 or num_rows < 1:
        logger.warning(f"Cannot generate paylines for invalid grid size: {num_reels}x{num_rows}")
        return {}

    paylines: Dict[str, Payline] = {}

    # Horizontal lines
    for r in range(num_rows):
        paylines[f"line_{r}"] = [(reel_idx, r) for reel_idx in range(num_reels)]

    # Simple diagonals (only if grid allows)
    if num_reels >= num_rows and num_rows > 1:
         # Top-left to bottom-right
         paylines["diag_down"] = [(i, i) for i in range(num_rows)]
         # Bottom-left to top-right
         paylines["diag_up"] = [(i, num_rows - 1 - i) for i in range(num_rows)]

    # TODO: Add more complex standard payline shapes (V, inverted V, zig-zags)
    # Example V-shape (for 5x3):
    # if num_reels >= 5 and num_rows >= 3:
    #    paylines["v_shape"] = [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0)]

    logger.debug(f"Generated {len(paylines)} standard paylines for {num_reels}x{num_rows} grid.")
    return paylines


# --- Stop Generation ---
# How the final reel positions (stops) are determined.
# Simplest: Independent random stop on each reel strip.
# More complex: Weighted stops, linked reels, anti-clustering logic, etc.

def generate_random_stops(reel_strips: List[ReelStrip], rng: Optional[random.Random] = None) -> List[int]:
    """
    Generates a random stop index (position) for each reel strip independently.

    Assumes uniform probability for each stop position on the strip.

    Args:
        reel_strips: The list of ReelStrip objects for the game.
        rng: Optional random number generator instance for reproducibility.
             If None, uses the default `random` module.

    Returns:
        A list of integer stop indices, one for each reel strip.
        Returns an empty list if reel_strips is empty.
    """
    if not reel_strips:
        return []

    if rng is None:
        rng = random # Use default random module

    stops = []
    for strip in reel_strips:
        if len(strip) == 0:
             logger.error(f"Cannot generate stop for empty ReelStrip: {strip.name}")
             # Or raise error? For now, append an invalid index or handle upstream.
             stops.append(-1) # Indicate error?
        else:
             stops.append(rng.randint(0, len(strip) - 1))
    return stops

def get_visible_window(reel_strips: List[ReelStrip], stops: List[int], num_rows: int) -> List[List[Symbol]]:
    """
    Determines the grid of symbols visible in the window based on reel stops.

    Args:
        reel_strips: The list of ReelStrip objects.
        stops: A list of stop indices, one for each reel strip. Must match length of reel_strips.
        num_rows: The number of visible rows (height of the window).

    Returns:
        A list of lists representing the visible window. Outer list represents columns (reels),
        inner list represents rows within a column (e.g., window[reel_idx][row_idx]).
        Returns Symbol objects within the lists.

    Raises:
        ValueError: If the length of stops does not match the number of reel strips,
                    or if num_rows is not positive.
    """
    if num_rows <= 0:
        raise ValueError("Number of rows (num_rows) must be positive.")

    num_reels = len(reel_strips)
    if len(stops) != num_reels:
        raise ValueError(f"Length of stops ({len(stops)}) must match number of reel strips ({num_reels}).")

    window: List[List[Symbol]] = []
    for r_idx in range(num_reels):
        strip = reel_strips[r_idx]
        stop_index = stops[r_idx]

        if len(strip) == 0:
             logger.warning(f"ReelStrip '{strip.name}' is empty. Generating empty column for window.")
             # Use a placeholder symbol for clarity, ensure it exists or create on the fly
             reel_column = [Symbol(name="EMPTY", display_name="-")] * num_rows
        elif stop_index < 0 or stop_index >= len(strip):
             logger.error(f"Invalid stop index {stop_index} for ReelStrip '{strip.name}' with length {len(strip)}. Generating error column.")
             # Use a placeholder symbol for clarity
             reel_column = [Symbol(name="ERROR", display_name="!")] * num_rows
        else:
            reel_column: List[Symbol] = []
            for row_idx in range(num_rows):
                # Symbol index on the strip: stop index + row index (adjusting for wrap-around)
                symbol_index_on_strip = (stop_index + row_idx) % len(strip)
                # Retrieve the actual Symbol object
                reel_column.append(strip.get_symbol_at(symbol_index_on_strip))
        window.append(reel_column)

    # The window is returned as List[Column], where Column = List[Symbol]
    # Example: window[0] is the first reel's visible symbols from top to bottom.
    return window