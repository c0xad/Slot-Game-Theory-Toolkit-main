# -*- coding: utf-8 -*-
"""
Defines classes and functions related to slot machine reels and game mechanics.

Includes:
- Reel strip representation and manipulation
- Advanced payline definitions and patterns
- Complex stop generation algorithms
- Visible window calculation
- Reel set configuration support
- Symbol distribution analysis
"""
import logging
import random
import math
import numpy as np
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Sequence, Dict, Any, Union, Tuple, Optional, Callable, Set, Iterator, TypeVar, Generic, Iterable

# Import the enhanced Symbol class
from .symbol import Symbol

# Configure logging
logger = logging.getLogger(__name__)


class StopSelectionMethod(Enum):
    """
    Enumeration of different methods for selecting reel stops.
    """
    UNIFORM = auto()  # Equal probability for all stops
    WEIGHTED = auto()  # Weighted probability based on stop weights
    VIRTUAL_MAPPING = auto()  # Virtual reel to physical reel mapping
    CLUSTERED = auto()  # Stops that tend to appear in groups
    ANTI_CLUSTERED = auto()  # Stops that tend to avoid appearing together


class ReelStrip:
    """
    Represents a single reel strip containing an ordered sequence of symbols.
    Provides methods for analyzing and manipulating reel contents.
    """
    def __init__(self, 
                 symbols: Sequence[Union[Symbol, str]], 
                 name: str = "Reel",
                 weights: Optional[List[float]] = None,
                 virtual_mapping: Optional[Dict[int, int]] = None):
        """
        Initializes a ReelStrip.

        Args:
            symbols: A sequence of Symbol objects or string names representing the symbols on the strip.
                     If strings are provided, basic Symbol objects (only name) will be created.
                     The sequence must not be empty.
            name: An optional name for the reel strip (e.g., "Reel 1", "Bonus Reel"). Defaults to "Reel".
            weights: Optional list of weights for each stop position. If provided, must match the length
                     of symbols. Used for weighted random selection.
            virtual_mapping: Optional mapping from virtual stops to physical stops. Useful for
                             implementing virtual reel systems.

        Raises:
            ValueError: If the symbols sequence is empty or weights don't match symbols length.
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
                logger.debug(f"Creating basic Symbol from string '{s_input}' for ReelStrip '{name}' at index {idx}")
                self.symbols.append(Symbol(name=s_input)) 
            elif isinstance(s_input, Symbol):
                self.symbols.append(s_input)
            else:
                raise TypeError(f"Invalid type in symbols sequence at index {idx}: {type(s_input)}. Expected Symbol or str.")

        self.name: str = name
        self.length: int = len(self.symbols)
        
        # Set up weights for non-uniform stop selection
        if weights is not None:
            if len(weights) != self.length:
                raise ValueError(f"Length of weights ({len(weights)}) must match length of symbols ({self.length}).")
            self.weights = weights
        else:
            # Default to uniform weights
            self.weights = [1.0] * self.length
            
        # Normalize weights to probabilities
        weight_sum = sum(self.weights)
        self.probabilities = [w / weight_sum for w in self.weights]
        
        # Set up virtual mapping
        self.virtual_mapping = virtual_mapping
        
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

    def get_symbols_window(self, start_index: int, window_size: int) -> List[Symbol]:
        """
        Gets a window of symbols starting from a specific index.

        Args:
            start_index: The starting index for the window.
            window_size: The number of symbols to include in the window.

        Returns:
            A list of Symbol objects in the specified window.
        """
        return [self.get_symbol_at(start_index + i) for i in range(window_size)]

    def get_symbol_distribution(self) -> Dict[str, float]:
        """
        Calculates the distribution of symbols on this reel strip.

        Returns:
            A dictionary mapping symbol names to their frequency (as a percentage).
        """
        counter = Counter(symbol.name for symbol in self.symbols)
        return {symbol: count / self.length for symbol, count in counter.items()}
    
    def get_symbol_positions(self, symbol_name: str) -> List[int]:
        """
        Find all positions of a given symbol on the reel strip.
        
        Args:
            symbol_name: Name of the symbol to find.
            
        Returns:
            List of indices where the symbol appears.
        """
        return [i for i, symbol in enumerate(self.symbols) if symbol.name == symbol_name]
    
    def calculate_rtp_contribution(self, symbol_payouts: Dict[str, List[float]]) -> float:
        """
        Calculate the theoretical Return-to-Player (RTP) contribution of this reel strip
        based on symbol distributions and payout values.
        
        Args:
            symbol_payouts: Dictionary mapping symbol names to payout values for different numbers
                           of matching symbols (e.g. {'CHERRY': [0, 0, 5, 10, 50]})
                           
        Returns:
            The RTP contribution as a value between 0 and 1 (e.g., 0.95 for 95% RTP)
        """
        distribution = self.get_symbol_distribution()
        total_rtp = 0.0
        
        for symbol_name, frequency in distribution.items():
            if symbol_name in symbol_payouts:
                # Simplified calculation - would need to be modified for actual game rules
                max_payout = max(symbol_payouts[symbol_name])
                total_rtp += frequency * max_payout
                
        return total_rtp
    
    def optimize_for_volatility(self, target_volatility: float, 
                              symbol_values: Dict[str, float]) -> 'ReelStrip':
        """
        Creates a new reel strip with reordered symbols to achieve a target volatility.
        
        Args:
            target_volatility: The desired volatility level.
            symbol_values: Dictionary mapping symbol names to their value/weight.
            
        Returns:
            A new ReelStrip with reordered symbols.
        """
        # Example of a complex optimization algorithm
        # In a real implementation, this would use advanced statistical methods
        
        # Create a copy of the symbols to reorder
        symbols_copy = self.symbols.copy()
        
        # Simple implementation: sort symbols by their value * random factor
        # to approximate desired volatility
        sorted_symbols = sorted(
            symbols_copy,
            key=lambda s: symbol_values.get(s.name, 0) * (1 + (random.random() - 0.5) * target_volatility)
        )
        
        return ReelStrip(sorted_symbols, name=f"{self.name}_optimized")

    def __iter__(self) -> Iterator[Symbol]:
        """Returns an iterator over the symbols in the reel strip."""
        return iter(self.symbols)
    
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


class ReelSet:
    """
    Represents a complete set of reel strips for a slot game.
    Manages multiple reel strips and their configuration.
    """
    def __init__(self, reel_strips: List[ReelStrip], name: str = "Standard"):
        """
        Initialize a ReelSet with a list of ReelStrip objects.
        
        Args:
            reel_strips: List of ReelStrip objects for each reel.
            name: Optional name for this reel set (e.g., "Base Game", "Free Spins").
        """
        if not reel_strips:
            raise ValueError("ReelSet must contain at least one ReelStrip.")
        
        self.reel_strips = reel_strips
        self.name = name
        self.num_reels = len(reel_strips)
    
    def get_reel(self, index: int) -> ReelStrip:
        """
        Get a specific reel strip by index.
        
        Args:
            index: Index of the reel to retrieve.
            
        Returns:
            The ReelStrip at the specified index.
            
        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= self.num_reels:
            raise IndexError(f"Reel index {index} out of range (0-{self.num_reels-1}).")
        return self.reel_strips[index]
    
    def analyze_symbol_distribution(self) -> Dict[str, List[float]]:
        """
        Analyze the distribution of symbols across all reels.
        
        Returns:
            Dictionary mapping symbol names to a list of their frequencies on each reel.
        """
        all_symbols = set()
        for reel in self.reel_strips:
            all_symbols.update(s.name for s in reel.symbols)
            
        distribution = {symbol: [] for symbol in all_symbols}
        
        for reel in self.reel_strips:
            reel_dist = reel.get_symbol_distribution()
            for symbol in all_symbols:
                distribution[symbol].append(reel_dist.get(symbol, 0))
                
        return distribution
    
    def __len__(self) -> int:
        """Returns the number of reel strips in this set."""
        return self.num_reels


# --- Payline Definitions ---
# Paylines define which positions on the visible window form winning lines.

Payline = List[Tuple[int, int]]  # List of (reel_idx, row_idx) coordinates

class PaylinePattern(Enum):
    """
    Enumeration of standard payline pattern types.
    """
    HORIZONTAL = auto()  # Straight horizontal lines
    DIAGONAL = auto()    # Diagonal lines
    V_SHAPE = auto()     # V-shaped lines
    W_SHAPE = auto()     # W-shaped lines
    ZIG_ZAG = auto()     # Zig-zag patterns
    CUSTOM = auto()      # Custom defined patterns


@dataclass
class PaylineConfig:
    """
    Configuration for generating complex payline patterns.
    """
    pattern: PaylinePattern
    num_reels: int
    num_rows: int
    start_row: Optional[int] = None  # For patterns that start at a specific row
    peak_reel: Optional[int] = None  # For V/W patterns, which reel is the peak
    zigzag_amplitude: Optional[int] = None  # For zigzag patterns


def get_standard_paylines(num_reels: int, num_rows: int) -> Dict[str, Payline]:
    """
    Generates a comprehensive set of standard paylines for a given grid size.

    Includes horizontal lines, diagonals, V-shapes, W-shapes, and zig-zags.

    Args:
        num_reels: The number of reels (width of the grid).
        num_rows: The number of visible rows (height of the grid).

    Returns:
        A dictionary mapping payline names to their coordinate lists.
    """
    if num_reels < 1 or num_rows < 1:
        logger.warning(f"Cannot generate paylines for invalid grid size: {num_reels}x{num_rows}")
        return {}

    paylines: Dict[str, Payline] = {}

    # Horizontal lines
    for r in range(num_rows):
        paylines[f"line_{r}"] = [(reel_idx, r) for reel_idx in range(num_reels)]

    # Simple diagonals (only if grid allows)
    if num_reels >= 3 and num_rows >= 2:
         # Top-left to bottom-right
         if num_reels >= num_rows:
             paylines["diag_down"] = [(i, min(i, num_rows-1)) for i in range(num_reels)]
         else:
             paylines["diag_down"] = [(i, i) for i in range(min(num_reels, num_rows))]
             
         # Bottom-left to top-right
         if num_reels >= num_rows:
             paylines["diag_up"] = [(i, max(0, num_rows-1-i)) for i in range(num_reels)]
         else:
             paylines["diag_up"] = [(i, num_rows-1-i) for i in range(min(num_reels, num_rows))]

    # V-shapes (for 5+ reels, 3+ rows)
    if num_reels >= 5 and num_rows >= 3:
        # V-shape with peak at middle reel
        mid_reel = num_reels // 2
        paylines["v_shape"] = []
        for i in range(num_reels):
            row = min(num_rows-1, abs(i - mid_reel))
            paylines["v_shape"].append((i, row))
            
        # Inverted V-shape
        paylines["inv_v_shape"] = []
        for i in range(num_reels):
            row = min(num_rows-1, num_rows-1-abs(i - mid_reel))
            paylines["inv_v_shape"].append((i, row))

    # W-shapes (for 5+ reels, 3+ rows)
    if num_reels >= 5 and num_rows >= 3:
        # W-shape
        segment_width = (num_reels - 1) // 2
        paylines["w_shape"] = []
        for i in range(num_reels):
            if i <= segment_width:
                # First half of W
                row = min(num_rows - 1, num_rows - 1 - i * (num_rows-1) // segment_width)
            else:
                # Second half of W
                relative_pos = i - segment_width
                period = min(num_rows - 1, (relative_pos * (num_rows-1) // segment_width) % (2 * (num_rows-1)))
                row = period if period <= num_rows-1 else 2*(num_rows-1) - period
            paylines["w_shape"].append((i, int(row)))

    # Zig-zag patterns
    if num_reels >= 4 and num_rows >= 3:
        # Simple zig-zag
        paylines["zigzag"] = []
        for i in range(num_reels):
            row = (i % 2) * (num_rows - 1)
            paylines["zigzag"].append((i, row))
            
        # Complex zig-zag
        paylines["complex_zigzag"] = []
        middle_row = num_rows // 2
        amplitude = min(1, (num_rows - 1) // 2)
        for i in range(num_reels):
            offset = amplitude * math.sin(i * math.pi / 2)
            row = middle_row + round(offset)
            paylines["complex_zigzag"].append((i, row))

    logger.debug(f"Generated {len(paylines)} standard paylines for {num_reels}x{num_rows} grid.")
    return paylines


def generate_custom_payline(config: PaylineConfig) -> Payline:
    """
    Generates a custom payline based on the specified configuration.
    
    Args:
        config: PaylineConfig object specifying pattern type and parameters.
        
    Returns:
        A list of (reel_idx, row_idx) coordinates defining the payline.
    """
    if config.pattern == PaylinePattern.HORIZONTAL:
        row = config.start_row if config.start_row is not None else 0
        return [(reel, row) for reel in range(config.num_reels)]
        
    elif config.pattern == PaylinePattern.DIAGONAL:
        if config.start_row is None:
            start_row = 0
        else:
            start_row = config.start_row
            
        payline = []
        for reel in range(config.num_reels):
            row = (start_row + reel) % config.num_rows
            payline.append((reel, row))
        return payline
        
    elif config.pattern == PaylinePattern.V_SHAPE:
        if config.peak_reel is None:
            peak_reel = config.num_reels // 2
        else:
            peak_reel = config.peak_reel
            
        payline = []
        for reel in range(config.num_reels):
            row = min(config.num_rows - 1, abs(reel - peak_reel))
            payline.append((reel, row))
        return payline
        
    elif config.pattern == PaylinePattern.W_SHAPE:
        payline = []
        segment_width = config.num_reels // 4
        
        for reel in range(config.num_reels):
            segment = reel // segment_width
            pos_in_segment = reel % segment_width
            
            if segment % 2 == 0:  # Downward segments
                row = pos_in_segment * (config.num_rows - 1) // segment_width
            else:  # Upward segments
                row = (config.num_rows - 1) - pos_in_segment * (config.num_rows - 1) // segment_width
                
            payline.append((reel, min(row, config.num_rows - 1)))
        return payline
        
    elif config.pattern == PaylinePattern.ZIG_ZAG:
        amplitude = config.zigzag_amplitude if config.zigzag_amplitude is not None else 1
        middle_row = config.num_rows // 2
        
        payline = []
        for reel in range(config.num_reels):
            # Use sine function to generate zig-zag
            offset = amplitude * math.sin(reel * math.pi)
            row = middle_row + round(offset)
            row = max(0, min(config.num_rows - 1, row))  # Clamp to valid rows
            payline.append((reel, row))
        return payline
    
    else:  # PaylinePattern.CUSTOM or unhandled case
        logger.warning(f"Unhandled payline pattern: {config.pattern}")
        return []


def get_ways_to_win_paylines(num_reels: int, num_rows: int) -> Dict[str, List[Set[Tuple[int, int]]]]:
    """
    Generates all possible ways-to-win combinations for a given grid size.
    
    In ways-to-win games, each winning combination includes one symbol position
    from each consecutive reel, starting from the leftmost reel.
    
    Args:
        num_reels: The number of reels (width of the grid).
        num_rows: The number of visible rows (height of the grid).
        
    Returns:
        A dictionary mapping length of win (2 to num_reels) to a list of all
        possible position combinations of that length.
    """
    if num_reels < 2 or num_rows < 1:
        logger.warning(f"Cannot generate ways-to-win for invalid grid size: {num_reels}x{num_rows}")
        return {}
    
    # Create all possible positions for each reel
    reel_positions = [
        {(reel, row) for row in range(num_rows)}
        for reel in range(num_reels)
    ]
    
    ways_to_win = {}
    
    # Generate all combinations for each win length (2 to num_reels)
    for win_length in range(2, num_reels + 1):
        combinations = []
        
        # Start with all positions on first reel
        current_combinations = [{pos} for pos in reel_positions[0]]
        
        # For each subsequent reel up to win_length
        for reel in range(1, win_length):
            new_combinations = []
            
            # Extend each current combination with each position on current reel
            for combo in current_combinations:
                for pos in reel_positions[reel]:
                    new_combo = combo.copy()
                    new_combo.add(pos)
                    new_combinations.append(new_combo)
                    
            current_combinations = new_combinations
            
        ways_to_win[win_length] = current_combinations
    
    return ways_to_win


# --- Stop Generation ---
# Advanced methods for determining the final reel positions (stops).

def generate_random_stops(reel_strips: List[ReelStrip], 
                         method: StopSelectionMethod = StopSelectionMethod.UNIFORM,
                         rng: Optional[random.Random] = None) -> List[int]:
    """
    Generates stop indices (positions) for each reel strip using the specified method.

    Args:
        reel_strips: The list of ReelStrip objects for the game.
        method: The method to use for selecting stops.
        rng: Optional random number generator instance for reproducibility.
             If None, uses the default `random` module.

    Returns:
        A list of integer stop indices, one for each reel strip.
    """
    if not reel_strips:
        return []

    if rng is None:
        rng = random

    stops = []
    
    for strip in reel_strips:
        if len(strip) == 0:
             logger.error(f"Cannot generate stop for empty ReelStrip: {strip.name}")
             stops.append(-1)
             continue
             
        if method == StopSelectionMethod.UNIFORM:
            # Simple uniform random selection
            stops.append(rng.randint(0, len(strip) - 1))
            
        elif method == StopSelectionMethod.WEIGHTED:
            # Weighted random selection based on strip's probability distribution
            stops.append(np.random.choice(range(len(strip)), p=strip.probabilities))
            
        elif method == StopSelectionMethod.VIRTUAL_MAPPING:
            # Use virtual mapping if available, otherwise fall back to uniform
            if strip.virtual_mapping:
                virtual_stop = rng.randint(0, max(strip.virtual_mapping.keys()))
                # Map virtual stop to physical stop
                physical_stop = strip.virtual_mapping.get(
                    virtual_stop, 
                    virtual_stop % len(strip)  # Fallback if mapping is incomplete
                )
                stops.append(physical_stop)
            else:
                logger.warning(f"No virtual mapping available for {strip.name}, using uniform selection.")
                stops.append(rng.randint(0, len(strip) - 1))
                
        elif method == StopSelectionMethod.CLUSTERED:
            # Generate clustered stops (more likely to be similar across reels)
            base_stop = rng.randint(0, len(strip) - 1)
            if stops:  # If we have previous stops, tend to cluster
                # Generate a stop that tends to be near the previous reel's stop
                # Adjust for different reel lengths
                previous_stop = stops[-1]
                previous_length = len(reel_strips[len(stops) - 1])
                relative_position = previous_stop / previous_length
                
                # Target similar relative position with some randomness
                target_position = (relative_position + rng.gauss(0, 0.1)) % 1.0
                target_stop = int(target_position * len(strip))
                
                # Mix between random and clustered
                cluster_weight = 0.7  # 70% clustering, 30% random
                if rng.random() < cluster_weight:
                    stops.append(target_stop)
                else:
                    stops.append(base_stop)
            else:
                stops.append(base_stop)
                
        elif method == StopSelectionMethod.ANTI_CLUSTERED:
            # Generate anti-clustered stops (more likely to be different across reels)
            base_stop = rng.randint(0, len(strip) - 1)
            if stops:  # If we have previous stops, tend to separate
                previous_stop = stops[-1]
                previous_length = len(reel_strips[len(stops) - 1])
                
                # Aim for opposite side of the reel
                relative_position = previous_stop / previous_length
                target_position = (relative_position + 0.5) % 1.0
                target_stop = int(target_position * len(strip))
                
                # Add some randomness
                final_stop = (target_stop + rng.randint(-len(strip)//8, len(strip)//8)) % len(strip)
                stops.append(final_stop)
            else:
                stops.append(base_stop)
                
        else:
            logger.warning(f"Unknown stop selection method: {method}, using uniform selection.")
            stops.append(rng.randint(0, len(strip) - 1))
            
    return stops


def generate_biased_stops(reel_strips: List[ReelStrip], 
                         bias_config: Dict[str, float],
                         rng: Optional[random.Random] = None) -> List[int]:
    """
    Generates stops biased towards or away from specific symbols.
    
    Args:
        reel_strips: The list of ReelStrip objects.
        bias_config: Dictionary mapping symbol names to bias values:
                    - Positive values increase chance of showing the symbol
                    - Negative values decrease chance of showing the symbol
                    - Zero means no bias for that symbol
        rng: Optional random number generator instance.
        
    Returns:
        A list of integer stop indices, one for each reel strip.
    """
    if not reel_strips:
        return []
        
    if rng is None:
        rng = random
        
    stops = []
    
    for strip in reel_strips:
        if len(strip) == 0:
            stops.append(-1)
            continue
            
        # Calculate bias scores for each position
        position_scores = []
        for pos in range(len(strip)):
            # Look at visible window (assuming 3 rows)
            visible_window = 3
            score = 0
            
            for offset in range(visible_window):
                symbol = strip.get_symbol_at(pos + offset)
                if symbol.name in bias_config:
                    score += bias_config[symbol.name]
                    
            position_scores.append(score)
            
        # Convert scores to probabilities (softmax)
        max_score = max(position_scores)
        exp_scores = [math.exp(score - max_score) for score in position_scores]
        sum_exp = sum(exp_scores)
        probabilities = [score / sum_exp for score in exp_scores]
        
        # Select stop based on calculated probabilities
        selected_pos = np.random.choice(range(len(strip)), p=probabilities)
        stops.append(selected_pos)
        
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
             reel_column = [Symbol(name="EMPTY", display_name="-")] * num_rows
        elif stop_index < 0 or stop_index >= len(strip):
             logger.error(f"Invalid stop index {stop_index} for ReelStrip '{strip.name}' with length {len(strip)}. Generating error column.")
             reel_column = [Symbol(name="ERROR", display_name="!")] * num_rows
        else:
            reel_column: List[Symbol] = []
            for row_idx in range(num_rows):
                # Symbol index on the strip: stop index + row index (adjusting for wrap-around)
                symbol_index_on_strip = (stop_index + row_idx) % len(strip)
                # Retrieve the actual Symbol object
                reel_column.append(strip.get_symbol_at(symbol_index_on_strip))
        window.append(reel_column)

    return window


def generate_scatter_optimized_stops(reel_strips: List[ReelStrip], 
                                   scatter_symbol: str,
                                   target_count: int,
                                   rng: Optional[random.Random] = None) -> List[int]:
    """
    Generates stops optimized to show a specific number of scatter symbols.
    
    Useful for feature triggers and bonus games.
    
    Args:
        reel_strips: List of ReelStrip objects.
        scatter_symbol: Name of the scatter symbol.
        target_count: Target number of scatter symbols to show.
        rng: Optional random number generator.
        
    Returns:
        List of stop indices optimized for scatter appearance.
    """
    if not reel_strips:
        return []
        
    if rng is None:
        rng = random
        
    # Find all positions of scatter symbols on each reel
    scatter_positions = []
    for strip in reel_strips:
        positions = [i for i, symbol in enumerate(strip.symbols) 
                    if symbol.name == scatter_symbol]
        scatter_positions.append(positions)
        
    num_reels = len(reel_strips)
    num_rows = 3  # Default visible window height
    
    # Calculate how many reels should show scatters
    reels_with_scatters = min(target_count, num_reels)
    
    # Randomly select which reels will show scatters
    scatter_reels = sorted(rng.sample(range(num_reels), reels_with_scatters))
    
    stops = []
    for i, strip in enumerate(reel_strips):
        if i in scatter_reels and scatter_positions[i]:
            # For reels that should show scatter, pick a position where scatter
            # will be visible in the window
            scatter_pos = rng.choice(scatter_positions[i])
            # Adjust stop position so scatter is visible in window
            row_offset = rng.randint(0, num_rows - 1)
            stop = (scatter_pos - row_offset) % len(strip)
            stops.append(stop)
        else:
            # For other reels, avoid showing scatters
            if not scatter_positions[i] or len(scatter_positions[i]) == len(strip):
                # If no scatters on this reel or all positions have scatters,
                # just pick a random stop
                stops.append(rng.randint(0, len(strip) - 1))
            else:
                # Otherwise, pick a stop that avoids showing scatters
                candidate_stops = []
                for stop in range(len(strip)):
                    visible_positions = [(stop + row) % len(strip) for row in range(num_rows)]
                    # Check if any visible position has a scatter
                    has_scatter = any(pos in scatter_positions[i] for pos in visible_positions)
                    if not has_scatter:
                        candidate_stops.append(stop)
                        
                if candidate_stops:
                    stops.append(rng.choice(candidate_stops))
                else:
                    # Fallback if all stops would show a scatter
                    stops.append(rng.randint(0, len(strip) - 1))
    
    return stops


class ReelAnalyzer:
    """
    Utility class for analyzing reel configurations and symbol distributions.
    """
    
    @staticmethod
    def calculate_symbol_hit_frequency(reel_strips: List[ReelStrip], 
                                    symbol_name: str,
                                    min_count: int = 1) -> float:
        """
        Calculate the frequency of hitting at least min_count of a specific symbol.
        
        Args:
            reel_strips: List of ReelStrip objects.
            symbol_name: Name of the symbol to analyze.
            min_count: Minimum number of symbols required.
            
        Returns:
            Hit frequency as a value between 0 and 1.
        """
        total_combinations = 1
        for strip in reel_strips:
            total_combinations *= len(strip)
            
        # Quick estimation for common cases
        if min_count == 1:
            # Probability of NOT hitting the symbol on any reel
            prob_not_hitting = 1.0
            for strip in reel_strips:
                symbol_count = sum(1 for s in strip.symbols if s.name == symbol_name)
                prob_not_hitting_on_reel = 1 - (symbol_count / len(strip))
                prob_not_hitting *= prob_not_hitting_on_reel
                
            return 1 - prob_not_hitting
            
        elif min_count > len(reel_strips):
            # Impossible to get more symbols than reels
            return 0.0
            
        else:
            # For complex cases, use Monte Carlo simulation
            # (simplified - a full implementation would be more accurate)
            num_trials = 10000
            hits = 0
            
            for _ in range(num_trials):
                stops = generate_random_stops(reel_strips)
                window = get_visible_window(reel_strips, stops, 3)  # Assuming 3 visible rows
                
                # Count symbol occurrences in window
                symbol_count = sum(
                    1 for reel in window
                    for symbol in reel
                    if symbol.name == symbol_name
                )
                
                if symbol_count >= min_count:
                    hits += 1
                    
            return hits / num_trials
    
    @staticmethod
    def estimate_volatility(reel_strips: List[ReelStrip], 
                          paylines: Dict[str, Payline],
                          symbol_payouts: Dict[str, List[float]]) -> float:
        """
        Estimate the volatility (variance) of a slot game configuration.
        
        Args:
            reel_strips: List of ReelStrip objects.
            paylines: Dictionary of paylines.
            symbol_payouts: Dictionary mapping symbol names to payouts.
            
        Returns:
            Estimated volatility value.
        """
        # Simplified volatility estimation based on payout distribution
        # A full implementation would run simulations and calculate variance
        
        # Calculate symbol distribution on first reel as baseline
        first_reel_dist = {}
        if reel_strips:
            first_reel = reel_strips[0]
            for symbol in first_reel.symbols:
                first_reel_dist[symbol.name] = first_reel_dist.get(symbol.name, 0) + 1
            first_reel_dist = {s: count / len(first_reel) for s, count in first_reel_dist.items()}
        
        # Estimate volatility based on symbol payouts and distribution
        volatility = 0.0
        for symbol, payouts in symbol_payouts.items():
            if symbol in first_reel_dist and len(payouts) > 0:
                # Use max payout as proxy for volatility contribution
                max_payout = max(payouts) if payouts else 0
                symbol_freq = first_reel_dist.get(symbol, 0)
                
                # Rare symbols with high payouts contribute more to volatility
                volatility += max_payout * max_payout * (1 - symbol_freq)
        
        # Normalize the result
        return math.sqrt(volatility) / 100.0