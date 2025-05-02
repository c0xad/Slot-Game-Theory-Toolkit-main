# -*- coding: utf-8 -*-
"""
Defines classes and functions related to slot machine paytables.

Includes:
- Representation of winning combinations (symbols, counts, positions).
- Payout calculation based on winning combinations and bet.
- Handling of special symbols (wilds, scatters) in win evaluation.
"""

import logging
from typing import List, Dict, Tuple, Optional, Any, Union # Added Union

# Import Symbol from symbol.py and Payline type from reels.py
from .symbol import Symbol
from .reels import Payline

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for a winning combination definition
# Example: (Symbol('7'), 3) -> Payout multiplier for three '7' symbols
# Example: (Symbol('CHERRY'), 2, 'left_to_right') -> Payout for two cherries starting from left
WinCondition = Tuple[Symbol, int] # Basic: Symbol + Count needed
WinConditionDetailed = Tuple[Symbol, int, str] # Symbol, Count, Direction/Requirement (e.g., 'left_to_right', 'scatter')
# Made Optional[str] into str for detailed, assuming detailed always has a type string.
# Use Union for the dictionary key type hint.
WinConditionKey = Union[WinCondition, WinConditionDetailed]

PayoutValue = Union[float, int] # Payout multiplier or fixed amount

class Paytable:
    """
    Represents the paytable mapping win conditions to payouts.

    Stores winning combinations (specific symbols and their required counts,
    potentially with directionality like 'left_to_right' or type like 'scatter')
    and their corresponding payout values (typically multipliers).
    """

    def __init__(self, win_payouts: Dict[WinConditionKey, PayoutValue]):
        """
        Initializes the Paytable.

        Args:
            win_payouts: A dictionary where keys are tuples representing winning
                         conditions (e.g., (Symbol('7'), 3) or
                         (Symbol('SCATTER'), 3, 'scatter')) and values are the
                         payout multipliers (e.g., 100 for 100x bet).
        """
        if not isinstance(win_payouts, dict):
            raise TypeError("win_payouts must be a dictionary.")
        self.win_payouts: Dict[WinConditionKey, PayoutValue] = win_payouts
        self._validate_paytable()

    def _validate_paytable(self):
        """Performs basic validation on the paytable structure and values."""
        if not self.win_payouts:
            logger.warning("Paytable initialized with no win conditions.")
            return

        for condition, payout in self.win_payouts.items():
            if not isinstance(condition, tuple) or not (2 <= len(condition) <= 3):
                 raise TypeError(f"Invalid paytable condition format: {condition}. Must be tuple of length 2 or 3.")

            symbol = condition[0]
            count = condition[1]

            if not isinstance(symbol, Symbol):
                raise TypeError(f"Invalid symbol type in condition {condition}. Expected Symbol, got {type(symbol)}.")
            if not isinstance(count, int) or count <= 0:
                raise ValueError(f"Invalid count '{count}' in condition {condition}. Must be a positive integer.")
            if len(condition) == 3 and not isinstance(condition[2], str):
                 raise TypeError(f"Invalid type/direction in condition {condition}. Expected string, got {type(condition[2])}.")

            if not isinstance(payout, (int, float)) or payout < 0:
                # Allow zero payout? Yes, could signify a trigger without direct pay.
                raise ValueError(f"Invalid payout value {payout} for condition {condition}. Must be non-negative number.")
        logger.debug("Paytable validation successful.")


    def get_payout(self, symbol: Symbol, count: int, condition_type: Optional[str] = None) -> PayoutValue:
        """
        Gets the payout for a specific symbol count and optional condition type.

        Checks for detailed conditions (Symbol, count, type) first, then falls
        back to simple conditions (Symbol, count).

        Args:
            symbol: The symbol to check.
            count: The number of symbols in the combination.
            condition_type: Optional string specifying the type (e.g., 'left_to_right', 'scatter').

        Returns:
            The payout value (multiplier or amount) if the combination is winning,
            otherwise 0.
        """
        # Try matching detailed condition first if type is provided
        if condition_type:
            detailed_condition: WinConditionDetailed = (symbol, count, condition_type)
            if detailed_condition in self.win_payouts:
                return self.win_payouts[detailed_condition]

        # Try matching simple condition (symbol, count) as fallback or if no type specified
        simple_condition: WinCondition = (symbol, count)
        if simple_condition in self.win_payouts:
            return self.win_payouts[simple_condition]

        # Note: This implementation requires an exact match for the count.
        # It does not automatically award payouts for lower counts if a higher count matches.
        # Win evaluation logic (e.g., evaluate_payline_win) should handle finding the highest paying match.
        return 0 # No payout for this specific combination

    def __repr__(self) -> str:
        return f"Paytable(num_entries={len(self.win_payouts)})"


def evaluate_payline_win(
    payline: Payline,
    window: List[List[Symbol]],
    paytable: Paytable,
    bet_per_line: float = 1.0,
    direction: str = 'left_to_right' # Added direction parameter
) -> float:
    """
    Evaluates a single payline against the visible window and paytable for a given direction.

    Args:
        payline: The list of (reel_idx, row_idx) coordinates defining the line.
        window: The visible grid of symbols (List[List[Symbol]]).
        paytable: The Paytable instance.
        bet_per_line: The bet amount allocated to this specific line.
        direction: The direction to evaluate ('left_to_right', 'right_to_left', 'any_adjacent').

    Returns:
        The payout amount for this line for the specified direction.
    """
    line_symbols: List[Symbol] = []
    for reel_idx, row_idx in payline:
        # Check bounds carefully
        if 0 <= reel_idx < len(window) and 0 <= row_idx < len(window[reel_idx]):
            line_symbols.append(window[reel_idx][row_idx])
        else:
             logger.error(f"Payline coordinate ({reel_idx}, {row_idx}) is outside window bounds.")
             return 0.0 # Invalid line coordinate for this window

    if not line_symbols:
        return 0.0

    num_symbols_on_line = len(line_symbols)
    highest_payout_for_line = 0.0

    if direction == 'left_to_right':
        first_symbol = line_symbols[0]
        is_first_wild = getattr(first_symbol, 'is_wild', False)
        match_count = 0

        for i, current_symbol in enumerate(line_symbols):
            is_current_wild = getattr(current_symbol, 'is_wild', False)

            # Determine the symbol type we are currently matching
            # If the line starts with wilds, the first non-wild sets the type.
            symbol_type_to_match = first_symbol
            if is_first_wild:
                 # Find first non-wild on the line to determine match type
                 first_non_wild_on_line = next((s for s in line_symbols[:i+1] if not getattr(s, 'is_wild', False)), None)
                 if first_non_wild_on_line:
                     symbol_type_to_match = first_non_wild_on_line
                 # If line is all wilds so far, symbol_type_to_match remains WILD

            # Check if the current symbol matches the type or is wild
            if current_symbol == symbol_type_to_match or is_current_wild:
                match_count = i + 1
                is_currently_all_wilds = (match_count == sum(1 for s in line_symbols[:match_count] if getattr(s, 'is_wild', False)))
                payout_for_this_count = 0

                # 1. Check for specific WILD payout if the line is all wilds so far
                #    We use symbol_type_to_match here because if the line started wild, it holds the WILD symbol.
                if is_currently_all_wilds and getattr(symbol_type_to_match, 'is_wild', False):
                    payout_for_this_count = paytable.get_payout(symbol_type_to_match, match_count, direction)

                # 2. If not all wilds, or if no specific WILD payout was found, check for the matched non-wild symbol type
                #    Ensure we check the actual non-wild type if the line started wild.
                actual_symbol_type = symbol_type_to_match
                if getattr(actual_symbol_type, 'is_wild', False) and not is_currently_all_wilds:
                    # Find the first non-wild again to be sure
                     first_non_wild_on_line = next((s for s in line_symbols[:match_count] if not getattr(s, 'is_wild', False)), None)
                     if first_non_wild_on_line:
                         actual_symbol_type = first_non_wild_on_line

                # Only check non-wild payout if not already paid as all-wild, and if the type isn't wild
                if payout_for_this_count == 0 and not getattr(actual_symbol_type, 'is_wild', False):
                    payout_for_this_count = paytable.get_payout(actual_symbol_type, match_count, direction)

                # Update the highest payout found for this line
                highest_payout_for_line = max(highest_payout_for_line, payout_for_this_count)
            else:
                break # Sequence broken

    elif direction == 'right_to_left':
        # --- Implement Right-to-Left Logic ---
        last_symbol = line_symbols[-1] # Start matching from the last symbol
        is_last_wild = getattr(last_symbol, 'is_wild', False)
        match_count = 0

        # Iterate from right to left (index n-1 down to 0)
        for i in range(num_symbols_on_line - 1, -1, -1):
            current_symbol = line_symbols[i]
            is_current_wild = getattr(current_symbol, 'is_wild', False)

            # Determine the symbol type we are currently matching (from the right)
            symbol_type_to_match = last_symbol
            if is_last_wild:
                 # Find the last non-wild on the line (from current position i to end) to determine match type
                 last_non_wild_on_line = next((s for s in reversed(line_symbols[i:]) if not getattr(s, 'is_wild', False)), None)
                 if last_non_wild_on_line:
                     symbol_type_to_match = last_non_wild_on_line
                 # If line is all wilds from right so far, symbol_type_to_match remains WILD

            # Check if the current symbol matches the type or is wild
            if current_symbol == symbol_type_to_match or is_current_wild:
                # Calculate match count from the right end
                match_count = num_symbols_on_line - i
                is_currently_all_wilds = (match_count == sum(1 for s in line_symbols[i:] if getattr(s, 'is_wild', False)))
                payout_for_this_count = 0

                # 1. Check for specific WILD payout if the sequence is all wilds so far (from the right)
                if is_currently_all_wilds and getattr(symbol_type_to_match, 'is_wild', False):
                     payout_for_this_count = paytable.get_payout(symbol_type_to_match, match_count, direction)

                # 2. If not all wilds, or no WILD payout, check for the matched non-wild symbol type
                actual_symbol_type = symbol_type_to_match
                if getattr(actual_symbol_type, 'is_wild', False) and not is_currently_all_wilds:
                    # Find the last non-wild again to be sure
                    last_non_wild_on_line = next((s for s in reversed(line_symbols[i:]) if not getattr(s, 'is_wild', False)), None)
                    if last_non_wild_on_line:
                        actual_symbol_type = last_non_wild_on_line

                if payout_for_this_count == 0 and not getattr(actual_symbol_type, 'is_wild', False):
                    payout_for_this_count = paytable.get_payout(actual_symbol_type, match_count, direction)

                # Update the highest payout found for this line
                highest_payout_for_line = max(highest_payout_for_line, payout_for_this_count)
            else:
                break # Sequence broken (from the right)

    elif direction == 'any_adjacent':
        # --- Implement "Ways to Win" or Adjacent Symbol Logic ---
        # This logic typically replaces standard paylines. It involves finding
        # matching symbols (or wilds) on consecutive reels starting from the left,
        # irrespective of their vertical position.
        # Example: Find counts of Symbol 'A' or WILD on reels 0, 1, 2...
        # If counts are [c0, c1, c2, ...], a 3-of-a-kind win exists if c0>0, c1>0, c2>0.
        # The number of "ways" is c0 * c1 * c2.
        # The paytable lookup uses the symbol ('A') and the length of the sequence (3).
        # The final payout is ways * paytable_multiplier * bet_factor.
        # This requires a different evaluation approach than iterating payline coordinates.
        # Consider implementing a separate evaluate_ways_wins function.
        logger.warning("Any-adjacent/Ways payline evaluation logic is complex and not implemented in evaluate_payline_win.")
        pass # Placeholder - Full implementation needed, likely in a separate function.

    else:
        logger.warning(f"Unsupported payline evaluation direction: {direction}")
        return 0.0

    # Final payout is the highest multiplier found * bet per line
    return highest_payout_for_line * bet_per_line


def evaluate_scatter_wins(
    window: List[List[Symbol]],
    paytable: Paytable,
    total_bet: float = 1.0
) -> float:
    """
    Evaluates scatter wins based on the entire visible window.

    Args:
        window: The visible grid of symbols.
        paytable: The Paytable instance.
        total_bet: The total bet amount for the spin. Scatter wins often multiply total bet.

    Returns:
        The total payout amount for scatter wins.
    """
    highest_scatter_payout_multiplier = 0.0
    symbol_counts: Dict[Symbol, int] = {}

    # Count all symbols in the window
    for column in window:
        for symbol in column:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    # Check paytable for scatter conditions
    for symbol, count in symbol_counts.items():
        if getattr(symbol, 'is_scatter', False):
            # Check paytable using the specific 'scatter' condition type
            payout_multiplier = paytable.get_payout(symbol, count, 'scatter')
            # Typically only the highest paying scatter combination pays out
            highest_scatter_payout_multiplier = max(highest_scatter_payout_multiplier, payout_multiplier)

    # Scatter wins usually multiply the *total* bet for the spin
    return highest_scatter_payout_multiplier * total_bet

def evaluate_all_wins(
    window: List[List[Symbol]],
    paylines: Dict[str, Payline],
    paytable: Paytable,
    bet_per_line: float,
    total_bet: float,
    line_directions: List[str] = ['left_to_right'] # Allow specifying directions
) -> Tuple[float, Dict[str, float], float]:
    """
    Evaluates all payline wins (for specified directions) and scatter wins for a given spin result.

    Args:
        window: The visible grid of symbols.
        paylines: Dictionary mapping payline names to their coordinate lists.
        paytable: The Paytable instance.
        bet_per_line: Bet amount per line.
        total_bet: Total bet amount for the spin.
        line_directions: List of directions to evaluate for paylines (e.g., ['left_to_right']).

    Returns:
        A tuple containing:
        - Total payout amount for the spin (sum of highest line wins + scatter win).
        - Dictionary mapping payline names to their individual *highest* payout amounts found across evaluated directions.
        - Total scatter payout amount.
    """
    total_line_win = 0.0
    line_win_details: Dict[str, float] = {} # Stores the highest win found for each line name

    if bet_per_line <= 0 and len(paylines) > 0:
        logger.warning("bet_per_line is zero, line wins will be zero.")
    elif len(paylines) == 0 and bet_per_line > 0:
         logger.warning("bet_per_line > 0 but no paylines defined.")


    for name, line in paylines.items():
        highest_win_this_line = 0.0
        # Store wins per direction for potential logging or detailed results
        wins_by_direction: Dict[str, float] = {}

        for direction in line_directions:
            # Calculate win for this specific direction
            line_win_for_direction = evaluate_payline_win(line, window, paytable, bet_per_line, direction)
            wins_by_direction[direction] = line_win_for_direction
            # Keep track of the highest win found for this line across all evaluated directions
            highest_win_this_line = max(highest_win_this_line, line_win_for_direction)

        # Optional: Log the wins found for each direction for this line
        # if highest_win_this_line > 0 and len(line_directions) > 1:
        #     logger.debug(f"Line '{name}': Wins by direction: {wins_by_direction}. Paying max: {highest_win_this_line}")

        if highest_win_this_line > 0:
            line_win_details[name] = highest_win_this_line
            # Standard slot rule: Only the highest win per line is paid, even if it wins
            # in multiple directions (e.g., LTR and RTL). The max() above handles this.
            # The total line win is the sum of these highest-per-line wins across different lines.
            total_line_win += highest_win_this_line

    # Evaluate scatter wins separately
    total_scatter_win = evaluate_scatter_wins(window, paytable, total_bet)

    # Total win is the sum of wins on each payline plus scatter wins
    total_win = total_line_win + total_scatter_win
    return total_win, line_win_details, total_scatter_win