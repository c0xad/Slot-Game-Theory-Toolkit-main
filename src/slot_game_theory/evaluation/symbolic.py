# -*- coding: utf-8 -*-
"""
Symbolic evaluation of slot game metrics (RTP, Variance, Hit Frequency).

Suitable for games with a small, enumerable state space (e.g., classic 3-reel slots).
Calculates exact values by iterating through all possible outcomes.
"""

import itertools
from typing import List, Tuple, Dict, Any, Optional
from ..core.reels import ReelStrip, Symbol, Payline, get_visible_window # Adjust import path as needed
from ..core.paytable import Paytable, evaluate_all_wins # Adjust import path
import sympy

# Type alias for metrics result
MetricsResult = Dict[str, float] # e.g., {"rtp": 0.96, "variance": 25.5, "hit_frequency": 0.2}

def calculate_metrics_classic_symbolic(
    reel_strips: List[ReelStrip],
    paylines: Dict[str, Payline],
    paytable: Paytable,
    num_rows: int,
    bet_per_line: float = 1.0,
    total_bet: Optional[float] = None # If None, calculated from bet_per_line * num_lines
) -> MetricsResult:
    """
    Calculates RTP, Variance, and Hit Frequency for a classic slot by enumerating all outcomes.

    Args:
        reel_strips: List of ReelStrip objects.
        paylines: Dictionary of payline definitions.
        paytable: Paytable object.
        num_rows: Number of visible rows on the slot window.
        bet_per_line: The bet amount per payline.
        total_bet: The total bet per spin (used for scatter payouts). If None,
                   it's calculated as bet_per_line * len(paylines).

    Returns:
        A dictionary containing calculated 'rtp', 'variance', and 'hit_frequency'.
    """
    if total_bet is None:
        total_bet = bet_per_line * len(paylines)
        if total_bet == 0 and len(paylines) > 0:
             print("Warning: total_bet is zero but paylines exist. Scatter wins might be incorrect.")
        elif total_bet == 0:
             total_bet = 1.0 # Avoid division by zero if no lines and no total bet given

    if total_bet <= 0:
        raise ValueError("Total bet must be positive for RTP calculation.")

    num_reels = len(reel_strips)
    reel_lengths = [len(strip) for strip in reel_strips]
    total_combinations = list(itertools.product(*[range(length) for length in reel_lengths]))
    num_total_combinations = len(total_combinations)

    if num_total_combinations == 0:
        return {"rtp": 0.0, "variance": 0.0, "hit_frequency": 0.0}

    total_payout_sum = 0.0
    total_payout_sq_sum = 0.0
    winning_spins_count = 0

    print(f"Starting symbolic evaluation for {num_total_combinations} combinations...")

    for i, stops in enumerate(total_combinations):
        # Get the visible window for this combination of stops
        window = get_visible_window(reel_strips, list(stops), num_rows)

        # Evaluate wins for this window
        spin_payout, _, _ = evaluate_all_wins(
            window=window,
            paylines=paylines,
            paytable=paytable,
            bet_per_line=bet_per_line,
            total_bet=total_bet
        )

        # Accumulate sums for metrics calculation
        total_payout_sum += spin_payout
        total_payout_sq_sum += spin_payout ** 2
        if spin_payout > 0:
            winning_spins_count += 1

        # Optional: Progress indicator
        # if (i + 1) % (num_total_combinations // 10 or 1) == 0:
        #     print(f"  ... evaluated {(i + 1) / num_total_combinations * 100:.0f}%")


    # Calculate metrics
    # E[Payout] = Sum(Payout_i) / N
    expected_payout = total_payout_sum / num_total_combinations

    # RTP = E[Payout] / TotalBet
    rtp = expected_payout / total_bet

    # Var(Payout) = E[Payout^2] - (E[Payout])^2
    expected_payout_sq = total_payout_sq_sum / num_total_combinations
    variance = expected_payout_sq - (expected_payout ** 2)

    # Hit Frequency = Number of Winning Spins / Total Spins
    hit_frequency = winning_spins_count / num_total_combinations

    print("Symbolic evaluation complete.")

    return {
        "rtp": rtp,
        "variance": variance, # This is variance of the payout amount, not necessarily normalized by bet
        "hit_frequency": hit_frequency,
        "expected_payout": expected_payout,
        "total_combinations": num_total_combinations,
    }

# Potential extensions:
# - Handle more complex state transitions (if using Markov models alongside)
# - Use generating functions for variance calculation (more complex implementation)
# - Add calculation for other metrics (e.g., element contribution to RTP)

# TODO: Define more sophisticated structures as needed.

class SymbolicReel:
    """Represents a reel strip with symbols as symbolic variables."""
    def __init__(self, symbols: List[str], name: str):
        self.name = name
        # Create symbolic variables for each unique symbol
        self.symbol_vars = {s: sympy.Symbol(f'{s}_{name}') for s in set(symbols)}
        self.strip = [self.symbol_vars[s] for s in symbols]
        self.length = len(self.strip)
        # Placeholder for probabilities - needs refinement based on actual weighting
        self.probabilities = {var: sympy.Rational(1, self.length) for var in self.symbol_vars.values()}

    def __repr__(self):
        return f"SymbolicReel(name='{self.name}', length={self.length})"


class SymbolicPayline:
    """Represents a payline as a sequence of reel indices."""
    def __init__(self, indices: List[int]):
        if not indices:
            raise ValueError("Payline indices cannot be empty.")
        self.indices = indices
        self.length = len(indices)

    def __repr__(self):
        return f"SymbolicPayline(indices={self.indices})"

class SymbolicPaytable:
    """Represents a paytable mapping winning combinations to payouts."""
    def __init__(self, payouts: Dict[Tuple[sympy.Symbol, ...], sympy.Number]):
        # Key: Tuple of symbolic symbols representing the winning combination
        # Value: Payout amount (can be symbolic or numeric)
        self.payouts = payouts

    def get_payout(self, combination: Tuple[sympy.Symbol, ...]) -> sympy.Number:
        """Returns the payout for a given symbolic combination."""
        # Needs refinement for partial matches, wildcards, etc.
        return self.payouts.get(combination, sympy.Integer(0))

    def __repr__(self):
        # Basic representation, might need improvement for large paytables
        return f"SymbolicPaytable(entries={len(self.payouts)})"


def calculate_combination_probability(
    reels: List[SymbolicReel],
    payline: SymbolicPayline,
    combination: Tuple[sympy.Symbol, ...]
) -> sympy.Expr:
    """Calculates the symbolic probability of a specific combination on a payline."""
    if len(combination) != payline.length or payline.length > len(reels):
        raise ValueError("Combination length must match payline length and number of reels.")

    prob = sympy.Integer(1)
    for i, symbol_var in enumerate(combination):
        reel = reels[i]
        line_index = payline.indices[i]
        # Simple probability assumption: equal chance for each symbol on the reel.
        # This needs refinement for weighted reels.
        symbol_prob = reel.probabilities.get(symbol_var)
        if symbol_prob is None:
            # Symbol not present on this reel for this combination
            return sympy.Integer(0)
        prob *= symbol_prob

    return prob


def calculate_symbolic_rtp(
    reels: List[SymbolicReel],
    paylines: List[SymbolicPayline],
    paytable: SymbolicPaytable
) -> sympy.Expr:
    """Calculates the total symbolic RTP across all paylines and winning combinations."""
    total_expected_value = sympy.Integer(0)

    # Iterate through all defined winning combinations in the paytable
    for combination, payout in paytable.payouts.items():
        if payout == 0:
            continue

        combination_len = len(combination)
        if combination_len == 0:
            continue # Skip empty combinations

        # Check which paylines match the length of the combination
        applicable_paylines = [pl for pl in paylines if pl.length == combination_len]

        for payline in applicable_paylines:
             # Ensure we don't try to access reels beyond the defined ones for the payline
            if combination_len > len(reels):
                continue

            # Extract the specific reels involved in this payline
            relevant_reels = [reels[i] for i in range(combination_len)]

            # Calculate probability for this specific combination on this payline
            # Note: This assumes the combination starts from the first reel of the payline.
            # More complex logic needed for combinations not starting at reel 1.
            prob = calculate_combination_probability(relevant_reels, payline, combination)

            total_expected_value += prob * payout

    # Assuming a bet of 1 unit per spin covering all lines (common simplification)
    # The RTP is the total expected value.
    return total_expected_value

# Example Usage (Illustrative - requires concrete symbol definitions and paytable)
if __name__ == '__main__':
    # Define symbols
    A, K, Q, J, W = sympy.symbols('A K Q J W') # W is Wild

    # Define Reels (simplified example, assumes equal weighting)
    reel1 = SymbolicReel(['A', 'K', 'Q', 'J', 'A', 'K', 'W'], name='R1')
    reel2 = SymbolicReel(['A', 'K', 'Q', 'J', 'Q', 'J', 'W'], name='R2')
    reel3 = SymbolicReel(['A', 'K', 'Q', 'J', 'A', 'K', 'Q'], name='R3')

    reels = [reel1, reel2, reel3]

    # Define Paylines (e.g., middle row)
    payline1 = SymbolicPayline([1, 1, 1]) # Assumes 0-based indexing within visible window (needs clarification)
                                         # Or absolute indices on the strip? Let's assume absolute for now.
    # Correcting payline interpretation: indices refer to the reel index for that position
    # Let's define paylines based on the *position* in the window mapping to a *reel* index.
    # Assuming a 3x3 window and standard paylines:
    # Top Line: indices relative to the start of the *visible* part of the strip
    # Mid Line: indices relative to the start+1
    # Bot Line: indices relative to the start+2
    # This symbolic model needs a clearer definition of the "window" vs "strip".
    # For now, let's simplify and assume payline indices directly map to reel strip indices.
    payline_middle = SymbolicPayline([reel1.length // 2, reel2.length // 2, reel3.length // 2])


    # Define Paytable (Symbolic Keys)
    paytable_dict = {
        (reel1.symbol_vars['A'], reel2.symbol_vars['A'], reel3.symbol_vars['A']): sympy.Integer(50),
        (reel1.symbol_vars['K'], reel2.symbol_vars['K'], reel3.symbol_vars['K']): sympy.Integer(40),
        # Add wild combinations if necessary (requires more complex logic)
    }
    paytable = SymbolicPaytable(paytable_dict)

    # Calculate probability of AAA on the middle line (simplified)
    # This current probability calculation assumes the combination *is* the symbols at the payline indices.
    # It doesn't account for the random stopping position.
    # The symbolic approach needs refinement to handle the random stop aspect.
    # A more correct symbolic approach might involve:
    # 1. Defining the window size.
    # 2. Calculating the probability of each possible *window* appearing.
    # 3. Evaluating paylines against each window configuration.
    # This is significantly more complex.

    # Let's recalculate based on the probability of symbols appearing at *any* stop position.
    # This is closer but still not fully capturing payline interactions correctly without window concept.

    prob_AAA = calculate_combination_probability(reels, payline_middle, (reel1.symbol_vars['A'], reel2.symbol_vars['A'], reel3.symbol_vars['A']))
    print(f"Simplified Symbolic Probability of AAA on middle line: {prob_AAA}")

    # Calculate Symbolic RTP (Simplified)
    rtp = calculate_symbolic_rtp(reels, [payline_middle], paytable)
    print(f"Simplified Symbolic RTP: {rtp}")

    # Note: The current symbolic implementation is a simplified starting point.
    # A full symbolic calculation often involves matrix methods or generating functions
    # to handle all possible stop combinations and window formations correctly, which is complex.
    # It doesn't currently handle weighted reels or the concept of a viewing window correctly.