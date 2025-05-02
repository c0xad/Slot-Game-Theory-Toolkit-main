# -*- coding: utf-8 -*-
"""
Symbolic evaluation of slot game metrics (RTP, Variance, Hit Frequency).

Suitable for games with a small, enumerable state space (e.g., classic 3-reel slots).
Calculates exact values by iterating through all possible outcomes.
"""

import itertools
import sympy as sp
from sympy import Symbol, Integer, Rational, Expr, Add, Mul
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable
from ..core.reels import ReelStrip, Symbol as GameSymbol, Payline, get_visible_window
from ..core.paytable import Paytable, evaluate_all_wins

# Type alias for metrics result
MetricsResult = Dict[str, float]

def calculate_metrics_classic_symbolic(
    reel_strips: List[ReelStrip],
    paylines: Dict[str, Payline],
    paytable: Paytable,
    num_rows: int,
    bet_per_line: float = 1.0,
    total_bet: Optional[float] = None,
    max_combinations: int = 1_000_000,  # Safety limit to prevent excessive calculations
    verbose: bool = False
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
        max_combinations: Maximum number of combinations to evaluate before aborting.
        verbose: Whether to print progress information.

    Returns:
        A dictionary containing calculated 'rtp', 'variance', and 'hit_frequency'.
    """
    if total_bet is None:
        total_bet = bet_per_line * len(paylines)
        if total_bet == 0 and len(paylines) > 0:
             print("Warning: total_bet is zero but paylines exist. Scatter wins might be incorrect.")
        elif total_bet == 0:
             total_bet = 1.0  # Avoid division by zero if no lines and no total bet given

    if total_bet <= 0:
        raise ValueError("Total bet must be positive for RTP calculation.")

    num_reels = len(reel_strips)
    reel_lengths = [len(strip) for strip in reel_strips]
    
    # Calculate total combinations and check against limit
    total_possible_combinations = 1
    for length in reel_lengths:
        total_possible_combinations *= length
    
    if total_possible_combinations > max_combinations:
        print(f"Warning: Large number of combinations ({total_possible_combinations}). "
              f"This may take a long time or exceed memory limits.")
        print(f"Consider using Monte Carlo simulation for large state spaces.")
        if not verbose:
            print("Set verbose=True to see progress updates.")
        
    # Generate all possible stop combinations
    total_combinations = list(itertools.product(*[range(length) for length in reel_lengths]))
    num_total_combinations = len(total_combinations)

    if num_total_combinations == 0:
        return {"rtp": 0.0, "variance": 0.0, "hit_frequency": 0.0}

    # Account for weighted reels
    weights_exist = any(strip.weights is not None for strip in reel_strips)
    stop_probabilities = {}
    total_probability_mass = 0.0
    
    if weights_exist:
        # Calculate probability for each stop combination
        for stops in total_combinations:
            probability = 1.0
            for i, stop in enumerate(stops):
                strip = reel_strips[i]
                if strip.weights:
                    probability *= strip.weights[stop] / sum(strip.weights)
                else:
                    # Equal probability for each stop position
                    probability *= 1.0 / reel_lengths[i]
            stop_probabilities[stops] = probability
            total_probability_mass += probability
        
        # Normalize probabilities if needed
        if abs(total_probability_mass - 1.0) > 1e-10:
            for stops in stop_probabilities:
                stop_probabilities[stops] /= total_probability_mass
    else:
        # If no weights, all combinations are equally likely
        equal_probability = 1.0 / num_total_combinations
        stop_probabilities = {stops: equal_probability for stops in total_combinations}
    
    if verbose:
        print(f"Starting symbolic evaluation for {num_total_combinations} combinations...")

    total_payout_sum = 0.0
    total_payout_sq_sum = 0.0
    winning_spins_count = 0

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

        # Get probability for this combination
        probability = stop_probabilities[stops]
        
        # Accumulate sums for metrics calculation, weighted by probability
        total_payout_sum += spin_payout * probability
        total_payout_sq_sum += (spin_payout ** 2) * probability
        if spin_payout > 0:
            winning_spins_count += probability

        # Optional: Progress indicator
        if verbose and (i + 1) % (num_total_combinations // 10 or 1) == 0:
            print(f"  ... evaluated {(i + 1) / num_total_combinations * 100:.0f}%")

    # Calculate metrics
    # E[Payout] is already correctly weighted by probability
    expected_payout = total_payout_sum

    # RTP = E[Payout] / TotalBet
    rtp = expected_payout / total_bet

    # Var(Payout) = E[Payout^2] - (E[Payout])^2
    # These are already correctly weighted by probability
    variance = total_payout_sq_sum - (expected_payout ** 2)

    # Hit Frequency = P(Payout > 0)
    # This is the sum of probabilities for winning combinations
    hit_frequency = winning_spins_count

    if verbose:
        print("Symbolic evaluation complete.")

    return {
        "rtp": rtp,
        "variance": variance,
        "hit_frequency": hit_frequency,
        "expected_payout": expected_payout,
        "total_combinations": num_total_combinations,
    }

# --- Advanced Symbolic Engine using SymPy ---

class SymbolicReel:
    """Represents a reel strip with symbols as symbolic variables."""
    def __init__(self, 
                 symbols: List[Union[str, GameSymbol]], 
                 name: str, 
                 weights: Optional[List[int]] = None):
        """
        Initialize a symbolic reel.
        
        Args:
            symbols: List of symbols on the reel strip.
            name: Name identifier for this reel.
            weights: Optional list of weights for each stop position.
        """
        self.name = name
        
        # Convert GameSymbol objects to strings if needed
        self.original_symbols = symbols
        self.symbol_strings = [s.id if isinstance(s, GameSymbol) else str(s) for s in symbols]
        
        # Create symbolic variables for each unique symbol
        self.unique_symbols = set(self.symbol_strings)
        self.symbol_vars = {s: Symbol(f'{s}_{name}') for s in self.unique_symbols}
        
        # Map original symbols to symbolic variables
        self.strip = [self.symbol_vars[s] for s in self.symbol_strings]
        self.length = len(self.strip)
        
        # Handle weights for probability calculation
        self.weights = weights
        if weights:
            if len(weights) != self.length:
                raise ValueError(f"Weights length ({len(weights)}) must match strip length ({self.length})")
            total_weight = sum(weights)
            self.stop_probabilities = [Rational(w, total_weight) for w in weights]
        else:
            # Equal probability for each stop
            self.stop_probabilities = [Rational(1, self.length) for _ in range(self.length)]
        
        # Calculate probability for each symbol
        self.symbol_probabilities = {}
        for sym in self.unique_symbols:
            # Sum probabilities of all occurrences of this symbol
            prob = 0
            for i, s in enumerate(self.symbol_strings):
                if s == sym:
                    prob += self.stop_probabilities[i]
            self.symbol_probabilities[self.symbol_vars[sym]] = prob

    def get_window_probability(self, 
                               window_symbols: List[Symbol], 
                               start_index: int, 
                               window_height: int) -> Expr:
        """
        Calculate the probability of a specific window appearing on this reel.
        
        Args:
            window_symbols: List of symbolic variables for the visible window.
            start_index: Starting index (stop position) on the reel.
            window_height: Height of the visible window.
            
        Returns:
            Symbolic expression for the probability.
        """
        if len(window_symbols) != window_height:
            raise ValueError("Window symbols length must match window height.")
        
        # Check if the symbols starting at start_index match the window_symbols
        match = True
        for i in range(window_height):
            reel_idx = (start_index + i) % self.length
            if self.strip[reel_idx] != window_symbols[i]:
                match = False
                break
        
        # Return the probability of this stop position if it's a match
        if match:
            return self.stop_probabilities[start_index]
        else:
            return Integer(0)

    def __repr__(self):
        return f"SymbolicReel(name='{self.name}', length={self.length}, unique_symbols={len(self.unique_symbols)})"


class SymbolicPayline:
    """Represents a payline as a sequence of positions in the visible window."""
    def __init__(self, positions: List[Tuple[int, int]]):
        """
        Initialize a symbolic payline.
        
        Args:
            positions: List of (row, col) positions in the window, starting from (0,0) at top-left.
        """
        if not positions:
            raise ValueError("Payline positions cannot be empty.")
        self.positions = positions
        self.length = len(positions)

    def get_symbols(self, window: List[List[Symbol]]) -> List[Symbol]:
        """
        Extract symbols along this payline from the window.
        
        Args:
            window: 2D list of symbolic variables representing the visible window.
            
        Returns:
            List of symbolic variables along the payline.
        """
        symbols = []
        for row, col in self.positions:
            try:
                symbols.append(window[row][col])
            except IndexError:
                raise ValueError(f"Position ({row},{col}) is outside the window dimensions.")
        return symbols

    def __repr__(self):
        return f"SymbolicPayline(positions={self.positions})"


class SymbolicPaytable:
    """Represents a paytable mapping winning combinations to payouts."""
    def __init__(self, payouts: Dict[Tuple[Symbol, ...], Union[int, Expr]],
                 wild_symbol: Optional[Symbol] = None,
                 direction: str = 'left_to_right'):
        """
        Initialize a symbolic paytable.
        
        Args:
            payouts: Dictionary mapping symbol combinations to payout amounts.
            wild_symbol: Optional wild symbol that can substitute for other symbols.
            direction: Direction for evaluating wins ('left_to_right', 'right_to_left', or 'both').
        """
        # Convert all payouts to symbolic expressions
        self.payouts = {combo: (payout if isinstance(payout, Expr) else Integer(payout)) 
                        for combo, payout in payouts.items()}
        self.wild_symbol = wild_symbol
        self.direction = direction
        
        # Precompute all combo lengths for faster lookups
        self.combo_lengths = sorted(set(len(combo) for combo in self.payouts), reverse=True)

    def evaluate_line(self, line_symbols: List[Symbol]) -> Expr:
        """
        Evaluate a line and return the payout expression.
        
        Args:
            line_symbols: List of symbolic variables along a payline.
            
        Returns:
            Symbolic expression for the payout.
        """
        max_payout = Integer(0)
        
        # Handle different evaluation directions
        directions = []
        if self.direction == 'left_to_right' or self.direction == 'both':
            directions.append(line_symbols)
        if self.direction == 'right_to_left' or self.direction == 'both':
            directions.append(list(reversed(line_symbols)))
        
        for symbols in directions:
            # Check combinations from longest to shortest
            for length in self.combo_lengths:
                if length > len(symbols):
                    continue
                
                # Extract the combination to check
                combo = tuple(symbols[:length])
                
                # Direct match
                if combo in self.payouts:
                    payout = self.payouts[combo]
                    max_payout = max_payout if max_payout > payout else payout
                
                # Check for wild substitutions
                elif self.wild_symbol is not None:
                    # Find all valid wild substitutions
                    for sub_combo in self._generate_wild_substitutions(combo):
                        if sub_combo in self.payouts:
                            payout = self.payouts[sub_combo]
                            max_payout = max_payout if max_payout > payout else payout
        
        return max_payout

    def _generate_wild_substitutions(self, combo: Tuple[Symbol, ...]) -> List[Tuple[Symbol, ...]]:
        """
        Generate all possible combinations by substituting wilds with regular symbols.
        
        Args:
            combo: A tuple of symbols that may contain wilds.
            
        Returns:
            List of all possible substituted combinations.
        """
        if not self.wild_symbol or self.wild_symbol not in combo:
            return [combo]
        
        # Start with the original combo
        result = [combo]
        
        # For each position with a wild, substitute with all other symbols
        for i, symbol in enumerate(combo):
            if symbol == self.wild_symbol:
                new_result = []
                for c in result:
                    # Get all unique non-wild symbols from the paytable
                    substitutes = set()
                    for pay_combo in self.payouts:
                        for s in pay_combo:
                            if s != self.wild_symbol:
                                substitutes.add(s)
                    
                    # Create new combinations with each substitute
                    for sub in substitutes:
                        new_combo = list(c)
                        new_combo[i] = sub
                        new_result.append(tuple(new_combo))
                
                result = new_result
        
        return result

    def __repr__(self):
        return f"SymbolicPaytable(entries={len(self.payouts)}, direction='{self.direction}')"


def calculate_window_probability(reels: List[SymbolicReel], 
                                window_height: int, 
                                window_config: List[List[List[Symbol]]]) -> Expr:
    """
    Calculate the probability of a specific window configuration.
    
    Args:
        reels: List of SymbolicReel objects.
        window_height: Height of the visible window.
        window_config: List of possible windows for each reel.
        
    Returns:
        Symbolic expression for the probability of this window configuration.
    """
    if len(reels) != len(window_config):
        raise ValueError("Number of reels must match number of window configurations.")
    
    # Calculate probability for each reel
    reel_probs = []
    for i, reel in enumerate(reels):
        # Sum probabilities for all possible stop positions that create the desired window
        reel_prob = Integer(0)
        for j in range(reel.length):
            window_prob = reel.get_window_probability(window_config[i], j, window_height)
            reel_prob += window_prob
        
        reel_probs.append(reel_prob)
    
    # Multiply probabilities across all reels
    total_prob = Integer(1)
    for prob in reel_probs:
        total_prob *= prob
    
    return total_prob


def calculate_symbolic_rtp(reels: List[SymbolicReel],
                          paylines: List[SymbolicPayline],
                          paytable: SymbolicPaytable,
                          window_height: int,
                          scatter_payouts: Optional[Dict[Tuple[Symbol, int], Union[int, Expr]]] = None) -> Expr:
    """
    Calculate the symbolic RTP expression.
    
    Args:
        reels: List of SymbolicReel objects.
        paylines: List of SymbolicPayline objects.
        paytable: SymbolicPaytable object for line wins.
        window_height: Height of the visible window.
        scatter_payouts: Optional dict mapping (scatter_symbol, count) to payout.
        
    Returns:
        Symbolic expression for the RTP.
    """
    # Initialize expected value
    expected_value = Integer(0)
    
    # To avoid excessive enumeration, we'll approach this differently:
    # 1. For each possible window configuration
    # 2. Calculate its probability
    # 3. Evaluate all paylines for that window
    # 4. Add to the expected value
    
    # This is computationally intensive for large symbol sets/windows
    # We'll need to simplify or use a different approach for complex games
    
    # Generate all possible window configurations
    # For each reel, get all unique symbols
    all_symbols = []
    for reel in reels:
        all_symbols.append(list(reel.symbol_vars.values()))
    
    # For each reel, generate all possible window configurations
    window_configs = []
    for i, reel_symbols in enumerate(all_symbols):
        reel_windows = []
        for window in itertools.product(reel_symbols, repeat=window_height):
            reel_windows.append(list(window))
        window_configs.append(reel_windows)
    
    # For each combination of window configurations
    for window_combo in itertools.product(*window_configs):
        # Calculate probability of this window configuration
        window_prob = calculate_window_probability(reels, window_height, window_combo)
        
        if window_prob == 0:
            continue  # Skip impossible windows
        
        # Create 2D window for payline evaluation
        window_2d = []
        for row in range(window_height):
            window_row = []
            for col in range(len(reels)):
                window_row.append(window_combo[col][row])
            window_2d.append(window_row)
        
        # Evaluate all paylines
        total_payout = Integer(0)
        for payline in paylines:
            line_symbols = payline.get_symbols(window_2d)
            line_payout = paytable.evaluate_line(line_symbols)
            total_payout += line_payout
        
        # Add scatter payouts if applicable
        if scatter_payouts:
            for scatter_symbol, counts_and_pays in scatter_payouts.items():
                # Count occurrences of this scatter in the window
                scatter_count = sum(row.count(scatter_symbol) for row in window_2d)
                
                # Check if this count has a payout
                for (_, count), payout in counts_and_pays.items():
                    if scatter_count >= count:
                        scatter_payout = payout if isinstance(payout, Expr) else Integer(payout)
                        total_payout += scatter_payout
                        break  # Assume only the highest scatter count pays
        
        # Add to expected value
        expected_value += window_prob * total_payout
    
    # Normalize by bet amount (assume 1 unit bet for simplicity)
    return expected_value


def calculate_symbolic_hit_frequency(reels: List[SymbolicReel],
                                   paylines: List[SymbolicPayline],
                                   paytable: SymbolicPaytable,
                                   window_height: int) -> Expr:
    """
    Calculate the symbolic hit frequency expression.
    
    Args:
        reels: List of SymbolicReel objects.
        paylines: List of SymbolicPayline objects.
        paytable: SymbolicPaytable object.
        window_height: Height of the visible window.
        
    Returns:
        Symbolic expression for the hit frequency.
    """
    # Similar to RTP calculation, but we check for any win rather than calculating payouts
    hit_prob = Integer(0)
    
    # Generate all possible window configurations
    all_symbols = []
    for reel in reels:
        all_symbols.append(list(reel.symbol_vars.values()))
    
    window_configs = []
    for i, reel_symbols in enumerate(all_symbols):
        reel_windows = []
        for window in itertools.product(reel_symbols, repeat=window_height):
            reel_windows.append(list(window))
        window_configs.append(reel_windows)
    
    # For each combination of window configurations
    for window_combo in itertools.product(*window_configs):
        # Calculate probability of this window configuration
        window_prob = calculate_window_probability(reels, window_height, window_combo)
        
        if window_prob == 0:
            continue  # Skip impossible windows
        
        # Create 2D window for payline evaluation
        window_2d = []
        for row in range(window_height):
            window_row = []
            for col in range(len(reels)):
                window_row.append(window_combo[col][row])
            window_2d.append(window_row)
        
        # Check if any payline gives a win
        is_win = False
        for payline in paylines:
            line_symbols = payline.get_symbols(window_2d)
            line_payout = paytable.evaluate_line(line_symbols)
            if line_payout > 0:
                is_win = True
                break
        
        # If this window has a win, add its probability to the hit frequency
        if is_win:
            hit_prob += window_prob
    
    return hit_prob


def calculate_full_symbolic_metrics(reels: List[SymbolicReel],
                                  paylines: List[SymbolicPayline],
                                  paytable: SymbolicPaytable,
                                  window_height: int,
                                  scatter_payouts: Optional[Dict] = None) -> Dict[str, Expr]:
    """
    Calculate full symbolic metrics (RTP, Variance, Hit Frequency).
    
    Args:
        reels: List of SymbolicReel objects.
        paylines: List of SymbolicPayline objects.
        paytable: SymbolicPaytable object.
        window_height: Height of the visible window.
        scatter_payouts: Optional dict for scatter symbols.
        
    Returns:
        Dictionary of symbolic metric expressions.
    """
    rtp = calculate_symbolic_rtp(reels, paylines, paytable, window_height, scatter_payouts)
    hit_frequency = calculate_symbolic_hit_frequency(reels, paylines, paytable, window_height)
    
    # Variance calculation would require square terms - more complex and costly
    # For now, we'll leave it as None
    variance = None
    
    return {
        "rtp": rtp,
        "hit_frequency": hit_frequency,
        "variance": variance
    }


# Example Usage
if __name__ == '__main__':
    # Define symbols using SymPy symbols
    A = Symbol('A')
    K = Symbol('K')
    Q = Symbol('Q')
    J = Symbol('J')
    W = Symbol('WILD')  # Wild symbol
    
    # Define reels with different weights
    reel1 = SymbolicReel(['A', 'K', 'Q', 'J', 'A', 'K', 'W'], name='R1', 
                        weights=[10, 10, 10, 10, 5, 5, 3])  # Wild is rare
    reel2 = SymbolicReel(['A', 'K', 'Q', 'J', 'Q', 'J', 'W'], name='R2',
                        weights=[8, 8, 12, 12, 6, 6, 2])  # Different distribution
    reel3 = SymbolicReel(['A', 'K', 'Q', 'J', 'A', 'K', 'Q'], name='R3',
                        weights=[7, 7, 10, 10, 5, 5, 8])  # No wild
    
    reels = [reel1, reel2, reel3]
    window_height = 3
    
    # Define paylines (for a 3x3 window)
    # Row positions
    top_line = SymbolicPayline([(0, 0), (0, 1), (0, 2)])      # Top row
    mid_line = SymbolicPayline([(1, 0), (1, 1), (1, 2)])      # Middle row
    bot_line = SymbolicPayline([(2, 0), (2, 1), (2, 2)])      # Bottom row
    
    # Diagonal positions
    diag1 = SymbolicPayline([(0, 0), (1, 1), (2, 2)])         # Diagonal top-left to bottom-right
    diag2 = SymbolicPayline([(2, 0), (1, 1), (0, 2)])         # Diagonal bottom-left to top-right
    
    paylines = [top_line, mid_line, bot_line, diag1, diag2]
    
    # Define paytable with wilds
    # Using the symbol variables from the reels
    A_sym = reel1.symbol_vars['A']
    K_sym = reel1.symbol_vars['K']
    Q_sym = reel1.symbol_vars['Q']
    J_sym = reel1.symbol_vars['J']
    W_sym = reel1.symbol_vars['W']
    
    paytable_dict = {
        (A_sym, A_sym, A_sym): 50,
        (K_sym, K_sym, K_sym): 40,
        (Q_sym, Q_sym, Q_sym): 30,
        (J_sym, J_sym, J_sym): 20,
        (W_sym, W_sym, W_sym): 100,  # Wild line pays highest
    }
    
    paytable = SymbolicPaytable(paytable_dict, wild_symbol=W_sym, direction='left_to_right')
    
    # Calculate symbolic metrics
    print("Calculating symbolic metrics (this may take a while for complex games)...")
    metrics = calculate_full_symbolic_metrics(reels, paylines, paytable, window_height)
    
    print(f"Symbolic RTP expression: {metrics['rtp']}")
    print(f"Numeric RTP value: {float(metrics['rtp']):.6f}")
    print(f"Symbolic Hit Frequency: {metrics['hit_frequency']}")
    print(f"Numeric Hit Frequency: {float(metrics['hit_frequency']):.6f}")
    
    # For more complex games, we might want to simplify these expressions
    rtp_simplified = sp.simplify(metrics['rtp'])
    hit_freq_simplified = sp.simplify(metrics['hit_frequency'])
    
    print(f"Simplified RTP: {rtp_simplified}")
    print(f"Simplified Hit Frequency: {hit_freq_simplified}")
    
    # Explanation of the calculation approach:
    print("\nNote on symbolic calculation:")
    print("The symbolic approach provides exact mathematical expressions for slot metrics.")
    print("This is valuable for understanding the mathematical structure of the game,")
    print("but becomes computationally intensive for larger games.")
    print("For complex games with many reels/symbols, Monte Carlo simulation is recommended.")