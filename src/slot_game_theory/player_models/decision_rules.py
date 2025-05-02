# -*- coding: utf-8 -*-
"""
Models player decision rules, such as when to stop playing or how much to bet.

These rules form the basis of the player's strategy in response to a game design.
They can range from simple heuristics (stop-loss) to complex dynamic programming solutions.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import math
from datetime import datetime, timedelta

# Assume access to game evaluation results and potentially prospect theory values
# from ..evaluation.simulation import SimulationResult
# from .prospect_theory import calculate_prospect_value

# Type alias for game metrics used by decision rules
GameMetrics = Dict[str, Any] # e.g., {"rtp": 0.96, "variance": 25.5, "hit_frequency": 0.2}

# Type alias for player state
PlayerState = Dict[str, Any] # e.g., {"current_balance": 100, "spins_played": 50, "initial_balance": 100}

# Type alias for the output of a player behavior model
PlayerBehaviorOutput = Dict[str, Any] # e.g., {"expected_total_spins": 1000, "average_bet": 1.0, "churn_probability": 0.75}


class StopRule:
    """Base class for player stopping rules."""
    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        """Determines if the player should stop playing based on the current state."""
        raise NotImplementedError

    def __call__(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        return self.should_stop(player_state, game_metrics)


class SimpleStopLossRule(StopRule):
    """Player stops if balance drops below a certain percentage of initial balance."""
    def __init__(self, loss_percentage: float = 0.5):
        if not 0 < loss_percentage <= 1:
            raise ValueError("loss_percentage must be between 0 (exclusive) and 1 (inclusive)")
        self.loss_percentage = loss_percentage
        self.stop_threshold = None

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        initial_balance = player_state.get("initial_balance")
        current_balance = player_state.get("current_balance")

        if initial_balance is None or current_balance is None:
            print("Warning: Balance information missing for StopLossRule.")
            return False # Cannot evaluate rule

        if self.stop_threshold is None:
             # Calculate threshold only once based on initial balance
             self.stop_threshold = initial_balance * (1 - self.loss_percentage)

        return current_balance <= self.stop_threshold


class SpinLimitRule(StopRule):
    """Player stops after a fixed number of spins."""
    def __init__(self, max_spins: int = 1000):
        if max_spins <= 0:
            raise ValueError("max_spins must be positive")
        self.max_spins = max_spins

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        spins_played = player_state.get("spins_played", 0)
        return spins_played >= self.max_spins


class WinGoalRule(StopRule):
    """Player stops if balance reaches or exceeds a certain multiple of initial balance."""
    def __init__(self, win_multiplier: float = 2.0):
        if win_multiplier <= 1.0:
            raise ValueError("win_multiplier must be greater than 1.0")
        self.win_multiplier = win_multiplier
        self.goal_threshold = None

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        initial_balance = player_state.get("initial_balance")
        current_balance = player_state.get("current_balance")

        if initial_balance is None or current_balance is None:
            print("Warning: Balance information missing for WinGoalRule.")
            return False # Cannot evaluate rule

        if self.goal_threshold is None:
             # Calculate threshold only once based on initial balance
             self.goal_threshold = initial_balance * self.win_multiplier

        return current_balance >= self.goal_threshold


class TimeLimitRule(StopRule):
    """Player stops after a set amount of time has elapsed."""
    def __init__(self, time_limit_minutes: float = 30.0):
        if time_limit_minutes <= 0:
            raise ValueError("time_limit_minutes must be positive")
        self.time_limit_minutes = time_limit_minutes
        self.time_limit = timedelta(minutes=time_limit_minutes)
        self.start_time = None

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        current_time = player_state.get("current_time")
        
        if current_time is None:
            print("Warning: Time information missing for TimeLimitRule.")
            return False # Cannot evaluate rule
            
        if self.start_time is None:
            self.start_time = player_state.get("session_start_time", current_time)
            
        elapsed_time = current_time - self.start_time
        return elapsed_time >= self.time_limit


class LossStreakRule(StopRule):
    """Player stops after experiencing a certain number of consecutive losses."""
    def __init__(self, max_consecutive_losses: int = 5):
        if max_consecutive_losses <= 0:
            raise ValueError("max_consecutive_losses must be positive")
        self.max_consecutive_losses = max_consecutive_losses

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        current_loss_streak = player_state.get("current_loss_streak", 0)
        return current_loss_streak >= self.max_consecutive_losses


class CombinedRule(StopRule):
    """Player stops if *any* of the contained rules trigger."""
    def __init__(self, rules: List[StopRule]):
        if not rules:
            raise ValueError("CombinedRule requires at least one rule.")
        self.rules = rules

    def should_stop(self, player_state: PlayerState, game_metrics: GameMetrics) -> bool:
        # Reset stateful thresholds in sub-rules if necessary (e.g., if initial_balance changes mid-session, though unlikely here)
        # For SimpleStopLossRule and WinGoalRule, threshold depends on initial_balance,
        # which is assumed constant for the session simulated here. If state could reset,
        # we might need logic here to reset sub-rule internal state.
        return any(rule(player_state, game_metrics) for rule in self.rules)


class BettingStrategy:
    """Base class for player betting strategies."""
    def choose_bet(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        """Determines the bet amount for the next spin."""
        raise NotImplementedError

    def __call__(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        return self.choose_bet(player_state, game_metrics)


class FixedBetStrategy(BettingStrategy):
    """Player always bets the same amount."""
    def __init__(self, bet_amount: float = 1.0):
        if bet_amount <= 0:
            raise ValueError("bet_amount must be positive")
        self.bet_amount = bet_amount

    def choose_bet(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        current_balance = player_state.get("current_balance", 0)
        return min(self.bet_amount, current_balance)


class ProportionalBetting(BettingStrategy):
    """Player bets a fixed percentage of their current balance."""
    def __init__(self, bet_percentage: float = 0.01):
        if not 0 < bet_percentage <= 1:
            raise ValueError("bet_percentage must be between 0 (exclusive) and 1 (inclusive)")
        self.bet_percentage = bet_percentage

    def choose_bet(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        current_balance = player_state.get("current_balance")
        if current_balance is None or current_balance <= 0:
            return 0 # Cannot bet if balance is unknown or zero/negative
        bet = current_balance * self.bet_percentage
        # Ensure a minimum bet while respecting available balance
        return max(min(0.01, current_balance), bet)


class KellyCriterionBetting(BettingStrategy):
    """
    Implements the Kelly Criterion for optimal bet sizing.
    
    The Kelly Criterion calculates the optimal bet size based on the probability of winning
    and the payout odds. It maximizes logarithmic wealth growth over time.
    
    For a slot game, this is an approximation since the win probability and odds are complex.
    """
    def __init__(self, conservative_factor: float = 0.5, min_bet: float = 0.01, max_bet_percentage: float = 0.1):
        """
        Args:
            conservative_factor: Factor to scale down the Kelly bet (typically 0.5 for "Half Kelly").
                                 A value of 1.0 is the full Kelly bet.
            min_bet: Minimum bet amount
            max_bet_percentage: Maximum bet as percentage of current balance
        """
        if not 0 < conservative_factor <= 1:
            raise ValueError("conservative_factor must be between 0 (exclusive) and 1 (inclusive)")
        if min_bet <= 0:
            raise ValueError("min_bet must be positive")
        if not 0 < max_bet_percentage <= 1:
            raise ValueError("max_bet_percentage must be between 0 (exclusive) and 1 (inclusive)")
            
        self.conservative_factor = conservative_factor
        self.min_bet = min_bet
        self.max_bet_percentage = max_bet_percentage
    
    def choose_bet(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        current_balance = player_state.get("current_balance", 0)
        if current_balance <= 0:
            return 0  # Cannot bet with zero or negative balance
        
        # Extract the required metrics
        rtp = game_metrics.get("rtp", 0.95)  # Return to player
        hit_frequency = game_metrics.get("hit_frequency", 0.3)  # Probability of any win
        
        # Edge case handling
        if hit_frequency <= 0 or hit_frequency >= 1:
            # Default to a safe minimum bet if hit frequency is invalid
            return min(self.min_bet, current_balance)
        
        # Calculate average payout odds when winning
        # If RTP = 0.95 and hit_freq = 0.3, then avg win multiplier = 0.95/0.3 = 3.17
        avg_win_multiplier = rtp / hit_frequency
        
        # Kelly formula: f* = (p*(b+1) - 1) / b
        # Where: f* is the fraction of bankroll to bet
        #        p is probability of winning
        #        b is the net odds received on the wager (payout-to-1)
        
        # For slots: 
        # - p = hit_frequency
        # - b = avg_win_multiplier - 1 (subtract 1 because Kelly expects net odds)
        b = avg_win_multiplier - 1
        
        # Calculate the Kelly fraction
        if b <= 0:
            # Game has negative expected value, bet minimum
            kelly_fraction = 0
        else:
            kelly_fraction = (hit_frequency * (b + 1) - 1) / b
            
        # Apply conservative factor and limits
        adjusted_fraction = max(0, min(self.max_bet_percentage, kelly_fraction * self.conservative_factor))
        
        # Calculate bet amount
        bet_amount = current_balance * adjusted_fraction
        
        # Ensure bet is at least min_bet but not more than current balance
        return max(min(bet_amount, current_balance), min(self.min_bet, current_balance))


class AdaptiveBetting(BettingStrategy):
    """
    Adaptively changes bet size based on recent results (win/loss streaks).
    Increases bets after wins, decreases after losses.
    """
    def __init__(
        self, 
        initial_bet: float = 1.0,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.7,
        max_increase_multiplier: float = 4.0,
        max_decrease_multiplier: float = 0.5,
        reset_threshold: int = 3  # Reset to initial bet after this many consecutive opposite results
    ):
        """
        Args:
            initial_bet: Starting bet amount
            increase_factor: Multiply bet by this factor after a win
            decrease_factor: Multiply bet by this factor after a loss
            max_increase_multiplier: Maximum bet can be initial_bet * max_increase_multiplier
            max_decrease_multiplier: Minimum bet can be initial_bet * max_decrease_multiplier
            reset_threshold: Reset to initial bet after this many consecutive opposite results
        """
        if initial_bet <= 0:
            raise ValueError("initial_bet must be positive")
        if increase_factor <= 1:
            raise ValueError("increase_factor must be greater than 1")
        if not 0 < decrease_factor < 1:
            raise ValueError("decrease_factor must be between 0 (exclusive) and 1 (exclusive)")
        if max_increase_multiplier < 1:
            raise ValueError("max_increase_multiplier must be at least 1")
        if not 0 < max_decrease_multiplier <= 1:
            raise ValueError("max_decrease_multiplier must be between 0 (exclusive) and 1 (inclusive)")
        if reset_threshold <= 0:
            raise ValueError("reset_threshold must be positive")
            
        self.initial_bet = initial_bet
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.max_bet = initial_bet * max_increase_multiplier
        self.min_bet = initial_bet * max_decrease_multiplier
        self.reset_threshold = reset_threshold
        
        # State tracking - initialize in choose_bet if not already set in player_state
        self.current_bet = initial_bet
        self.consecutive_wins = 0
        self.consecutive_losses = 0
    
    def choose_bet(self, player_state: PlayerState, game_metrics: GameMetrics) -> float:
        # Get the current balance
        current_balance = player_state.get("current_balance", 0)
        if current_balance <= 0:
            return 0  # Cannot bet with zero or negative balance
            
        # Initialize state from player_state or use defaults if this is first call
        last_spin_result = player_state.get("last_spin_result", None)
        self.current_bet = player_state.get("current_bet", self.initial_bet)
        self.consecutive_wins = player_state.get("consecutive_wins", 0)
        self.consecutive_losses = player_state.get("consecutive_losses", 0)
        
        # Adjust bet based on win/loss streak
        if last_spin_result is not None:
            # If last spin was a win
            if last_spin_result["net_win"] > 0:
                # Increase bet after a win
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                
                if self.consecutive_wins >= self.reset_threshold:
                    # Reset after too many consecutive wins
                    self.current_bet = self.initial_bet
                    self.consecutive_wins = 0
                else:
                    # Increase bet within limits
                    self.current_bet = min(self.max_bet, self.current_bet * self.increase_factor)
            else:
                # Decrease bet after a loss
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                
                if self.consecutive_losses >= self.reset_threshold:
                    # Reset after too many consecutive losses
                    self.current_bet = self.initial_bet
                    self.consecutive_losses = 0
                else:
                    # Decrease bet within limits
                    self.current_bet = max(self.min_bet, self.current_bet * self.decrease_factor)
        
        # Update the player state with our internal tracking
        player_state["current_bet"] = self.current_bet
        player_state["consecutive_wins"] = self.consecutive_wins
        player_state["consecutive_losses"] = self.consecutive_losses
        
        # Return the bet, limited by available balance
        return min(self.current_bet, current_balance)


# --- Simulating Player Session ---

def simulate_player_session(
    game_metrics: GameMetrics,
    initial_state: PlayerState,
    stop_rule: StopRule,
    betting_strategy: BettingStrategy,
    # Need a way to simulate game outcomes based on metrics (RTP, variance)
    outcome_simulator: Callable[[float], Tuple[float, Dict[str, Any]]] # Function: bet -> (payout, details)
) -> PlayerState:
    """
    Simulates a single player session until a stop rule is met.

    Args:
        game_metrics: Characteristics of the slot game.
        initial_state: Starting state of the player (balance, etc.).
        stop_rule: The rule determining when the player stops.
        betting_strategy: The rule determining the player's bet size.
        outcome_simulator: A function that takes a bet amount and returns a
                           stochastically generated payout and additional details.

    Returns:
        The final state of the player after the session ends.
    """
    player_state = initial_state.copy()
    
    # Initialize state variables if not present
    player_state["spins_played"] = player_state.get("spins_played", 0)
    player_state["session_start_time"] = player_state.get("session_start_time", datetime.now())
    player_state["current_time"] = player_state.get("current_time", player_state["session_start_time"])
    player_state["total_wagered"] = player_state.get("total_wagered", 0.0)
    player_state["total_won"] = player_state.get("total_won", 0.0)
    player_state["total_lost"] = player_state.get("total_lost", 0.0)
    player_state["consecutive_wins"] = player_state.get("consecutive_wins", 0)
    player_state["consecutive_losses"] = player_state.get("consecutive_losses", 0)
    player_state["current_loss_streak"] = player_state.get("current_loss_streak", 0)
    player_state["current_win_streak"] = player_state.get("current_win_streak", 0)
    player_state["max_win_streak"] = player_state.get("max_win_streak", 0)
    player_state["max_loss_streak"] = player_state.get("max_loss_streak", 0)
    player_state["largest_win"] = player_state.get("largest_win", 0.0)
    player_state["win_count"] = player_state.get("win_count", 0)
    player_state["loss_count"] = player_state.get("loss_count", 0)
    player_state["spin_history"] = player_state.get("spin_history", [])

    while not stop_rule(player_state, game_metrics):
        # 1. Choose Bet
        bet_amount = betting_strategy(player_state, game_metrics)
        if bet_amount <= 0 or bet_amount > player_state.get("current_balance", 0):
            # Cannot afford bet or invalid bet amount
            break
            
        # 2. Simulate Spin Outcome
        payout, outcome_details = outcome_simulator(bet_amount)
        
        # 3. Update Time
        # Assume each spin takes between 3-5 seconds (randomize for realism)
        time_per_spin = 3 + 2 * np.random.random()  # seconds
        player_state["current_time"] += timedelta(seconds=time_per_spin)
        
        # 4. Calculate win/loss
        net_win = payout - bet_amount
        is_win = net_win > 0
        
        # 5. Update State with detailed information
        player_state["current_balance"] = player_state.get("current_balance", 0) - bet_amount + payout
        player_state["spins_played"] += 1
        player_state["total_wagered"] += bet_amount
        
        if is_win:
            player_state["total_won"] += payout
            player_state["win_count"] += 1
            player_state["current_win_streak"] += 1
            player_state["current_loss_streak"] = 0
            player_state["max_win_streak"] = max(player_state["max_win_streak"], player_state["current_win_streak"])
            player_state["largest_win"] = max(player_state["largest_win"], payout)
        else:
            player_state["total_lost"] += bet_amount
            player_state["loss_count"] += 1
            player_state["current_loss_streak"] += 1
            player_state["current_win_streak"] = 0
            player_state["max_loss_streak"] = max(player_state["max_loss_streak"], player_state["current_loss_streak"])
            
        # Record details about this spin
        spin_record = {
            "spin_number": player_state["spins_played"],
            "bet_amount": bet_amount,
            "payout": payout,
            "net_win": net_win,
            "is_win": is_win,
            "balance_after": player_state["current_balance"],
            "timestamp": player_state["current_time"],
            "outcome_details": outcome_details
        }
        player_state["spin_history"].append(spin_record)
        player_state["last_spin_result"] = spin_record
        
        # Safety break for infinite loops (should be handled by stop rule ideally)
        if player_state["spins_played"] > 1_000_000:
            print("Warning: Spin limit reached in player session simulation.")
            break

    # Calculate session statistics before returning
    session_duration = player_state["current_time"] - player_state["session_start_time"]
    player_state["session_duration_seconds"] = session_duration.total_seconds()
    player_state["actual_rtp"] = player_state["total_won"] / player_state["total_wagered"] if player_state["total_wagered"] > 0 else 0
    player_state["net_result"] = player_state["current_balance"] - player_state.get("initial_balance", 0)
    
    return player_state


# --- Advanced Outcome Simulators --- 

def multi_tier_outcome_simulator(game_metrics: GameMetrics) -> Callable[[float], Tuple[float, Dict[str, Any]]]:
    """
    Creates a realistic outcome simulator based on a multi-tier payout structure.
    
    This simulator creates a discrete distribution with multiple win tiers:
    - Base game small wins
    - Base game medium wins
    - Base game large wins
    - Bonus/feature game wins (if specified)
    
    The distribution is constructed to match the specified RTP, hit frequency, and variance.
    
    Args:
        game_metrics: Must contain 'rtp', 'hit_frequency', 'variance', and optionally
                    'bonus_frequency', 'bonus_contribution', 'max_win_multiplier'
    Returns:
        Function that takes a bet amount and returns a tuple of (payout, details)
    """
    # Extract required metrics
    rtp = game_metrics.get("rtp")
    hit_freq = game_metrics.get("hit_frequency")
    variance = game_metrics.get("variance")
    
    # Optional parameters with defaults
    bonus_freq = game_metrics.get("bonus_frequency", 0.001)  # Default 1 in 1000 spins
    bonus_contribution = game_metrics.get("bonus_contribution", 0.3)  # 30% of RTP comes from bonus
    base_contribution = 1.0 - bonus_contribution
    max_win_multiplier = game_metrics.get("max_win_multiplier", 500.0)  # Maximum win as multiplier of bet
    
    # Validate inputs
    if rtp is None or hit_freq is None or variance is None:
        raise ValueError("multi_tier_outcome_simulator requires 'rtp', 'hit_frequency', and 'variance' in game_metrics")
    if not 0 < hit_freq <= 1:
        raise ValueError("hit_frequency must be between 0 (exclusive) and 1 (inclusive)")
    if rtp < 0:
        raise ValueError("RTP cannot be negative")
    if variance < 0:
        raise ValueError("Variance cannot be negative")
    if not 0 <= bonus_freq <= hit_freq:
        raise ValueError("bonus_frequency must be between 0 and hit_frequency")
    if not 0 <= bonus_contribution <= 1:
        raise ValueError("bonus_contribution must be between 0 and 1")
    
    # Calculate base game parameters (excluding bonus)
    base_game_rtp = rtp * base_contribution
    base_game_hit_freq = hit_freq - bonus_freq
    
    # Payout tiers for base game as multipliers of bet amount
    # Distribution is designed to roughly match real slot games
    # We'll use 4 tiers for base game wins
    base_win_tiers = {
        "small": {"min": 0.1, "max": 2.0, "frequency": 0.80},  # Most common, small wins (0.1x to 2x bet)
        "medium": {"min": 2.0, "max": 10.0, "frequency": 0.15},  # Medium wins (2x to 10x bet)
        "large": {"min": 10.0, "max": 50.0, "frequency": 0.04},  # Large wins (10x to 50x bet)
        "epic": {"min": 50.0, "max": 200.0, "frequency": 0.01}   # Epic wins (50x to 200x bet)
    }

    # Calculate frequencies based on base game hit frequency
    for tier in base_win_tiers.values():
        tier["actual_freq"] = base_game_hit_freq * tier["frequency"]
    
    # Calculate average win multiplier for each tier to achieve target RTP
    # This is a simplified approach - real slot math would fine-tune these values more precisely
    # but this gives a reasonable approximation
    
    # First, distribute the base game RTP proportionally to frequencies
    total_base_freq = sum(tier["actual_freq"] for tier in base_win_tiers.values())
    
    for tier in base_win_tiers.values():
        tier_rtp_proportion = tier["actual_freq"] / total_base_freq if total_base_freq > 0 else 0
        tier_target_rtp = base_game_rtp * tier_rtp_proportion
        
        # Average payout multiplier = tier_rtp / tier_frequency
        tier_avg_multiplier = tier_target_rtp / tier["actual_freq"] if tier["actual_freq"] > 0 else 0
        
        # Constrain to tier min/max
        tier_avg_multiplier = max(min(tier_avg_multiplier, tier["max"]), tier["min"])
        tier["avg_multiplier"] = tier_avg_multiplier
    
    # Bonus game parameters
    # For simplicity, let's model the bonus as a range from 10x to max_win_multiplier
    # with an average that achieves the bonus_contribution of RTP
    bonus_target_rtp = rtp * bonus_contribution
    bonus_avg_multiplier = bonus_target_rtp / bonus_freq if bonus_freq > 0 else 0
    
    # Constrain bonus average to a reasonable range
    bonus_min_multiplier = 10.0
    bonus_avg_multiplier = max(bonus_min_multiplier, min(bonus_avg_multiplier, max_win_multiplier * 0.7))
    
    # Function to simulate the outcome of a single spin
    def simulator(bet: float) -> Tuple[float, Dict[str, Any]]:
        if bet <= 0:
            return 0, {"outcome_type": "invalid_bet", "win_tier": None, "multiplier": 0}
            
        # Generate a random number to determine outcome
        roll = np.random.random()
        
        # Initialize outcome details
        details = {
            "outcome_type": "loss",
            "win_tier": None,
            "multiplier": 0.0,
            "is_bonus": False
        }
        
        # Check if it's a bonus game
        if roll < bonus_freq:
            # It's a bonus
            details["outcome_type"] = "win"
            details["win_tier"] = "bonus"
            details["is_bonus"] = True
            
            # For bonus wins, use a distribution that's weighted toward smaller values
            # but with a long tail toward max_win_multiplier
            # Approximated with a clipped exponential distribution
            mean_param = 1.0 / (bonus_avg_multiplier - bonus_min_multiplier)
            raw_multiplier = np.random.exponential(1.0 / mean_param)
            bonus_multiplier = min(max_win_multiplier, bonus_min_multiplier + raw_multiplier)
            details["multiplier"] = bonus_multiplier
            
            return bet * bonus_multiplier, details
            
        # Check for base game wins
        cumulative_prob = bonus_freq
        for tier_name, tier in base_win_tiers.items():
            cumulative_prob += tier["actual_freq"]
            if roll < cumulative_prob:
                # Win in this tier
                details["outcome_type"] = "win"
                details["win_tier"] = tier_name
                
                # Generate a payout multiplier within the tier's range
                # Use triangular distribution centered on the average multiplier for more realistic distribution
                mode = tier["avg_multiplier"]
                min_val = tier["min"]
                max_val = tier["max"]
                
                # Ensure mode is between min and max
                mode = max(min_val, min(mode, max_val))
                
                # Generate the multiplier using triangular distribution
                multiplier = np.random.triangular(min_val, mode, max_val)
                details["multiplier"] = multiplier
                
                return bet * multiplier, details
                
        # If we get here, it's a loss
        return 0.0, details
        
    return simulator


def volatility_adjusted_outcome_simulator(game_metrics: GameMetrics) -> Callable[[float], Tuple[float, Dict[str, Any]]]:
    """
    Creates an outcome simulator with a simplified approach to match RTP, hit frequency, and variance.
    
    Instead of using a complex multi-tier system, this simulator uses two parameters:
    - hit_frequency: Probability of any win
    - win_multiplier_variance: Variance in the win multiplier
    
    The distribution is constructed to match the specified RTP, hit frequency, and volatility parameters.
    
    Args:
        game_metrics: Must contain 'rtp', 'hit_frequency', and 'volatility_index'
                      where volatility_index is a value from 1 (low) to 10 (extreme)
    Returns:
        Function that takes a bet amount and returns a tuple of (payout, details)
    """
    # Extract required metrics
    rtp = game_metrics.get("rtp")
    hit_freq = game_metrics.get("hit_frequency")
    volatility_index = game_metrics.get("volatility_index", 5)  # Default to medium volatility
    
    # Validate inputs
    if rtp is None or hit_freq is None:
        raise ValueError("volatility_adjusted_outcome_simulator requires 'rtp' and 'hit_frequency' in game_metrics")
    if not 0 < hit_freq <= 1:
        raise ValueError("hit_frequency must be between 0 (exclusive) and 1 (inclusive)")
    if rtp < 0:
        raise ValueError("RTP cannot be negative")
    if not 1 <= volatility_index <= 10:
        raise ValueError("volatility_index must be between 1 and 10")
    
    # Calculate average win multiplier
    avg_win_multiplier = rtp / hit_freq if hit_freq > 0 else 0
    
    # Volatility parameters
    # For low volatility games (1), most wins are close to the average
    # For high volatility games (10), wins have a wide range with occasional very large wins
    # This affects both the shape of the distribution and the max possible win
    
    # Map volatility index to distribution parameters
    # Higher volatility means more right-skewed distribution with fatter tails
    skewness = 0.5 + (volatility_index - 1) * 0.5  # Ranges from 0.5 to 5.0
    
    # Map volatility to max win multiplier
    # Low volatility (1) -> max ~20x
    # Medium volatility (5) -> max ~200x
    # High volatility (10) -> max ~2000x
    max_win_multiplier = 20 * math.pow(10, (volatility_index - 1) / 4)
    
    # Minimum win is typically the bet amount (1x) for medium/high volatility games,
    # but can be less for low volatility games to increase hit frequency
    min_win_multiplier = 1.0 if volatility_index >= 5 else 0.1
    
    # Define feature/bonus frequency based on volatility
    # Higher volatility games tend to have more of their RTP in features
    bonus_freq = 0.001 * volatility_index  # 0.1% for index 1, 1% for index 10
    
    # Function to simulate the outcome of a single spin
    def simulator(bet: float) -> Tuple[float, Dict[str, Any]]:
        if bet <= 0:
            return 0, {"outcome_type": "invalid_bet", "win_tier": None, "multiplier": 0, "is_feature": False}
            
        # Generate a random number to determine outcome
        roll = np.random.random()
        
        # Initialize outcome details
        details = {
            "outcome_type": "loss",
            "win_tier": None,
            "multiplier": 0.0,
            "is_feature": False
        }
        
        # Check if it's a win
        if roll < hit_freq:
            details["outcome_type"] = "win"
            
            # Check if it's a feature/bonus win
            if roll < bonus_freq:
                details["is_feature"] = True
                details["win_tier"] = "feature"
                
                # Feature wins can be much larger and have more variance
                # Use a heavily right-skewed distribution
                feature_shape = 0.7  # Lower values give more right skew
                max_feature_mult = max_win_multiplier
                
                # Generate win multiplier using Weibull distribution
                # (good for modeling right-skewed data with long tails)
                raw_multiplier = np.random.weibull(feature_shape) * avg_win_multiplier * 2
                multiplier = min(max_feature_mult, raw_multiplier + min_win_multiplier)
                
                details["multiplier"] = multiplier
                return bet * multiplier, details
            else:
                # Regular win - determine tier based on multiplier
                # Use a lognormal distribution which is naturally right-skewed
                # Adjust mean and sigma to control the shape
                
                # Calculate lognormal parameters to match target avg and skewness
                # Higher volatility = higher sigma
                sigma = 0.2 + (volatility_index / 10) * 0.8  # 0.2 - 1.0
                
                # Required mean of lognormal to achieve target average
                # E[X] = exp(mu + sigma^2/2)
                # => mu = ln(E[X]) - sigma^2/2
                mu = math.log(avg_win_multiplier) - (sigma**2) / 2
                
                # Generate win multiplier
                raw_multiplier = np.random.lognormal(mean=mu, sigma=sigma)
                multiplier = min(max_win_multiplier, max(min_win_multiplier, raw_multiplier))
                
                # Determine win tier based on multiplier
                if multiplier < 2.0:
                    details["win_tier"] = "small"
                elif multiplier < 10.0:
                    details["win_tier"] = "medium"
                elif multiplier < 50.0:
                    details["win_tier"] = "large"
                else:
                    details["win_tier"] = "epic"
                    
                details["multiplier"] = multiplier
                return bet * multiplier, details
        
        # If we get here, it's a loss
        return 0.0, details
        
    return simulator


# --- Predicting Behavior (Inner Loop Output) ---

def predict_player_behavior(
    game_metrics: GameMetrics,
    player_population_params: Dict[str, Any] | None = None
) -> PlayerBehaviorOutput:
    """
    Predicts key player behavior metrics by simulating sessions for a sample
    player population based on the provided game characteristics.

    This function represents the output needed by the outer optimization loop.

    Args:
        game_metrics: The metrics of the game design being evaluated. 
                      Requires at least 'rtp', 'hit_frequency', and 'volatility_index'.
        player_population_params: Optional parameters to define the player population.
                                  If provided, overrides the default player profiles.

    Returns:
        A dictionary containing predicted behavior metrics averaged over the population.
    """
    # Validate required game metrics for the simulator
    required_metrics = ['rtp', 'hit_frequency']
    missing_metrics = [metric for metric in required_metrics if metric not in game_metrics]
    if missing_metrics:
        raise ValueError(f"predict_player_behavior requires these missing metrics: {', '.join(missing_metrics)}")

    print(f"Predicting player behavior for game RTP={game_metrics.get('rtp')}, HitFreq={game_metrics.get('hit_frequency')}, Volatility={game_metrics.get('volatility_index', 5)}")

    # --- Define Player Population --- 
    # Default player profiles if none provided
    default_profiles = [
        # Conservative players (40% of population)
        # Low risk tolerance, moderate bankroll, fixed betting
        {"name": "Conservative-Fixed-Small", "weight": 0.20, "initial_balance": 100, "stop_loss_perc": 0.3, 
         "win_goal_mult": 1.5, "bet_type": "fixed", "bet_params": {"bet_amount": 0.5}},
        {"name": "Conservative-Fixed-Medium", "weight": 0.15, "initial_balance": 100, "stop_loss_perc": 0.4, 
         "win_goal_mult": 1.6, "bet_type": "fixed", "bet_params": {"bet_amount": 1.0}},
        {"name": "Conservative-PropBet", "weight": 0.05, "initial_balance": 80, "stop_loss_perc": 0.35, 
         "win_goal_mult": 1.5, "bet_type": "proportional", "bet_params": {"bet_percentage": 0.01}},
        
        # Moderate players (35% of population)
        # Medium risk tolerance, mixed bankrolls, various betting strategies
        {"name": "Moderate-Fixed", "weight": 0.15, "initial_balance": 150, "stop_loss_perc": 0.5, 
         "win_goal_mult": 2.0, "bet_type": "fixed", "bet_params": {"bet_amount": 1.0}},
        {"name": "Moderate-Kelly", "weight": 0.10, "initial_balance": 150, "stop_loss_perc": 0.5, 
         "win_goal_mult": 2.0, "bet_type": "kelly", "bet_params": {"conservative_factor": 0.5, 
         "min_bet": 0.5, "max_bet_percentage": 0.05}},
        {"name": "Moderate-PropBet", "weight": 0.10, "initial_balance": 120, "stop_loss_perc": 0.5, 
         "win_goal_mult": 2.0, "bet_type": "proportional", "bet_params": {"bet_percentage": 0.02}},
         
        # Risk-seeking players (25% of population)
        # Higher risk tolerance, larger bankrolls, aggressive betting
        {"name": "Risk-Fixed-Large", "weight": 0.10, "initial_balance": 200, "stop_loss_perc": 0.7, 
         "win_goal_mult": 3.0, "bet_type": "fixed", "bet_params": {"bet_amount": 2.0}},
        {"name": "Risk-Adaptive", "weight": 0.08, "initial_balance": 200, "stop_loss_perc": 0.7, 
         "win_goal_mult": 3.0, "bet_type": "adaptive", "bet_params": {"initial_bet": 1.0, 
         "increase_factor": 1.5, "decrease_factor": 0.7}},
        {"name": "Risk-Kelly", "weight": 0.07, "initial_balance": 250, "stop_loss_perc": 0.8, 
         "win_goal_mult": 3.5, "bet_type": "kelly", "bet_params": {"conservative_factor": 0.7, 
         "min_bet": 1.0, "max_bet_percentage": 0.1}}
    ]
    
    # Use provided params or defaults
    player_profiles = player_population_params.get("player_profiles", default_profiles) if player_population_params else default_profiles
    
    # Normalize weights if needed
    total_weight = sum(profile["weight"] for profile in player_profiles)
    if total_weight != 1.0:
        for profile in player_profiles:
            profile["weight"] = profile["weight"] / total_weight
    
    # Simulation parameters
    num_sessions_per_profile = player_population_params.get("num_sessions_per_profile", 50) if player_population_params else 50
    
    # Initialize accumulators for metrics
    total_spins_weighted = 0
    total_final_balance_weighted = 0
    total_wagered_weighted = 0
    total_time_played_weighted = 0
    total_sessions_simulated = 0
    
    # Additional player behavior metrics
    player_breakdown_by_profile = {}  # Detailed metrics by player profile
    churn_probability = 0  # Estimated probability player will not return
    total_churn_weighted = 0
    
    # Create the outcome simulator
    try:
        # Prefer the more sophisticated simulator if volatility_index is provided
        if "volatility_index" in game_metrics:
            outcome_sim = volatility_adjusted_outcome_simulator(game_metrics)
        else:
            # Fallback to multi-tier simulator with a default volatility
            game_metrics_with_volatility = game_metrics.copy()
            game_metrics_with_volatility["volatility_index"] = 5  # Medium volatility
            outcome_sim = multi_tier_outcome_simulator(game_metrics_with_volatility)
    except ValueError as e:
        print(f"Error creating outcome simulator: {e}")
        # Return error state
        return {
            "error": str(e),
            "expected_total_spins": 0,
            "average_final_balance": 0,
            "average_bet": 0,
            "churn_probability": 1.0  # Assume 100% churn if simulation fails
        } 

    # --- Simulation Loop --- 
    for profile in player_profiles:
        profile_weight = profile["weight"]
        profile_name = profile["name"]
        
        # Create initial state
        initial_state = {
            "initial_balance": profile["initial_balance"],
            "current_balance": profile["initial_balance"]
        }
        
        # Define stop rule - CombinedRule with StopLoss, WinGoal, and a safety SpinLimitRule
        stop_rules = [
            SimpleStopLossRule(loss_percentage=profile["stop_loss_perc"]),
            WinGoalRule(win_multiplier=profile["win_goal_mult"]),
            SpinLimitRule(max_spins=10000)  # Safety limit
        ]
        
        # Add streak-based rules for some profiles (simulate player frustration/excitement)
        if "streak_rule" in profile:
            streak_params = profile["streak_rule"]
            if streak_params.get("type") == "loss":
                stop_rules.append(LossStreakRule(max_consecutive_losses=streak_params.get("max_streak", 5)))
        
        stop_rule = CombinedRule(stop_rules)
        
        # Create betting strategy based on profile
        bet_type = profile["bet_type"]
        if bet_type == "fixed":
            betting_strategy = FixedBetStrategy(**profile["bet_params"])
        elif bet_type == "proportional":
            betting_strategy = ProportionalBetting(**profile["bet_params"])
        elif bet_type == "kelly":
            betting_strategy = KellyCriterionBetting(**profile["bet_params"])
        elif bet_type == "adaptive":
            betting_strategy = AdaptiveBetting(**profile["bet_params"])
        else:
            raise ValueError(f"Unknown betting strategy type: {bet_type}")
        
        # Simulate sessions for this profile
        profile_total_spins = 0
        profile_total_final_balance = 0
        profile_total_wagered = 0
        profile_total_time_played_seconds = 0
        profile_churn_count = 0  # Count sessions that would likely lead to churn
        
        profile_sessions = []  # Store detailed session data
        
        for _ in range(num_sessions_per_profile):
            final_state = simulate_player_session(
                game_metrics, initial_state.copy(), stop_rule, betting_strategy, outcome_sim
            )
            
            # Extract key metrics
            spins_played = final_state.get("spins_played", 0)
            final_balance = final_state.get("current_balance", 0)
            total_wagered = final_state.get("total_wagered", 0)
            time_played_seconds = final_state.get("session_duration_seconds", 0)
            
            # Store session data
            profile_sessions.append({
                "spins": spins_played,
                "final_balance": final_balance,
                "wagered": total_wagered,
                "time_played_seconds": time_played_seconds,
                "initial_balance": initial_state["initial_balance"],
                "net_result": final_state.get("net_result", 0),
                "largest_win": final_state.get("largest_win", 0)
            })
            
            # Update profile totals
            profile_total_spins += spins_played
            profile_total_final_balance += final_balance
            profile_total_wagered += total_wagered
            profile_total_time_played_seconds += time_played_seconds
            
            # Churn estimation
            # Players are more likely to churn if:
            # 1. They lost too quickly (high loss rate)
            # 2. They had no meaningful wins (no excitement)
            # 3. They reached their stop-loss (frustrating experience)
            
            net_result = final_state.get("net_result", 0)
            largest_win = final_state.get("largest_win", 0)
            largest_win_multiplier = largest_win / betting_strategy.choose_bet(initial_state, game_metrics) if largest_win > 0 else 0
            
            # Simple churn model
            is_likely_churn = (
                (net_result < -0.5 * initial_state["initial_balance"] and spins_played < 20) or  # Quick large loss
                (largest_win_multiplier < 5 and net_result < 0) or  # No exciting wins and lost money
                (final_balance <= initial_state["initial_balance"] * (1 - profile["stop_loss_perc"]))  # Hit stop-loss
            )
            
            if is_likely_churn:
                profile_churn_count += 1
            
            # Increment session counter
            total_sessions_simulated += 1
            
        # Calculate averages for this profile
        avg_spins = profile_total_spins / num_sessions_per_profile if num_sessions_per_profile > 0 else 0
        avg_final_balance = profile_total_final_balance / num_sessions_per_profile if num_sessions_per_profile > 0 else 0
        avg_wagered = profile_total_wagered / num_sessions_per_profile if num_sessions_per_profile > 0 else 0
        avg_time_played = profile_total_time_played_seconds / num_sessions_per_profile if num_sessions_per_profile > 0 else 0
        profile_churn_probability = profile_churn_count / num_sessions_per_profile if num_sessions_per_profile > 0 else 0
        
        # Add weighted values to overall totals
        total_spins_weighted += avg_spins * profile_weight
        total_final_balance_weighted += avg_final_balance * profile_weight
        total_wagered_weighted += avg_wagered * profile_weight
        total_time_played_weighted += avg_time_played * profile_weight
        total_churn_weighted += profile_churn_probability * profile_weight
        
        # Store detailed profile metrics
        player_breakdown_by_profile[profile_name] = {
            "avg_spins": avg_spins,
            "avg_final_balance": avg_final_balance,
            "avg_wagered": avg_wagered,
            "avg_session_minutes": avg_time_played / 60,
            "churn_probability": profile_churn_probability,
            "weight": profile_weight,
            "initial_balance": profile["initial_balance"],
            "stop_loss_perc": profile["stop_loss_perc"],
            "win_goal_mult": profile["win_goal_mult"],
            "bet_type": profile["bet_type"],
            "detailed_sessions": profile_sessions[:5]  # Store first 5 sessions for reference
        }
    
    # --- Calculate Overall Behavior Metrics ---
    avg_bet = total_wagered_weighted / total_spins_weighted if total_spins_weighted > 0 else 0
    churn_probability = total_churn_weighted
    
    # Player lifetime value estimation:
    # Based on expected spins per session, avg bet, house edge, and churn probability
    house_edge = 1 - game_metrics.get("rtp", 0.9)
    avg_profit_per_spin = avg_bet * house_edge
    avg_sessions_before_churn = 1 / churn_probability if churn_probability > 0 else 10  # Default to 10 if churn is 0
    
    # Adjust for diminishing returns in later sessions (players tend to play less over time)
    # This is a simplification - real LTV would need a proper retention curve
    diminishing_factor = 0.9  # Each session is 90% as valuable as the previous one
    effective_sessions = (1 - diminishing_factor ** avg_sessions_before_churn) / (1 - diminishing_factor) if diminishing_factor < 1 else avg_sessions_before_churn
    
    player_ltv = total_spins_weighted * avg_profit_per_spin * effective_sessions
    
    # Return comprehensive behavior metrics
    return {
        "expected_total_spins": total_spins_weighted,
        "average_final_balance": total_final_balance_weighted,
        "average_wagered": total_wagered_weighted,
        "average_session_minutes": total_time_played_weighted / 60,
        "average_bet": avg_bet,
        "churn_probability": churn_probability,
        "estimated_player_ltv": player_ltv,
        "house_edge": house_edge,
        "avg_profit_per_spin": avg_profit_per_spin,
        "expected_sessions_before_churn": avg_sessions_before_churn,
        "num_profiles_simulated": len(player_profiles),
        "num_sessions_per_profile": num_sessions_per_profile,
        "total_sessions_simulated": total_sessions_simulated,
        "player_breakdown": player_breakdown_by_profile
    }