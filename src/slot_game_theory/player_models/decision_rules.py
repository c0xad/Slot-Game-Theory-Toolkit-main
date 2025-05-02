# -*- coding: utf-8 -*-
"""
Models player decision rules, such as when to stop playing or how much to bet.

These rules form the basis of the player's strategy in response to a game design.
They can range from simple heuristics (stop-loss) to complex dynamic programming solutions.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np

# Assume access to game evaluation results and potentially prospect theory values
# from ..evaluation.simulation import SimulationResult
# from .prospect_theory import calculate_prospect_value

# Type alias for game metrics used by decision rules
GameMetrics = Dict[str, Any] # e.g., {"rtp": 0.96, "variance": 25.5, "hit_frequency": 0.2}

# Type alias for player state
PlayerState = Dict[str, Any] # e.g., {"current_balance": 100, "spins_played": 50, "initial_balance": 100}

# Type alias for the output of a player behavior model
PlayerBehaviorOutput = Dict[str, Any] # e.g., {"expected_total_spins": 1000, "average_bet": 1.0}


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

# Add other rules: WinGoalRule, TimeLimitRule, CombinedRule, etc.


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
        # Ensure player has enough balance? Or assume sufficient funds for model.
        # current_balance = player_state.get("current_balance", self.bet_amount)
        # return min(self.bet_amount, current_balance)
        return self.bet_amount

# Add other strategies: ProportionalBetting, Martingale (discouraged!), etc.


# --- Simulating Player Session ---

def simulate_player_session(
    game_metrics: GameMetrics,
    initial_state: PlayerState,
    stop_rule: StopRule,
    betting_strategy: BettingStrategy,
    # Need a way to simulate game outcomes based on metrics (RTP, variance)
    # This is tricky without the full simulation engine.
    # Option 1: Pass the simulation engine itself (complex dependency)
    # Option 2: Use simplified stochastic model based on metrics
    outcome_simulator: Callable[[float], float] # Function: bet -> payout
) -> PlayerState:
    """
    Simulates a single player session until a stop rule is met.

    Args:
        game_metrics: Characteristics of the slot game.
        initial_state: Starting state of the player (balance, etc.).
        stop_rule: The rule determining when the player stops.
        betting_strategy: The rule determining the player's bet size.
        outcome_simulator: A function that takes a bet amount and returns a
                           stochastically generated payout based on game_metrics.

    Returns:
        The final state of the player after the session ends.
    """
    player_state = initial_state.copy()
    player_state["spins_played"] = player_state.get("spins_played", 0) # Ensure spins_played exists

    while not stop_rule(player_state, game_metrics):
        # 1. Choose Bet
        bet_amount = betting_strategy(player_state, game_metrics)
        if bet_amount <= 0 or bet_amount > player_state.get("current_balance", 0):
            # print("Player cannot afford bet or invalid bet amount. Stopping session.")
            break # Stop if bet is invalid or unaffordable

        # 2. Simulate Spin Outcome
        payout = outcome_simulator(bet_amount)

        # 3. Update State
        player_state["current_balance"] = player_state.get("current_balance", 0) - bet_amount + payout
        player_state["spins_played"] += 1
        # Add other state updates (e.g., tracking wins/losses)

        # Safety break for infinite loops (should be handled by stop rule ideally)
        if player_state["spins_played"] > 1_000_000: # Arbitrary large number
            print("Warning: Spin limit reached in player session simulation.")
            break

    return player_state


# --- Simplified Outcome Simulator (Example) ---
# This needs to be statistically sound and reflect RTP and Variance.
# A simple normal distribution might not capture skewness of slot payouts well.
# Using a more appropriate distribution (e.g., shifted log-normal, or custom discrete) is better.

def simple_normal_outcome_simulator(game_metrics: GameMetrics) -> Callable[[float], float]:
    """Creates a simple outcome simulator based on Normal distribution."""
    rtp = game_metrics.get("rtp", 0.96)
    variance = game_metrics.get("variance", 25.0)
    std_dev = np.sqrt(variance) if variance >= 0 else 0

    # Mean payout = bet * rtp
    # We need mean and std dev of the *payout*, not necessarily normalized payout multiplier
    # Assuming variance provided is for payout amount with bet=1? Needs clarification.
    # Let's assume variance is for payout multiplier, std_dev_multiplier = std_dev

    def simulator(bet: float) -> float:
        mean_payout = bet * rtp
        # Scale std_dev by bet? Var(k*X) = k^2 * Var(X) => StdDev(k*X) = k * StdDev(X)
        payout_std_dev = bet * std_dev
        payout = np.random.normal(loc=mean_payout, scale=payout_std_dev)
        return max(0, payout) # Payout cannot be negative

    return simulator


# --- Predicting Behavior (Inner Loop Output) ---

def predict_player_behavior(
    game_metrics: GameMetrics,
    # Parameters describing the representative player population
    player_population_params: Dict[str, Any] = None
) -> PlayerBehaviorOutput:
    """
    Predicts key player behavior metrics based on game characteristics.
    This function represents the output needed by the outer optimization loop.

    Args:
        game_metrics: The metrics of the game design being evaluated.
        player_population_params: Parameters describing the player type (risk aversion, etc.).

    Returns:
        A dictionary containing predicted behavior (e.g., expected spins).
    """
    # This is where the core player modeling happens.
    # It could involve:
    # 1. Defining representative player(s) (initial balance, stop rules, betting strategy).
    # 2. Simulating many sessions for these players using an outcome simulator based on game_metrics.
    # 3. Averaging the results (e.g., average spins played).
    # OR
    # 4. Using analytical models if available (e.g., relating variance to playtime).

    print(f"Predicting player behavior for game RTP={game_metrics.get('rtp', 'N/A')}, Var={game_metrics.get('variance', 'N/A')}")

    # --- Placeholder Implementation ---
    # Simulate N sessions for a representative player
    num_sessions_to_sim = 100 # Number of players/sessions to average over
    total_spins_across_sessions = 0

    # Define the representative player
    initial_state = {"initial_balance": 100, "current_balance": 100}
    stop_rule = SimpleStopLossRule(loss_percentage=0.8) # Stop if balance drops to 20
    # stop_rule = SpinLimitRule(max_spins=500) # Alternative rule
    betting_strategy = FixedBetStrategy(bet_amount=1.0)
    outcome_sim = simple_normal_outcome_simulator(game_metrics) # Use the simplified simulator

    for _ in range(num_sessions_to_sim):
        final_state = simulate_player_session(
            game_metrics, initial_state, stop_rule, betting_strategy, outcome_sim
        )
        total_spins_across_sessions += final_state.get("spins_played", 0)

    avg_spins = total_spins_across_sessions / num_sessions_to_sim if num_sessions_to_sim > 0 else 0
    # --- End Placeholder ---

    return {
        "expected_total_spins": avg_spins,
        "average_bet": 1.0, # Assuming fixed bet from strategy for now
        # Add other predicted metrics: churn probability, lifetime value, etc.
    }