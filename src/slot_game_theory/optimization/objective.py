# -*- coding: utf-8 -*-
"""
Defines objective functions for the slot game design optimization problem.

These functions typically represent the quantity the designer (Player 0) wants to
maximize or minimize, such as operator profit, player lifetime value, or time-on-device,
subject to player behavior and regulatory constraints.
"""

from typing import Dict, Any, Callable
# Assume access to evaluation results and player models
# from ..evaluation.simulation import SimulationResult # Or symbolic result type
# from ..player_models.decision_rules import PlayerResponse # Type for predicted player behavior

# Define a type for the design parameters being optimized
# This could be a complex structure or a flattened numpy array
DesignParameters = Any # e.g., np.ndarray, Dict[str, Any]

# Define a type for the output of an evaluation function (e.g., simulation results)
EvaluationOutput = Dict[str, Any] # e.g., SimulationResult

# Define a type for the output of a player behavior model
PlayerBehaviorOutput = Dict[str, Any] # e.g., {"expected_playtime": 100, "churn_prob": 0.1}


def calculate_operator_profit(
    eval_results: EvaluationOutput,
    player_behavior: PlayerBehaviorOutput,
    # Add other relevant parameters like operational costs, taxes, etc.
    cost_per_spin: float = 0.01 # Example operational cost
) -> float:
    """
    Calculates the estimated operator profit based on game metrics and player behavior.

    Args:
        eval_results: Output from the game evaluation (e.g., simulation results).
                      Expected keys: 'rtp', 'expected_payout_per_spin', 'total_spins'.
        player_behavior: Output from the player behavior model.
                         Expected keys: 'expected_total_spins', 'expected_lifetime_value', etc.
        cost_per_spin: Operational cost associated with each spin.

    Returns:
        Estimated profit (or a related metric to be optimized).
    """
    # Example Profit Calculation:
    # Profit = (Total Bet In - Total Payout Out) - Operational Costs
    # Profit = Expected Spins * (Avg Bet * (1 - RTP) - Cost Per Spin)

    # Need consistent metrics from evaluation and player model
    # Let's assume player_behavior gives 'expected_total_spins' and eval_results gives 'rtp'
    # and we assume a constant average bet 'avg_bet' (this might come from player model too)

    expected_total_spins = player_behavior.get("expected_total_spins", 1000) # Default if not provided
    rtp = eval_results.get("rtp", 0.96) # Default if not provided
    avg_bet = eval_results.get("average_bet", 1.0) # Should ideally come from player model based on design

    # Theoretical House Edge per spin
    house_edge_per_spin = avg_bet * (1 - rtp)

    # Total expected profit over player lifetime
    total_profit = expected_total_spins * (house_edge_per_spin - cost_per_spin)

    # The optimizer might minimize negative profit, so return -total_profit if needed.
    return total_profit


def calculate_time_on_device(
    eval_results: EvaluationOutput,
    player_behavior: PlayerBehaviorOutput
) -> float:
    """
    Estimates the player's time-on-device (engagement metric).

    Often correlated with lower volatility.
    """
    # This would likely be a direct output from the player behavior model
    expected_playtime_spins = player_behavior.get("expected_total_spins", 0)
    # Could also be estimated based on variance from eval_results if no player model
    # variance = eval_results.get("variance", 0)
    # time_on_device = some_function_of_variance(variance)
    return expected_playtime_spins # Or a time-based equivalent


# --- Wrapper for use in optimization libraries ---

def create_objective_function(
    evaluation_func: Callable[[DesignParameters], EvaluationOutput],
    player_model_func: Callable[[EvaluationOutput], PlayerBehaviorOutput],
    objective_metric_func: Callable[[EvaluationOutput, PlayerBehaviorOutput], float],
    maximize: bool = True
) -> Callable[[DesignParameters], float]:
    """
    Creates a single objective function suitable for optimization libraries.

    This function takes only the design parameters as input, runs the evaluation
    and player modeling, and returns the final scalar objective value.

    Args:
        evaluation_func: A function that takes design parameters and returns evaluation metrics.
        player_model_func: A function that takes evaluation metrics and predicts player behavior.
        objective_metric_func: A function that calculates the final scalar objective
                               (e.g., profit) from evaluation and player behavior results.
        maximize: If True, the function returns the objective value directly.
                  If False, it returns the negative objective value (for minimization).

    Returns:
        A callable function `f(design_params) -> objective_value`.
    """
    def objective_function(design_params: DesignParameters) -> float:
        eval_results = evaluation_func(design_params)
        player_behavior = player_model_func(eval_results)
        objective_value = objective_metric_func(eval_results, player_behavior)

        # Ensure finite value (handle potential NaN or Inf from calculations)
        if not np.isfinite(objective_value):
             # Return a very bad value if maximization, very good if minimization
             return -np.inf if maximize else np.inf

        return objective_value if maximize else -objective_value

    return objective_function