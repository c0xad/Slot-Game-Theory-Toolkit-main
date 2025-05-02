# -*- coding: utf-8 -*-
"""
Implements components of Prospect Theory relevant to gambling decisions.

Includes:
- Value function (S-shaped, reference-dependent).
- Probability weighting function (overweighting small probabilities).
"""

import numpy as np
from typing import Callable

# --- Value Function ---
# v(x) = x^alpha if x >= 0
# v(x) = -lambda * (-x)^beta if x < 0
# Where x is the outcome relative to a reference point.

DEFAULT_ALPHA = 0.88  # Exponent for gains (typically < 1, risk aversion)
DEFAULT_BETA = 0.88   # Exponent for losses (typically < 1, risk seeking in losses)
DEFAULT_LAMBDA = 2.25 # Loss aversion coefficient (typically > 1)

def value_function(
    outcome: float,
    reference_point: float = 0.0,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    lambda_loss_aversion: float = DEFAULT_LAMBDA
) -> float:
    """
    Calculates the subjective value of an outcome based on Prospect Theory.

    Args:
        outcome: The objective outcome (e.g., win/loss amount).
        reference_point: The reference point against which the outcome is evaluated.
        alpha: Parameter for the curvature of the value function for gains.
        beta: Parameter for the curvature of the value function for losses.
        lambda_loss_aversion: Loss aversion parameter.

    Returns:
        The subjective value (utility) of the outcome.
    """
    x = outcome - reference_point

    if x >= 0:
        # Ensure base is non-negative for fractional power
        return np.power(x, alpha) if x > 0 else 0.0
    else:
        # Ensure base is non-negative for fractional power
        return -lambda_loss_aversion * np.power(-x, beta)


# --- Probability Weighting Function ---
# Common form (Tversky & Kahneman, 1992): w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)
# Simpler form (Prelec): w(p) = exp(-(-ln(p))^delta)

DEFAULT_GAMMA = 0.61 # Parameter for probability weighting (typical value for gains)
DEFAULT_DELTA = 0.69 # Parameter for probability weighting (typical value for losses)
# Note: Often different parameters are used for gains and losses.

def probability_weighting_tversky_kahneman(
    probability: float,
    gamma: float = DEFAULT_GAMMA
) -> float:
    """
    Applies the Tversky & Kahneman (1992) probability weighting function.

    Args:
        probability: The objective probability (must be between 0 and 1).
        gamma: The parameter controlling the shape of the weighting function.

    Returns:
        The subjective decision weight for the probability.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")
    if probability == 0:
        return 0.0
    if probability == 1:
        return 1.0

    p_gamma = np.power(probability, gamma)
    one_minus_p_gamma = np.power(1 - probability, gamma)
    denominator = np.power(p_gamma + one_minus_p_gamma, 1 / gamma)

    # Handle potential division by zero or numerical instability
    if denominator == 0:
        # This case is unlikely with standard parameters but handle defensively
        return probability # Fallback or handle as error

    return p_gamma / denominator

# Add other weighting functions like Prelec if needed.

# --- Calculating Prospect Theory Value of a Gamble ---

def calculate_prospect_value(
    outcomes: np.ndarray, # Array of possible outcomes (relative to reference)
    probabilities: np.ndarray, # Array of corresponding objective probabilities
    value_func: Callable[[float], float] = value_function, # Pass configured value func
    weighting_func: Callable[[float], float] = probability_weighting_tversky_kahneman # Pass configured weighting func
    # Optional: Separate weighting functions for gains/losses
) -> float:
    """
    Calculates the overall subjective value of a prospect (gamble).

    Args:
        outcomes: Array of potential outcomes (e.g., [-1, 10, 100]). Assumed relative to ref point.
        probabilities: Array of probabilities for each outcome. Must sum to 1.
        value_func: The subjective value function to apply to outcomes.
        weighting_func: The probability weighting function to apply.

    Returns:
        The overall Prospect Theory value (decision utility) of the gamble.
    """
    if not np.isclose(np.sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1")
    if len(outcomes) != len(probabilities):
        raise ValueError("Outcomes and probabilities must have the same length")

    subjective_value = 0.0
    for i in range(len(outcomes)):
        outcome = outcomes[i]
        prob = probabilities[i]

        # Apply value function to outcome
        v_outcome = value_func(outcome)

        # Apply weighting function to probability
        # TODO: Handle gain/loss separation for weighting if needed
        w_prob = weighting_func(prob)

        subjective_value += w_prob * v_outcome

    return subjective_value


# Example Usage:
# Define a simple gamble: 50% chance to win 100, 50% chance to lose 50
# outcomes = np.array([100.0, -50.0])
# probabilities = np.array([0.5, 0.5])
#
# # Use default parameters
# pt_value = calculate_prospect_value(outcomes, probabilities)
# print(f"Prospect Theory Value (default params): {pt_value}")
#
# # Define custom parameters
# custom_value_func = lambda x: value_function(x, alpha=0.9, beta=0.9, lambda_loss_aversion=2.0)
# custom_weighting_func = lambda p: probability_weighting_tversky_kahneman(p, gamma=0.65)
# pt_value_custom = calculate_prospect_value(outcomes, probabilities, custom_value_func, custom_weighting_func)
# print(f"Prospect Theory Value (custom params): {pt_value_custom}")