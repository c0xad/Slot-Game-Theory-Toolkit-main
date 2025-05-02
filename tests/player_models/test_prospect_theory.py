# -*- coding: utf-8 -*-
"""Tests for the prospect theory module."""

import pytest
import numpy as np
from slot_game_theory.player_models import prospect_theory # Adjust import path

# --- Test Value Function ---

def test_value_function_defaults():
    """Test value function with default parameters."""
    # Gains (alpha=0.88) - concave
    assert prospect_theory.value_function(0) == 0.0
    assert prospect_theory.value_function(10) == pytest.approx(10**0.88)
    assert prospect_theory.value_function(100) == pytest.approx(100**0.88)
    assert prospect_theory.value_function(10) < 10 # Risk aversion for gains
    assert prospect_theory.value_function(100) < 100

    # Losses (beta=0.88, lambda=2.25) - convex, steeper
    assert prospect_theory.value_function(-10) == pytest.approx(-2.25 * (10**0.88))
    assert prospect_theory.value_function(-100) == pytest.approx(-2.25 * (100**0.88))
    # Check loss aversion: |-v(-10)| > v(10)
    assert abs(prospect_theory.value_function(-10)) > prospect_theory.value_function(10)
    # Check risk seeking for losses: slope decreases for larger losses
    # v(-10) - v(-20) < v(-100) - v(-110) (approximately, due to convexity)
    v_neg10 = prospect_theory.value_function(-10)
    v_neg20 = prospect_theory.value_function(-20)
    v_neg100 = prospect_theory.value_function(-100)
    v_neg110 = prospect_theory.value_function(-110)
    # Value increases (less negative) as loss decreases, so v(-10) > v(-20)
    # We expect the *decrease* in value per unit loss to be smaller for larger losses
    # i.e., |v(-20) - v(-10)| / 10 > |v(-110) - v(-100)| / 10
    assert (v_neg10 - v_neg20) > (v_neg100 - v_neg110)


def test_value_function_custom_params():
    """Test value function with custom parameters."""
    alpha, beta, lambda_ = 0.5, 0.6, 3.0
    vf = lambda x: prospect_theory.value_function(x, alpha=alpha, beta=beta, lambda_loss_aversion=lambda_)

    assert vf(16) == pytest.approx(16**alpha) == 4.0
    assert vf(-100) == pytest.approx(-lambda_ * (100**beta)) == pytest.approx(-3.0 * (100**0.6))

def test_value_function_reference_point():
    """Test value function with a non-zero reference point."""
    rp = 50.0
    vf = lambda x: prospect_theory.value_function(x, reference_point=rp)

    # Outcome 60 is a gain of 10 relative to rp
    assert vf(60) == pytest.approx(prospect_theory.value_function(10))
    # Outcome 40 is a loss of 10 relative to rp
    assert vf(40) == pytest.approx(prospect_theory.value_function(-10))
    # Outcome 50 is zero relative to rp
    assert vf(50) == 0.0

# --- Test Probability Weighting Function ---

def test_probability_weighting_defaults():
    """Test Tversky-Kahneman weighting with default gamma."""
    gamma = prospect_theory.DEFAULT_GAMMA # 0.61
    w = prospect_theory.probability_weighting_tversky_kahneman

    assert w(0.0) == 0.0
    assert w(1.0) == 1.0

    # Check overweighting of small probabilities
    assert w(0.01) > 0.01
    assert w(0.1) > 0.1

    # Check underweighting of moderate/high probabilities
    assert w(0.5) < 0.5
    assert w(0.9) < 0.9

    # Check approximate shape based on typical gamma < 1
    assert w(0.2) / 0.2 > w(0.8) / 0.8 # Overweighting effect stronger for small p

def test_probability_weighting_custom_gamma():
    """Test weighting with different gamma values."""
    w_low_gamma = lambda p: prospect_theory.probability_weighting_tversky_kahneman(p, gamma=0.4)
    w_high_gamma = lambda p: prospect_theory.probability_weighting_tversky_kahneman(p, gamma=0.9)

    # Lower gamma -> more distortion (more overweighting of small p)
    assert w_low_gamma(0.1) > w_high_gamma(0.1)
    # Higher gamma -> closer to linear
    assert abs(w_high_gamma(0.5) - 0.5) < abs(w_low_gamma(0.5) - 0.5)

def test_probability_weighting_invalid_input():
    """Test invalid probability inputs."""
    w = prospect_theory.probability_weighting_tversky_kahneman
    with pytest.raises(ValueError):
        w(-0.1)
    with pytest.raises(ValueError):
        w(1.1)

# --- Test Calculating Prospect Value ---

@pytest.fixture
def simple_gamble():
    """A simple gamble: 50% win 100, 50% lose 50."""
    return np.array([100.0, -50.0]), np.array([0.5, 0.5])

def test_calculate_prospect_value_defaults(simple_gamble):
    """Test calculating prospect value with default functions."""
    outcomes, probs = simple_gamble
    pt_value = prospect_theory.calculate_prospect_value(outcomes, probs)

    # Manual calculation with defaults:
    # v(100) = 100^0.88
    # v(-50) = -2.25 * 50^0.88
    # w(0.5) = 0.5^0.61 / (0.5^0.61 + (1-0.5)^0.61)^(1/0.61)
    v_100 = prospect_theory.value_function(100)
    v_neg50 = prospect_theory.value_function(-50)
    w_05 = prospect_theory.probability_weighting_tversky_kahneman(0.5)
    expected_value = w_05 * v_100 + w_05 * v_neg50

    assert pt_value == pytest.approx(expected_value)
    # Check sign - usually negative for such gambles due to loss aversion
    assert pt_value < 0

def test_calculate_prospect_value_custom_funcs(simple_gamble):
    """Test calculating prospect value with custom functions."""
    outcomes, probs = simple_gamble

    # Custom funcs: Linear value, linear weighting (expected value)
    linear_value = lambda x: x
    linear_weighting = lambda p: p
    pt_value_ev = prospect_theory.calculate_prospect_value(
        outcomes, probs, linear_value, linear_weighting
    )
    expected_ev = 0.5 * 100 + 0.5 * (-50) = 50 - 25 = 25.0
    assert pt_value_ev == pytest.approx(25.0)

    # Custom funcs: Example risk seeking
    risk_seeking_value = lambda x: prospect_theory.value_function(x, alpha=1.2, beta=1.2, lambda_loss_aversion=1.0)
    pt_value_rs = prospect_theory.calculate_prospect_value(
        outcomes, probs, risk_seeking_value, prospect_theory.probability_weighting_tversky_kahneman
    )
    # Expect value to be higher than default due to less loss aversion and risk seeking shape
    pt_value_default = prospect_theory.calculate_prospect_value(outcomes, probs)
    assert pt_value_rs > pt_value_default


def test_calculate_prospect_value_invalid_probs(simple_gamble):
    """Test prospect value calculation with invalid probabilities."""
    outcomes, _ = simple_gamble
    invalid_probs1 = np.array([0.6, 0.5]) # Sum > 1
    invalid_probs2 = np.array([0.4, 0.5]) # Sum < 1

    with pytest.raises(ValueError, match="Probabilities must sum to 1"):
        prospect_theory.calculate_prospect_value(outcomes, invalid_probs1)
    with pytest.raises(ValueError, match="Probabilities must sum to 1"):
        prospect_theory.calculate_prospect_value(outcomes, invalid_probs2)

def test_calculate_prospect_value_mismatched_lengths(simple_gamble):
    """Test prospect value calculation with mismatched outcomes/probs lengths."""
    outcomes, probs = simple_gamble
    with pytest.raises(ValueError, match="must have the same length"):
        prospect_theory.calculate_prospect_value(outcomes[:1], probs)
    with pytest.raises(ValueError, match="must have the same length"):
        prospect_theory.calculate_prospect_value(outcomes, probs[:1])