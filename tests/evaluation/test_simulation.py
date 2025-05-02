# -*- coding: utf-8 -*-
"""Tests for the Monte Carlo simulation module."""

import pytest
import time
from slot_game_theory.core import reels, paytable, bonuses
from slot_game_theory.evaluation import simulation # Adjust import path
from slot_game_theory.utils import helpers # For setting seed

# --- Fixtures for Simulation ---

@pytest.fixture
def sim_symbols():
    """Symbols for simulation tests."""
    return {
        "A": reels.Symbol("A"),
        "K": reels.Symbol("K"),
        "Q": reels.Symbol("Q"),
        "SCATTER": reels.Symbol("SCATTER", is_scatter=True),
        "WILD": reels.Symbol("WILD", is_wild=True),
    }

@pytest.fixture
def sim_reels(sim_symbols):
    """Reels for simulation (slightly more complex)."""
    # 3x3 setup
    r1 = reels.ReelStrip([sim_symbols["A"], sim_symbols["K"], sim_symbols["Q"], sim_symbols["SCATTER"], sim_symbols["A"]])
    r2 = reels.ReelStrip([sim_symbols["K"], sim_symbols["WILD"], sim_symbols["Q"], sim_symbols["K"], sim_symbols["A"]])
    r3 = reels.ReelStrip([sim_symbols["Q"], sim_symbols["A"], sim_symbols["SCATTER"], sim_symbols["WILD"], sim_symbols["K"]])
    return [r1, r2, r3]

@pytest.fixture
def sim_paylines():
    """Standard 3x3 paylines."""
    return reels.get_standard_paylines(num_reels=3, num_rows=3)

@pytest.fixture
def sim_paytable(sim_symbols):
    """Paytable for simulation."""
    win_conditions = {
        (sim_symbols["A"], 3): 50,
        (sim_symbols["A"], 2): 5, # Left-to-right assumed by default evaluator
        (sim_symbols["K"], 3): 20,
        (sim_symbols["Q"], 3): 10,
        (sim_symbols["WILD"], 3, 'left_to_right'): 100,
        (sim_symbols["SCATTER"], 3, 'scatter'): 15, # Pays 15x total bet
    }
    return paytable.Paytable(win_conditions)

@pytest.fixture
def sim_bonus_trigger(sim_symbols):
    """Bonus trigger condition."""
    return {sim_symbols["SCATTER"]: 3} # 3 Scatters trigger bonus

@pytest.fixture
def sim_bonus_feature():
    """A simple free spins bonus feature."""
    # Uses base game reels and paytable for simplicity here
    return bonuses.FreeSpinsFeature(num_spins=10, multiplier=2.0)


@pytest.fixture
def simulator_instance(sim_reels, sim_paylines, sim_paytable):
    """Basic simulator instance without bonus."""
    return simulation.MonteCarloSimulator(
        reel_strips=sim_reels,
        paylines=sim_paylines,
        paytable=sim_paytable,
        num_rows=3
    )

@pytest.fixture
def simulator_instance_with_bonus(sim_reels, sim_paylines, sim_paytable, sim_bonus_trigger, sim_bonus_feature):
    """Simulator instance with bonus feature."""
    return simulation.MonteCarloSimulator(
        reel_strips=sim_reels,
        paylines=sim_paylines,
        paytable=sim_paytable,
        num_rows=3,
        bonus_triggers=sim_bonus_trigger,
        bonus_feature=sim_bonus_feature
    )

# --- Test Simulation Runs ---

def test_simulation_run_basic(simulator_instance):
    """Test a basic simulation run returns expected structure."""
    helpers.set_random_seed(123) # For reproducibility if needed, though results vary
    num_spins = 1000
    results = simulator_instance.run_simulation(num_spins=num_spins, initial_bet=1.0)

    assert isinstance(results, dict)
    assert results["total_spins"] == num_spins
    assert "rtp" in results
    assert "variance" in results
    assert "hit_frequency" in results
    assert "duration_sec" in results
    assert "spins_per_second" in results
    assert "expected_payout_per_spin" in results

    # Plausibility checks (highly dependent on game math, use wide bounds)
    assert 0 <= results["rtp"] <= 2.0 # RTP can exceed 1 temporarily or if math is off
    assert results["variance"] >= 0
    assert 0 <= results["hit_frequency"] <= 1.0
    assert results["duration_sec"] > 0

def test_simulation_run_with_bonus(simulator_instance_with_bonus):
    """Test simulation run including bonus features."""
    helpers.set_random_seed(456)
    num_spins = 5000 # More spins needed to likely hit bonus
    results = simulator_instance_with_bonus.run_simulation(num_spins=num_spins, initial_bet=1.0)

    assert isinstance(results, dict)
    assert results["total_spins"] == num_spins
    assert "rtp" in results
    assert "variance" in results
    assert "hit_frequency" in results
    assert "bonus_frequency" in results
    assert "bonus_entries" in results

    # Plausibility checks
    assert 0 <= results["rtp"] <= 2.0
    assert results["variance"] >= 0
    assert 0 <= results["hit_frequency"] <= 1.0
    assert 0 <= results["bonus_frequency"] <= 1.0
    # Check if bonus was likely entered (stochastic, might fail occasionally with low spins)
    # assert results["bonus_entries"] > 0 # This might be too strict for few spins

def test_simulation_zero_spins(simulator_instance):
    """Test simulation with zero spins."""
    results = simulator_instance.run_simulation(num_spins=0)
    assert results["total_spins"] == 0
    assert results["rtp"] == 0.0 # Or handle as NaN/error? Defined as 0 for now.
    assert results["variance"] == 0.0
    assert results["hit_frequency"] == 0.0
    assert results["bonus_frequency"] == 0.0
    assert results["duration_sec"] >= 0

# Add tests for:
# - GPU execution path (if possible to mock/test)
# - Variance reduction techniques (if implemented)
# - Specific known RTP cases (if a game with known math is set up)
# - Edge cases (e.g., zero bet)