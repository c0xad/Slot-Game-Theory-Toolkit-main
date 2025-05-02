# -*- coding: utf-8 -*-
"""Tests for the symbolic evaluation module."""

import pytest
import numpy as np
from slot_game_theory.core import reels, paytable
from slot_game_theory.evaluation import symbolic # Adjust import path

# --- Fixtures for a Simple Test Case ---

@pytest.fixture
def simple_symbols():
    """Very simple symbol set."""
    return {
        "A": reels.Symbol("A"),
        "B": reels.Symbol("B"),
    }

@pytest.fixture
def simple_reels(simple_symbols):
    """Very simple 2x2 reel setup."""
    # R1: A, B
    # R2: A, B
    r1 = reels.ReelStrip([simple_symbols["A"], simple_symbols["B"]])
    r2 = reels.ReelStrip([simple_symbols["A"], simple_symbols["B"]])
    return [r1, r2]

@pytest.fixture
def simple_paylines():
    """Single middle payline for 2x2."""
    # Window rows are 0, 1. Middle line is row 0? Or row 1? Let's use row 0.
    return {"line_top": [(0, 0), (1, 0)]} # Top row

@pytest.fixture
def simple_paytable(simple_symbols):
    """Very simple paytable."""
    # 2xA pays 10
    # 2xB pays 5
    win_conditions = {
        (simple_symbols["A"], 2): 10.0,
        (simple_symbols["B"], 2): 5.0,
    }
    return paytable.Paytable(win_conditions)

# --- Test Symbolic Calculation ---

def test_calculate_metrics_classic_symbolic(
    simple_reels, simple_paylines, simple_paytable, simple_symbols
):
    """Test the symbolic calculation on a minimal example."""
    num_rows = 2 # Window height
    bet_per_line = 1.0
    total_bet = 1.0 # Only one line

    # Expected Outcomes (Stops: [R1_idx, R2_idx]):
    # [0, 0]: Stop R1=A, R2=A -> Window=[[A,B],[A,B]] -> Line0=[A,A] -> Win=10
    # [0, 1]: Stop R1=A, R2=B -> Window=[[A,B],[B,A]] -> Line0=[A,B] -> Win=0
    # [1, 0]: Stop R1=B, R2=A -> Window=[[B,A],[A,B]] -> Line0=[B,A] -> Win=0
    # [1, 1]: Stop R1=B, R2=B -> Window=[[B,A],[B,A]] -> Line0=[B,B] -> Win=5
    # Total Combinations = 2 * 2 = 4

    # Total Payout = 10 + 0 + 0 + 5 = 15
    # Expected Payout = 15 / 4 = 3.75
    # Expected RTP = Expected Payout / Total Bet = 3.75 / 1.0 = 3.75 (This seems high, check logic)
    # Let's re-check window calculation: get_visible_window(reels, stops, num_rows)
    # stop_index = stops[r_idx]
    # symbol_index_on_strip = (stop_index + row_idx) % len(strip)
    # Window is List[ReelColumn], where ReelColumn is List[Symbol]

    # Stops [0, 0]: R1 stop A, R2 stop A
    #   R1 Col: strip1[(0+0)%2]=A, strip1[(0+1)%2]=B -> [A, B]
    #   R2 Col: strip2[(0+0)%2]=A, strip2[(0+1)%2]=B -> [A, B]
    #   Window = [[A, B], [A, B]]
    #   Line Top (0,0), (1,0): Window[0][0]=A, Window[1][0]=A -> 2xA -> Win=10
    # Stops [0, 1]: R1 stop A, R2 stop B
    #   R1 Col: [A, B]
    #   R2 Col: strip2[(1+0)%2]=B, strip2[(1+1)%2]=A -> [B, A]
    #   Window = [[A, B], [B, A]]
    #   Line Top (0,0), (1,0): Window[0][0]=A, Window[1][0]=B -> No win -> Win=0
    # Stops [1, 0]: R1 stop B, R2 stop A
    #   R1 Col: strip1[(1+0)%2]=B, strip1[(1+1)%2]=A -> [B, A]
    #   R2 Col: [A, B]
    #   Window = [[B, A], [A, B]]
    #   Line Top (0,0), (1,0): Window[0][0]=B, Window[1][0]=A -> No win -> Win=0
    # Stops [1, 1]: R1 stop B, R2 stop B
    #   R1 Col: [B, A]
    #   R2 Col: [B, A]
    #   Window = [[B, A], [B, A]]
    #   Line Top (0,0), (1,0): Window[0][0]=B, Window[1][0]=B -> 2xB -> Win=5

    # Calculation seems correct. Total Payout = 15. Expected Payout = 3.75. RTP = 3.75.
    # This RTP > 1.0 is possible if the paytable is generous relative to combinations.

    # Variance Calculation:
    # Payouts: [10, 0, 0, 5]
    # E[Payout^2] = (10^2 + 0^2 + 0^2 + 5^2) / 4 = (100 + 0 + 0 + 25) / 4 = 125 / 4 = 31.25
    # Var(Payout) = E[Payout^2] - (E[Payout])^2 = 31.25 - (3.75)^2
    # Var(Payout) = 31.25 - 14.0625 = 17.1875

    # Hit Frequency Calculation:
    # Winning spins = 2 (payouts 10 and 5)
    # Hit Freq = 2 / 4 = 0.5

    metrics = symbolic.calculate_metrics_classic_symbolic(
        reel_strips=simple_reels,
        paylines=simple_paylines,
        paytable=simple_paytable,
        num_rows=num_rows,
        bet_per_line=bet_per_line,
        total_bet=total_bet
    )

    assert metrics["total_combinations"] == 4
    assert metrics["expected_payout"] == pytest.approx(3.75)
    assert metrics["rtp"] == pytest.approx(3.75)
    assert metrics["variance"] == pytest.approx(17.1875)
    assert metrics["hit_frequency"] == pytest.approx(0.5)

def test_symbolic_zero_bet():
    """Test that symbolic calculation handles zero bet correctly."""
    # Use fixtures from previous test
    fixtures = test_calculate_metrics_classic_symbolic.__pytest_fixtures__
    simple_reels = fixtures['simple_reels']
    simple_paylines = fixtures['simple_paylines']
    simple_paytable = fixtures['simple_paytable']

    with pytest.raises(ValueError, match="Total bet must be positive"):
        symbolic.calculate_metrics_classic_symbolic(
            reel_strips=simple_reels,
            paylines=simple_paylines,
            paytable=simple_paytable,
            num_rows=2,
            bet_per_line=0,
            total_bet=0
        )

def test_symbolic_no_combinations(simple_paylines, simple_paytable):
    """Test symbolic calculation with empty reels."""
    metrics = symbolic.calculate_metrics_classic_symbolic(
        reel_strips=[], # No reels
        paylines=simple_paylines,
        paytable=simple_paytable,
        num_rows=2,
        bet_per_line=1.0
    )
    assert metrics["total_combinations"] == 0
    assert metrics["rtp"] == 0.0
    assert metrics["variance"] == 0.0
    assert metrics["hit_frequency"] == 0.0

# Add more tests:
# - Test with scatter symbols
# - Test with wild symbols
# - Test with different payline configurations
# - Test edge cases (e.g., paytable with no winning lines)