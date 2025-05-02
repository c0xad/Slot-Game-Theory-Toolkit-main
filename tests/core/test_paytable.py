# -*- coding: utf-8 -*-
"""Tests for the paytable module."""

import pytest
from slot_game_theory.core import paytable, reels # Adjust import path

# --- Fixtures ---

@pytest.fixture
def symbols():
    """Common symbols."""
    return {
        "A": reels.Symbol("A"),
        "K": reels.Symbol("K"),
        "WILD": reels.Symbol("WILD", is_wild=True),
        "SCATTER": reels.Symbol("SCATTER", is_scatter=True),
    }

@pytest.fixture
def sample_paytable(symbols):
    """A sample paytable for testing."""
    win_conditions = {
        # Simple conditions (Symbol, Count) -> Payout Multiplier
        (symbols["A"], 3): 50,
        (symbols["A"], 2): 5,
        (symbols["K"], 3): 20,
        # Detailed conditions (Symbol, Count, Direction/Type) -> Payout Multiplier
        (symbols["WILD"], 3, 'left_to_right'): 100, # Wild line pays differently
        (symbols["SCATTER"], 3, 'scatter'): 10, # 3 scatters anywhere pay 10x total bet
        (symbols["SCATTER"], 2, 'scatter'): 2,  # 2 scatters anywhere pay 2x total bet
    }
    return paytable.Paytable(win_conditions)

@pytest.fixture
def sample_window(symbols):
    """A sample 3x3 window."""
    # Reel 1: A, K, A
    # Reel 2: WILD, A, SCATTER
    # Reel 3: A, K, SCATTER
    return [
        [symbols["A"], symbols["K"], symbols["A"]],
        [symbols["WILD"], symbols["A"], symbols["SCATTER"]],
        [symbols["A"], symbols["K"], symbols["SCATTER"]],
    ]

@pytest.fixture
def sample_paylines():
    """Sample paylines for a 3x3 grid."""
    return {
        "line_mid": [(0, 1), (1, 1), (2, 1)], # K, A, K
        "line_top": [(0, 0), (1, 0), (2, 0)], # A, WILD, A
        "line_bot": [(0, 2), (1, 2), (2, 2)], # A, SCATTER, SCATTER
        "diag_down": [(0, 0), (1, 1), (2, 2)], # A, A, SCATTER
    }

# --- Test Paytable Class ---

def test_paytable_creation(sample_paytable):
    """Test basic paytable creation."""
    assert isinstance(sample_paytable, paytable.Paytable)
    assert len(sample_paytable.win_payouts) == 6

def test_paytable_get_payout(sample_paytable, symbols):
    """Test retrieving payouts from the paytable."""
    # Exact matches
    assert sample_paytable.get_payout(symbols["A"], 3) == 50
    assert sample_paytable.get_payout(symbols["A"], 2) == 5
    assert sample_paytable.get_payout(symbols["K"], 3) == 20
    assert sample_paytable.get_payout(symbols["WILD"], 3, 'left_to_right') == 100
    assert sample_paytable.get_payout(symbols["SCATTER"], 3, 'scatter') == 10
    assert sample_paytable.get_payout(symbols["SCATTER"], 2, 'scatter') == 2

    # Non-winning combinations
    assert sample_paytable.get_payout(symbols["K"], 2) == 0
    assert sample_paytable.get_payout(symbols["A"], 4) == 0 # Count too high
    assert sample_paytable.get_payout(symbols["WILD"], 3) == 0 # Requires direction
    assert sample_paytable.get_payout(symbols["SCATTER"], 3) == 0 # Requires type 'scatter'

def test_paytable_invalid_payout():
    """Test that invalid payout values raise errors."""
    with pytest.raises(ValueError):
        paytable.Paytable({(reels.Symbol("X"), 3): -10})

# --- Test Win Evaluation Functions ---

def test_evaluate_payline_win_simple(sample_paytable, symbols):
    """Test simple payline win evaluation (no wilds)."""
    # Line: K, K, K (should not exist in window, but test logic)
    line_symbols = [symbols["K"], symbols["K"], symbols["K"]]
    line_coords = [(0,0), (1,0), (2,0)] # Dummy coords
    # Create a dummy window for this specific test
    window = [[symbols["K"]], [symbols["K"]], [symbols["K"]]]

    # Need to mock get_visible_window or pass symbols directly?
    # Let's test the logic by constructing line_symbols directly for now.
    # This requires refactoring evaluate_payline_win or testing it via evaluate_all_wins.

    # Re-thinking: Test evaluate_all_wins which uses evaluate_payline_win internally.
    pass # See tests below for evaluate_all_wins

def test_evaluate_payline_win_with_wilds(sample_paytable, symbols, sample_window, sample_paylines):
    """Test payline evaluation involving wild symbols."""
    # line_top: A, WILD, A -> Should count as 3 'A's
    win, _, _ = paytable.evaluate_all_wins(sample_window, sample_paylines, sample_paytable, 1.0, 3.0)
    # Check specific line payout (needs modification to evaluate_all_wins to return details easily)
    # For now, check if the total win includes the expected line win.
    # Expected: 3xA -> 50x bet_per_line = 50
    # Let's manually call evaluate_payline_win for this line
    line_top_win = paytable.evaluate_payline_win(sample_paylines["line_top"], sample_window, sample_paytable, 1.0)
    assert line_top_win == 50.0

    # Test wild line: WILD, WILD, WILD (if it existed)
    wild_window = [
        [symbols["WILD"]],
        [symbols["WILD"]],
        [symbols["WILD"]],
    ]
    wild_line = [(0,0), (1,0), (2,0)]
    wild_line_win = paytable.evaluate_payline_win(wild_line, wild_window, sample_paytable, 1.0)
    assert wild_line_win == 100.0 # Specific payout for 3 wilds

def test_evaluate_scatter_wins(sample_paytable, symbols, sample_window):
    """Test scatter win evaluation."""
    # Window has SCATTER at (1, 2) and (2, 2) -> 2 scatters total
    scatter_win = paytable.evaluate_scatter_wins(sample_window, sample_paytable, total_bet=3.0)
    # Expected: 2 scatters pay 2x total bet = 2 * 3.0 = 6.0
    assert scatter_win == 6.0

    # Test with 3 scatters
    window_3scatter = [
        [symbols["A"], symbols["K"], symbols["SCATTER"]],
        [symbols["WILD"], symbols["A"], symbols["SCATTER"]],
        [symbols["A"], symbols["K"], symbols["SCATTER"]],
    ]
    scatter_win_3 = paytable.evaluate_scatter_wins(window_3scatter, sample_paytable, total_bet=5.0)
    # Expected: 3 scatters pay 10x total bet = 10 * 5.0 = 50.0
    assert scatter_win_3 == 50.0

def test_evaluate_all_wins(sample_paytable, symbols, sample_window, sample_paylines):
    """Test the combined evaluation of all wins."""
    bet_per_line = 1.0
    total_bet = bet_per_line * len(sample_paylines) # Assume total bet = lines * bet_per_line

    total_win, line_details, scatter_win = paytable.evaluate_all_wins(
        sample_window, sample_paylines, sample_paytable, bet_per_line, total_bet
    )

    # Expected Line Wins (bet_per_line = 1.0):
    # line_mid: K, A, K -> No win (0)
    # line_top: A, WILD, A -> 3xA win (50)
    # line_bot: A, SCATTER, SCATTER -> No win (0)
    # diag_down: A, A, SCATTER -> 2xA win (5)
    # Total Line Win = 50 + 5 = 55

    # Expected Scatter Wins (total_bet = 4.0):
    # 2 Scatters -> 2x total_bet = 2 * 4.0 = 8.0

    # Total Expected Win = 55 + 8.0 = 63.0

    assert line_details.get("line_top", 0) == 50.0
    assert line_details.get("diag_down", 0) == 5.0
    assert len(line_details) == 2 # Only winning lines included

    assert scatter_win == 8.0
    assert total_win == 63.0
@pytest.fixture
def paytable_with_rtl_wild(symbols):
    """Paytable including RTL wins and specific WILD payouts."""
    win_conditions = {
        # LTR
        (symbols["A"], 3, 'left_to_right'): 50,
        (symbols["A"], 2, 'left_to_right'): 5,
        (symbols["K"], 3, 'left_to_right'): 20,
        (symbols["WILD"], 3, 'left_to_right'): 100, # 3 Wilds LTR
        # RTL
        (symbols["A"], 3, 'right_to_left'): 40, # Different payout for RTL
        (symbols["A"], 2, 'right_to_left'): 4,
        (symbols["K"], 3, 'right_to_left'): 15,
        (symbols["WILD"], 3, 'right_to_left'): 90, # Different payout for 3 Wilds RTL
        # Scatter
        (symbols["SCATTER"], 3, 'scatter'): 10,
        (symbols["SCATTER"], 2, 'scatter'): 2,
    }
    return paytable.Paytable(win_conditions)

@pytest.fixture
def window_rtl(symbols):
    """Window designed for RTL tests."""
    # Reel 1: K, A, K
    # Reel 2: WILD, A, WILD
    # Reel 3: A, K, A
    return [
        [symbols["K"], symbols["A"], symbols["K"]],
        [symbols["WILD"], symbols["A"], symbols["WILD"]],
        [symbols["A"], symbols["K"], symbols["A"]],
    ]

@pytest.fixture
def window_wild_ltr(symbols):
    """Window for testing LTR wild logic."""
    # Reel 1: WILD, K, A
    # Reel 2: WILD, A, WILD
    # Reel 3: WILD, K, A
    return [
        [symbols["WILD"], symbols["K"], symbols["A"]],
        [symbols["WILD"], symbols["A"], symbols["WILD"]],
        [symbols["WILD"], symbols["K"], symbols["A"]],
    ]

@pytest.fixture
def window_wild_rtl(symbols):
    """Window for testing RTL wild logic."""
    # Reel 1: A, K, WILD
    # Reel 2: WILD, A, WILD
    # Reel 3: K, K, WILD
    return [
        [symbols["A"], symbols["K"], symbols["WILD"]],
        [symbols["WILD"], symbols["A"], symbols["WILD"]],
        [symbols["K"], symbols["K"], symbols["WILD"]],
    ]


def test_evaluate_payline_win_right_to_left(paytable_with_rtl_wild, symbols, window_rtl, sample_paylines):
    """Test right-to-left payline evaluation."""
    # line_mid: A, A, K -> 2xA RTL (4.0)
    win_mid = paytable.evaluate_payline_win(sample_paylines["line_mid"], window_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_mid == 4.0

    # line_top: K, WILD, A -> 3xA RTL (40.0)
    win_top = paytable.evaluate_payline_win(sample_paylines["line_top"], window_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_top == 40.0

    # line_bot: K, WILD, A -> 3xA RTL (40.0)
    win_bot = paytable.evaluate_payline_win(sample_paylines["line_bot"], window_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_bot == 40.0

    # diag_down: K, A, A -> 2xA RTL (4.0)
    win_diag = paytable.evaluate_payline_win(sample_paylines["diag_down"], window_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_diag == 4.0

    # Test LTR on same window (should be different or zero)
    # line_top: K, WILD, A -> No LTR win defined for K
    win_top_ltr = paytable.evaluate_payline_win(sample_paylines["line_top"], window_rtl, paytable_with_rtl_wild, 1.0, direction='left_to_right')
    assert win_top_ltr == 0.0


def test_evaluate_payline_win_refined_wilds(paytable_with_rtl_wild, symbols, window_wild_ltr, window_wild_rtl, sample_paylines):
    """Test refined wild handling (all-wild vs substitution)."""
    # --- LTR Tests (using window_wild_ltr) ---
    # line_top: WILD, WILD, WILD -> Should pay 100 (specific WILD LTR payout)
    win_top_ltr = paytable.evaluate_payline_win(sample_paylines["line_top"], window_wild_ltr, paytable_with_rtl_wild, 1.0, direction='left_to_right')
    assert win_top_ltr == 100.0

    # line_mid: K, A, K -> No LTR win
    win_mid_ltr = paytable.evaluate_payline_win(sample_paylines["line_mid"], window_wild_ltr, paytable_with_rtl_wild, 1.0, direction='left_to_right')
    assert win_mid_ltr == 0.0

    # line_bot: A, WILD, A -> Should pay as 3xA LTR (50)
    win_bot_ltr = paytable.evaluate_payline_win(sample_paylines["line_bot"], window_wild_ltr, paytable_with_rtl_wild, 1.0, direction='left_to_right')
    assert win_bot_ltr == 50.0

    # --- RTL Tests (using window_wild_rtl) ---
    # line_top: A, WILD, K -> No RTL win
    win_top_rtl = paytable.evaluate_payline_win(sample_paylines["line_top"], window_wild_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_top_rtl == 0.0

    # line_mid: K, A, K -> No RTL win
    win_mid_rtl = paytable.evaluate_payline_win(sample_paylines["line_mid"], window_wild_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_mid_rtl == 0.0

    # line_bot: WILD, WILD, WILD -> Should pay 90 (specific WILD RTL payout)
    win_bot_rtl = paytable.evaluate_payline_win(sample_paylines["line_bot"], window_wild_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_bot_rtl == 90.0

    # diag_down: A, A, WILD -> Should pay as 3xA RTL (40)
    win_diag_rtl = paytable.evaluate_payline_win(sample_paylines["diag_down"], window_wild_rtl, paytable_with_rtl_wild, 1.0, direction='right_to_left')
    assert win_diag_rtl == 40.0