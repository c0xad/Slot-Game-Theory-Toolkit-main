# -*- coding: utf-8 -*-
"""Tests for the core bonuses module."""

import pytest
from typing import List, Dict, Any
from slot_game_theory.core import reels, paytable, bonuses # Adjust import path

# --- Fixtures ---

@pytest.fixture
def symbols():
    """Common symbols fixture."""
    return {
        "A": reels.Symbol("A"),
        "K": reels.Symbol("K"),
        "SCATTER": reels.Symbol("SCATTER", is_scatter=True),
        "WILD": reels.Symbol("WILD", is_wild=True),
    }

@pytest.fixture
def base_game_state():
    """Basic GameState instance."""
    return bonuses.GameState(current_bet=1.0, balance=100.0)

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
def window_3scatter(symbols):
    """Window with 3 scatters."""
    return [
        [symbols["A"], symbols["K"], symbols["SCATTER"]],
        [symbols["WILD"], symbols["A"], symbols["SCATTER"]],
        [symbols["A"], symbols["K"], symbols["SCATTER"]],
    ]

@pytest.fixture
def trigger_3scatter(symbols):
    """Trigger condition for 3 scatters."""
    return {symbols["SCATTER"]: 3}

@pytest.fixture
def free_spins_feature(symbols, trigger_3scatter):
    """Instance of FreeSpinsFeature."""
    return bonuses.FreeSpinsFeature(
        num_spins=10,
        multiplier=2.0,
        can_retrigger=True,
        retrigger_condition=trigger_3scatter,
        retrigger_spins=5
    )

@pytest.fixture
def pick_feature():
    """Instance of PickAndWinFeature."""
    return bonuses.PickAndWinFeature(num_picks=3, possible_prizes=[5.0, 10.0, 20.0, 5.0])

# --- Test GameState ---

def test_gamestate_init(base_game_state):
    """Test GameState initialization."""
    assert base_game_state.base_bet == 1.0
    assert base_game_state.current_bet == 1.0
    assert base_game_state.balance == 100.0
    assert base_game_state.bonus_active is False
    assert base_game_state.active_bonus_feature is None
    assert base_game_state.free_spins_remaining == 0
    assert base_game_state.bonus_multiplier == 1.0
    assert base_game_state.state_data == {}

def test_gamestate_init_invalid():
    """Test GameState initialization with invalid values."""
    with pytest.raises(ValueError):
        bonuses.GameState(current_bet=0, balance=100)
    with pytest.raises(ValueError):
        bonuses.GameState(current_bet=-1, balance=100)
    with pytest.raises(TypeError):
        bonuses.GameState(current_bet=1.0, balance="abc")

def test_gamestate_enter_exit_bonus(base_game_state):
    """Test entering and exiting bonus mode."""
    assert base_game_state.bonus_active is False
    initial_data = {"level": 1}
    base_game_state.enter_bonus("TestBonus", initial_data)
    assert base_game_state.bonus_active is True
    assert base_game_state.active_bonus_feature == "TestBonus"
    assert base_game_state.state_data["level"] == 1

    # Test trying to re-enter while active (should likely do nothing or warn)
    base_game_state.enter_bonus("AnotherBonus")
    assert base_game_state.active_bonus_feature == "TestBonus" # Should not change

    base_game_state.exit_bonus()
    assert base_game_state.bonus_active is False
    assert base_game_state.active_bonus_feature is None
    assert base_game_state.state_data == {} # Data cleared
    assert base_game_state.current_bet == base_game_state.base_bet # Bet restored

    # Test exiting when not active (should likely do nothing or warn)
    base_game_state.exit_bonus()
    assert base_game_state.bonus_active is False

def test_gamestate_update_balance(base_game_state):
    """Test updating the balance."""
    base_game_state.update_balance(win_amount=10.5, cost=1.0)
    assert base_game_state.balance == pytest.approx(100.0 - 1.0 + 10.5)
    base_game_state.update_balance(win_amount=5.0) # Cost defaults to 0
    assert base_game_state.balance == pytest.approx(100.0 - 1.0 + 10.5 + 5.0)
    # Test invalid amounts (should log error and not change balance significantly)
    initial_balance = base_game_state.balance
    base_game_state.update_balance(win_amount=-5.0)
    assert base_game_state.balance == initial_balance # Negative win ignored
    base_game_state.update_balance(win_amount=10.0, cost=-1.0)
    assert base_game_state.balance == initial_balance + 10.0 # Negative cost ignored

def test_gamestate_repr(base_game_state):
    """Test the __repr__ method."""
    assert "Base Game" in repr(base_game_state)
    assert "Bet=1.0" in repr(base_game_state)
    assert "Balance=100.00" in repr(base_game_state)

    base_game_state.enter_bonus("FreeSpins")
    base_game_state.free_spins_remaining = 5
    base_game_state.bonus_multiplier = 3.0
    base_game_state.state_data["extra"] = True
    rep = repr(base_game_state)
    assert "Mode='FreeSpins'" in rep
    assert "FreeSpinsLeft=5" in rep
    assert "Multiplier=3.0x" in rep
    assert "Data={'extra': True}" in rep

# --- Test check_bonus_trigger ---

def test_check_bonus_trigger_simple_met(symbols, sample_window, trigger_3scatter):
    """Test simple trigger condition that is met."""
    # sample_window has 2 scatters, trigger needs 3 -> False
    trigger_2scatter = {symbols["SCATTER"]: 2}
    assert bonuses.check_bonus_trigger(sample_window, trigger_2scatter) == trigger_2scatter

def test_check_bonus_trigger_simple_not_met(symbols, sample_window, trigger_3scatter):
    """Test simple trigger condition that is not met."""
    # sample_window has 2 scatters, trigger needs 3 -> False
    assert bonuses.check_bonus_trigger(sample_window, trigger_3scatter) is None

def test_check_bonus_trigger_list_met(symbols, window_3scatter, trigger_3scatter):
    """Test list of triggers where one is met."""
    trigger_4A = {symbols["A"]: 4}
    trigger_list = [trigger_4A, trigger_3scatter] # Needs 4 A's OR 3 Scatters
    # window_3scatter has 3 scatters
    assert bonuses.check_bonus_trigger(window_3scatter, trigger_list) == trigger_3scatter

def test_check_bonus_trigger_list_not_met(symbols, sample_window, trigger_3scatter):
    """Test list of triggers where none are met."""
    trigger_4A = {symbols["A"]: 4}
    trigger_list = [trigger_4A, trigger_3scatter] # Needs 4 A's OR 3 Scatters
    # sample_window has 3 A's and 2 Scatters
    assert bonuses.check_bonus_trigger(sample_window, trigger_list) is None

def test_check_bonus_trigger_empty_invalid():
    """Test edge cases for check_bonus_trigger."""
    assert bonuses.check_bonus_trigger([], {}) is None
    assert bonuses.check_bonus_trigger([[]], {}) is None
    assert bonuses.check_bonus_trigger([[]], {reels.Symbol("S"): 3}) is None
    assert bonuses.check_bonus_trigger([[]], []) is None # Empty list of triggers

# --- Test FreeSpinsFeature ---

def test_free_spins_init():
    """Test FreeSpinsFeature initialization."""
    fs = bonuses.FreeSpinsFeature(num_spins=10, multiplier=2.0)
    assert fs.initial_spins == 10
    assert fs.multiplier == 2.0
    assert fs.bonus_reels is None
    assert fs.bonus_paytable is None
    assert fs.can_retrigger is False

    fs_retrigger = bonuses.FreeSpinsFeature(10, can_retrigger=True, retrigger_condition={"S":3}, retrigger_spins=5)
    assert fs_retrigger.can_retrigger is True

def test_free_spins_init_invalid():
    """Test FreeSpinsFeature initialization with invalid values."""
    with pytest.raises(ValueError): bonuses.FreeSpinsFeature(num_spins=0)
    with pytest.raises(ValueError): bonuses.FreeSpinsFeature(num_spins=10, multiplier=-1.0)
    with pytest.raises(TypeError): bonuses.FreeSpinsFeature(num_spins=10, bonus_reels="not a list")
    with pytest.raises(TypeError): bonuses.FreeSpinsFeature(num_spins=10, bonus_paytable="not a paytable")

def test_free_spins_start(base_game_state, free_spins_feature):
    """Test the start method of FreeSpinsFeature."""
    assert base_game_state.bonus_active is False
    free_spins_feature.start(base_game_state)
    assert base_game_state.bonus_active is True
    assert base_game_state.active_bonus_feature == "FreeSpins"
    assert base_game_state.free_spins_remaining == 10
    assert base_game_state.bonus_multiplier == 2.0
    assert base_game_state.current_bet == 0 # Bet set to 0 for free spins
    assert base_game_state.state_data["total_bonus_win"] == 0.0

def test_free_spins_play_step(base_game_state, free_spins_feature):
    """Test the play_step method returns correct context."""
    free_spins_feature.start(base_game_state)
    assert base_game_state.free_spins_remaining == 10

    # First step
    context = free_spins_feature.play_step(base_game_state)
    assert base_game_state.free_spins_remaining == 9
    assert isinstance(context, tuple)
    reels_ctx, paytable_ctx, multiplier_ctx = context
    assert reels_ctx is free_spins_feature.bonus_reels # Should be None in this fixture
    assert paytable_ctx is free_spins_feature.bonus_paytable # Should be None
    assert multiplier_ctx == 2.0

    # Play remaining steps
    for _ in range(9):
        free_spins_feature.play_step(base_game_state)
    assert base_game_state.free_spins_remaining == 0

    # Play step when no spins left
    context_after_end = free_spins_feature.play_step(base_game_state)
    assert context_after_end == 0.0 # Should return 0 payout

def test_free_spins_is_complete(base_game_state, free_spins_feature):
    """Test the is_complete method."""
    free_spins_feature.start(base_game_state)
    assert free_spins_feature.is_complete(base_game_state) is False
    base_game_state.free_spins_remaining = 1
    assert free_spins_feature.is_complete(base_game_state) is False
    base_game_state.free_spins_remaining = 0
    assert free_spins_feature.is_complete(base_game_state) is True
    base_game_state.free_spins_remaining = -1 # Should also be complete
    assert free_spins_feature.is_complete(base_game_state) is True

def test_free_spins_retrigger(base_game_state, free_spins_feature, window_3scatter):
    """Test the handle_retrigger method."""
    free_spins_feature.start(base_game_state)
    assert base_game_state.free_spins_remaining == 10
    # Simulate a spin result that triggers retrigger
    retriggered = free_spins_feature.handle_retrigger(base_game_state, window_3scatter)
    assert retriggered is True
    assert base_game_state.free_spins_remaining == 15 # 10 initial + 5 retrigger

    # Simulate a non-retriggering spin
    non_retrigger_window = [[reels.Symbol("A")]]*3 # Dummy window
    retriggered_again = free_spins_feature.handle_retrigger(base_game_state, non_retrigger_window)
    assert retriggered_again is False
    assert base_game_state.free_spins_remaining == 15

    # Test with retrigger disabled
    fs_no_retrigger = bonuses.FreeSpinsFeature(num_spins=10, can_retrigger=False)
    fs_no_retrigger.start(base_game_state)
    retriggered_disabled = fs_no_retrigger.handle_retrigger(base_game_state, window_3scatter)
    assert retriggered_disabled is False
    assert base_game_state.free_spins_remaining == 10


# --- Test PickAndWinFeature ---

def test_pick_feature_init():
    """Test PickAndWinFeature initialization."""
    pf = bonuses.PickAndWinFeature(num_picks=3, possible_prizes=[1, 2, 3])
    assert pf.num_picks == 3
    assert pf.possible_prizes == [1, 2, 3]

def test_pick_feature_init_invalid():
    """Test PickAndWinFeature initialization with invalid values."""
    with pytest.raises(ValueError): bonuses.PickAndWinFeature(num_picks=0, possible_prizes=[1])
    with pytest.raises(ValueError): bonuses.PickAndWinFeature(num_picks=3, possible_prizes=[])
    with pytest.raises(TypeError): bonuses.PickAndWinFeature(num_picks=3, possible_prizes=[1, "a"])

def test_pick_feature_start(base_game_state, pick_feature):
    """Test the start method of PickAndWinFeature."""
    pick_feature.start(base_game_state)
    assert base_game_state.bonus_active is True
    assert base_game_state.active_bonus_feature == "PickAndWin"
    assert base_game_state.state_data["picks_remaining"] == 3
    assert base_game_state.state_data["total_bonus_win"] == 0.0
    assert len(base_game_state.state_data["available_prizes"]) == 4
    assert base_game_state.state_data["picks_made"] == []

def test_pick_feature_play_step(base_game_state, pick_feature):
    """Test the play_step method returns direct win amount."""
    pick_feature.start(base_game_state)
    base_game_state.base_bet = 2.0 # Set base bet for multiplier calculation

    total_win_value = 0
    picks_made = []
    # Play 3 steps (picks)
    for i in range(3):
        assert pick_feature.is_complete(base_game_state) is False
        win_amount = pick_feature.play_step(base_game_state)
        assert isinstance(win_amount, float)
        assert win_amount >= 0 # Should be prize * base_bet
        assert base_game_state.state_data["picks_remaining"] == 3 - (i + 1)
        picks_made = base_game_state.state_data["picks_made"]
        total_win_value = base_game_state.state_data["total_bonus_win"] # Tracks sum of prize *values*

    assert len(picks_made) == 3
    assert total_win_value == sum(picks_made)
    assert pick_feature.is_complete(base_game_state) is True

    # Play step when no picks left
    win_after_end = pick_feature.play_step(base_game_state)
    assert win_after_end == 0.0

def test_pick_feature_is_complete(base_game_state, pick_feature):
    """Test the is_complete method."""
    pick_feature.start(base_game_state)
    assert pick_feature.is_complete(base_game_state) is False
    base_game_state.state_data["picks_remaining"] = 1
    assert pick_feature.is_complete(base_game_state) is False
    base_game_state.state_data["picks_remaining"] = 0
    assert pick_feature.is_complete(base_game_state) is True
    base_game_state.state_data["picks_remaining"] = -1
    assert pick_feature.is_complete(base_game_state) is True