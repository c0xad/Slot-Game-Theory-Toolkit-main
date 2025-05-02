# -*- coding: utf-8 -*-
"""Tests for the reels module."""

import pytest
from slot_game_theory.core import reels # Adjust import path as necessary

# --- Test Symbol Class ---

def test_symbol_creation():
    """Test basic symbol creation and attributes."""
    s1 = reels.Symbol("WILD", is_wild=True, value=10)
    assert s1.name == "WILD"
    assert s1.is_wild is True
    assert s1.is_scatter is False
    assert s1.value == 10
    assert repr(s1) == "Symbol(name='WILD')"

def test_symbol_equality():
    """Test symbol equality based on name."""
    s1 = reels.Symbol("A")
    s2 = reels.Symbol("A")
    s3 = reels.Symbol("K")
    assert s1 == s2
    assert s1 != s3
    assert s1 != "A" # Should not be equal to string

def test_symbol_hashing():
    """Test symbol hashing for use in sets/dicts."""
    s1 = reels.Symbol("A")
    s2 = reels.Symbol("A")
    s3 = reels.Symbol("K")
    symbol_set = {s1, s2, s3}
    assert len(symbol_set) == 2 # s1 and s2 should hash to the same value
    assert s1 in symbol_set
    assert s3 in symbol_set

def test_symbol_creation_invalid():
    """Test Symbol creation with invalid parameters."""
    with pytest.raises(ValueError, match="Symbol name must be a non-empty string"):
        reels.Symbol("") # Empty name
    with pytest.raises(TypeError, match="Symbol value must be numeric"):
        reels.Symbol("A", value="not a number")
    with pytest.raises(TypeError, match="Symbol is_wild must be a boolean"):
        reels.Symbol("A", is_wild="True")
    with pytest.raises(TypeError, match="Symbol is_scatter must be a boolean"):
        reels.Symbol("A", is_scatter=1)

def test_symbol_str():
    """Test the __str__ method of Symbol."""
    s = reels.Symbol("WILD", is_wild=True)
    assert str(s) == "WILD"

def test_symbol_properties():
    """Test accessing custom properties via kwargs."""
    s = reels.Symbol("BONUS", is_scatter=True, trigger_value=5, color="red")
    assert s.properties.get("trigger_value") == 5
    assert s.properties.get("color") == "red"
    assert "Props={'trigger_value': 5, 'color': 'red'}" in repr(s)

# --- Test ReelStrip Class ---

@pytest.fixture
def sample_symbols():
    """Fixture for common symbols."""
    return [reels.Symbol("A"), reels.Symbol("K"), reels.Symbol("Q"), reels.Symbol("J")]

@pytest.fixture
def sample_reel_strip(sample_symbols):
    """Fixture for a sample reel strip."""
    return reels.ReelStrip(sample_symbols, name="TestReel")

def test_reel_strip_creation(sample_reel_strip, sample_symbols):
    """Test reel strip creation and basic properties."""
    assert sample_reel_strip.name == "TestReel"
    assert len(sample_reel_strip) == 4
    assert sample_reel_strip.length == 4
    assert sample_reel_strip.symbols == sample_symbols
    assert "TestReel" in repr(sample_reel_strip)
    assert "length=4" in repr(sample_reel_strip)

def test_reel_strip_creation_with_strings():
    """Test reel strip creation directly from strings."""
    strip = reels.ReelStrip(["A", "K", "A"])
    assert len(strip) == 3
    assert strip.symbols[0] == reels.Symbol("A")
    assert strip.symbols[1] == reels.Symbol("K")
    assert strip.symbols[2] == reels.Symbol("A")

def test_reel_strip_get_symbol_at(sample_reel_strip, sample_symbols):
    """Test getting symbols with wrap-around."""
    assert sample_reel_strip.get_symbol_at(0) == sample_symbols[0] # A
    assert sample_reel_strip.get_symbol_at(3) == sample_symbols[3] # J
    assert sample_reel_strip.get_symbol_at(4) == sample_symbols[0] # Wrap-around to A
    assert sample_reel_strip.get_symbol_at(7) == sample_symbols[3] # Wrap-around to J
    assert sample_reel_strip.get_symbol_at(-1) == sample_symbols[3] # Negative index wrap-around

def test_reel_strip_creation_invalid():
    """Test ReelStrip creation with invalid parameters."""
    with pytest.raises(ValueError, match="Reel strip symbols sequence cannot be empty"):
        reels.ReelStrip([]) # Empty sequence
    with pytest.raises(TypeError, match="ReelStrip name must be a string"):
        reels.ReelStrip(["A"], name=123) # Invalid name type
    with pytest.raises(TypeError, match="Invalid type in symbols sequence"):
        reels.ReelStrip(["A", 10, "K"]) # Invalid type in sequence

# --- Test Payline Functions ---

def test_get_standard_paylines():
    """Test generation of standard paylines."""
    # 3x3 grid
    paylines_3x3 = reels.get_standard_paylines(num_reels=3, num_rows=3)
    assert "line_0" in paylines_3x3 # Updated name
    assert "line_1" in paylines_3x3 # Updated name
    assert "line_2" in paylines_3x3 # Updated name
    assert "diag_down" in paylines_3x3
    assert "diag_up" in paylines_3x3
    assert paylines_3x3["line_1"] == [(0, 1), (1, 1), (2, 1)] # Check line_1 (old mid)
    assert paylines_3x3["diag_down"] == [(0, 0), (1, 1), (2, 2)]
    assert paylines_3x3["diag_up"] == [(0, 2), (1, 1), (2, 0)]

    # 5x3 grid
    paylines_5x3 = reels.get_standard_paylines(num_reels=5, num_rows=3)
    assert len(paylines_5x3) == 5 # line_0, line_1, line_2, DiagDown, DiagUp
    assert paylines_5x3["line_0"] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)] # Check line_0 (old top)
    assert paylines_5x3["diag_down"] == [(0, 0), (1, 1), (2, 2)] # Only 3 elements deep

def test_get_standard_paylines_invalid_size():
    """Test payline generation with invalid grid size."""
    assert reels.get_standard_paylines(num_reels=0, num_rows=3) == {}
    assert reels.get_standard_paylines(num_reels=3, num_rows=0) == {}
    assert reels.get_standard_paylines(num_reels=-1, num_rows=3) == {}

# --- Test Stop Generation and Window ---

@pytest.fixture
def multi_reel_strips():
    """Fixture for multiple reel strips."""
    s = [reels.Symbol(c) for c in "AKQJ7"]
    r1 = reels.ReelStrip([s[0], s[1], s[2]], name="R1") # AKQ (len 3)
    r2 = reels.ReelStrip([s[1], s[2], s[3], s[4]], name="R2") # KQJ7 (len 4)
    r3 = reels.ReelStrip([s[4], s[0]], name="R3") # 7A (len 2)
    return [r1, r2, r3]

def test_generate_random_stops(multi_reel_strips):
    """Test random stop generation (basic check)."""
    stops = reels.generate_random_stops(multi_reel_strips)
    assert len(stops) == 3
    assert 0 <= stops[0] < 3
    assert 0 <= stops[1] < 4
    assert 0 <= stops[2] < 2

def test_generate_random_stops_rng(multi_reel_strips):
    """Test random stop generation with a seeded RNG."""
    import random
    rng1 = random.Random(12345)
    rng2 = random.Random(12345)
    stops1 = reels.generate_random_stops(multi_reel_strips, rng=rng1)
    stops2 = reels.generate_random_stops(multi_reel_strips, rng=rng2)
    assert stops1 == stops2 # Should be reproducible with same seed

def test_generate_random_stops_empty():
    """Test random stop generation with empty input."""
    assert reels.generate_random_stops([]) == []

def test_get_visible_window(multi_reel_strips):
    """Test calculating the visible window based on stops."""
    # R1=[A,K,Q], R2=[K,Q,J,7], R3=[7,A]
    # Stops: R1 at index 1 (K), R2 at index 2 (J), R3 at index 0 (7)
    stops = [1, 2, 0]
    num_rows = 3

    window = reels.get_visible_window(multi_reel_strips, stops, num_rows)

    # Expected window (columns):
    # Reel 1: K (stop 1), Q (stop 1+1=2), A (stop 1+2=3 -> wrap 0)
    # Reel 2: J (stop 2), 7 (stop 2+1=3), K (stop 2+2=4 -> wrap 0)
    # Reel 3: 7 (stop 0), A (stop 0+1=1), 7 (stop 0+2=2 -> wrap 0)

    assert len(window) == 3 # 3 columns (reels)

    # Check Reel 1 column
    assert len(window[0]) == 3
    assert window[0][0].name == "K" # Row 0
    assert window[0][1].name == "Q" # Row 1
    assert window[0][2].name == "A" # Row 2

    # Check Reel 2 column
    assert len(window[1]) == 3
    assert window[1][0].name == "J" # Row 0
    assert window[1][1].name == "7" # Row 1
    assert window[1][2].name == "K" # Row 2

    # Check Reel 3 column
    assert len(window[2]) == 3
    assert window[2][0].name == "7" # Row 0
    assert window[2][1].name == "A" # Row 1
    assert window[2][2].name == "7" # Row 2

def test_get_visible_window_invalid_input(multi_reel_strips):
    """Test get_visible_window with invalid inputs."""
    # Mismatched stops length
    with pytest.raises(ValueError, match="Length of stops"):
        reels.get_visible_window(multi_reel_strips, stops=[1, 2], num_rows=3)
    # Non-positive num_rows
    with pytest.raises(ValueError, match="Number of rows"):
        reels.get_visible_window(multi_reel_strips, stops=[1, 2, 0], num_rows=0)
    with pytest.raises(ValueError, match="Number of rows"):
        reels.get_visible_window(multi_reel_strips, stops=[1, 2, 0], num_rows=-1)

def test_get_visible_window_edge_cases(multi_reel_strips):
    """Test get_visible_window with edge cases like invalid stops."""
    # Invalid stop index (too high)
    window_invalid_stop = reels.get_visible_window(multi_reel_strips, stops=[1, 10, 0], num_rows=3)
    assert window_invalid_stop[1][0].name == "ERROR" # Check placeholder symbol

    # Empty reel strip (create a modified fixture)
    empty_strip = reels.ReelStrip([reels.Symbol("X")], name="Temp") # Need one symbol to init
    empty_strip.symbols = [] # Force empty
    empty_strip.length = 0
    strips_with_empty = [multi_reel_strips[0], empty_strip, multi_reel_strips[2]]
    window_empty_strip = reels.get_visible_window(strips_with_empty, stops=[1, 0, 0], num_rows=3)
    assert window_empty_strip[1][0].name == "EMPTY" # Check placeholder symbol