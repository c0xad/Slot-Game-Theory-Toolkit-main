# -*- coding: utf-8 -*-
"""
Core game definitions and mechanics for the Slot Game Theory Toolkit.

This package provides the fundamental building blocks for defining and simulating
slot machine games.
"""

# Import key classes for easier access
from .symbol import Symbol
from .reels import ReelStrip, Payline, generate_random_stops, get_visible_window, get_standard_paylines
from .paytable import Paytable, evaluate_all_wins, evaluate_payline_win, evaluate_scatter_wins
from .bonuses import GameState, BonusFeature, FreeSpinsFeature, PickAndWinFeature, check_bonus_trigger, BonusSpinContext
from .game import SlotGame, SpinResult

# Optionally import markov if it's considered part of the core user-facing API
# from . import markov

# Define what gets imported with "from slot_game_theory.core import *"
__all__ = [
    # Symbol
    "Symbol",
    # Reels
    "ReelStrip",
    "Payline",
    "generate_random_stops",
    "get_visible_window",
    "get_standard_paylines",
    # Paytable & Evaluation
    "Paytable",
    "evaluate_all_wins",
    "evaluate_payline_win",
    "evaluate_scatter_wins",
    # Bonuses & State
    "GameState",
    "BonusFeature", # Protocol
    "FreeSpinsFeature",
    "PickAndWinFeature",
    "check_bonus_trigger",
    "BonusSpinContext",
    # Game Orchestration
    "SlotGame",
    "SpinResult",
    # Markov (Optional)
    # "markov",
]