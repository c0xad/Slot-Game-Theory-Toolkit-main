# -*- coding: utf-8 -*-
"""
Defines classes and functions related to slot machine bonus features.

Includes logic for:
- Triggering bonus rounds (e.g., based on scatter symbols).
- Managing game state during bonuses (free spins, multipliers).
- Defining interfaces and example implementations for bonus features.
"""
import logging
import random
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union

# Import required components
from .symbol import Symbol
from .reels import ReelStrip, Payline # Payline might not be strictly needed here, but keep for now
from .paytable import Paytable

# Configure logging
logger = logging.getLogger(__name__)

class GameState:
    """
    Represents the mutable state of the game session, including bonus status.

    Tracks current bet, balance, active bonus features, remaining spins/picks,
    multipliers, and other dynamic bonus data.
    """
    def __init__(self, current_bet: float, balance: float, **kwargs):
        if not isinstance(current_bet, (int, float)) or current_bet <= 0:
            raise ValueError("current_bet must be a positive number.")
        if not isinstance(balance, (int, float)):
             raise TypeError("balance must be numeric.")

        self.base_bet: float = current_bet # Store the bet that triggered the state (or initial bet)
        self.current_bet: float = current_bet # The effective bet for the current spin (might be 0 in free spins)
        self.balance: float = balance
        self.bonus_active: bool = False
        self.active_bonus_feature: Optional[str] = None # Name/ID of the active bonus
        self.free_spins_remaining: int = 0
        self.bonus_multiplier: float = 1.0
        # General purpose state data for complex bonuses
        self.state_data: Dict[str, Any] = kwargs.copy()
        logger.debug(f"GameState initialized: Bet={self.current_bet}, Balance={self.balance}, StateData={self.state_data}")

    def enter_bonus(self, feature_name: str, initial_data: Optional[Dict[str, Any]] = None):
        """Activates a generic bonus mode."""
        if self.bonus_active:
            logger.warning(f"Attempting to enter bonus '{feature_name}' while bonus '{self.active_bonus_feature}' is already active.")
            # Handle nested bonuses or ignore? For now, ignore re-entry.
            return
        self.bonus_active = True
        self.active_bonus_feature = feature_name
        if initial_data:
            self.state_data.update(initial_data)
        logger.info(f"Entering bonus mode: {feature_name} with data {initial_data or {}}")

    def exit_bonus(self):
        """Resets bonus state variables when a bonus feature concludes."""
        if not self.bonus_active:
            logger.warning("Attempting to exit bonus mode when not active.")
            return
        logger.info(f"Exiting bonus mode: {self.active_bonus_feature}")
        self.bonus_active = False
        self.active_bonus_feature = None
        self.free_spins_remaining = 0 # Reset common bonus counters
        self.bonus_multiplier = 1.0
        self.state_data = {} # Clear custom data on exit (or selectively clear)
        self.current_bet = self.base_bet # Restore bet for next base game spin

    def update_balance(self, win_amount: float, cost: float = 0.0):
        """Updates the player's balance, accounting for win and cost."""
        if not isinstance(win_amount, (int, float)) or win_amount < 0:
             logger.error(f"Invalid win_amount: {win_amount}")
             win_amount = 0
        if not isinstance(cost, (int, float)) or cost < 0:
             logger.error(f"Invalid cost: {cost}")
             cost = 0

        self.balance += (win_amount - cost)
        logger.debug(f"Balance updated: Win={win_amount}, Cost={cost}, New Balance={self.balance:.2f}")


    def __repr__(self) -> str:
        mode = self.active_bonus_feature if self.bonus_active else "Base Game"
        details = ""
        if self.free_spins_remaining > 0:
             details += f", FreeSpinsLeft={self.free_spins_remaining}"
        if self.bonus_multiplier != 1.0:
             details += f", Multiplier={self.bonus_multiplier}x"
        if self.state_data:
             details += f", Data={self.state_data}"

        return f"GameState(Mode='{mode}', Bet={self.current_bet}, Balance={self.balance:.2f}{details})"


# --- Bonus Trigger Logic ---

def check_bonus_trigger(
    window: List[List[Symbol]],
    trigger_conditions: Union[Dict[Symbol, int], List[Dict[Symbol, int]]] # Allow single dict or list of dicts (for multiple triggers)
) -> Optional[Dict[Symbol, int]]:
    """
    Checks if the visible window triggers a bonus based on symbol counts.

    Args:
        window: The visible grid of symbols.
        trigger_conditions: A dictionary (or list of dictionaries) where keys are
                            triggering Symbols and values are the minimum required count.
                            If a list is provided, checks if *any* condition is met.

    Returns:
        The dictionary representing the specific condition that triggered the bonus,
        or None if no trigger occurred.
    """
    if not window or not trigger_conditions:
        return None

    symbol_counts: Dict[Symbol, int] = {}
    for column in window:
        for symbol in column:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

    conditions_to_check = []
    if isinstance(trigger_conditions, dict):
        conditions_to_check.append(trigger_conditions)
    elif isinstance(trigger_conditions, list):
        conditions_to_check = trigger_conditions
    else:
        logger.error(f"Invalid trigger_conditions type: {type(trigger_conditions)}")
        return None

    for condition in conditions_to_check:
        triggered = True
        if not isinstance(condition, dict):
             logger.warning(f"Skipping invalid trigger condition item: {condition}")
             continue
        for trigger_symbol, required_count in condition.items():
            if symbol_counts.get(trigger_symbol, 0) < required_count:
                triggered = False
                break # This specific condition dict is not met
        if triggered:
            count_str = ", ".join(f"{symbol_counts.get(ts, 0)}x {ts.name}" for ts in condition)
            logger.info(f"Bonus triggered by condition {condition} (Counts: {count_str})")
            return condition # Return the specific condition that triggered

    return None # No trigger condition met


# --- Bonus Feature Interface and Implementations ---

# Define context needed by the simulation loop to run a bonus step/spin
BonusSpinContext = Tuple[Optional[List[ReelStrip]], Optional[Paytable], float] # (reels, paytable, multiplier) - Made Optional

class BonusFeature(Protocol):
    """Defines a common interface for different bonus features."""

    feature_name: str # Class attribute to identify the feature type

    def start(self, game_state: GameState, trigger_context: Any = None) -> None:
        """
        Initializes the bonus feature state within the GameState.
        Args:
            game_state: The current game state object to modify.
            trigger_context: Optional data from the trigger (e.g., triggering symbols, window).
        """
        ...

    def play_step(self, game_state: GameState) -> Union[float, BonusSpinContext]:
        """
        Executes one logical step (e.g., a spin, a pick) within the bonus.

        Args:
            game_state: The current game state, potentially modified by the step.

        Returns:
            - If the step results in a direct win amount (e.g., pick bonus), return float.
            - If the step requires a spin to be simulated by the main loop, return
              a BonusSpinContext tuple: (reels_to_use, paytable_to_use, win_multiplier).
              Reels/paytable can be None to indicate using base game components.
            - Can potentially return other types for more complex interactions.
        """
        ...

    def handle_retrigger(self, game_state: GameState, window: List[List[Symbol]]) -> bool:
        """
        Checks for and handles retriggers based on the current bonus spin window.
        Should be called by the simulation loop after a bonus spin.

        Args:
            game_state: The current game state.
            window: The symbol window resulting from the bonus spin.

        Returns:
            True if a retrigger occurred, False otherwise.
        """
        ... # Default implementation or specific logic in subclasses

    def is_complete(self, game_state: GameState) -> bool:
        """Checks if the bonus feature has concluded based on the game state."""
        ...


class FreeSpinsFeature:
    """
    A standard free spins bonus feature implementation.
    On each step, provides the context for the simulation loop to run a free spin.
    """
    feature_name: str = "FreeSpins"

    def __init__(self, num_spins: int, multiplier: float = 1.0,
                 bonus_reels: Optional[List[ReelStrip]] = None,
                 bonus_paytable: Optional[Paytable] = None,
                 can_retrigger: bool = False,
                 retrigger_condition: Optional[Union[Dict[Symbol, int], List[Dict[Symbol, int]]]] = None, # Allow list
                 retrigger_spins: int = 0):

        if not isinstance(num_spins, int) or num_spins <= 0:
            raise ValueError("num_spins must be a positive integer.")
        if not isinstance(multiplier, (int, float)) or multiplier < 0:
            raise ValueError("multiplier must be a non-negative number.")
        if bonus_reels is not None and not isinstance(bonus_reels, list):
             raise TypeError("bonus_reels must be a list of ReelStrip objects or None.")
        if bonus_paytable is not None and not isinstance(bonus_paytable, Paytable):
             raise TypeError("bonus_paytable must be a Paytable object or None.")
        if can_retrigger and not retrigger_condition:
             logger.warning("FreeSpinsFeature set to can_retrigger=True but no retrigger_condition provided.")
        if can_retrigger and retrigger_spins <= 0:
             logger.warning("FreeSpinsFeature set to can_retrigger=True but retrigger_spins is not positive.")


        self.initial_spins = num_spins
        self.multiplier = multiplier
        self.bonus_reels = bonus_reels # If None, simulation should use base game reels
        self.bonus_paytable = bonus_paytable # If None, simulation should use base game paytable
        self.can_retrigger = can_retrigger
        self.retrigger_condition = retrigger_condition
        self.retrigger_spins = retrigger_spins
        logger.debug(f"FreeSpinsFeature created: Spins={num_spins}, Multiplier={multiplier}, Retrigger={can_retrigger}")


    def start(self, game_state: GameState, trigger_context: Any = None) -> None:
        """Initializes free spins in the game state."""
        initial_data = {
            "free_spins_remaining": self.initial_spins,
            "bonus_multiplier": self.multiplier,
            "total_bonus_win": 0.0 # Track win within the bonus feature
        }
        game_state.enter_bonus(self.feature_name, initial_data)
        game_state.free_spins_remaining = self.initial_spins # Also set direct attribute for convenience
        game_state.bonus_multiplier = self.multiplier
        game_state.current_bet = 0 # Free spins typically cost nothing

    def play_step(self, game_state: GameState) -> Union[float, BonusSpinContext]:
        """Provides context for the next free spin."""
        if game_state.free_spins_remaining <= 0:
            logger.warning("play_step called on FreeSpinsFeature with no spins remaining.")
            return 0.0 # Or raise error? Return 0 payout for safety.

        game_state.free_spins_remaining -= 1
        logger.debug(f"Playing free spin. Remaining: {game_state.free_spins_remaining}")

        # Return the context needed by the simulation loop
        # The simulation loop must handle the case where bonus_reels/paytable are None
        # and default to the base game versions passed to the simulator.
        context: BonusSpinContext = (
            self.bonus_reels, # Can be None
            self.bonus_paytable, # Can be None
            game_state.bonus_multiplier # Current multiplier
        )
        return context

    def handle_retrigger(self, game_state: GameState, window: List[List[Symbol]]) -> bool:
         """Checks for and handles retriggers based on the current bonus spin window."""
         if not self.can_retrigger or not self.retrigger_condition or self.retrigger_spins <= 0:
              return False # Cannot retrigger

         trigger_met = check_bonus_trigger(window, self.retrigger_condition)
         if trigger_met:
              game_state.free_spins_remaining += self.retrigger_spins
              logger.info(f"Free spins retriggered! Added {self.retrigger_spins} spins. New total: {game_state.free_spins_remaining}")
              return True
         return False

    def is_complete(self, game_state: GameState) -> bool:
        """Checks if free spins have run out."""
        # Check the direct attribute for simplicity, assuming it's managed correctly
        return game_state.free_spins_remaining <= 0


class PickAndWinFeature:
    """Placeholder for a pick-and-win style bonus."""
    feature_name: str = "PickAndWin"

    def __init__(self, num_picks: int, possible_prizes: List[float], picks_data: Optional[List[Any]] = None):
        if not isinstance(num_picks, int) or num_picks <= 0: raise ValueError("num_picks must be positive integer.")
        if not isinstance(possible_prizes, list) or not possible_prizes: raise ValueError("possible_prizes cannot be empty list.")
        if not all(isinstance(p, (int, float)) for p in possible_prizes): raise TypeError("All items in possible_prizes must be numeric.")

        self.num_picks = num_picks
        self.possible_prizes = possible_prizes # List of potential prize multipliers or amounts
        self.picks_data = picks_data # Optional structure defining pick field, special items etc.
        logger.debug(f"PickAndWinFeature created: Picks={num_picks}, Prizes={possible_prizes}")


    def start(self, game_state: GameState, trigger_context: Any = None) -> None:
        initial_data = {
            "picks_remaining": self.num_picks,
            "total_bonus_win": 0.0,
            "available_prizes": self.possible_prizes.copy(), # Copy prizes for this instance
            "picks_made": []
        }
        # Ensure state_data exists before updating
        if not hasattr(game_state, 'state_data') or game_state.state_data is None:
             game_state.state_data = {}
        game_state.enter_bonus(self.feature_name, initial_data)

    def play_step(self, game_state: GameState) -> Union[float, BonusSpinContext]:
        """Simulates one pick."""
        if game_state.state_data.get("picks_remaining", 0) <= 0:
            logger.warning("play_step called on PickAndWinFeature with no picks remaining.")
            return 0.0

        # Simple random pick logic placeholder
        available_prizes = game_state.state_data.get("available_prizes", [])
        if not available_prizes:
             logger.error("No available prizes left in PickAndWinFeature.")
             game_state.state_data["picks_remaining"] = 0 # End bonus if prizes run out
             return 0.0

        # In a real implementation, might involve player choice simulation or complex prize reveal logic
        chosen_prize_value = random.choice(available_prizes)
        # Optional: Remove chosen prize if picks are unique (needs careful handling if prizes are not unique values)
        # Example: available_prizes.remove(chosen_prize_value) # This removes only the first occurrence

        game_state.state_data["picks_remaining"] -= 1
        # Accumulate the prize value itself in total_bonus_win
        current_total_win = game_state.state_data.get("total_bonus_win", 0.0)
        game_state.state_data["total_bonus_win"] = current_total_win + chosen_prize_value
        game_state.state_data.setdefault("picks_made", []).append(chosen_prize_value) # Use setdefault

        logger.debug(f"Pick #{self.num_picks - game_state.state_data['picks_remaining']}: Chose prize value {chosen_prize_value}. "
                     f"Picks left: {game_state.state_data['picks_remaining']}. "
                     f"Total bonus win value: {game_state.state_data['total_bonus_win']}")

        # Return the direct win amount for this pick step
        # Assume prize value is a multiplier of the base bet that triggered the bonus
        win_amount = chosen_prize_value * game_state.base_bet
        return win_amount


    def handle_retrigger(self, game_state: GameState, window: List[List[Symbol]]) -> bool:
         """Pick bonuses typically don't retrigger based on spin results."""
         return False

    def is_complete(self, game_state: GameState) -> bool:
        """Checks if picks have run out."""
        # Check state_data safely
        return game_state.state_data is None or game_state.state_data.get("picks_remaining", 0) <= 0

# TODO: Add other bonus types like HoldAndSpinFeature, TrailBonusFeature, etc.