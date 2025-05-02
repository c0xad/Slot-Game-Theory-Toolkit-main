"""
Defines the main SlotGame class orchestrating the game mechanics.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional, Sequence, Any, Union

from .symbol import Symbol
from .reels import ReelStrip, Payline, get_visible_window, generate_random_stops, get_standard_paylines
from .paytable import Paytable, evaluate_all_wins, evaluate_scatter_wins # Import scatter eval separately
from .bonuses import (
    GameState,
    BonusFeature, # Protocol
    FreeSpinsFeature, # Example implementation
    check_bonus_trigger,
    BonusSpinContext
)

logger = logging.getLogger(__name__)


class SpinResult:
    """
    Holds the detailed results of a single slot game spin (base or bonus).
    """
    def __init__(self,
                 window: Optional[List[List[Symbol]]], # Can be None for non-spin bonus steps
                 stops: Optional[List[int]], # Can be None for non-spin bonus steps
                 total_win: float,
                 line_wins: Dict[str, float],
                 scatter_wins: float,
                 bonus_win: float, # Win specifically from a bonus step/feature
                 is_bonus_step: bool,
                 active_bonus_feature: Optional[str],
                 bet_per_line: float,
                 total_bet: float, # Actual cost deducted for this step
                 triggering_feature: Optional[BonusFeature] = None # Feature triggered this spin
                 ):
        self.window = window
        self.stops = stops
        self.total_win = total_win # Overall win for this step (line + scatter + bonus)
        self.line_wins = line_wins
        self.scatter_wins = scatter_wins
        self.bonus_win = bonus_win # Portion of total_win attributed purely to bonus mechanic
        self.is_bonus_step = is_bonus_step
        self.active_bonus_feature = active_bonus_feature
        self.bet_per_line = bet_per_line
        self.total_bet = total_bet # The cost incurred for this step
        self.triggering_feature = triggering_feature # The bonus feature activated by this spin (if any)

    def __repr__(self) -> str:
        win_details = f"(Line: {sum(self.line_wins.values()):.2f}, Scatter: {self.scatter_wins:.2f}, Bonus: {self.bonus_win:.2f})"
        state = f"Bonus Step ({self.active_bonus_feature})" if self.is_bonus_step else "Base Game"
        trigger_info = f" -> Triggered: {self.triggering_feature.feature_name}" if self.triggering_feature else ""

        if self.window:
            window_str = "\n".join([" | ".join(map(str, row)) for row in self._transpose_window()])
            display = (
                f"SpinResult ({state}{trigger_info}):\n"
                f" Window:\n{window_str}\n"
                f" Stops: {self.stops}\n"
                f" Total Win: {self.total_win:.2f} {win_details}\n"
                f" Cost: {self.total_bet:.2f}"
            )
        else:
            # Display for non-spin bonus steps (e.g., pick win)
             display = (
                f"SpinResult ({state}{trigger_info}):\n"
                f" Type: Non-spin step\n"
                f" Total Win: {self.total_win:.2f} {win_details}\n"
                f" Cost: {self.total_bet:.2f}"
            )
        return display

    def _transpose_window(self) -> List[List[Symbol]]:
        """Transposes the window for easier horizontal display."""
        if not self.window or not self.window[0]:
            return []
        num_rows = len(self.window[0])
        num_reels = len(self.window)
        return [[self.window[reel][row] for reel in range(num_reels)] for row in range(num_rows)]


class SlotGame:
    """
    Represents a complete slot game configuration.
    Note: This class holds the *configuration*. Runtime state (balance, bonus state)
          is managed by a GameState object passed to simulation methods.
    """
    def __init__(self,
                 reel_strips: List[ReelStrip],
                 paytable: Paytable,
                 num_rows: int,
                 paylines: Optional[Dict[str, Payline]] = None,
                 bonus_features: Optional[List[BonusFeature]] = None,
                 # Trigger mapping: Define how symbols trigger specific features
                 bonus_triggers: Optional[Dict[BonusFeature, Union[Dict[Symbol, int], List[Dict[Symbol, int]]]]] = None,
                 rng_seed: Optional[int] = None):
        """
        Initializes the SlotGame Configuration.

        Args:
            reel_strips: List of ReelStrip objects defining the base game reels.
            paytable: The Paytable instance for the base game.
            num_rows: The number of visible rows in the game window.
            paylines: Dictionary of paylines for the base game.
            bonus_features: List of configured BonusFeature objects available in the game.
            bonus_triggers: Mapping defining how features are triggered.
                            Keys are BonusFeature instances from bonus_features.
                            Values are the trigger conditions (Dict[Symbol, int] or List[Dict[Symbol, int]])
                            as expected by check_bonus_trigger.
            rng_seed: Optional seed for the random number generator.
        """
        if not reel_strips:
            raise ValueError("At least one reel strip must be provided.")
        if num_rows <= 0:
            raise ValueError("Number of rows must be positive.")

        self.reel_strips = reel_strips
        self.num_reels = len(reel_strips)
        self.num_rows = num_rows
        self.paytable = paytable
        self.paylines = paylines if paylines is not None else {}
        self.bonus_features = bonus_features if bonus_features is not None else []
        self.bonus_triggers = bonus_triggers if bonus_triggers is not None else {}

        # Validate bonus_triggers map keys against bonus_features list
        if self.bonus_triggers:
            configured_features = set(self.bonus_features)
            for feature in self.bonus_triggers.keys():
                if feature not in configured_features:
                    raise ValueError(f"Bonus trigger defined for feature '{feature.feature_name}' which is not in the bonus_features list.")

        self.rng = random.Random(rng_seed)

        logger.info(f"SlotGame configured: {self.num_reels} reels, {self.num_rows} rows, {len(self.paylines)} lines, {len(self.bonus_features)} bonus features.")

    def play_step(self, game_state: GameState) -> SpinResult:
        """
        Plays a single step of the game, either a base game spin or a bonus step.
        Manages the GameState accordingly.

        Args:
            game_state: The current mutable state of the game session.

        Returns:
            A SpinResult object containing the outcome of the step.
        """
        triggered_feature_instance: Optional[BonusFeature] = None

        if game_state.bonus_active:
            # --- Handle Active Bonus Step ---
            result = self._execute_bonus_step(game_state)
        else:
            # --- Handle Base Game Spin ---
            result = self._execute_base_spin(game_state)
            # Check for bonus triggers after base spin resolves
            if result.window:
                triggered_feature_instance = self._check_and_start_bonus(result.window, game_state)
                result.triggering_feature = triggered_feature_instance

        # Update balance (cost already deducted in execute methods)
        game_state.update_balance(win_amount=result.total_win, cost=0)

        # Clean up if bonus ended this step
        if game_state.bonus_active and game_state.active_bonus_feature:
            active_feature = next((f for f in self.bonus_features if f.feature_name == game_state.active_bonus_feature), None)
            if active_feature and active_feature.is_complete(game_state):
                 logger.info(f"Bonus feature '{game_state.active_bonus_feature}' completed.")
                 game_state.exit_bonus()
                 # Reset game state for next base spin if needed
                 game_state.current_bet = game_state.base_bet # Restore original bet

        return result

    def _execute_base_spin(self, game_state: GameState) -> SpinResult:
        """
        Executes a standard base game spin.
        """
        logger.debug(f"Executing base game spin. Bet: {game_state.current_bet}")
        bet_per_line = game_state.current_bet / len(self.paylines) if self.paylines else 0
        total_bet_cost = game_state.current_bet # Cost for this spin

        # Deduct cost immediately before spin resolves
        # game_state.update_balance(win_amount=0, cost=total_bet_cost)
        # We deduct cost via GameState.update_balance *after* win is known in play_step

        # 1. Generate Stops
        stops = generate_random_stops(self.reel_strips, self.rng)

        # 2. Get Visible Window
        window = get_visible_window(self.reel_strips, stops, self.num_rows)

        # 3. Evaluate Wins (using base game paytable/lines)
        total_win, line_wins, scatter_wins = evaluate_all_wins(
            window=window,
            paylines=self.paylines,
            paytable=self.paytable,
            bet_per_line=bet_per_line,
            total_bet=total_bet_cost, # Use actual cost for scatter calc
            line_directions=['left_to_right'] # Make configurable?
        )

        # 4. Create SpinResult
        result = SpinResult(
            window=window,
            stops=stops,
            total_win=total_win,
            line_wins=line_wins,
            scatter_wins=scatter_wins,
            bonus_win=0.0, # No bonus win component in base spin itself
            is_bonus_step=False,
            active_bonus_feature=None,
            bet_per_line=bet_per_line,
            total_bet=total_bet_cost
        )
        logger.debug(f"Base spin finished. Stops: {stops}, Total Win: {result.total_win}")
        return result

    def _execute_bonus_step(self, game_state: GameState) -> SpinResult:
        """
        Executes a step of the currently active bonus feature.
        """
        if not game_state.bonus_active or not game_state.active_bonus_feature:
            raise RuntimeError("Attempted to execute bonus step but no bonus is active.")

        active_feature = next((f for f in self.bonus_features if f.feature_name == game_state.active_bonus_feature), None)
        if not active_feature:
             raise RuntimeError(f"Active bonus feature '{game_state.active_bonus_feature}' not found in game configuration.")

        logger.debug(f"Executing step for bonus: {active_feature.feature_name}")

        # Cost for a bonus step is usually 0 (e.g., free spin)
        step_cost = 0.0
        game_state.current_bet = 0 # Reflect zero cost during bonus steps

        # Execute the bonus logic for one step
        step_outcome = active_feature.play_step(game_state)

        window: Optional[List[List[Symbol]]] = None
        stops: Optional[List[int]] = None
        line_wins: Dict[str, float] = {}
        scatter_wins: float = 0.0
        bonus_win_component: float = 0.0 # Win directly from bonus mechanic (e.g. pick)
        total_win: float = 0.0

        if isinstance(step_outcome, (float, int)): # Direct win (e.g., pick bonus)
            bonus_win_component = float(step_outcome)
            total_win = bonus_win_component
            logger.debug(f"Bonus step resulted in direct win: {total_win}")

        elif isinstance(step_outcome, tuple) and len(step_outcome) == 3: # BonusSpinContext
            bonus_spin_context: BonusSpinContext = step_outcome
            reels_to_use, paytable_to_use, win_multiplier = bonus_spin_context

            # Use bonus-specific reels/paytable if provided, else default to base game
            current_reels = reels_to_use if reels_to_use is not None else self.reel_strips
            current_paytable = paytable_to_use if paytable_to_use is not None else self.paytable

            # 1. Generate Stops (on potentially different reels)
            stops = generate_random_stops(current_reels, self.rng)
            # 2. Get Visible Window
            window = get_visible_window(current_reels, stops, self.num_rows)
            # 3. Evaluate Wins (using potentially different paytable, NO bet cost, but apply multiplier)
            #    Base bet is needed for scatter calculations if they multiply triggering bet
            spin_line_win, spin_line_details, spin_scatter_win = evaluate_all_wins(
                 window=window,
                 paylines=self.paylines, # Assume bonus uses same lines unless overridden?
                 paytable=current_paytable,
                 bet_per_line=0, # Free spin
                 total_bet=game_state.base_bet, # Scatters might scale with original bet
                 line_directions=['left_to_right']
            )
            # Apply multiplier
            line_wins = {ln: lw * win_multiplier for ln, lw in spin_line_details.items()}
            scatter_wins = spin_scatter_win * win_multiplier # Assume multiplier applies to scatters too?
            total_win = sum(line_wins.values()) + scatter_wins
            logger.debug(f"Bonus spin finished. Multiplier: {win_multiplier}x, Win: {total_win}")

            # Check for retrigger *after* the bonus spin resolves
            if window and active_feature.handle_retrigger(game_state, window):
                 logger.info(f"Bonus feature '{active_feature.feature_name}' retriggered.")
                 # Retrigger logic is handled within the feature's handle_retrigger method

        else:
             logger.error(f"Unexpected return type from {active_feature.feature_name}.play_step: {type(step_outcome)}")
             total_win = 0.0

        # Update bonus-specific total win tracking if the feature uses it (e.g., state_data)
        if "total_bonus_win" in game_state.state_data:
            game_state.state_data["total_bonus_win"] += total_win

        # Create SpinResult for the bonus step
        result = SpinResult(
            window=window,
            stops=stops,
            total_win=total_win,
            line_wins=line_wins,
            scatter_wins=scatter_wins,
            bonus_win=bonus_win_component, # Only non-zero for direct win steps typically
            is_bonus_step=True,
            active_bonus_feature=game_state.active_bonus_feature,
            bet_per_line=0, # Bonus steps are usually free
            total_bet=step_cost # Cost is typically 0
        )
        return result

    def _check_and_start_bonus(self, window: List[List[Symbol]], game_state: GameState) -> Optional[BonusFeature]:
        """
        Checks the window against configured triggers and starts the first matching bonus.
        Returns the BonusFeature instance that was triggered, or None.
        """
        if not self.bonus_triggers:
            return None

        for feature, trigger_condition in self.bonus_triggers.items():
            if check_bonus_trigger(window, trigger_condition):
                logger.info(f"Trigger condition met for bonus: {feature.feature_name}")
                feature.start(game_state, trigger_context=window) # Pass window as context
                # Assume only one bonus triggers per spin for now
                return feature
        return None

# --- Example Usage Update ---
if __name__ == '__main__':
    # --- Basic Setup (Symbols, Reels) ---
    symbols = {
        "WILD": Symbol(name="WILD", display_name="W", is_wild=True),
        "SCATTER": Symbol(name="SCATTER", display_name="S", is_scatter=True), # Used for bonus trigger
        "BONUS": Symbol(name="BONUS", display_name="B", is_scatter=True), # Potentially another trigger
        "A": Symbol(name="A", display_name="A"),
        "K": Symbol(name="K", display_name="K"),
        "Q": Symbol(name="Q", display_name="Q"),
    }
    reel1 = ReelStrip([symbols[s] for s in ["A","K","Q","WILD","SCATTER","A","Q","K"] * 4], "R1")
    reel2 = ReelStrip([symbols[s] for s in ["K","Q","A","SCATTER","WILD","K","A","Q"] * 4], "R2")
    reel3 = ReelStrip([symbols[s] for s in ["Q","A","K","WILD","A","SCATTER","K","Q"] * 4], "R3")
    strips = [reel1, reel2, reel3]
    num_rows = 3
    paylines = { "line_1": [(0,1), (1,1), (2,1)] } # Middle line only

    # --- Paytable --- (Added Scatter pay)
    paytable_data = {
        (symbols["A"], 3): 50,
        (symbols["K"], 3): 40,
        (symbols["Q"], 3): 30,
        (symbols["WILD"], 3): 100,
        # Scatter pay defined separately if needed (or handled by bonus trigger)
        (symbols["SCATTER"], 3, 'scatter'): 5, # Small scatter win 5x total bet
    }
    paytable = Paytable(paytable_data)

    # --- Bonus Feature Configuration ---
    free_spins_feature = FreeSpinsFeature(
        num_spins=10,
        multiplier=2.0,
        can_retrigger=True,
        retrigger_condition={symbols["SCATTER"]: 3}, # Retrigger on 3 scatters during FS
        retrigger_spins=10
    )
    # Future: Add PickAndWinFeature instance here

    bonus_features = [free_spins_feature]
    bonus_triggers = {
        # Trigger FreeSpinsFeature when 3 SCATTER symbols land
        free_spins_feature: {symbols["SCATTER"]: 3}
    }

    # --- Create Slot Game Configuration ---
    game_config = SlotGame(
        reel_strips=strips,
        paytable=paytable,
        num_rows=num_rows,
        paylines=paylines,
        bonus_features=bonus_features,
        bonus_triggers=bonus_triggers,
        rng_seed=42
    )

    # --- Simulation Loop ---
    num_total_steps = 30 # Simulate steps (can be base spins or bonus steps)
    initial_balance = 1000.0
    base_bet_amount = 1.0 # Corresponds to bet_per_line * num_lines for base game

    # Initialize Game State
    game_state = GameState(current_bet=base_bet_amount, balance=initial_balance)

    print(f"--- Running Simulation ({num_total_steps} steps) ---")
    print(f"Initial State: {game_state}")

    cumulative_win = 0.0

    for i in range(num_total_steps):
        print(f"\n--- Step {i+1} ---")
        print(f"State Before: {game_state}")

        # Store bet *before* play_step modifies it (esp. for bonus)
        cost_for_this_step = game_state.current_bet if not game_state.bonus_active else 0

        spin_result = game_config.play_step(game_state) # Pass mutable game_state

        print(spin_result) # SpinResult now shows state and cost
        print(f"State After : {game_state}")

        cumulative_win += spin_result.total_win

        # Check if game ended (e.g., balance depleted - basic check)
        if game_state.balance <= 0:
            print("\n--- Simulation Ended (Balance Depleted) ---")
            break

        # Stop if a bonus completed and we don't need more base spins
        # (This requires more sophisticated loop control based on simulation goals)

    print(f"\n--- Simulation Summary ---")
    print(f"Initial Balance : {initial_balance:.2f}")
    print(f"Final Balance   : {game_state.balance:.2f}")
    print(f"Cumulative Win  : {cumulative_win:.2f}")
    # Note: RTP calc needs careful consideration of total wagered across base/bonus
    # total_cost = initial_balance - game_state.balance + cumulative_win
    # print(f"Total Cost Est. : {total_cost:.2f}") 