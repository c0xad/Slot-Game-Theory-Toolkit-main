# -*- coding: utf-8 -*-
"""
Monte Carlo simulation engine for estimating slot game metrics.

Suitable for complex games where symbolic evaluation is infeasible.
Estimates RTP, Variance, Hit Frequency, etc., by simulating a large number of spins.
Includes GPU acceleration and variance reduction techniques.
"""
import logging
import random
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import multiprocessing as mp
from functools import partial

# GPU acceleration libraries
try:
    import cupy as cp
    from numba import jit, cuda
    GPU_ENABLED = True
    # Use environment variable to control GPU messaging
    import os
    if os.environ.get('SLOT_DEBUG_GPU') == '1':
        print("GPU acceleration enabled using CuPy and Numba.")
except ImportError:
    cp = np
    GPU_ENABLED = False
    # Define dummy decorators when numba isn't available
    def jit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper if args and callable(args[0]) else wrapper
    
    class cuda:
        @staticmethod
        def jit(*args, **kwargs):
            def wrapper(func):
                return func
            return wrapper if args and callable(args[0]) else wrapper

from ..core.reels import ReelStrip, Symbol, Payline, generate_random_stops, get_visible_window
from ..core.paytable import Paytable, evaluate_all_wins
# Updated bonus imports
from ..core.bonuses import GameState, check_bonus_trigger, BonusFeature, BonusSpinContext

# Configure logging
logger = logging.getLogger(__name__)

# Type alias for simulation results
SimulationResult = Dict[str, Any] # e.g., {"rtp": 0.96, "variance": 25.5, ..., "total_spins": 1e9, "duration_sec": 120.5}

class MonteCarloSimulator:
    """Runs Monte Carlo simulations to estimate slot game metrics."""

    def __init__(
        self,
        reel_strips: List[ReelStrip],
        paylines: Dict[str, Payline],
        paytable: Paytable,
        num_rows: int,
        # Updated type hint for triggers
        bonus_triggers: Optional[Union[Dict[Symbol, int], List[Dict[Symbol, int]]]] = None,
        bonus_feature: Optional[BonusFeature] = None, # The bonus feature logic object
        use_gpu: bool = False, # Flag to attempt GPU usage
        parallel_cpu: bool = False,
        num_processes: int = None,
        variance_reduction: Optional[str] = None, # e.g., 'antithetic'
        rng_seed: Optional[int] = None # Added RNG seed
    ):
        """
        Initializes the simulator.

        Args:
            reel_strips: List of ReelStrip objects for the base game.
            paylines: Dictionary of payline definitions.
            paytable: Paytable object for the base game.
            num_rows: Number of visible rows.
            bonus_triggers: Condition(s) to trigger the bonus feature. Can be a single dict
                            or a list of dicts {Symbol: count}.
            bonus_feature: An object implementing the BonusFeature protocol.
            use_gpu: If True, attempt to use GPU acceleration (requires compatible libraries).
            parallel_cpu: If True and use_gpu is False, use multiprocessing for CPU parallelism.
            num_processes: Number of CPU processes to use. If None, uses CPU count.
            variance_reduction: Name of variance reduction technique to apply (if any).
            rng_seed: Optional random seed for reproducibility.
        """
        # Store base game components
        self.base_reel_strips = reel_strips
        self.base_paylines = paylines # Assuming paylines don't change for now
        self.base_paytable = paytable
        self.num_rows = num_rows
        self.num_reels = len(reel_strips)

        # Store bonus components
        self.bonus_triggers = bonus_triggers
        self.bonus_feature = bonus_feature

        # Performance options
        self.use_gpu = use_gpu and GPU_ENABLED
        self.parallel_cpu = parallel_cpu and not self.use_gpu
        self.num_processes = num_processes or mp.cpu_count()
        
        # Analysis options
        self.variance_reduction = variance_reduction

        # Initialize RNG
        self.rng = random.Random(rng_seed)
        self.rng_seed = rng_seed
        
        # For batched GPU execution
        self.gpu_batch_size = 10000 if self.use_gpu else 0
        
        logger.info(f"Simulator initialized. RNG Seed: {rng_seed}. "
                   f"GPU Enabled: {self.use_gpu}. "
                   f"CPU Parallel: {self.parallel_cpu} with {self.num_processes} processes.")

        # Prepare data structures
        self._prepare_data()

    def _prepare_data(self):
        """Prepares data structures for efficient simulation."""
        # Preprocess paylines for faster access
        self.payline_indices = []
        for payline in self.base_paylines.values():
            self.payline_indices.append([(pos.row, pos.reel) for pos in payline.positions])
        
        # Prepare GPU data if needed
        if self.use_gpu:
            self._prepare_gpu_data()
    
    def _prepare_gpu_data(self):
        """Prepare data structures specifically for GPU computation."""
        # Convert reels to a GPU-friendly format
        # For each reel strip, we need:
        # 1. Symbol indices array
        # 2. Probability distribution for each stop
        self.gpu_reel_data = []
        
        for strip in self.base_reel_strips:
            # Convert strip to symbol indices (more efficient than object references)
            symbol_map = {sym: i for i, sym in enumerate(set(strip.symbols))}
            indices = [symbol_map[sym] for sym in strip.symbols]
            
            # Calculate stop probabilities based on weights
            if strip.weights:
                total_weight = sum(strip.weights)
                probs = [w / total_weight for w in strip.weights]
            else:
                # Equal probability for each stop
                probs = [1.0 / len(indices)] * len(indices)
            
            # Store in GPU memory
            if GPU_ENABLED:
                self.gpu_reel_data.append({
                    'indices': cp.array(indices, dtype=cp.int32),
                    'probs': cp.array(probs, dtype=cp.float32),
                    'length': len(indices),
                    'symbol_map': symbol_map,
                    'reverse_map': {i: sym for sym, i in symbol_map.items()}
                })
            else:
                self.gpu_reel_data.append({
                    'indices': np.array(indices, dtype=np.int32),
                    'probs': np.array(probs, dtype=np.float32),
                    'length': len(indices),
                    'symbol_map': symbol_map,
                    'reverse_map': {i: sym for sym, i in symbol_map.items()}
                })
        
        # Prepare payline data for GPU
        max_payline_length = max(len(p) for p in self.payline_indices)
        num_paylines = len(self.payline_indices)
        
        # Create a padded array of payline indices
        payline_array = np.zeros((num_paylines, max_payline_length, 2), dtype=np.int32)
        payline_lengths = np.zeros(num_paylines, dtype=np.int32)
        
        for i, payline in enumerate(self.payline_indices):
            payline_lengths[i] = len(payline)
            for j, (row, reel) in enumerate(payline):
                if j < max_payline_length:
                    payline_array[i, j, 0] = row
                    payline_array[i, j, 1] = reel
        
        if GPU_ENABLED:
            self.gpu_payline_array = cp.array(payline_array)
            self.gpu_payline_lengths = cp.array(payline_lengths)
        else:
            self.gpu_payline_array = payline_array
            self.gpu_payline_lengths = payline_lengths
        
        # Prepare paytable data for GPU
        # This is complex due to varied symbol combinations
        # For now, we'll convert it during evaluation

    @jit(forceobj=True)
    def _internal_spin_eval(
        self,
        reels_to_use: List[ReelStrip],
        paytable_to_use: Paytable,
        bet_per_line: float,
        total_bet: float
    ) -> Tuple[float, List[List[Symbol]]]:
        """
        Internal helper to perform core spin mechanics: stops, window, win eval.

        Args:
            reels_to_use: The reel strips for this spin.
            paytable_to_use: The paytable for this spin.
            bet_per_line: Bet per line for win calculation.
            total_bet: Total bet for scatter win calculation.

        Returns:
            A tuple containing: (spin_payout_amount, visible_window)
        """
        # 1. Generate Stops
        if self.variance_reduction == 'antithetic':
            # Generate antithetic variates for variance reduction
            u_randoms = [self.rng.random() for _ in range(len(reels_to_use))]
            stops1 = [int(u * strip.length) for u, strip in zip(u_randoms, reels_to_use)]
            stops2 = [int((1-u) * strip.length) for u, strip in zip(u_randoms, reels_to_use)]
            stops = stops1  # Use the first set, accumulate both results in caller
        else:
            stops = generate_random_stops(reels_to_use, rng=self.rng)

        # 2. Get Visible Window
        window = get_visible_window(reels_to_use, stops, self.num_rows)

        # 3. Evaluate Wins
        spin_payout, _, _ = evaluate_all_wins(
            window=window,
            paylines=self.base_paylines,
            paytable=paytable_to_use,
            bet_per_line=bet_per_line,
            total_bet=total_bet
        )
        return spin_payout, window

    @cuda.jit
    def _gpu_generate_stops(stops_out, lengths, rng_states):
        """CUDA kernel to generate random stops for reels."""
        i = cuda.grid(1)
        if i < stops_out.shape[0]:
            for r in range(stops_out.shape[1]):
                # Use cuRAND for random numbers
                rand = cuda.random.xoroshiro128p_uniform_float32(rng_states, i)
                stops_out[i, r] = int(rand * lengths[r])

    @cuda.jit
    def _gpu_build_windows(stops, reel_data, window_out, num_rows):
        """CUDA kernel to build visible windows from stops."""
        batch_idx = cuda.grid(1)
        if batch_idx < stops.shape[0]:
            for r in range(stops.shape[1]):  # For each reel
                stop = stops[batch_idx, r]
                reel_length = lengths[r]
                for row in range(num_rows):
                    idx = (stop + row) % reel_length
                    window_out[batch_idx, row, r] = reel_data[r][idx]

    def _gpu_batch_simulate(self, batch_size, bet_per_line, total_bet):
        """
        Simulate a batch of spins on GPU for better parallelism.
        
        Returns:
            Array of payouts for each spin in the batch
        """
        if not GPU_ENABLED:
            raise RuntimeError("GPU simulation requested but GPU libraries not available")

        # Allocate arrays on GPU
        stops = cp.zeros((batch_size, self.num_reels), dtype=cp.int32)
        
        # Setup random states for CUDA
        rng_states = cuda.random.create_xoroshiro128p_states(batch_size, seed=self.rng_seed)
        
        # Extract reel lengths for the kernel
        lengths = cp.array([reel['length'] for reel in self.gpu_reel_data], dtype=cp.int32)
        
        # Generate random stops
        threads_per_block = 256
        blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block
        self._gpu_generate_stops[blocks_per_grid, threads_per_block](
            stops, lengths, rng_states
        )
        
        # Create window array
        windows = cp.zeros((batch_size, self.num_rows, self.num_reels), dtype=cp.int32)
        
        # Extract reel data arrays
        reel_data = [reel['indices'] for reel in self.gpu_reel_data]
        
        # Build windows
        self._gpu_build_windows[blocks_per_grid, threads_per_block](
            stops, reel_data, windows, self.num_rows
        )
        
        # Evaluate wins (this part is complex and may require custom CUDA kernels)
        # For now, transfer back to CPU for evaluation
        cpu_windows = windows.get()
        
        # Convert numeric windows back to symbol objects
        symbol_windows = []
        for window in cpu_windows:
            symbol_window = []
            for row in range(self.num_rows):
                symbol_row = []
                for reel in range(self.num_reels):
                    sym_idx = window[row, reel]
                    symbol = self.gpu_reel_data[reel]['reverse_map'][int(sym_idx)]
                    symbol_row.append(symbol)
                symbol_window.append(symbol_row)
            symbol_windows.append(symbol_window)
        
        # Evaluate wins on CPU
        payouts = []
        for window in symbol_windows:
            spin_payout, _, _ = evaluate_all_wins(
                window=window,
                paylines=self.base_paylines,
                paytable=self.base_paytable,
                bet_per_line=bet_per_line,
                total_bet=total_bet
            )
            payouts.append(spin_payout)
        
        return np.array(payouts)

    def _parallel_cpu_simulate(self, num_spins, bet_per_line, total_bet, processes):
        """Run simulation using multiprocessing on CPU."""
        spins_per_process = num_spins // processes
        remainder = num_spins % processes
        
        # Create a partial function with fixed parameters
        sim_func = partial(
            self._simulate_batch, 
            reels=self.base_reel_strips,
            paylines=self.base_paylines,
            paytable=self.base_paytable,
            num_rows=self.num_rows,
            bet_per_line=bet_per_line,
            total_bet=total_bet
        )
        
        # Distribute spins across processes
        batch_sizes = [spins_per_process + (1 if i < remainder else 0) 
                      for i in range(processes)]
        
        # Create seed sequence for each process
        if self.rng_seed is not None:
            from numpy.random import SeedSequence
            seeds = SeedSequence(self.rng_seed).spawn(processes)
            seed_args = [(batch, seed.entropy) for batch, seed in zip(batch_sizes, seeds)]
        else:
            seed_args = [(batch, None) for batch in batch_sizes]
        
        # Run in parallel
        with mp.Pool(processes) as pool:
            results = pool.starmap(sim_func, seed_args)
        
        # Combine results
        all_payouts = []
        winning_spins = 0
        for payouts, wins in results:
            all_payouts.extend(payouts)
            winning_spins += wins
        
        return all_payouts, winning_spins
    
    @staticmethod
    def _simulate_batch(batch_size, seed, reels, paylines, paytable, num_rows, bet_per_line, total_bet):
        """Helper function for parallel CPU simulation."""
        # Create a local RNG with the provided seed
        local_rng = random.Random(seed)
        
        payouts = []
        winning_spins = 0
        
        for _ in range(batch_size):
            # Generate stops
            stops = [local_rng.randint(0, len(strip) - 1) for strip in reels]
            
            # Get window
            window = get_visible_window(reels, stops, num_rows)
            
            # Evaluate wins
            spin_payout, _, _ = evaluate_all_wins(
                window=window,
                paylines=paylines,
                paytable=paytable,
                bet_per_line=bet_per_line,
                total_bet=total_bet
            )
            
            payouts.append(spin_payout)
            if spin_payout > 0:
                winning_spins += 1
        
        return payouts, winning_spins

    def run_simulation(
        self,
        num_spins: int,
        initial_bet: float = 1.0,
        initial_balance: float = 0.0
    ) -> SimulationResult:
        """
        Runs the full Monte Carlo simulation for a specified number of spins.

        Args:
            num_spins: The total number of *base game equivalent* spins to simulate.
                       Note: Free spins do not count towards this total.
            initial_bet: The bet amount for base game spins.
            initial_balance: Starting balance for the simulation run.

        Returns:
            A dictionary containing estimated metrics and simulation info.
        """
        if num_spins <= 0:
             logger.warning("Number of spins must be positive.")
             return {}

        start_time = time.time()

        # Accumulators
        total_payout = 0.0
        total_payout_sq = 0.0
        total_bet_cost = 0.0
        winning_steps_count = 0
        bonus_entries = 0
        base_spins_completed = 0
        total_steps = 0
        
        # For advanced statistics
        payout_histogram = {}  # To track payout distribution
        largest_win = 0.0
        streak_data = {'current_win_streak': 0, 'current_loss_streak': 0,
                       'max_win_streak': 0, 'max_loss_streak': 0}

        # Check if we should use GPU batch simulation for base game
        if self.use_gpu and not self.bonus_feature:
            # Simple case: no bonus features, can use pure GPU acceleration
            logger.info(f"Using GPU batch simulation for {num_spins} spins...")
            
            # Calculate bet per line
            bet_per_line = initial_bet / len(self.base_paylines) if self.base_paylines else initial_bet
            
            # Process in batches for better GPU utilization
            remaining_spins = num_spins
            while remaining_spins > 0:
                batch_size = min(self.gpu_batch_size, remaining_spins)
                batch_payouts = self._gpu_batch_simulate(batch_size, bet_per_line, initial_bet)
                
                # Update accumulators
                total_payout += np.sum(batch_payouts)
                total_payout_sq += np.sum(batch_payouts ** 2)
                total_bet_cost += batch_size * initial_bet
                winning_steps_count += np.count_nonzero(batch_payouts > 0)
                
                # Update histogram
                for payout in batch_payouts:
                    payout_histogram[payout] = payout_histogram.get(payout, 0) + 1
                    largest_win = max(largest_win, payout)
                
                remaining_spins -= batch_size
                base_spins_completed += batch_size
                total_steps += batch_size
            
            # Don't need to handle bonus features in this code path
        
        elif self.parallel_cpu and not self.bonus_feature:
            # Use CPU parallelism for base game without bonus features
            logger.info(f"Using parallel CPU simulation with {self.num_processes} processes...")
            
            # Calculate bet per line
            bet_per_line = initial_bet / len(self.base_paylines) if self.base_paylines else initial_bet
            
            payouts, winning_spins = self._parallel_cpu_simulate(
                num_spins, bet_per_line, initial_bet, self.num_processes
            )
            
            # Update accumulators
            total_payout = np.sum(payouts)
            total_payout_sq = np.sum(np.array(payouts) ** 2)
            total_bet_cost = num_spins * initial_bet
            winning_steps_count = winning_spins
            base_spins_completed = num_spins
            total_steps = num_spins
            
            # Update histogram
            for payout in payouts:
                payout_histogram[payout] = payout_histogram.get(payout, 0) + 1
                largest_win = max(largest_win, payout)
        
        else:
            # Use sequential simulation with bonus features
            # Initialize game state
            game_state = GameState(current_bet=initial_bet, balance=initial_balance)
            game_state.base_bet = initial_bet

            logger.info(f"Starting sequential simulation for {num_spins} base game spins...")

            # Loop until desired number of base game spins are completed
            while base_spins_completed < num_spins:
                total_steps += 1
                spin_payout = 0.0
                spin_cost = 0.0
                window = None

                if game_state.bonus_active and self.bonus_feature:
                    # --- Handle Bonus Step ---
                    step_result = self.bonus_feature.play_step(game_state)
                    spin_cost = 0 # Bonus steps/spins are typically free

                    if isinstance(step_result, (float, int)):
                        # Direct win from bonus step (e.g., pick bonus)
                        spin_payout = float(step_result)
                    elif isinstance(step_result, tuple): # Expecting BonusSpinContext
                        # Context provided for a bonus spin (e.g., free spin)
                        try:
                            bonus_reels, bonus_paytable, multiplier = step_result
                        except ValueError:
                             logger.error(f"Invalid BonusSpinContext structure: {step_result}")
                             continue # Skip this step

                        reels_to_use = bonus_reels if bonus_reels is not None else self.base_reel_strips
                        paytable_to_use = bonus_paytable if bonus_paytable is not None else self.base_paytable

                        # Simulate the spin using the provided context
                        spin_win_raw, window = self._internal_spin_eval(
                            reels_to_use, paytable_to_use,
                            bet_per_line=game_state.base_bet / len(self.base_paylines) if self.base_paylines else 0,
                            total_bet=game_state.base_bet # Wins based on original triggering bet
                        )
                        spin_payout = spin_win_raw * multiplier

                        # Check for retrigger after the spin
                        if window and hasattr(self.bonus_feature, 'handle_retrigger'):
                             self.bonus_feature.handle_retrigger(game_state, window)
                    else:
                         logger.error(f"Unexpected result type from bonus_feature.play_step: {type(step_result)}")

                    # Check if bonus feature is now complete
                    if self.bonus_feature.is_complete(game_state):
                        game_state.exit_bonus()

                else:
                    # --- Handle Base Game Spin ---
                    base_spins_completed += 1
                    spin_cost = game_state.current_bet

                    # Simulate the spin
                    spin_payout, window = self._internal_spin_eval(
                        self.base_reel_strips, self.base_paytable,
                        bet_per_line=game_state.current_bet / len(self.base_paylines) if self.base_paylines else 0,
                        total_bet=game_state.current_bet
                    )

                    # Check for Bonus Trigger
                    if window and self.bonus_triggers and self.bonus_feature:
                        trigger_condition_met = check_bonus_trigger(window, self.bonus_triggers)
                        if trigger_condition_met:
                            bonus_entries += 1
                            # Pass trigger context if needed by the bonus start method
                            trigger_context = {"window": window, "condition": trigger_condition_met}
                            self.bonus_feature.start(game_state, trigger_context=trigger_context)

                # --- Update accumulators ---
                total_payout += spin_payout
                total_payout_sq += spin_payout ** 2
                total_bet_cost += spin_cost
                
                # Update histogram and largest win
                payout_histogram[spin_payout] = payout_histogram.get(spin_payout, 0) + 1
                largest_win = max(largest_win, spin_payout)
                
                # Update streak data
                if spin_payout > 0:
                    winning_steps_count += 1
                    streak_data['current_win_streak'] += 1
                    streak_data['current_loss_streak'] = 0
                    streak_data['max_win_streak'] = max(streak_data['max_win_streak'], 
                                                     streak_data['current_win_streak'])
                else:
                    streak_data['current_loss_streak'] += 1
                    streak_data['current_win_streak'] = 0
                    streak_data['max_loss_streak'] = max(streak_data['max_loss_streak'], 
                                                      streak_data['current_loss_streak'])

                # Update balance (optional but good for stateful checks)
                game_state.update_balance(spin_payout, spin_cost)

        end_time = time.time()
        duration_sec = end_time - start_time

        # --- Calculate final metrics ---
        if total_bet_cost <= 0 and total_payout > 0:
             logger.warning("Total bet cost is zero but payout occurred. RTP is infinite.")
             rtp = float('inf')
             variance = float('nan')
             expected_payout = float('nan')
        elif total_bet_cost <= 0:
             logger.warning("Total bet cost is zero. Cannot calculate RTP.")
             rtp = 0.0
             variance = 0.0
             expected_payout = 0.0
        else:
            # RTP = Total Payout / Total Amount Bet (Cost)
            rtp = total_payout / total_bet_cost

            # Calculate variance and expected payout
            expected_payout_per_base_spin = total_payout / num_spins
            expected_payout_sq_per_base_spin = total_payout_sq / num_spins
            variance = max(0.0, expected_payout_sq_per_base_spin - (expected_payout_per_base_spin ** 2))
            expected_payout = expected_payout_per_base_spin

        # Frequencies
        hit_frequency = winning_steps_count / total_steps if total_steps > 0 else 0.0
        bonus_frequency = bonus_entries / num_spins if num_spins > 0 else 0.0
        
        # Calculate volatility index
        # Standard formula: standard deviation / mean
        volatility_index = np.sqrt(variance) / expected_payout if expected_payout > 0 else 0.0
        
        # Calculate payout distribution metrics
        payout_distribution = {float(k): v / total_steps for k, v in payout_histogram.items()}
        
        # Calculate median payout (for non-zero wins)
        non_zero_payouts = [payout for payout in payout_histogram.keys() if payout > 0]
        median_win = np.median(non_zero_payouts) if non_zero_payouts else 0.0

        # Expanded results dictionary with more detailed metrics
        return {
            # Core metrics
            "rtp": rtp,
            "variance": variance,
            "hit_frequency": hit_frequency,
            "bonus_frequency": bonus_frequency,
            "expected_payout_per_base_spin": expected_payout,
            "volatility_index": volatility_index,
            
            # Raw totals
            "total_base_spins": num_spins,
            "total_steps": total_steps,
            "total_payout": total_payout,
            "total_bet_cost": total_bet_cost,
            "winning_steps": winning_steps_count,
            "bonus_entries": bonus_entries,
            
            # Payout distribution info
            "largest_win": largest_win,
            "largest_win_multiple": largest_win / initial_bet if initial_bet > 0 else 0.0,
            "median_win": median_win,
            "median_win_multiple": median_win / initial_bet if initial_bet > 0 else 0.0,
            "payout_distribution": payout_distribution,
            
            # Streak data
            "max_win_streak": streak_data.get('max_win_streak', 0),
            "max_loss_streak": streak_data.get('max_loss_streak', 0),
            
            # Performance info
            "duration_sec": duration_sec,
            "spins_per_second": num_spins / duration_sec if duration_sec > 0 else float('inf'),
            "gpu_used": self.use_gpu,
            "parallel_cpu": self.parallel_cpu,
            "variance_reduction": self.variance_reduction,
        }

# --- Variance Reduction Techniques ---

def antithetic_sampling(rng, reels):
    """
    Generates stops using antithetic variates for variance reduction.
    
    This technique uses the fact that if U is uniform(0,1), then (1-U) is also uniform(0,1),
    but negatively correlated with U.
    """
    u_randoms = [rng.random() for _ in range(len(reels))]
    stops1 = [int(u * len(strip)) for u, strip in zip(u_randoms, reels)]
    stops2 = [int((1-u) * len(strip)) for u, strip in zip(u_randoms, reels)]
    return stops1, stops2

def control_variates(primary_result, control_result, expected_control, beta=None):
    """
    Apply control variates method to reduce variance.
    
    Args:
        primary_result: The result we want to estimate (e.g., total RTP)
        control_result: A correlated result that we know the expected value of
        expected_control: The known expected value of the control
        beta: Scaling parameter (if None, calculated from data)
        
    Returns:
        An adjusted estimate with reduced variance
    """
    if beta is None:
        # Optimal beta = Cov(X,Y) / Var(Y)
        # Where X is primary, Y is control
        # In practice, calculate from sample data
        pass
    
    # Adjust the primary result using the control
    return primary_result - beta * (control_result - expected_control)

# --- Helper functions for advanced metrics ---

def calculate_hit_win_metrics(payouts, bet_amount):
    """Calculate additional metrics about hit/win distribution."""
    non_zero_wins = [p for p in payouts if p > 0]
    if not non_zero_wins:
        return {
            "avg_win_size": 0,
            "median_win_size": 0,
            "win_size_variance": 0,
            "max_win": 0
        }
    
    avg_win = sum(non_zero_wins) / len(non_zero_wins)
    median_win = sorted(non_zero_wins)[len(non_zero_wins) // 2]
    win_variance = sum((w - avg_win) ** 2 for w in non_zero_wins) / len(non_zero_wins)
    max_win = max(non_zero_wins)
    
    # Calculate win multiples relative to bet
    if bet_amount > 0:
        win_multiples = [w / bet_amount for w in non_zero_wins]
        avg_multiple = sum(win_multiples) / len(win_multiples)
        median_multiple = sorted(win_multiples)[len(win_multiples) // 2]
        max_multiple = max(win_multiples)
    else:
        avg_multiple = median_multiple = max_multiple = 0
    
    return {
        "avg_win_size": avg_win,
        "median_win_size": median_win,
        "win_size_variance": win_variance,
        "max_win": max_win,
        "avg_win_multiple": avg_multiple,
        "median_win_multiple": median_multiple,
        "max_win_multiple": max_multiple
    }

@jit(nopython=True)
def calculate_streak_metrics(win_sequence):
    """
    Calculate metrics about winning and losing streaks.
    
    Args:
        win_sequence: A binary array where 1=win, 0=loss
        
    Returns:
        Dictionary of streak metrics
    """
    max_win_streak = 0
    max_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for outcome in win_sequence:
        if outcome > 0:  # Win
            current_win_streak += 1
            current_loss_streak = 0
            max_win_streak = max(max_win_streak, current_win_streak)
        else:  # Loss
            current_loss_streak += 1
            current_win_streak = 0
            max_loss_streak = max(max_loss_streak, current_loss_streak)
    
    return {
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak
    }

# --- Main entrypoint for direct execution ---

if __name__ == "__main__":
    import argparse
    
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description="Run slot game simulation")
    parser.add_argument("--spins", type=int, default=1000000, help="Number of spins to simulate")
    parser.add_argument("--bet", type=float, default=1.0, help="Bet amount per spin")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    parser.add_argument("--parallel", action="store_true", help="Use CPU parallelism")
    parser.add_argument("--processes", type=int, help="Number of CPU processes to use")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Create a simple game configuration for testing
    from ..core.reels import ReelStrip, Symbol
    from ..core.paytable import Paytable
    
    # Define symbols
    symbols = [
        Symbol("A", "High"),
        Symbol("K", "High"),
        Symbol("Q", "High"),
        Symbol("J", "High"),
        Symbol("10", "Low"),
        Symbol("9", "Low"),
        Symbol("WILD", "Special", is_wild=True),
        Symbol("SCATTER", "Special", is_scatter=True)
    ]
    
    # Define reels
    # Weights make some symbols rarer than others
    reel1 = ReelStrip(
        symbols=[symbols[i % len(symbols)] for i in range(30)],
        weights=[10 if i < 20 else 5 for i in range(30)]
    )
    reel2 = ReelStrip(
        symbols=[symbols[(i + 2) % len(symbols)] for i in range(30)],
        weights=[8 if i < 15 else 12 for i in range(30)]
    )
    reel3 = ReelStrip(
        symbols=[symbols[(i + 4) % len(symbols)] for i in range(30)],
        weights=[7 if i < 10 else 13 for i in range(30)]
    )
    
    # Define paylines
    # For simplicity, use just horizontal lines in a 3x3 grid
    from ..core.reels import Payline, Position
    
    paylines = {
        "top": Payline([Position(0, 0), Position(0, 1), Position(0, 2)]),
        "middle": Payline([Position(1, 0), Position(1, 1), Position(1, 2)]),
        "bottom": Payline([Position(2, 0), Position(2, 1), Position(2, 2)])
    }
    
    # Define paytable
    # Simple structure: {(symbol, count): payout}
    paytable_data = {
        (symbols[0], 3): 50,  # AAA pays 50
        (symbols[1], 3): 40,  # KKK pays 40
        (symbols[2], 3): 30,  # QQQ pays 30
        (symbols[3], 3): 20,  # JJJ pays 20
        (symbols[4], 3): 15,  # 10-10-10 pays 15
        (symbols[5], 3): 10,  # 999 pays 10
        (symbols[6], 3): 100,  # WILD-WILD-WILD pays 100
        (symbols[7], 3): 0,   # SCATTER-SCATTER-SCATTER triggers bonus
    }
    
    paytable = Paytable(paytable_data, wild_symbol=symbols[6])
    
    # Create and run the simulator
    simulator = MonteCarloSimulator(
        reel_strips=[reel1, reel2, reel3],
        paylines=paylines,
        paytable=paytable,
        num_rows=3,
        bonus_triggers={symbols[7]: 3},  # 3 scatters trigger bonus
        bonus_feature=None,  # No bonus feature for this example
        use_gpu=args.gpu,
        parallel_cpu=args.parallel,
        num_processes=args.processes,
        rng_seed=args.seed
    )
    
    # Run the simulation
    results = simulator.run_simulation(
        num_spins=args.spins,
        initial_bet=args.bet
    )
    
    # Print results
    print("\n=== Simulation Results ===")
    print(f"Spins: {args.spins:,}")
    print(f"RTP: {results['rtp'] * 100:.2f}%")
    print(f"Hit Frequency: {results['hit_frequency'] * 100:.2f}%")
    print(f"Volatility Index: {results['volatility_index']:.2f}")
    print(f"Largest Win: {results['largest_win']:.2f} ({results['largest_win_multiple']:.1f}x bet)")
    print(f"Duration: {results['duration_sec']:.2f} seconds")
    print(f"Performance: {results['spins_per_second']:.0f} spins/second")
    print(f"GPU Used: {results['gpu_used']}")