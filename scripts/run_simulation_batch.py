# -*- coding: utf-8 -*-
"""
Example script to run multiple simulation batches.

This script demonstrates how to use the simulation engine to run simulations
for different configurations or parameter sets and save the results.
"""

import argparse
import json
import os
import sys
import time
import pandas as pd

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from src
from src.slot_game_theory.core import reels, paytable, bonuses
from src.slot_game_theory.evaluation import simulation
from src.slot_game_theory.utils import helpers

# --- Configuration Loading ---

def load_game_config(config_path: str) -> dict:
    """Loads game configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded game configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        exit(1)

def setup_game_from_config(config: dict) -> tuple:
    """Sets up game components based on the loaded configuration."""
    # This function needs to parse the config dict and instantiate
    # ReelStrip, Paytable, BonusFeature objects etc.
    # This is highly dependent on the config file format.
    print("Setting up game components from configuration (Placeholder)...")

    # Placeholder setup - replace with actual parsing logic
    symbols = {name: reels.Symbol(name, **props) for name, props in config.get("symbols", {}).items()}
    reel_strips = [reels.ReelStrip([symbols.get(s, reels.Symbol(s)) for s in strip_def], name=f"Reel{i+1}")
                   for i, strip_def in enumerate(config.get("reel_strips", []))]
    paylines = {name: coords for name, coords in config.get("paylines", {}).items()} # Assuming coords are List[Tuple[int, int]]

    # Paytable parsing needs care with symbol objects
    parsed_win_payouts = {}
    for condition_list, payout in config.get("paytable", {}).items():
        # Assuming condition_list is like ["SYMBOL_NAME", count, "type"]
        symbol_name = condition_list[0]
        count = condition_list[1]
        cond_type = condition_list[2] if len(condition_list) > 2 else None
        symbol_obj = symbols.get(symbol_name)
        if not symbol_obj:
            print(f"Warning: Symbol '{symbol_name}' not found in symbols definition for paytable.")
            continue
        if cond_type:
            parsed_win_payouts[(symbol_obj, count, cond_type)] = payout
        else:
             parsed_win_payouts[(symbol_obj, count)] = payout
    paytable_obj = paytable.Paytable(parsed_win_payouts)

    # Bonus setup (simplified)
    bonus_trigger_conf = config.get("bonus_trigger")
    bonus_triggers = None
    if bonus_trigger_conf:
         trigger_symbol = symbols.get(bonus_trigger_conf["symbol"])
         if trigger_symbol:
             bonus_triggers = {trigger_symbol: bonus_trigger_conf["count"]}

    bonus_feature_conf = config.get("bonus_feature")
    bonus_feature = None
    if bonus_feature_conf and bonus_feature_conf["type"] == "FreeSpins":
        bonus_feature = bonuses.FreeSpinsFeature(
            num_spins=bonus_feature_conf["num_spins"],
            multiplier=bonus_feature_conf.get("multiplier", 1.0)
            # Add bonus reels/paytable parsing if needed
        )

    num_rows = config.get("num_rows", 3)

    return reel_strips, paylines, paytable_obj, num_rows, bonus_triggers, bonus_feature


# --- Main Simulation Logic ---

def run_batch(config_path: str, num_spins: int, num_batches: int, output_dir: str, seed: int):
    """Runs a batch of simulations."""
    helpers.set_random_seed(seed)
    config = load_game_config(config_path)
    game_components = setup_game_from_config(config)
    reel_strips, paylines, paytable_obj, num_rows, bonus_triggers, bonus_feature = game_components

    simulator = simulation.MonteCarloSimulator(
        reel_strips=reel_strips,
        paylines=paylines,
        paytable=paytable_obj,
        num_rows=num_rows,
        bonus_triggers=bonus_triggers,
        bonus_feature=bonus_feature
        # Add GPU/variance reduction flags if needed
    )

    all_results = []
    start_batch_time = time.time()

    print(f"\n--- Starting Simulation Batch ---")
    print(f"Config: {config_path}")
    print(f"Spins per Batch: {num_spins:,}")
    print(f"Number of Batches: {num_batches}")
    print(f"Output Directory: {output_dir}")
    print(f"-------------------------------")

    for i in range(num_batches):
        print(f"\nRunning Batch {i+1}/{num_batches}...")
        with helpers.Timer(f"Batch {i+1}"):
            results = simulator.run_simulation(num_spins=num_spins, initial_bet=config.get("default_bet", 1.0))
        print(f"Batch {i+1} Results:\n{helpers.format_metrics(results)}")
        results['batch_number'] = i + 1
        results['config_file'] = os.path.basename(config_path)
        all_results.append(results)

    total_batch_duration = time.time() - start_batch_time
    print(f"\n--- Batch Complete ---")
    print(f"Total duration: {total_batch_duration:.2f} seconds")

    # --- Save Results ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    results_df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(config_path))[0]
    output_filename = f"{base_filename}_sim_results_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    try:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


# --- Function to be called from main.py ---

def run_simulation(config_path: str, output_dir: str, num_runs: int = 10000, seed: int = None):
    """
    Wrapper function called from main.py to run a simulation.
    
    Args:
        config_path: Path to the configuration file
        output_dir: Directory to save simulation results
        num_runs: Number of simulation runs (spins)
        seed: Random seed for reproducibility
    """
    if seed is None:
        seed = int(time.time())
    
    # Use a single batch for simpler execution from main.py
    run_batch(
        config_path=config_path,
        num_spins=num_runs,
        num_batches=1,
        output_dir=output_dir,
        seed=seed
    )
    
    print("\nSimulation completed.")


# --- Command Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch simulations for slot game analysis.")
    parser.add_argument("config", help="Path to the game configuration JSON file.")
    parser.add_argument("-s", "--spins", type=int, default=1_000_000, help="Number of spins per simulation batch.")
    parser.add_argument("-n", "--num-batches", type=int, default=1, help="Number of simulation batches to run.")
    parser.add_argument("-o", "--output-dir", default="data/simulation_results", help="Directory to save simulation results.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())

    run_batch(
        config_path=args.config,
        num_spins=args.spins,
        num_batches=args.num_batches,
        output_dir=args.output_dir,
        seed=seed
    )

    print("\nScript finished.")

# Example Usage:
# python scripts/run_simulation_batch.py configs/my_slot_game.json -s 10000000 -n 5 -o results/my_game --seed 42