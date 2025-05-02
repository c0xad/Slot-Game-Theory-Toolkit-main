#!/usr/bin/env python3
"""
Run optimization batch script for the Slot Game Theory Toolkit.

This script loads optimization configuration from a JSON file and runs 
the optimization process with specified parameters.
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from src.slot_game_theory.optimization.solver import ScipyDifferentialEvolutionOptimizer
from src.slot_game_theory.optimization.objective import create_objective_function
from src.slot_game_theory.optimization.constraints import create_constraint_function_for_optimizer, ConstraintSet, RTPConstraint, VarianceConstraint, HitFrequencyConstraint, MaxPayoutConstraint
from src.slot_game_theory.core import SlotGame, Symbol, ReelStrip, Paytable, get_standard_paylines

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load optimization configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary with optimization configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise

def save_results(results, output_dir: str, config: Dict[str, Any]):
    """
    Save optimization results to output directory.
    
    Args:
        results: Optimization results object
        output_dir: Directory to save results to
        config: Original configuration used for optimization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    
    # Prepare results for serialization
    serializable_results = {
        "success": results.success,
        "best_value": float(results.best_value) if results.best_value is not None else None,
        "message": results.message,
        "best_parameters": results.best_params.tolist() if hasattr(results.best_params, 'tolist') else results.best_params,
        "timestamp": timestamp,
        "configuration": config
    }
    
    with open(result_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    
    # Optionally save a summary file for quick reference
    summary_file = os.path.join(output_dir, f"optimization_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Optimization Results Summary\n")
        f.write(f"==========================\n\n")
        f.write(f"Status: {'Success' if results.success else 'Failed'}\n")
        f.write(f"Best objective value: {results.best_value}\n")
        f.write(f"Message: {results.message}\n\n")
        f.write(f"Best parameters:\n")
        if results.best_params is not None:
            for i, param in enumerate(results.best_params):
                f.write(f"  Parameter {i+1}: {param}\n")
    
    logger.info(f"Summary saved to {summary_file}")

def create_constraints_from_config(config, game):
    """
    Create a ConstraintSet from the configuration.
    
    Args:
        config: The optimization configuration
        game: The SlotGame instance
        
    Returns:
        A ConstraintSet object
    """
    constraints = []
    
    # Extract constraints from config
    constraint_configs = config.get("constraints", {})
    
    # RTP constraints
    if "rtp" in constraint_configs:
        rtp_config = constraint_configs["rtp"]
        min_rtp = rtp_config.get("min", -float('inf'))
        max_rtp = rtp_config.get("max", float('inf'))
        constraints.append(RTPConstraint(min_rtp=min_rtp, max_rtp=max_rtp))
    
    # Volatility/Variance constraints
    if "volatility" in constraint_configs:
        vol_config = constraint_configs["volatility"]
        min_vol = vol_config.get("min", -float('inf'))
        max_vol = vol_config.get("max", float('inf'))
        constraints.append(VarianceConstraint(min_variance=min_vol, max_variance=max_vol))
    
    # Max win constraints
    if "max_win" in constraint_configs:
        max_win_config = constraint_configs["max_win"]
        max_win = max_win_config.get("max", float('inf'))
        constraints.append(MaxPayoutConstraint(max_multiplier=max_win))
    
    # Hit frequency constraints
    if "hit_frequency" in constraint_configs:
        hf_config = constraint_configs["hit_frequency"]
        min_hf = hf_config.get("min", -float('inf'))
        max_hf = hf_config.get("max", float('inf'))
        constraints.append(HitFrequencyConstraint(min_hf=min_hf, max_hf=max_hf))
    
    return ConstraintSet(constraints)

def create_slot_game_from_config(config):
    """
    Create a SlotGame instance from the configuration.
    
    Args:
        config: The optimization configuration
        
    Returns:
        A SlotGame instance
    """
    game_config = config.get("base_game_config", {})
    
    # Create symbols
    symbols = {}
    for symbol_name in game_config.get("reels", {}).get("symbols", []):
        symbols[symbol_name] = Symbol(name=symbol_name)
    
    # Create paytable
    paytable_data = game_config.get("paytable_base", {})
    # Create payouts dictionary
    win_payouts = {}
    for symbol_name, pays in paytable_data.items():
        if symbol_name in symbols:
            for count, value in enumerate(pays):
                if count > 0 and value > 0:  # Skip 0 count or 0 value entries
                    win_payouts[(symbols[symbol_name], count)] = value
    
    # Create paytable using the payouts dictionary
    paytable = Paytable(win_payouts=win_payouts)
    
    # Create reels (with minimal setup for demo)
    num_reels = game_config.get("reels", {}).get("num_reels", 5)
    reel_strips = []
    for _ in range(num_reels):
        # Create a simple reel strip with equal distribution of all symbols
        strip = ReelStrip([symbols[s] for s in game_config.get("reels", {}).get("symbols", []) for _ in range(3)])
        reel_strips.append(strip)
    
    # Create paylines
    num_rows = game_config.get("reels", {}).get("num_rows", 3)
    paylines = get_standard_paylines(num_reels, num_rows)
    
    # Create SlotGame
    return SlotGame(
        reel_strips=reel_strips,
        paytable=paytable,
        num_rows=num_rows,
        paylines=paylines
    )

def run_optimization(config_path: str, output_dir: str):
    """
    Run the optimization process based on configuration.
    
    Args:
        config_path: Path to the optimization configuration file
        output_dir: Directory to save optimization results
    """
    logger.info(f"Starting optimization with config: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract optimization parameters
    opt_params = config.get("algorithm_params", {})
    max_iter = opt_params.get("generations", 50)
    pop_size = opt_params.get("population_size", 15)
    strategy = "best1bin"  # Default DE strategy
    
    # Create parameter bounds
    bounds = []
    for param in config.get("optimization_params", {}).get("optimizable_params", {}).values():
        if param.get("enabled", False):
            if "range" in param:
                bounds.append(param["range"])
            elif "min" in param and "max" in param:
                bounds.append((param["min"], param["max"]))
    
    # Ensure we have at least 4 parameters for our dummy evaluation function
    if len(bounds) < 4:
        # Add dummy bounds if needed
        for i in range(len(bounds), 4):
            bounds.append((0.0, 1.0))
    
    if not bounds:
        logger.error("No parameter bounds defined in configuration")
        raise ValueError("No parameter bounds defined in configuration")
    
    # Create game instance
    game = create_slot_game_from_config(config)
    
    # Create constraints
    constraint_set = create_constraints_from_config(config["optimization_params"], game)
    
    # TODO: Create proper evaluation function based on simulation or analytical methods
    # For now, create a simple dummy function for demonstration
    def evaluate_design(params):
        # Dummy evaluation - in real implementation, this would run simulations
        return {
            "rtp": 0.92 + params[0] * 0.06,  # Between 0.92 and 0.98
            "variance": 5.0 + params[1] * 10.0,  # Between 5 and 15
            "hit_frequency": 0.2 + params[2] * 0.3,  # Between 0.2 and 0.5
            "max_observed_payout_multiplier": 100 + params[3] * 1900  # Between 100 and 2000
        }
    
    # Create constraint function
    constraint_function = create_constraint_function_for_optimizer(
        evaluation_func=evaluate_design,
        constraint_set=constraint_set
    )
    
    # TODO: Create proper player model function
    # For now, create a simple dummy function
    def player_model(eval_results):
        # Dummy player model - in real implementation this would use player behavior models
        variance = eval_results.get("variance", 10.0)
        rtp = eval_results.get("rtp", 0.92)
        hit_freq = eval_results.get("hit_frequency", 0.3)
        
        # Simple model: players play longer with higher RTP, lower variance, higher hit frequency
        expected_spins = 1000 * (rtp / 0.92) * (10.0 / variance) * (hit_freq / 0.3)
        return {
            "expected_total_spins": expected_spins,
            "average_bet": 1.0
        }
    
    # Create objective function
    def objective_metric(eval_results, player_behavior):
        # Simple profit metric: total profit over player lifetime
        rtp = eval_results.get("rtp", 0.92)
        spins = player_behavior.get("expected_total_spins", 1000)
        bet = player_behavior.get("average_bet", 1.0)
        
        return spins * bet * (1 - rtp)  # House edge * total bet amount
    
    # Wrap the create_objective_function to inject the np reference
    def create_objective_function_wrapper(evaluation_func, player_model_func, objective_metric_func, maximize=True):
        def objective_function(design_params):
            eval_results = evaluation_func(design_params)
            player_behavior = player_model_func(eval_results)
            objective_value = objective_metric_func(eval_results, player_behavior)
            
            # Ensure finite value (handle potential NaN or Inf from calculations)
            if not np.isfinite(objective_value):
                # Return a very bad value if maximization, very good if minimization
                return -np.inf if maximize else np.inf
            
            return objective_value if maximize else -objective_value
        
        return objective_function
    
    objective_function = create_objective_function_wrapper(
        evaluation_func=evaluate_design,
        player_model_func=player_model,
        objective_metric_func=objective_metric,
        maximize=True
    )
    
    # Create and run optimizer
    try:
        optimizer = ScipyDifferentialEvolutionOptimizer(
            objective_func=objective_function, 
            constraint_func=constraint_function
        )
        
        logger.info(f"Running optimization with {len(bounds)} parameters")
        results = optimizer.solve(
            bounds=bounds, 
            max_iter=max_iter,
            pop_size=pop_size,
            strategy=strategy
        )
        
        # Save results
        save_results(results, output_dir, config)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise

if __name__ == "__main__":
    # When run as a script, parse args and run
    import argparse
    
    parser = argparse.ArgumentParser(description="Run slot game parameter optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to optimization configuration")
    parser.add_argument("--output", type=str, default="data/optimization_results", help="Output directory")
    
    args = parser.parse_args()
    run_optimization(config_path=args.config, output_dir=args.output) 