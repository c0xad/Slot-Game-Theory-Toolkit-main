#!/usr/bin/env python3
"""
Slot Game Theory Toolkit - Main Entry Point

This script serves as the main entry point for the Slot Game Theory Toolkit.
It provides a command-line interface to access different functionalities of the project.
"""

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).resolve().parent)
sys.path.insert(0, project_root)

def main():
    """Main entry point for the Slot Game Theory Toolkit."""
    parser = argparse.ArgumentParser(
        description="Slot Game Theory Toolkit - A framework for analyzing and designing slot games"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Run simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run slot game simulations")
    sim_parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_simulation.json",
        help="Path to simulation configuration file"
    )
    sim_parser.add_argument(
        "--output", 
        type=str, 
        default="data/simulation_results",
        help="Output directory for simulation results"
    )
    sim_parser.add_argument(
        "--runs", 
        type=int, 
        default=10000,
        help="Number of simulation runs"
    )
    
    # Optimization command
    opt_parser = subparsers.add_parser("optimize", help="Optimize slot game parameters")
    opt_parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_optimization.json",
        help="Path to optimization configuration file"
    )
    opt_parser.add_argument(
        "--output", 
        type=str, 
        default="data/optimization_results",
        help="Output directory for optimization results"
    )
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate analysis reports")
    report_parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to simulation or optimization results data"
    )
    report_parser.add_argument(
        "--output", 
        type=str, 
        default="data/reports",
        help="Output directory for reports"
    )
    report_parser.add_argument(
        "--template", 
        type=str, 
        default="standard",
        help="Report template to use"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "simulate":
        # Import and run simulation script
        try:
            from scripts.run_simulation_batch import run_simulation
            run_simulation(config_path=args.config, output_dir=args.output, num_runs=args.runs)
        except ImportError as e:
            print(f"Error: Could not import simulation module. Make sure the project is set up correctly.")
            print(f"Details: {e}")
            return 1
        except Exception as e:
            print(f"Error running simulation: {e}")
            return 1
            
    elif args.command == "optimize":
        # Import and run optimization script 
        try:
            from scripts.run_optimization import run_optimization
            run_optimization(config_path=args.config, output_dir=args.output)
        except ImportError as e:
            print(f"Error: Could not import optimization module. Make sure the project is set up correctly.")
            print(f"Details: {e}")
            return 1
        except Exception as e:
            print(f"Error running optimization: {e}")
            return 1
            
    elif args.command == "report":
        # Import and run report generation script
        try:
            from scripts.generate_report import generate_report
            generate_report(data_path=args.data, output_dir=args.output, template=args.template)
        except ImportError as e:
            print(f"Error: Could not import report generation module. Make sure the project is set up correctly.")
            print(f"Details: {e}")
            return 1
        except Exception as e:
            print(f"Error generating report: {e}")
            return 1
            
    else:
        # If no command specified, show help
        parser.print_help()
        return 0
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 