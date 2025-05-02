# Using the Slot Game Theory Toolkit

This document explains how to use the main entry point (`main.py`) to run various components of the Slot Game Theory Toolkit.

## Prerequisites

1. Ensure you have Python 3.8+ installed
2. Install the project and its dependencies:
   ```
   pip install -e .
   ```

## Running the Project

The main entry point provides a command-line interface with several commands for different functionalities.

### General Usage

```
python main.py [command] [options]
```

Available commands:
- `simulate`: Run slot game simulations
- `optimize`: Optimize slot game parameters
- `report`: Generate analysis reports

If you run `main.py` without any arguments, it will display the help information.

### Running Simulations

To run slot game simulations:

```
python main.py simulate [options]
```

Options:
- `--config`: Path to the simulation configuration file (default: "configs/default_simulation.json")
- `--output`: Output directory for simulation results (default: "data/simulation_results")
- `--runs`: Number of simulation runs (default: 10000)

Example:
```
python main.py simulate --config configs/custom_config.json --runs 50000
```

### Running Optimization

To optimize slot game parameters:

```
python main.py optimize [options]
```

Options:
- `--config`: Path to the optimization configuration file (default: "configs/default_optimization.json")
- `--output`: Output directory for optimization results (default: "data/optimization_results")

Example:
```
python main.py optimize --config configs/custom_optimization.json
```

### Generating Reports

To generate analysis reports from simulation or optimization results:

```
python main.py report [options]
```

Options:
- `--data`: Path to simulation or optimization results data (required)
- `--output`: Output directory for reports (default: "data/reports")
- `--template`: Report template to use (default: "standard")

Example:
```
python main.py report --data data/simulation_results/run_20231015 --template detailed
```

## Configuration Files

The toolkit uses JSON configuration files to define parameters for simulations and optimizations. Example configuration files can be found in the `configs/` directory.

## Extending the Toolkit

To add new functionality, you can:

1. Create new modules in the appropriate directories under `src/slot_game_theory/`
2. Add new commands to the `main.py` script by extending the argument parser
3. Add new scripts in the `scripts/` directory for specific workflows

## Troubleshooting

If you encounter errors:

1. Ensure the project is correctly installed
2. Check that all required dependencies are installed
3. Verify that configuration files exist and are correctly formatted
4. Check console output for specific error messages 