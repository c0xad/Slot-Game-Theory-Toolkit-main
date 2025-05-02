# Project Plan: Slot Game Theory Toolkit & Research

This document outlines the proposed structure for the project, combining a Python toolkit for slot game analysis and optimization with supporting research artifacts.

## Proposed Project Structure

```
casino/
├── .gitignore             # Specifies intentionally untracked files that Git should ignore
├── LICENSE                # Project license file (e.g., MIT as mentioned)
├── README.md              # Top-level project description, setup, usage guide
├── pyproject.toml         # Configuration for build system, dependencies (e.g., Poetry, Flit)
│
├── data/                  # Directory for all data related to the project
│   ├── raw/               # Original, immutable data sources (e.g., anonymized logs)
│   │   └── .gitkeep       # Placeholder to ensure Git tracks the empty directory
│   ├── processed/         # Data after cleaning, transformation, feature engineering
│   │   └── .gitkeep
│   └── simulation_results/ # Outputs from simulation runs (parameters, metrics)
│       └── .gitkeep
│
├── docs/                  # Project documentation source files (e.g., for Sphinx/MkDocs)
│   ├── index.md           # Main documentation landing page
│   ├── theory.md          # Detailed explanation of the game-theoretic model
│   ├── toolkit_usage.md   # Guide on using the Python library functions
│   ├── methodology.md     # Details on simulation, optimization methods
│   └── research_notes.md  # Misc notes, literature review summaries
│
├── notebooks/             # Jupyter notebooks for analysis, visualization, prototyping
│   ├── 01_data_exploration.ipynb  # Initial exploration of raw data
│   ├── 02_model_prototyping.ipynb # Testing core logic, symbolic evaluation
│   ├── 03_simulation_analysis.ipynb # Analyzing results from Monte Carlo runs
│   ├── 04_optimization_runs.ipynb # Setting up and analyzing optimization results
│   └── 05_validation_analysis.ipynb # Comparing model predictions to actual data
│
├── research/              # Files related to the research process and outputs
│   ├── literature/        # PDFs, notes, summaries of reviewed papers
│   │   └── .gitkeep
│   ├── manuscript/        # Drafts, figures, tables for papers or reports
│   │   └── .gitkeep
│   └── presentations/     # Slides or materials for talks/posters
│       └── .gitkeep
│
├── src/                   # Source code for the installable Python package
│   └── slot_game_theory/  # The main Python package name
│       ├── __init__.py    # Makes the directory a Python package
│       │
│       ├── core/            # Core game definitions and mechanics
│       │   ├── __init__.py
│       │   ├── reels.py       # Classes/functions for reel strips, symbols, paylines
│       │   ├── paytable.py    # Handling paytable definitions and lookups
│       │   ├── bonuses.py     # Logic for bonus rounds, free spins, features
│       │   └── markov.py      # Markov chain models if applicable
│       │
│       ├── evaluation/      # Modules for calculating game metrics
│       │   ├── __init__.py
│       │   ├── symbolic.py    # Analytical calculation for simple cases (RTP, Var, HF)
│       │   └── simulation.py  # Monte Carlo simulation engine (GPU-accelerated?)
│       │
│       ├── optimization/    # Modules for optimizing game design parameters
│       │   ├── __init__.py
│       │   ├── objective.py   # Defining the objective function (e.g., operator profit)
│       │   ├── constraints.py # Implementing regulatory/design constraints
│       │   └── solver.py      # Equilibrium solver (bi-level optimization logic)
│       │
│       ├── player_models/   # Models of player behavior
│       │   ├── __init__.py
│       │   ├── prospect_theory.py # Prospect theory value functions, weighting
│       │   └── decision_rules.py # Player stop-rules, bet sizing logic
│       │
│       └── utils/           # Common utility functions (math, data loading, etc.)
│           ├── __init__.py
│           └── helpers.py
│
├── tests/                 # Automated tests for the source code
│   ├── __init__.py
│   ├── core/              # Tests for core mechanics
│   │   ├── __init__.py
│   │   └── test_reels.py
│   │   └── test_paytable.py
│   ├── evaluation/        # Tests for evaluation modules
│   │   ├── __init__.py
│   │   └── test_symbolic.py
│   │   └── test_simulation.py
│   ├── optimization/      # Tests for optimization logic
│   │   ├── __init__.py
│   │   └── test_constraints.py
│   └── player_models/     # Tests for player models
│       ├── __init__.py
│       └── test_prospect_theory.py
│
└── scripts/               # Standalone scripts for running tasks (e.g., batch simulations)
    ├── run_simulation_batch.py # Example script to run multiple simulations
    └── generate_report.py    # Example script to generate analysis reports
```

## Mermaid Diagram

```mermaid
graph TD
    subgraph Project Root
        A[casino]
    end

    subgraph Core Files
        B[.gitignore]
        C[LICENSE]
        D[README.md]
        E[pyproject.toml]
    end

    subgraph Data
        F[data] --> F1[raw/.gitkeep]
        F --> F2[processed/.gitkeep]
        F --> F3[simulation_results/.gitkeep]
    end

    subgraph Documentation
        G[docs] --> G1[index.md]
        G --> G2[theory.md]
        G --> G3[toolkit_usage.md]
        G --> G4[methodology.md]
        G --> G5[research_notes.md]
    end

    subgraph Notebooks
        H[notebooks] --> H1[01_data_exploration.ipynb]
        H --> H2[02_model_prototyping.ipynb]
        H --> H3[03_simulation_analysis.ipynb]
        H --> H4[04_optimization_runs.ipynb]
        H --> H5[05_validation_analysis.ipynb]
    end

    subgraph Research Artifacts
        I[research] --> I1[literature/.gitkeep]
        I --> I2[manuscript/.gitkeep]
        I --> I3[presentations/.gitkeep]
    end

    subgraph Source Code (src/slot_game_theory)
        J[src] --> J1[slot_game_theory]
        J1 --> J1_init[__init__.py]
        J1 --> J1_core[core] --> J1_core_files[reels.py, paytable.py, bonuses.py, markov.py]
        J1 --> J1_eval[evaluation] --> J1_eval_files[symbolic.py, simulation.py]
        J1 --> J1_opt[optimization] --> J1_opt_files[objective.py, constraints.py, solver.py]
        J1 --> J1_player[player_models] --> J1_player_files[prospect_theory.py, decision_rules.py]
        J1 --> J1_utils[utils] --> J1_utils_files[helpers.py]
    end

    subgraph Tests
        K[tests] --> K1[core] --> K1_files[test_reels.py, ...]
        K --> K2[evaluation] --> K2_files[test_symbolic.py, ...]
        K --> K3[optimization] --> K3_files[test_constraints.py, ...]
        K --> K4[player_models] --> K4_files[test_prospect_theory.py, ...]
    end

    subgraph Scripts
        L[scripts] --> L1[run_simulation_batch.py]
        L --> L2[generate_report.py]
    end

    A --> B; A --> C; A --> D; A --> E;
    A --> F; A --> G; A --> H; A --> I; A --> J; A --> K; A --> L;
```

## Explanation of Key Directories

*   **`data/`**: Holds all data, separated into raw inputs, processed/cleaned data, and simulation outputs. Keeping raw data separate and immutable is good practice.
*   **`docs/`**: Contains the source files for generating project documentation. This could use tools like Sphinx or MkDocs.
*   **`notebooks/`**: Ideal for exploratory data analysis, model prototyping, and visualizing results interactively. Numbering helps suggest a workflow.
*   **`research/`**: Stores non-code research materials like papers, notes, and presentations.
*   **`src/slot_game_theory/`**: The heart of the Python toolkit. It's structured as an installable package (`slot_game_theory`). Subdirectories group related functionality (core mechanics, evaluation, optimization, player modeling, utilities). Placeholder `.py` files are included based on the proposal's methodology.
*   **`tests/`**: Contains unit and integration tests mirroring the `src/` structure. Essential for ensuring the toolkit's correctness and reliability.
*   **`scripts/`**: For helper scripts that might perform larger tasks using the toolkit, like running batches of simulations or generating specific reports.