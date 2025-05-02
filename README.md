# Slot Game Theory: Analysis and Design Toolkit

This project provides a Python toolkit and research framework for the mathematical analysis, simulation, and optimal design of slot machine games. It leverages principles from game theory, probability, simulation, and optimization to model the complex interplay between game mechanics, player experience, and operator objectives.

The toolkit aims to model the slot designer as a strategic agent balancing multiple objectives, including:
*   **Operator Profit:** Maximizing revenue generation within regulatory bounds.
*   **Player Engagement:** Designing games that are enjoyable and retain players.
*   **Regulatory Compliance:** Ensuring metrics like Return-to-Player (RTP) meet jurisdictional requirements.

## Core Concepts

*   **Slot Mechanics:** Modeling reels, symbols, paylines, paytables, and various bonus features (free spins, pick bonuses, jackpots).
*   **Mathematical Analysis:** Calculating key performance indicators (KPIs) like RTP, variance, hit frequency, and feature frequency using probability theory and techniques like Markov chains.
*   **Simulation:** Employing Monte Carlo methods to estimate KPIs and analyze game dynamics over millions or billions of spins.
*   **Optimization:** Using algorithms (e.g., genetic algorithms) to find optimal game parameters (reel strip configurations, paytable values, bonus triggers) that satisfy specific objectives and constraints.
*   **Player Modeling:** Incorporating models like Prospect Theory to understand and predict player behavior and risk preferences.

## Key Features

*   **Flexible Game Configuration:** Define slot game parameters using JSON configuration files (`configs/`).
*   **Core Mathematical Engine:**
    *   Calculate exact or simulated RTP, variance, hit frequency.
    *   **Markov Chain Analysis:** (`src/slot_game_theory/core/markov.py`) Model game states and transitions to analyze long-term probabilities, expected feature times, and state values.
*   **Simulation Framework:** (`src/slot_game_theory/simulation/`) Run large-scale simulations to validate analytical results and explore game dynamics.
*   **Optimization Module:** (`src/slot_game_theory/optimization/`) Optimize game parameters against defined objectives and constraints.
*   **Data Visualization:** Generate plots for transition diagrams, simulation results, and optimization progress (requires `matplotlib` and `networkx`).

## Project Structure

```
├── configs/             # JSON configuration files for games, simulations, optimizations
│   ├── sample_slot_game.json
│   ├── default_simulation.json
│   └── default_optimization.json
├── data/                # (Optional) For storing simulation results or real-world data
├── notebooks/           # Jupyter notebooks for examples, analysis, and visualization
├── src/slot_game_theory/ # Main source code
│   ├── core/            # Core game mechanics, probability, math (e.g., markov.py)
│   ├── simulation/      # Simulation engine and components
│   ├── optimization/    # Optimization algorithms and setup
│   ├── utils/           # Utility functions
│   └── __init__.py
├── tests/               # Unit and integration tests
├── .gitignore
├── LICENSE
├── PLAN.md              # Detailed project plan and architecture
├── README.md            # This file
└── requirements.txt     # Project dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Slot-Game-Theory-Toolkit.git # Replace with actual URL
    cd Slot-Game-Theory-Toolkit
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` may need to be created or updated)*

## Usage Example (Markov Analysis)

```python
# Example usage (within a script or notebook)
import numpy as np
from src.slot_game_theory.core.markov import MarkovChain, slot_game_markov_example

# Run the pre-defined slot game example analysis
# This will print metrics and generate a transition diagram plot
slot_game_mc = slot_game_markov_example()

# --- Or define a simple custom chain ---
states = ['StateA', 'StateB']
P = np.array([
    [0.8, 0.2], # Transitions from StateA
    [0.1, 0.9]  # Transitions from StateB
])
custom_mc = MarkovChain(states, P)
stationary_dist = custom_mc.get_stationary_distribution()
print(f"\nCustom Chain Stationary Distribution: {stationary_dist}")

# You can now use custom_mc methods like:
# custom_mc.simulate_chain('StateA', 100)
# custom_mc.first_passage_time('StateA', 'StateB')
# custom_mc.plot_transition_diagram()
```

*(Refer to `notebooks/` for more detailed examples when available)*

## Contributing

Contributions are welcome! Please follow these general guidelines:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Add tests for your changes.
5.  Ensure all tests pass.
6.  Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.