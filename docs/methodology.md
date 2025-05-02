# Methodology Details

This section outlines the methods used for evaluation, simulation, optimization, and validation.

## Symbolic Evaluation Module

*   For small state spaces (e.g., classic 3x5 reels).
*   Enumerates the full outcome space.
*   Calculates exact RTP, variance (σ²), and hit frequency (hf).
*   Leverages generating functions where applicable.

## High-Dimensional Simulation Module

*   For complex video slots and progressive jackpots.
*   GPU-accelerated Monte Carlo methods.
*   Variance reduction techniques:
    *   Antithetic variates.
    *   Control variates (especially for jackpot estimation).
*   Efficient sampling strategies.

## Equilibrium Solver

*   Bi-level optimization framework:
    *   **Outer Loop (Designer):** Maximizes operator objective (e.g., profit) by choosing `(s, p, β)` within regulatory region `R`. Uses algorithms like Differential Evolution.
    *   **Inner Loop (Player):** Minimizes player objective (or maximizes prospect theory value) by choosing `(τ, b)`. Solved via dynamic programming / backward induction on the player's decision process.
*   Embeds the Symbolic/Simulation modules to evaluate metrics for given designs.

## Regulatory Stress-Testing

*   Simulates play sessions under specific designs.
*   Applies statistical tests based on regulatory requirements (e.g., UKGC, MGA live-RTP variance limits).
*   Flags designs that fail confidence interval tests over simulated time windows.

## Validation

*   Compares model predictions (RTP trajectory, variance, player lifetime, churn) against anonymized real-world game log data.
*   Requires access to partner studio data (>10M spins target).
*   Statistical analysis of prediction accuracy.