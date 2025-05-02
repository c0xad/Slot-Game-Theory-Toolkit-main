# Theoretical Framework

This section details the game-theoretic model underpinning the slot design analysis.

## Players

*   **Player 0 (Designer):** Chooses reel strips (`s`), paytable (`p`), bonus parameters (`β`).
*   **Player 1 (Gambler):** Chooses stake (`b`), stop-rule (`τ`) based on prospect theory value `V(·)`.

## Game Form

1.  **Stage 1:** Player 0 commits to `(s, p, β)` within regulatory constraints `R`.
2.  **Stage 2:** Nature draws outcomes based on transition matrix `P(s, β)`.
3.  **Stage 3:** Player 1 updates wealth, decides continue/quit, chooses next bet `b`.

## Solution Concept

*   Stackelberg equilibrium with a risk-biased follower (Player 1).

## Key Metrics

*   **RTP (Return to Player):** `E[Δw] / E[b]`
*   **Variance (σ²):** `Var[Δw | b]`
*   **Hit Frequency (hf):** `Pr{payout > 0}`

*(Refer to the original research proposal for more detailed formulations)*