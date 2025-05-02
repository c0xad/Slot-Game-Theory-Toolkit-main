# -*- coding: utf-8 -*-
"""
Provides tools for modeling slot game mechanics using Markov chains.

Useful for analyzing games with distinct states (e.g., base game, different bonus stages)
and calculating long-term probabilities or expected values.

Requires careful state definition and transition probability calculation.
Can become complex for games with many features or continuous variables.
"""

import numpy as np
from typing import List, Dict, Hashable, Optional

# Define a type for states (can be integers, strings, tuples, etc.)
StateType = Hashable

class MarkovChain:
    """
    Represents a discrete-time Markov chain for modeling game states.
    """
    def __init__(self, states: List[StateType], transition_matrix: Optional[np.ndarray] = None):
        """
        Initializes the Markov chain.

        Args:
            states: An ordered list of the possible states in the chain.
            transition_matrix: A square NumPy array where P[i, j] is the probability
                               of transitioning from state i to state j.
                               If None, it must be set later.
        """
        self.states = list(states) # Ensure it's a list
        self.state_to_index: Dict[StateType, int] = {state: i for i, state in enumerate(self.states)}
        self.num_states = len(states)

        if transition_matrix is not None:
            self.set_transition_matrix(transition_matrix)
        else:
            self._transition_matrix: Optional[np.ndarray] = None

    def set_transition_matrix(self, transition_matrix: np.ndarray):
        """Sets and validates the transition matrix."""
        if not isinstance(transition_matrix, np.ndarray):
            raise TypeError("Transition matrix must be a NumPy array.")
        if transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError(f"Transition matrix shape must be ({self.num_states}, {self.num_states}), "
                             f"got {transition_matrix.shape}")
        # Check if rows sum to 1 (approximately)
        if not np.allclose(transition_matrix.sum(axis=1), 1.0):
            print("Warning: Rows of the transition matrix do not sum to 1.")
            # Depending on tolerance, might raise ValueError here instead
            # raise ValueError("Rows of the transition matrix must sum to 1.")
        self._transition_matrix = transition_matrix

    @property
    def transition_matrix(self) -> np.ndarray:
        """Returns the transition matrix."""
        if self._transition_matrix is None:
            raise ValueError("Transition matrix has not been set.")
        return self._transition_matrix

    def get_stationary_distribution(self) -> Optional[np.ndarray]:
        """
        Calculates the stationary distribution (long-term probabilities of being in each state).

        Assumes the chain is irreducible and aperiodic.
        Returns None if the calculation fails or the matrix is not set.
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return None

        try:
            # We need to solve pi * P = pi, subject to sum(pi) = 1.
            # This is equivalent to finding the left eigenvector of P corresponding to eigenvalue 1.
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

            # Find the index of the eigenvalue closest to 1
            one_indices = np.isclose(eigenvalues, 1.0)

            if not np.any(one_indices):
                print("Warning: No eigenvalue close to 1 found. Stationary distribution might not exist or be unique.")
                return None

            # Get the eigenvector corresponding to eigenvalue 1
            # There might be multiple indices close to 1 due to floating point issues, take the first
            stationary_vector = eigenvectors[:, one_indices].T[0]

            # Normalize the eigenvector to sum to 1
            stationary_distribution = stationary_vector / stationary_vector.sum()

            # Return the real part (imaginary part should be negligible if calculation is correct)
            return np.real(stationary_distribution)

        except np.linalg.LinAlgError:
            print("Error: Eigenvalue decomposition failed.")
            return None

    def expected_steps_to_absorption(self, transient_states: List[StateType], absorbing_states: List[StateType]) -> Optional[np.ndarray]:
        """
        Calculates the expected number of steps to reach an absorbing state,
        starting from each transient state.

        Requires the chain to have absorbing states.
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return None

        transient_indices = [self.state_to_index[s] for s in transient_states]
        absorbing_indices = [self.state_to_index[s] for s in absorbing_states]

        if not transient_indices or not absorbing_indices:
            print("Error: Both transient and absorbing states must be specified.")
            return None

        # Extract the submatrix Q corresponding to transitions between transient states
        Q = self.transition_matrix[np.ix_(transient_indices, transient_indices)]

        # Calculate the fundamental matrix N = (I - Q)^-1
        try:
            I = np.identity(len(transient_indices))
            N = np.linalg.inv(I - Q)
        except np.linalg.LinAlgError:
            print("Error: Could not invert (I - Q). Matrix might be singular.")
            return None

        # The expected number of steps is the sum of each row in N
        expected_steps = N.sum(axis=1)
        return expected_steps

    def __repr__(self) -> str:
        return f"MarkovChain(num_states={self.num_states}, states={self.states})"


# Example Usage (Conceptual)
# states = ['BaseGame', 'FreeSpins', 'PickBonus']
# P = np.array([
#     [0.95, 0.04, 0.01], # Transitions from BaseGame
#     [1.00, 0.00, 0.00], # Transitions from FreeSpins (returns to BaseGame after completion)
#     [1.00, 0.00, 0.00]  # Transitions from PickBonus (returns to BaseGame)
# ])
# mc = MarkovChain(states, P)
# stationary_dist = mc.get_stationary_distribution()
# print(f"Stationary Distribution: {stationary_dist}")