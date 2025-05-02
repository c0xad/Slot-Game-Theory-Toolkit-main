# -*- coding: utf-8 -*-
"""
Provides tools for modeling slot game mechanics using Markov chains.

Useful for analyzing games with distinct states (e.g., base game, different bonus stages)
and calculating long-term probabilities or expected values.

Requires careful state definition and transition probability calculation.
Can become complex for games with many features or continuous variables.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Hashable, Optional, Tuple, Union

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
        
    def expected_return_from_state(self, state_values: Dict[StateType, float]) -> Dict[StateType, float]:
        """
        Calculates the expected return starting from each state, given values for each state.
        
        Args:
            state_values: Dictionary mapping states to their immediate value/return
            
        Returns:
            Dictionary mapping each state to its expected long-term return
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return {}
            
        # Convert state values to vector form
        value_vector = np.zeros(self.num_states)
        for state, value in state_values.items():
            if state in self.state_to_index:
                value_vector[self.state_to_index[state]] = value
            else:
                print(f"Warning: State '{state}' not found in Markov chain states.")
        
        # Calculate stationary distribution
        stationary_dist = self.get_stationary_distribution()
        if stationary_dist is None:
            print("Error: Could not calculate stationary distribution.")
            return {}
            
        # Calculate expected returns using the fundamental matrix
        I = np.identity(self.num_states)
        try:
            fundamental_matrix = np.linalg.inv(I - self.transition_matrix)
        except np.linalg.LinAlgError:
            print("Error: Could not calculate fundamental matrix.")
            return {}
            
        expected_returns = fundamental_matrix @ value_vector
        
        # Convert back to dictionary
        return {state: expected_returns[self.state_to_index[state]] for state in self.states}
    
    def first_passage_time(self, from_state: StateType, to_state: StateType) -> Optional[float]:
        """
        Calculates the expected number of steps to reach 'to_state' starting from 'from_state'.
        
        Args:
            from_state: The starting state
            to_state: The target state
            
        Returns:
            Expected number of steps, or None if calculation fails
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return None
            
        try:
            from_idx = self.state_to_index[from_state]
            to_idx = self.state_to_index[to_state]
        except KeyError as e:
            print(f"Error: State {e} not found in Markov chain states.")
            return None
            
        # If starting at the target state, expected time is 0
        if from_state == to_state:
            return 0.0
            
        # Create a modified chain where the target state is absorbing
        modified_matrix = self.transition_matrix.copy()
        modified_matrix[to_idx, :] = 0
        modified_matrix[to_idx, to_idx] = 1
        
        # Calculate the fundamental matrix for the modified chain
        Q = np.delete(np.delete(modified_matrix, to_idx, 0), to_idx, 1)
        I = np.identity(self.num_states - 1)
        
        try:
            N = np.linalg.inv(I - Q)
            # The expected time is the sum of the row corresponding to from_state
            from_idx_adjusted = from_idx if from_idx < to_idx else from_idx - 1
            return float(np.sum(N[from_idx_adjusted, :]))
        except np.linalg.LinAlgError:
            print("Error: Could not calculate first passage time.")
            return None
    
    def simulate_chain(self, start_state: StateType, num_steps: int) -> List[StateType]:
        """
        Simulates a random walk through the Markov chain.
        
        Args:
            start_state: The starting state for the simulation
            num_steps: The number of steps to simulate
            
        Returns:
            A list of states visited during the simulation
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return []
            
        try:
            current_idx = self.state_to_index[start_state]
        except KeyError:
            print(f"Error: Start state '{start_state}' not found in Markov chain states.")
            return []
            
        states_visited = [start_state]
        
        for _ in range(num_steps):
            # Get transition probabilities from current state
            transition_probs = self.transition_matrix[current_idx, :]
            
            # Choose next state based on transition probabilities
            next_idx = np.random.choice(self.num_states, p=transition_probs)
            next_state = self.states[next_idx]
            
            states_visited.append(next_state)
            current_idx = next_idx
            
        return states_visited
        
    def plot_transition_diagram(self, figsize: Tuple[int, int] = (10, 8), min_prob: float = 0.01):
        """
        Plots a graphical representation of the Markov chain.
        
        Args:
            figsize: Figure size as (width, height)
            min_prob: Minimum probability to display as a connection
        """
        if self._transition_matrix is None:
            print("Error: Transition matrix not set.")
            return
            
        try:
            import networkx as nx
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for state in self.states:
                G.add_node(state)
                
            # Add edges with weights based on transition probabilities
            for i, from_state in enumerate(self.states):
                for j, to_state in enumerate(self.states):
                    prob = self.transition_matrix[i, j]
                    if prob > min_prob:  # Only add edges with probability above threshold
                        G.add_edge(from_state, to_state, weight=prob, label=f"{prob:.3f}")
            
            # Plot
            plt.figure(figsize=figsize)
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
            
            # Draw edges
            edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, arrowsize=20, alpha=0.7)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            # Draw edge labels (probabilities)
            edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
            
            plt.axis('off')
            plt.title("Markov Chain Transition Diagram")
            plt.tight_layout()
            
        except ImportError:
            print("Error: NetworkX and/or matplotlib required for visualization.")

    def __repr__(self) -> str:
        return f"MarkovChain(num_states={self.num_states}, states={self.states})"


# Example Usage: Modern Slot Game with Multiple Features
def slot_game_markov_example():
    """Example demonstrating a realistic slot game model using Markov chains."""
    # Define game states
    states = [
        'BaseGame',              # Main game play
        'FreeSpins10',           # 10 free spins awarded
        'FreeSpins15',           # 15 free spins awarded
        'FreeSpins20',           # 20 free spins awarded
        'PickBonus_Low',         # Pick bonus with low multipliers
        'PickBonus_High',        # Pick bonus with high multipliers
        'Jackpot_Mini',          # Mini jackpot triggered
        'Jackpot_Major',         # Major jackpot triggered
        'Jackpot_Grand',         # Grand jackpot triggered
        'Respin_Feature',        # Hold and respin feature
        'Gamble_Feature'         # Gamble feature after any win
    ]
    
    # Transition matrix with realistic probabilities
    # Rows and columns correspond to states list order
    P = np.array([
        # BaseGame  FS10    FS15    FS20    Pick_L  Pick_H  J_Mini  J_Major J_Grand Respin  Gamble
        [0.920,     0.020,  0.010,  0.005,  0.015,  0.005,  0.010,  0.003,  0.001,  0.006,  0.005],  # From BaseGame
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From FreeSpins10
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From FreeSpins15
        [0.980,     0.000,  0.000,  0.020,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From FreeSpins20 (small retrigger chance)
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From PickBonus_Low
        [0.980,     0.000,  0.000,  0.000,  0.000,  0.000,  0.010,  0.007,  0.003,  0.000,  0.000],  # From PickBonus_High (might lead to jackpots)
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From Jackpot_Mini
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From Jackpot_Major
        [1.000,     0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000],  # From Jackpot_Grand
        [0.850,     0.050,  0.030,  0.020,  0.020,  0.010,  0.010,  0.005,  0.002,  0.003,  0.000],  # From Respin_Feature (can trigger other features)
        [0.875,     0.050,  0.025,  0.015,  0.015,  0.010,  0.005,  0.003,  0.001,  0.001,  0.000]   # From Gamble_Feature
    ])
    
    # Create Markov chain model
    mc = MarkovChain(states, P)
    
    # Calculate and display stationary distribution
    stationary_dist = mc.get_stationary_distribution()
    if stationary_dist is not None:
        print("Long-term probability of being in each state:")
        for state, prob in zip(states, stationary_dist):
            print(f"  {state}: {prob:.6f} ({prob*100:.4f}%)")
            
        # Calculate expected values
        state_values = {
            'BaseGame': 0.85,           # Average return during base game (85% RTP)
            'FreeSpins10': 20.0,        # Expected value of 10 free spins
            'FreeSpins15': 35.0,        # Expected value of 15 free spins
            'FreeSpins20': 55.0,        # Expected value of 20 free spins
            'PickBonus_Low': 15.0,      # Expected value of low pick bonus
            'PickBonus_High': 40.0,     # Expected value of high pick bonus
            'Jackpot_Mini': 50.0,       # Mini jackpot value
            'Jackpot_Major': 250.0,     # Major jackpot value
            'Jackpot_Grand': 5000.0,    # Grand jackpot value
            'Respin_Feature': 25.0,     # Expected value of respin feature
            'Gamble_Feature': 1.0       # Gamble has 1:1 expected value (fair gamble)
        }
        
        # Calculate expected return from each state
        expected_returns = mc.expected_return_from_state(state_values)
        print("\nExpected return starting from each state:")
        for state, value in expected_returns.items():
            print(f"  {state}: {value:.4f}")
            
        # Calculate expected features per session (100 spins)
        spins_per_session = 100
        print(f"\nExpected number of features per {spins_per_session} spins:")
        for state, prob in zip(states[1:], stationary_dist[1:]):  # Skip base game
            expected_occurrences = prob * spins_per_session / stationary_dist[0]
            print(f"  {state}: {expected_occurrences:.4f}")
            
        # Simulate a player session
        print("\nSimulating 200 spins starting from BaseGame:")
        simulation = mc.simulate_chain('BaseGame', 200)
        feature_counts = {state: simulation.count(state) for state in states}
        print("Feature occurrences in simulation:")
        for state, count in feature_counts.items():
            if state != 'BaseGame':  # Skip base game
                print(f"  {state}: {count}")
                
        # Calculate first passage times
        print("\nExpected spins to trigger features from BaseGame:")
        for state in states[1:]:  # Skip base game
            passage_time = mc.first_passage_time('BaseGame', state)
            if passage_time is not None:
                print(f"  {state}: {passage_time:.2f} spins")
        
        # Visualize the Markov chain
        mc.plot_transition_diagram(figsize=(12, 10), min_prob=0.005)
        
    return mc


if __name__ == "__main__":
    # Run the example when the script is executed directly
    slot_game_mc = slot_game_markov_example()