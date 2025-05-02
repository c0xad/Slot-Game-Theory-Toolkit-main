# -*- coding: utf-8 -*-
"""
Defines constraints for the slot game design optimization problem.

These constraints represent regulatory requirements (e.g., RTP bands, max payout)
or design choices (e.g., maximum volatility, specific feature frequency).
They limit the feasible region for the optimization solver.
"""

from typing import Dict, Any, Callable, List, Tuple, Union
import numpy as np

# Assume access to evaluation results
# from ..evaluation.simulation import SimulationResult # Or symbolic result type

# Define a type for the design parameters being optimized
DesignParameters = Any # e.g., np.ndarray, Dict[str, Any]

# Define a type for the output of an evaluation function
EvaluationOutput = Dict[str, Any] # e.g., SimulationResult

class Constraint:
    """Base class for constraints."""
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        """
        Evaluates whether the constraint is satisfied.

        Args:
            design_params: The current design parameters being evaluated.
            eval_results: The results from evaluating the design (e.g., simulation metrics).

        Returns:
            True if the constraint is satisfied, False otherwise.
        """
        raise NotImplementedError

    def __call__(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        return self.evaluate(design_params, eval_results)

    def __repr__(self) -> str:
        return f"Constraint(name='{self.name}')"


class RTPConstraint(Constraint):
    """Constraint on the Return to Player (RTP)."""
    def __init__(self, min_rtp: float = -np.inf, max_rtp: float = np.inf):
        super().__init__(f"RTP Range [{min_rtp}, {max_rtp}]")
        if min_rtp > max_rtp:
            raise ValueError("min_rtp cannot be greater than max_rtp")
        self.min_rtp = min_rtp
        self.max_rtp = max_rtp

    def evaluate(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        rtp = eval_results.get("rtp")
        if rtp is None:
            print(f"Warning: RTP not found in evaluation results for constraint '{self.name}'. Constraint fails.")
            return False # Cannot satisfy if metric is missing
        return self.min_rtp <= rtp <= self.max_rtp


class MaxPayoutConstraint(Constraint):
    """Constraint on the maximum possible payout multiplier."""
    def __init__(self, max_multiplier: float):
        super().__init__(f"Max Payout Multiplier <= {max_multiplier}")
        if max_multiplier <= 0:
            raise ValueError("max_multiplier must be positive")
        self.max_multiplier = max_multiplier

    def evaluate(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        # This might require a different type of evaluation, e.g., iterating through
        # the paytable directly or running specific simulation checks.
        # For now, assume 'max_observed_payout' might be in eval_results from simulation.
        max_obs_payout = eval_results.get("max_observed_payout_multiplier")
        if max_obs_payout is None:
            # Need a way to determine theoretical max payout from design_params (paytable)
            # Placeholder: Assume constraint passes if not directly observed in simulation
            # print(f"Warning: Max observed payout not found for constraint '{self.name}'. Requires theoretical check.")
            # return check_theoretical_max_payout(design_params, self.max_multiplier)
            return True # Placeholder - needs proper implementation
        return max_obs_payout <= self.max_multiplier


class VarianceConstraint(Constraint):
    """Constraint on the payout variance (volatility index)."""
    def __init__(self, min_variance: float = -np.inf, max_variance: float = np.inf):
        super().__init__(f"Variance Range [{min_variance}, {max_variance}]")
        if min_variance > max_variance:
            raise ValueError("min_variance cannot be greater than max_variance")
        self.min_variance = min_variance
        self.max_variance = max_variance

    def evaluate(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        variance = eval_results.get("variance")
        if variance is None:
            print(f"Warning: Variance not found in evaluation results for constraint '{self.name}'. Constraint fails.")
            return False
        return self.min_variance <= variance <= self.max_variance

class HitFrequencyConstraint(Constraint):
    """Constraint on the hit frequency."""
    def __init__(self, min_hf: float = -np.inf, max_hf: float = np.inf):
        super().__init__(f"Hit Frequency Range [{min_hf}, {max_hf}]")
        if min_hf > max_hf:
            raise ValueError("min_hf cannot be greater than max_hf")
        self.min_hf = min_hf
        self.max_hf = max_hf

    def evaluate(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        hf = eval_results.get("hit_frequency")
        if hf is None:
            print(f"Warning: Hit Frequency not found for evaluation results for constraint '{self.name}'. Constraint fails.")
            return False
        return self.min_hf <= hf <= self.max_hf


# --- Constraint Set Management ---

class ConstraintSet:
    """Manages a collection of constraints."""
    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

    def evaluate_all(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        """Checks if all constraints in the set are satisfied."""
        for constraint in self.constraints:
            if not constraint.evaluate(design_params, eval_results):
                # print(f"Constraint failed: {constraint.name}") # Optional logging
                return False
        return True

    def __call__(self, design_params: DesignParameters, eval_results: EvaluationOutput) -> bool:
        return self.evaluate_all(design_params, eval_results)

    def __repr__(self) -> str:
        return f"ConstraintSet(num_constraints={len(self.constraints)})"


# --- Helper for Optimization Libraries ---

def create_constraint_function_for_optimizer(
    evaluation_func: Callable[[DesignParameters], EvaluationOutput],
    constraint_set: ConstraintSet
) -> Callable[[DesignParameters], Union[List[float], bool]]:
    """
    Creates a constraint evaluation function suitable for optimization libraries.

    Some libraries expect a function that returns a list of values (where <= 0 means feasible),
    while others might expect a boolean. This function provides a structure, but the
    exact output format needs to be adapted to the specific optimizer (e.g., scipy.optimize).

    Args:
        evaluation_func: Function to get metrics for a given design.
        constraint_set: The ConstraintSet instance.

    Returns:
        A callable function `g(design_params) -> constraint_values_or_bool`.
    """
    def constraint_function(design_params: DesignParameters) -> bool: # -> List[float] for scipy
        eval_results = evaluation_func(design_params)
        is_feasible = constraint_set.evaluate_all(design_params, eval_results)

        # Adapt the return value based on the optimizer's requirements:
        # Option 1: Boolean feasibility
        return is_feasible

        # Option 2: List of values for scipy.optimize (inequality constraints g(x) >= 0)
        # constraint_values = []
        # for constraint in constraint_set.constraints:
        #     # Need to reformulate each constraint into g(x) >= 0 form
        #     # Example: RTPConstraint (min <= rtp <= max) becomes:
        #     # g1(x) = rtp - min >= 0
        #     # g2(x) = max - rtp >= 0
        #     rtp = eval_results.get("rtp", -np.inf) # Handle missing metric
        #     if isinstance(constraint, RTPConstraint):
        #         constraint_values.append(rtp - constraint.min_rtp)
        #         constraint_values.append(constraint.max_rtp - rtp)
        #     # ... add conversions for other constraint types ...
        # return constraint_values

    return constraint_function