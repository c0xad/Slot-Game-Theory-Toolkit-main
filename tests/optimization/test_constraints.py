# -*- coding: utf-8 -*-
"""Tests for the optimization constraints module."""

import pytest
import numpy as np
from slot_game_theory.optimization import constraints # Adjust import path

# --- Sample Data ---

@pytest.fixture
def sample_eval_results():
    """Sample evaluation results dictionary."""
    return {
        "rtp": 0.965,
        "variance": 35.2,
        "hit_frequency": 0.21,
        "max_observed_payout_multiplier": 500.0,
        # Add other metrics if needed by constraints
    }

@pytest.fixture
def sample_design_params():
    """Sample design parameters (content doesn't matter for these tests)."""
    return {"reel_strip_1": [0, 1, 2], "paytable": {}}

# --- Test Individual Constraints ---

def test_rtp_constraint(sample_design_params, sample_eval_results):
    """Test the RTPConstraint."""
    # Case 1: Within bounds
    c1 = constraints.RTPConstraint(min_rtp=0.95, max_rtp=0.97)
    assert c1.evaluate(sample_design_params, sample_eval_results) is True

    # Case 2: Below min bound
    c2 = constraints.RTPConstraint(min_rtp=0.97, max_rtp=0.98)
    assert c2.evaluate(sample_design_params, sample_eval_results) is False

    # Case 3: Above max bound
    c3 = constraints.RTPConstraint(min_rtp=0.95, max_rtp=0.96)
    assert c3.evaluate(sample_design_params, sample_eval_results) is False

    # Case 4: Exact boundary
    c4 = constraints.RTPConstraint(min_rtp=0.965, max_rtp=0.97)
    assert c4.evaluate(sample_design_params, sample_eval_results) is True
    c5 = constraints.RTPConstraint(min_rtp=0.95, max_rtp=0.965)
    assert c5.evaluate(sample_design_params, sample_eval_results) is True

    # Case 5: Metric missing
    c6 = constraints.RTPConstraint(min_rtp=0.95, max_rtp=0.97)
    assert c6.evaluate(sample_design_params, {}) is False # Empty results

    # Case 6: Invalid bounds
    with pytest.raises(ValueError):
        constraints.RTPConstraint(min_rtp=0.98, max_rtp=0.97)

def test_max_payout_constraint(sample_design_params, sample_eval_results):
    """Test the MaxPayoutConstraint."""
    # Case 1: Within bound
    c1 = constraints.MaxPayoutConstraint(max_multiplier=600.0)
    assert c1.evaluate(sample_design_params, sample_eval_results) is True

    # Case 2: Above bound
    c2 = constraints.MaxPayoutConstraint(max_multiplier=400.0)
    assert c2.evaluate(sample_design_params, sample_eval_results) is False

    # Case 3: Exact boundary
    c3 = constraints.MaxPayoutConstraint(max_multiplier=500.0)
    assert c3.evaluate(sample_design_params, sample_eval_results) is True

    # Case 4: Metric missing (current implementation returns True - needs refinement)
    c4 = constraints.MaxPayoutConstraint(max_multiplier=1000.0)
    assert c4.evaluate(sample_design_params, {}) is True # Placeholder behavior

    # Case 5: Invalid bound
    with pytest.raises(ValueError):
        constraints.MaxPayoutConstraint(max_multiplier=-100.0)
    with pytest.raises(ValueError):
        constraints.MaxPayoutConstraint(max_multiplier=0.0)


def test_variance_constraint(sample_design_params, sample_eval_results):
    """Test the VarianceConstraint."""
    c1 = constraints.VarianceConstraint(min_variance=30.0, max_variance=40.0)
    assert c1.evaluate(sample_design_params, sample_eval_results) is True

    c2 = constraints.VarianceConstraint(min_variance=40.0, max_variance=50.0)
    assert c2.evaluate(sample_design_params, sample_eval_results) is False

    c3 = constraints.VarianceConstraint(min_variance=30.0, max_variance=35.0)
    assert c3.evaluate(sample_design_params, sample_eval_results) is False

    c4 = constraints.VarianceConstraint(min_variance=35.2, max_variance=40.0)
    assert c4.evaluate(sample_design_params, sample_eval_results) is True

    c5 = constraints.VarianceConstraint(min_variance=30.0, max_variance=35.2)
    assert c5.evaluate(sample_design_params, sample_eval_results) is True

    c6 = constraints.VarianceConstraint(min_variance=30.0, max_variance=40.0)
    assert c6.evaluate(sample_design_params, {}) is False # Metric missing

    with pytest.raises(ValueError):
        constraints.VarianceConstraint(min_variance=50.0, max_variance=40.0)


def test_hit_frequency_constraint(sample_design_params, sample_eval_results):
    """Test the HitFrequencyConstraint."""
    c1 = constraints.HitFrequencyConstraint(min_hf=0.20, max_hf=0.25)
    assert c1.evaluate(sample_design_params, sample_eval_results) is True

    c2 = constraints.HitFrequencyConstraint(min_hf=0.22, max_hf=0.25)
    assert c2.evaluate(sample_design_params, sample_eval_results) is False

    c3 = constraints.HitFrequencyConstraint(min_hf=0.20, max_hf=0.205)
    assert c3.evaluate(sample_design_params, sample_eval_results) is False

    c4 = constraints.HitFrequencyConstraint(min_hf=0.21, max_hf=0.25)
    assert c4.evaluate(sample_design_params, sample_eval_results) is True

    c5 = constraints.HitFrequencyConstraint(min_hf=0.20, max_hf=0.21)
    assert c5.evaluate(sample_design_params, sample_eval_results) is True

    c6 = constraints.HitFrequencyConstraint(min_hf=0.20, max_hf=0.25)
    assert c6.evaluate(sample_design_params, {}) is False # Metric missing

    with pytest.raises(ValueError):
        constraints.HitFrequencyConstraint(min_hf=0.3, max_hf=0.2)


# --- Test ConstraintSet ---

def test_constraint_set(sample_design_params, sample_eval_results):
    """Test the ConstraintSet evaluation."""
    c_rtp_ok = constraints.RTPConstraint(min_rtp=0.95, max_rtp=0.97)
    c_rtp_fail = constraints.RTPConstraint(min_rtp=0.97, max_rtp=0.98)
    c_var_ok = constraints.VarianceConstraint(min_variance=30.0, max_variance=40.0)
    c_hf_ok = constraints.HitFrequencyConstraint(min_hf=0.20, max_hf=0.25)
    c_hf_fail = constraints.HitFrequencyConstraint(min_hf=0.22, max_hf=0.25)

    # Set 1: All constraints OK
    set1 = constraints.ConstraintSet([c_rtp_ok, c_var_ok, c_hf_ok])
    assert set1.evaluate_all(sample_design_params, sample_eval_results) is True
    assert set1(sample_design_params, sample_eval_results) is True # Test __call__

    # Set 2: One constraint fails (RTP)
    set2 = constraints.ConstraintSet([c_rtp_fail, c_var_ok, c_hf_ok])
    assert set2.evaluate_all(sample_design_params, sample_eval_results) is False

    # Set 3: One constraint fails (HF)
    set3 = constraints.ConstraintSet([c_rtp_ok, c_var_ok, c_hf_fail])
    assert set3.evaluate_all(sample_design_params, sample_eval_results) is False

    # Set 4: Empty set
    set4 = constraints.ConstraintSet([])
    assert set4.evaluate_all(sample_design_params, sample_eval_results) is True

    # Set 5: Metric missing for one constraint
    set5 = constraints.ConstraintSet([c_rtp_ok, c_var_ok])
    assert set5.evaluate_all(sample_design_params, {"rtp": 0.96}) is False # Variance missing