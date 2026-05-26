from __future__ import annotations

import numpy as np

from mvp_biomarker_state_twin.src.biomarkers import (
    compute_mobility_index,
    compute_occupancy_imbalance_index,
    compute_spatial_freedom_index,
    compute_transition_proxy,
)


def test_entropy_like_spatial_freedom_uniform_distribution_is_one() -> None:
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_spatial_freedom_index(probabilities) == 1.0


def test_occupancy_imbalance_uniform_distribution_is_zero() -> None:
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_occupancy_imbalance_index(probabilities) == 0.0


def test_transition_proxy_uses_half_l1_distance() -> None:
    previous = np.array([0.5, 0.5, 0.0, 0.0])
    current = np.array([0.0, 0.5, 0.5, 0.0])
    assert compute_transition_proxy(previous, current) == 0.5


def test_mobility_index_weighting_is_deterministic() -> None:
    mobility = compute_mobility_index(0.8, 0.25, activity_weight=0.7, transition_weight=0.3)
    assert round(mobility, 4) == 0.635
