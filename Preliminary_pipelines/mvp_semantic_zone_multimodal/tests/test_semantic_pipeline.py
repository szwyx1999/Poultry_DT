from __future__ import annotations

import pandas as pd

from mvp_semantic_zone_multimodal.src.config import load_config
from mvp_semantic_zone_multimodal.src.hmm_model import _determine_state_count
from mvp_semantic_zone_multimodal.src.merge_multimodal import attach_event_labels
from mvp_semantic_zone_multimodal.src.semantic_biomarkers import _imbalance, _spatial_freedom
from mvp_semantic_zone_multimodal.src.semantic_zone_builder import build_zone_masks


def test_build_zone_masks_general_remainder() -> None:
    zone_config = {
        "image_width": 10,
        "image_height": 10,
        "zones": [
            {
                "zone_id": "drinking_zone",
                "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
            },
            {
                "zone_id": "feeding_zone",
                "polygon": [[2, 0], [4, 0], [4, 2], [2, 2]],
            },
            {
                "zone_id": "general_zone",
                "polygon": None,
            },
        ],
    }
    masks = build_zone_masks(zone_config, image_width=10, image_height=10)
    assert masks["drinking_zone"].sum() > 0
    assert masks["feeding_zone"].sum() > 0
    occupied = masks["drinking_zone"] | masks["feeding_zone"]
    assert masks["general_zone"].sum() == 100 - occupied.sum()


def test_spatial_entropy_and_imbalance_are_consistent() -> None:
    distributed = pd.Series([1 / 3, 1 / 3, 1 / 3], dtype=float).to_numpy()
    concentrated = pd.Series([0.9, 0.05, 0.05], dtype=float).to_numpy()
    assert _spatial_freedom(distributed) > _spatial_freedom(concentrated)
    assert _imbalance(distributed) < _imbalance(concentrated)


def test_attach_event_labels_marks_baseline_during_and_recovery() -> None:
    config = load_config("mvp_semantic_zone_multimodal/config/default.yaml")
    multimodal_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "room_id": "room_1",
                "session_id": "room_1_16_17_aug",
                "start_time": "2025-08-17T08:40:00-03:00",
                "end_time": "2025-08-17T08:40:30-03:00",
            },
            {
                "window_id": "w2",
                "room_id": "room_1",
                "session_id": "room_1_16_17_aug",
                "start_time": "2025-08-17T08:49:15-03:00",
                "end_time": "2025-08-17T08:49:45-03:00",
            },
            {
                "window_id": "w3",
                "room_id": "room_1",
                "session_id": "room_1_16_17_aug",
                "start_time": "2025-08-17T08:51:10-03:00",
                "end_time": "2025-08-17T08:51:40-03:00",
            },
        ]
    )
    event_log_df = pd.DataFrame(
        [
            {
                "event_id": config.caretaker_event_id,
                "event_type": config.caretaker_event_type,
                "room_id": "room_1",
                "event_start_time": "2025-08-17T08:49:00-03:00",
                "event_end_time": "2025-08-17T08:50:57-03:00",
            }
        ]
    )
    labelled_df = attach_event_labels(multimodal_df, event_log_df, config)
    phase_lookup = dict(zip(labelled_df["window_id"], labelled_df["event_phase"]))
    assert phase_lookup["w1"] == "pre_entry_baseline"
    assert phase_lookup["w2"] == "during_entry"
    assert phase_lookup["w3"] == "post_entry_recovery"


def test_auto_state_count_thresholds() -> None:
    config = load_config("mvp_semantic_zone_multimodal/config/default.yaml")
    assert _determine_state_count(5, config) == 2
    assert _determine_state_count(20, config) == 3
    assert _determine_state_count(200, config) == 4
