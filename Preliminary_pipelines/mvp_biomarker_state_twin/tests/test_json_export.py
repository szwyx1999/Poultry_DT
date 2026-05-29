from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mvp_biomarker_state_twin.src.config import load_config
from mvp_biomarker_state_twin.src.unity_export import export_unity_json


def test_unity_json_schema_sanity(tmp_path: Path) -> None:
    config = load_config()
    config = config.__class__(**{**config.__dict__, "unity_json_dir": tmp_path, "unity_copy_paths": tuple()})
    canonical_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "room_id": "room_1",
                "zone_id": "zone_A",
                "activity_mean": 0.2,
                "activity_proxy_raw": 0.2,
            },
            {
                "window_id": "w1",
                "room_id": "room_1",
                "zone_id": "zone_B",
                "activity_mean": 0.4,
                "activity_proxy_raw": 0.4,
            },
        ]
    )
    biomarker_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "room_id": "room_1",
                "start_time": "2025-08-16T09:00:00-03:00",
                "end_time": "2025-08-16T09:00:30-03:00",
                "start_time_dt": pd.Timestamp("2025-08-16T09:00:00-03:00"),
                "mobility_index": 0.4,
                "spatial_freedom_index": 0.9,
                "occupancy_imbalance_index": 0.1,
                "activity_mean": 0.3,
                "event_id": None,
                "event_phase": "normal",
                "event_type": None,
            }
        ]
    )
    state_df = pd.DataFrame(
        [
            {
                "window_id": "w1",
                "room_id": "room_1",
                "state_id": 0,
                "state_label": "stable_low_activity",
                "state_probability_max": 0.95,
                "welfare_risk_score": 0.2,
                "risk_level": "low",
                "sustained_risk_flag": False,
            }
        ]
    )
    zone_config = {
        "room_id": "room_1",
        "zones": [
            {"zone_id": "zone_A", "row": 0, "col": 0},
            {"zone_id": "zone_B", "row": 0, "col": 1},
        ],
    }
    output_path, _ = export_unity_json(canonical_df, biomarker_df, state_df, zone_config, config, "fallback_gmm_state_model")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["schema_version"] == "mvp_biomarker_state_twin_v1"
    assert len(payload["rooms"]) == 1
    assert len(payload["timeline"]) == 1
    assert payload["timeline"][0]["state"]["state_label"] == "stable_low_activity"
