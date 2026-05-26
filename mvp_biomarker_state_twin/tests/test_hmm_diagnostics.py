from __future__ import annotations

from pathlib import Path

import pandas as pd

from mvp_biomarker_state_twin.src.diagnostics import (
    analyze_hmm_sequence,
    validate_hmm_sequence_for_plotting,
    validate_window_level_hmm_input,
    write_hmm_sequence_diagnostics_report,
)


def test_validate_window_level_hmm_input_rejects_duplicate_room_window_rows() -> None:
    dataframe = pd.DataFrame(
        [
            {"room_id": "room_1", "window_id": "w1", "start_time": "2025-08-16T09:00:00-03:00"},
            {"room_id": "room_1", "window_id": "w1", "start_time": "2025-08-16T09:00:30-03:00"},
        ]
    )
    try:
        validate_window_level_hmm_input(dataframe, "biomarker_window_table.csv")
    except ValueError as exc:
        assert "one row per room/window" in str(exc)
    else:
        raise AssertionError("Expected duplicate room/window validation failure.")


def test_hmm_sequence_diagnostics_report_tracks_duplicates(tmp_path: Path) -> None:
    sequence_df = pd.DataFrame(
        [
            {"room_id": "room_1", "session_id": "s1", "media_id": "m1", "window_id": "w1", "start_time": "2025-08-16T09:00:00-03:00", "state_id": 0, "state_label": "a"},
            {"room_id": "room_1", "session_id": "s1", "media_id": "m1", "window_id": "w2", "start_time": "2025-08-16T09:00:00-03:00", "state_id": 1, "state_label": "b"},
        ]
    )
    diagnostics = analyze_hmm_sequence(
        canonical_df=pd.DataFrame([{"window_id": "w1"}, {"window_id": "w2"}]),
        biomarker_df=pd.DataFrame([{"window_id": "w1"}, {"window_id": "w2"}]),
        sequence_df=sequence_df,
    )
    assert diagnostics.duplicate_room_session_start_time_row_count == 2
    assert diagnostics.multi_state_same_room_session_start_time_group_count == 1
    assert diagnostics.is_valid_temporal_sequence is False

    output_path = tmp_path / "hmm_sequence_diagnostics.md"
    write_hmm_sequence_diagnostics_report(diagnostics, str(output_path))
    report_text = output_path.read_text(encoding="utf-8")
    assert "Total rows in `hmm_state_sequence.csv`: 2" in report_text
    assert "Any time window with multiple assigned states: yes" in report_text

    try:
        validate_hmm_sequence_for_plotting(diagnostics)
    except ValueError as exc:
        assert "zone-level rows leaked" in str(exc)
    else:
        raise AssertionError("Expected invalid HMM plotting validation failure.")
