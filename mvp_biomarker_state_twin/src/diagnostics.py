from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class HmmSequenceDiagnostics:
    canonical_row_count: int
    biomarker_row_count: int
    hmm_sequence_row_count: int
    unique_room_count: int
    unique_session_count: int
    unique_window_count: int
    unique_room_window_count: int
    duplicate_room_window_row_count: int
    duplicate_room_session_start_time_row_count: int
    multi_state_same_room_session_start_time_group_count: int
    is_valid_temporal_sequence: bool
    model_type: str
    requested_state_setting: str
    fitted_state_count: int
    occupied_state_count: int
    event_labelled_window_count: int
    caretaker_event_covered: bool
    sequence_summary_df: pd.DataFrame
    duplicate_examples_df: pd.DataFrame


def validate_window_level_hmm_input(window_df: pd.DataFrame, table_name: str) -> None:
    if window_df.empty:
        return

    duplicate_room_window_rows = int(window_df.duplicated(["room_id", "window_id"], keep=False).sum())
    duplicate_room_session_start_rows = int(
        window_df.duplicated(_session_aware_group_columns(window_df, include_window_id=False), keep=False).sum()
    )
    if duplicate_room_window_rows > 0 or duplicate_room_session_start_rows > 0:
        raise ValueError(
            f"{table_name} must contain one row per room/window before HMM fitting. "
            f"Found duplicate room_id+window_id rows={duplicate_room_window_rows}, "
            f"duplicate room_id+session_id+start_time rows={duplicate_room_session_start_rows}."
        )


def analyze_hmm_sequence(
    canonical_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    sequence_df: pd.DataFrame,
    model_type: str = "",
    requested_state_setting: str | int | None = None,
    fitted_state_count: int = 0,
    occupied_state_count: int = 0,
    event_labelled_window_count: int = 0,
    caretaker_event_covered: bool = False,
) -> HmmSequenceDiagnostics:
    if sequence_df.empty:
        return HmmSequenceDiagnostics(
            canonical_row_count=len(canonical_df),
            biomarker_row_count=len(biomarker_df),
            hmm_sequence_row_count=0,
            unique_room_count=0,
            unique_session_count=0,
            unique_window_count=0,
            unique_room_window_count=0,
            duplicate_room_window_row_count=0,
            duplicate_room_session_start_time_row_count=0,
            multi_state_same_room_session_start_time_group_count=0,
            is_valid_temporal_sequence=True,
            model_type=model_type or "empty",
            requested_state_setting=str(requested_state_setting if requested_state_setting is not None else "auto"),
            fitted_state_count=fitted_state_count,
            occupied_state_count=occupied_state_count,
            event_labelled_window_count=event_labelled_window_count,
            caretaker_event_covered=caretaker_event_covered,
            sequence_summary_df=pd.DataFrame(columns=["room_id", "session_id", "row_count", "start_time_min", "start_time_max"]),
            duplicate_examples_df=pd.DataFrame(),
        )

    start_time_group_columns = _session_aware_group_columns(sequence_df, include_window_id=False)
    room_window_group_columns = ["room_id", "window_id"]

    duplicate_room_window_mask = sequence_df.duplicated(room_window_group_columns, keep=False)
    duplicate_room_start_mask = sequence_df.duplicated(start_time_group_columns, keep=False)
    multi_state_group_mask = sequence_df.groupby(start_time_group_columns, dropna=False)["state_id"].nunique().gt(1)
    multi_state_group_count = int(multi_state_group_mask.sum())
    valid_sequence = (
        int(duplicate_room_window_mask.sum()) == 0
        and int(duplicate_room_start_mask.sum()) == 0
        and multi_state_group_count == 0
    )

    sequence_summary_df = (
        sequence_df.groupby([column for column in ("room_id", "session_id") if column in sequence_df.columns], dropna=False, sort=False)
        .agg(
            row_count=("window_id", "count"),
            unique_windows=("window_id", "nunique"),
            start_time_min=("start_time", "min"),
            start_time_max=("start_time", "max"),
        )
        .reset_index()
    )

    duplicate_examples_df = sequence_df[duplicate_room_window_mask | duplicate_room_start_mask].copy()
    if not duplicate_examples_df.empty:
        example_columns = [
            column
            for column in ("room_id", "session_id", "media_id", "window_id", "start_time", "state_id", "state_label")
            if column in duplicate_examples_df.columns
        ]
        duplicate_examples_df = duplicate_examples_df[example_columns].sort_values(
            [column for column in ("room_id", "session_id", "start_time", "window_id") if column in duplicate_examples_df.columns],
            kind="stable",
        )

    return HmmSequenceDiagnostics(
        canonical_row_count=len(canonical_df),
        biomarker_row_count=len(biomarker_df),
        hmm_sequence_row_count=len(sequence_df),
        unique_room_count=int(sequence_df["room_id"].nunique()),
        unique_session_count=int(sequence_df["session_id"].nunique()) if "session_id" in sequence_df.columns else 0,
        unique_window_count=int(sequence_df["window_id"].nunique()),
        unique_room_window_count=int(sequence_df[room_window_group_columns].drop_duplicates().shape[0]),
        duplicate_room_window_row_count=int(duplicate_room_window_mask.sum()),
        duplicate_room_session_start_time_row_count=int(duplicate_room_start_mask.sum()),
        multi_state_same_room_session_start_time_group_count=multi_state_group_count,
        is_valid_temporal_sequence=valid_sequence,
        model_type=model_type or "unknown",
        requested_state_setting=str(requested_state_setting if requested_state_setting is not None else "auto"),
        fitted_state_count=fitted_state_count,
        occupied_state_count=occupied_state_count,
        event_labelled_window_count=event_labelled_window_count,
        caretaker_event_covered=caretaker_event_covered,
        sequence_summary_df=sequence_summary_df,
        duplicate_examples_df=duplicate_examples_df,
    )


def write_hmm_sequence_diagnostics_report(
    diagnostics: HmmSequenceDiagnostics,
    output_path: str,
) -> None:
    lines = [
        "# HMM Sequence Diagnostics",
        "",
        "This report validates that the latent-state sequence is window-level and temporally well-defined before plotting.",
        "",
        "## Summary",
        "",
        f"- Total rows in `canonical_zone_feature_table.csv`: {diagnostics.canonical_row_count}",
        f"- Total rows in `biomarker_window_table.csv`: {diagnostics.biomarker_row_count}",
        f"- Total rows in `hmm_state_sequence.csv`: {diagnostics.hmm_sequence_row_count}",
        f"- Model type: {diagnostics.model_type}",
        f"- Requested HMM state setting: {diagnostics.requested_state_setting}",
        f"- Fitted HMM components used: {diagnostics.fitted_state_count}",
        f"- Occupied latent states observed: {diagnostics.occupied_state_count}",
        f"- Number of unique rooms: {diagnostics.unique_room_count}",
        f"- Number of unique sessions: {diagnostics.unique_session_count}",
        f"- Number of unique windows: {diagnostics.unique_window_count}",
        f"- Number of event-labelled windows: {diagnostics.event_labelled_window_count}",
        f"- Caretaker event covered by processed windows: {'yes' if diagnostics.caretaker_event_covered else 'no'}",
        f"- Number of unique room/window pairs: {diagnostics.unique_room_window_count}",
        f"- Duplicate `room_id` + `window_id` row count: {diagnostics.duplicate_room_window_row_count}",
        f"- Duplicate `room_id` + `session_id` + `start_time` row count: {diagnostics.duplicate_room_session_start_time_row_count}",
        f"- Any time window with multiple assigned states: {'yes' if diagnostics.multi_state_same_room_session_start_time_group_count > 0 else 'no'}",
        f"- Multiple-state `room_id` + `session_id` + `start_time` group count: {diagnostics.multi_state_same_room_session_start_time_group_count}",
        f"- Valid temporal sequence for plotting: {'yes' if diagnostics.is_valid_temporal_sequence else 'no'}",
    ]

    if not diagnostics.sequence_summary_df.empty:
        lines.extend(
            [
                "",
                "## Sequence Breakdown",
                "",
                "```text",
                diagnostics.sequence_summary_df.to_string(index=False),
                "```",
            ]
        )

    if not diagnostics.duplicate_examples_df.empty:
        lines.extend(
            [
                "",
                "## Duplicate Examples",
                "",
                "```text",
                diagnostics.duplicate_examples_df.head(20).to_string(index=False),
                "```",
            ]
        )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def validate_hmm_sequence_for_plotting(diagnostics: HmmSequenceDiagnostics) -> None:
    if diagnostics.is_valid_temporal_sequence:
        return
    raise ValueError(
        "hmm_state_sequence contains duplicate room/window or room/session/start_time rows, "
        "or multiple states for the same room/session/start_time. "
        "This usually indicates zone-level rows leaked into the HMM output."
    )


def _session_aware_group_columns(dataframe: pd.DataFrame, include_window_id: bool) -> list[str]:
    columns = ["room_id"]
    if "session_id" in dataframe.columns:
        columns.append("session_id")
    if include_window_id:
        columns.append("window_id")
    else:
        columns.append("start_time")
    return columns
