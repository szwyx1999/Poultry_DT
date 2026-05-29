from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BiomarkerTwinConfig


LOGGER = logging.getLogger(__name__)

IDENTIFIER_COLUMNS = [
    "window_id",
    "media_id",
    "system_id",
    "room_id",
    "session_id",
    "zone_id",
    "start_time",
    "end_time",
    "zone_config_id",
]

IDENTIFIER_CANDIDATES = {
    "window_id": ["window_id"],
    "media_id": ["media_id"],
    "system_id": ["system_id"],
    "room_id": ["room_id"],
    "session_id": ["session_id"],
    "zone_id": ["zone_id"],
    "start_time": ["start_time"],
    "end_time": ["end_time"],
    "zone_config_id": ["zone_config_id"],
}

ACTIVITY_FIELD_CANDIDATES = {
    "activity_mean": ["activity_mean", "activity", "zone_activity_mean"],
    "activity_std": ["activity_std", "activity_sd"],
    "motion_pixel_ratio": ["motion_pixel_ratio", "motion_ratio"],
    "frame_count": ["frame_count", "n_frames"],
}

OPTIONAL_METADATA_CANDIDATES = {
    "video_path": ["video_path", "file_path"],
    "video_start_offset_sec": ["video_start_offset_sec"],
    "duration_seconds": ["duration_seconds"],
    "has_audio": ["has_audio"],
    "quality_status": ["quality_status"],
    "selection_rank": ["selection_rank"],
    "warnings": ["warnings"],
}


@dataclass(frozen=True)
class CanonicalBuildResult:
    canonical_df: pd.DataFrame
    schema_mapping_markdown: str


def build_canonical_zone_feature_table(
    zone_features_df: pd.DataFrame,
    selected_windows_df: pd.DataFrame,
    window_index_df: pd.DataFrame,
    media_manifest_df: pd.DataFrame,
) -> CanonicalBuildResult:
    if zone_features_df.empty:
        empty_columns = IDENTIFIER_COLUMNS + list(ACTIVITY_FIELD_CANDIDATES) + list(OPTIONAL_METADATA_CANDIDATES)
        return CanonicalBuildResult(
            canonical_df=pd.DataFrame(columns=empty_columns),
            schema_mapping_markdown=_build_schema_mapping_markdown({}, {}, {}, "The MVP1 feature table was empty."),
        )

    base_df = zone_features_df.copy()
    supplemental_df = _merge_window_metadata(selected_windows_df, window_index_df, media_manifest_df)
    merged_df = base_df.merge(
        supplemental_df,
        on="window_id",
        how="left",
        suffixes=("", "__supplemental"),
    )
    canonical_data: dict[str, pd.Series] = {}
    identifier_mapping: dict[str, str] = {}
    field_mapping: dict[str, str] = {}

    for canonical_name, candidates in IDENTIFIER_CANDIDATES.items():
        canonical_data[canonical_name], source_name = _extract_preferred_series(merged_df, candidates)
        identifier_mapping[canonical_name] = source_name

    for canonical_name, candidates in ACTIVITY_FIELD_CANDIDATES.items():
        canonical_data[canonical_name], source_name = _extract_preferred_series(merged_df, candidates)
        field_mapping[canonical_name] = source_name

    for canonical_name, candidates in OPTIONAL_METADATA_CANDIDATES.items():
        canonical_data[canonical_name], source_name = _extract_preferred_series(merged_df, candidates)
        field_mapping[canonical_name] = source_name

    canonical_df = pd.DataFrame(canonical_data)
    canonical_df["activity_proxy_raw"] = _choose_activity_proxy(canonical_df)
    canonical_df["source_activity_field"] = _choose_source_activity_field(canonical_df)
    canonical_df["start_time_dt"] = pd.to_datetime(canonical_df["start_time"], errors="coerce", utc=False)
    canonical_df["end_time_dt"] = pd.to_datetime(canonical_df["end_time"], errors="coerce", utc=False)
    sort_columns = [column for column in ("room_id", "start_time_dt", "window_id", "zone_id") if column in canonical_df.columns]
    canonical_df = canonical_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    notes = _build_notes(canonical_df)
    schema_mapping_markdown = _build_schema_mapping_markdown(identifier_mapping, field_mapping, canonical_df, notes)
    return CanonicalBuildResult(canonical_df=canonical_df, schema_mapping_markdown=schema_mapping_markdown)


def compute_spatial_freedom_index(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        return float("nan")
    clipped = np.clip(probabilities, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        clipped = np.repeat(1.0 / probabilities.size, probabilities.size)
    else:
        clipped = clipped / total
    if probabilities.size == 1:
        return 1.0
    mask = clipped > 0
    entropy = -np.sum(clipped[mask] * np.log(clipped[mask]))
    return float(entropy / np.log(probabilities.size))


def compute_occupancy_imbalance_index(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        return float("nan")
    clipped = np.clip(probabilities, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        clipped = np.repeat(1.0 / probabilities.size, probabilities.size)
    else:
        clipped = clipped / total
    uniform = np.repeat(1.0 / probabilities.size, probabilities.size)
    return float(0.5 * np.abs(clipped - uniform).sum())


def compute_transition_proxy(previous_probabilities: np.ndarray | None, current_probabilities: np.ndarray) -> float:
    current = np.asarray(current_probabilities, dtype=float)
    if previous_probabilities is None:
        return 0.0
    previous = np.asarray(previous_probabilities, dtype=float)
    return float(0.5 * np.abs(current - previous).sum())


def compute_mobility_index(normalized_activity: float, transition_proxy: float, activity_weight: float, transition_weight: float) -> float:
    return float((activity_weight * normalized_activity) + (transition_weight * transition_proxy))


def compute_room_window_biomarkers(canonical_df: pd.DataFrame, config: BiomarkerTwinConfig) -> pd.DataFrame:
    if canonical_df.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "room_id",
                "session_id",
                "media_id",
                "system_id",
                "start_time",
                "end_time",
                "duration_seconds",
                "zone_config_id",
                "zone_count",
                "missing_zone_count",
                "activity_total",
                "activity_mean",
                "activity_std_mean",
                "motion_pixel_ratio_mean",
                "distribution_fallback_used",
                "distribution_source",
                "spatial_freedom_index",
                "occupancy_imbalance_index",
                "transition_proxy",
                "normalized_activity",
                "mobility_index",
            ]
        )

    rows: list[dict] = []
    group_columns = ["room_id", "start_time_dt", "window_id"]
    grouped = canonical_df.sort_values(group_columns + ["zone_id"], kind="stable").groupby(group_columns, dropna=False, sort=False)

    for (_, start_time_dt, window_id), window_group in grouped:
        zone_weights = pd.to_numeric(window_group["activity_proxy_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        zone_count = int(window_group["zone_id"].nunique())
        total_weight = float(zone_weights.sum())
        distribution_fallback_used = total_weight <= 0 or zone_count <= 0
        if distribution_fallback_used:
            probabilities = np.repeat(1.0 / max(zone_count, 1), max(zone_count, 1))
            distribution_source = "uniform_zero_activity_fallback"
        else:
            probabilities = (zone_weights / total_weight).to_numpy(dtype=float)
            distribution_source = "activity_proxy_distribution"

        first_row = window_group.iloc[0]
        rows.append(
            {
                "window_id": window_id,
                "media_id": _safe_text(first_row.get("media_id")),
                "system_id": _safe_text(first_row.get("system_id")),
                "room_id": _safe_text(first_row.get("room_id")),
                "session_id": _safe_text(first_row.get("session_id")),
                "start_time": _safe_text(first_row.get("start_time")),
                "end_time": _safe_text(first_row.get("end_time")),
                "duration_seconds": _safe_float(first_row.get("duration_seconds")),
                "zone_config_id": _safe_text(first_row.get("zone_config_id")),
                "zone_count": zone_count,
                "missing_zone_count": int(window_group["activity_proxy_raw"].isna().sum()),
                "activity_total": total_weight,
                "activity_mean": _safe_float(pd.to_numeric(window_group.get("activity_mean"), errors="coerce").mean()),
                "activity_std_mean": _safe_float(pd.to_numeric(window_group.get("activity_std"), errors="coerce").mean()),
                "motion_pixel_ratio_mean": _safe_float(pd.to_numeric(window_group.get("motion_pixel_ratio"), errors="coerce").mean()),
                "distribution_fallback_used": bool(distribution_fallback_used),
                "distribution_source": distribution_source,
                "distribution_vector": probabilities.tolist(),
                "spatial_freedom_index": compute_spatial_freedom_index(probabilities),
                "occupancy_imbalance_index": compute_occupancy_imbalance_index(probabilities),
                "warnings": _merge_text_values(window_group.get("warnings")),
            }
        )

    window_df = pd.DataFrame(rows)
    window_df["start_time_dt"] = pd.to_datetime(window_df["start_time"], errors="coerce", utc=False)
    window_df["end_time_dt"] = pd.to_datetime(window_df["end_time"], errors="coerce", utc=False)
    window_df = window_df.sort_values(["room_id", "start_time_dt", "window_id"], kind="stable").reset_index(drop=True)

    transition_values: list[float] = []
    for _, room_group in window_df.groupby("room_id", sort=False):
        previous_distribution: np.ndarray | None = None
        for _, row in room_group.iterrows():
            distribution = np.asarray(row["distribution_vector"], dtype=float)
            transition_values.append(compute_transition_proxy(previous_distribution, distribution))
            previous_distribution = distribution
    window_df["transition_proxy"] = transition_values
    window_df["normalized_activity"] = window_df.groupby("room_id", sort=False)["activity_mean"].transform(_min_max_scale)
    window_df["normalized_activity"] = window_df["normalized_activity"].fillna(window_df.groupby("room_id", sort=False)["activity_total"].transform(_min_max_scale))
    window_df["normalized_activity"] = window_df["normalized_activity"].fillna(0.0)
    window_df["mobility_index"] = window_df.apply(
        lambda row: compute_mobility_index(
            normalized_activity=float(row["normalized_activity"]),
            transition_proxy=float(row["transition_proxy"]),
            activity_weight=config.mobility_activity_weight,
            transition_weight=config.mobility_transition_weight,
        ),
        axis=1,
    )
    return window_df


def _merge_window_metadata(
    selected_windows_df: pd.DataFrame,
    window_index_df: pd.DataFrame,
    media_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    candidate_frames = [frame.copy() for frame in (selected_windows_df, window_index_df) if not frame.empty]
    merged_df = pd.DataFrame(columns=["window_id"])
    if candidate_frames:
        merged_df = candidate_frames[0]
        for frame in candidate_frames[1:]:
            supplemental_columns = [column for column in frame.columns if column == "window_id" or column not in merged_df.columns]
            merged_df = merged_df.merge(
                frame[supplemental_columns],
                on="window_id",
                how="outer",
                suffixes=("", "__window"),
            )
    if not media_manifest_df.empty and "media_id" in merged_df.columns:
        supplemental_columns = [column for column in media_manifest_df.columns if column == "media_id" or column not in merged_df.columns]
        merged_df = merged_df.merge(
            media_manifest_df[supplemental_columns],
            on="media_id",
            how="left",
            suffixes=("", "__media"),
        )
    return merged_df


def _extract_preferred_series(dataframe: pd.DataFrame, candidates: list[str]) -> tuple[pd.Series, str]:
    for candidate in candidates:
        matching_columns = [column for column in dataframe.columns if _normalize_column_name(column) == _normalize_column_name(candidate)]
        direct_first = [column for column in matching_columns if "__" not in column]
        ordered_matches = direct_first + [column for column in matching_columns if column not in direct_first]
        for column in ordered_matches:
            series = dataframe[column]
            if series.notna().any() or candidate == candidates[0]:
                return series, column
    fallback_name = candidates[0]
    LOGGER.warning("Missing expected column candidates for %s", fallback_name)
    return pd.Series([pd.NA] * len(dataframe), index=dataframe.index, dtype=object), "(missing)"


def _build_schema_mapping_markdown(
    identifier_mapping: dict[str, str],
    field_mapping: dict[str, str],
    canonical_df: pd.DataFrame | dict,
    notes: str,
) -> str:
    lines = [
        "# Schema Mapping",
        "",
        "This report documents how the MVP1 feature exports were normalized into the canonical zone feature table.",
        "",
        "## Identifier Mapping",
        "",
        "| Canonical field | Source column |",
        "| --- | --- |",
    ]
    for canonical_name in IDENTIFIER_COLUMNS:
        lines.append(f"| `{canonical_name}` | `{identifier_mapping.get(canonical_name, '(missing)')}` |")

    lines.extend(
        [
            "",
            "## Activity And Metadata Mapping",
            "",
            "| Canonical field | Source column |",
            "| --- | --- |",
        ]
    )
    for canonical_name in list(ACTIVITY_FIELD_CANDIDATES) + list(OPTIONAL_METADATA_CANDIDATES):
        lines.append(f"| `{canonical_name}` | `{field_mapping.get(canonical_name, '(missing)')}` |")

    if isinstance(canonical_df, pd.DataFrame) and not canonical_df.empty:
        lines.extend(
            [
                "",
                "## Canonical Table Summary",
                "",
                f"- Rows: {len(canonical_df)}",
                f"- Unique windows: {canonical_df['window_id'].nunique()}",
                f"- Unique rooms: {canonical_df['room_id'].nunique()}",
                f"- Unique zones: {canonical_df['zone_id'].nunique()}",
                f"- Activity proxy priority: `activity_mean` -> `motion_pixel_ratio` -> `activity_std`",
            ]
        )

    lines.extend(["", "## Notes", "", notes.strip(), ""])
    return "\n".join(lines)


def _build_notes(canonical_df: pd.DataFrame) -> str:
    source_field_counts = canonical_df["source_activity_field"].fillna("missing").value_counts(dropna=False).to_dict()
    return (
        "The canonical table preserves one row per `window_id` and `zone_id`.\n\n"
        f"Observed activity proxy source counts: {source_field_counts}.\n\n"
        "When `zone_config_id` was absent from `video_zone_features.csv`, it was backfilled from the selected-window and preprocessing metadata tables."
    )


def _choose_activity_proxy(dataframe: pd.DataFrame) -> pd.Series:
    proxy_columns = [
        pd.to_numeric(dataframe.get("activity_mean"), errors="coerce"),
        pd.to_numeric(dataframe.get("motion_pixel_ratio"), errors="coerce"),
        pd.to_numeric(dataframe.get("activity_std"), errors="coerce"),
    ]
    proxy_frame = pd.concat(proxy_columns, axis=1)
    return proxy_frame.bfill(axis=1).iloc[:, 0]


def _choose_source_activity_field(dataframe: pd.DataFrame) -> pd.Series:
    source_labels = []
    for _, row in dataframe.iterrows():
        if pd.notna(row.get("activity_mean")):
            source_labels.append("activity_mean")
        elif pd.notna(row.get("motion_pixel_ratio")):
            source_labels.append("motion_pixel_ratio")
        elif pd.notna(row.get("activity_std")):
            source_labels.append("activity_std")
        else:
            source_labels.append("missing")
    return pd.Series(source_labels, index=dataframe.index)


def _normalize_column_name(column_name: str) -> str:
    return "".join(character.lower() for character in str(column_name) if character.isalnum())


def _min_max_scale(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    valid_series = numeric_series.dropna()
    if valid_series.empty:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    minimum = float(valid_series.min())
    maximum = float(valid_series.max())
    if minimum == maximum:
        fill_value = 0.0 if minimum == 0 else 1.0
        return pd.Series([fill_value if pd.notna(value) else np.nan for value in numeric_series], index=series.index, dtype=float)
    return (numeric_series - minimum) / (maximum - minimum)


def _merge_text_values(series: pd.Series | None) -> str:
    if series is None:
        return ""
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_value in series.fillna("").tolist():
        for token in str(raw_value).split("|"):
            normalized = token.strip()
            if normalized and normalized not in seen:
                tokens.append(normalized)
                seen.add(normalized)
    return " | ".join(tokens)


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _safe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
