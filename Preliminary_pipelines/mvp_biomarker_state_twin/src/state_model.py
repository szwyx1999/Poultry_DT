from __future__ import annotations

import logging
import os
from dataclasses import dataclass

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BiomarkerTwinConfig, config_as_dict


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StateModelResult:
    state_sequence_df: pd.DataFrame
    state_summary_df: pd.DataFrame
    model_path: str
    model_type: str
    effective_states: int
    occupied_states: int
    configured_state_setting: str | int
    feature_columns: list[str]
    risk_component_weights: dict[str, float]


def fit_state_model(biomarker_df: pd.DataFrame, config: BiomarkerTwinConfig, model_path: str) -> StateModelResult:
    sorted_df = biomarker_df.sort_values(_sequence_sort_columns(biomarker_df), kind="stable").reset_index(drop=True)
    if sorted_df.empty:
        empty_sequence = pd.DataFrame(
            columns=[
                "window_id",
                "room_id",
                "session_id",
                "start_time",
                "end_time",
                "state_id",
                "state_label",
                "state_probability_max",
                "welfare_risk_score",
                "risk_level",
                "sustained_risk_flag",
            ]
        )
        empty_summary = pd.DataFrame(columns=["state_id", "state_label", "window_count", "state_risk_weight"])
        joblib.dump({"model": None, "model_type": "empty", "config": config_as_dict(config)}, model_path)
        return StateModelResult(
            empty_sequence,
            empty_summary,
            model_path,
            "empty",
            effective_states=0,
            occupied_states=0,
            configured_state_setting=config.hmm_n_states,
            feature_columns=[],
            risk_component_weights=_risk_component_weights(config),
        )

    feature_columns = _select_model_feature_columns(sorted_df)
    feature_frame = sorted_df[feature_columns].copy()
    if config.add_missingness_indicators:
        for column in feature_columns:
            feature_frame[f"{column}__missing"] = feature_frame[column].isna().astype(float)
    model_feature_columns = feature_frame.columns.tolist()

    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    feature_matrix = preprocessing.fit_transform(feature_frame)
    effective_states = _determine_effective_state_count(len(sorted_df), config)
    output_state_columns = max(config.max_hmm_states, effective_states)
    sequence_group_columns = _sequence_group_columns(sorted_df)
    sequence_lengths = sorted_df.groupby(sequence_group_columns, sort=False, dropna=False).size().tolist()

    model_object, posterior_matrix, predicted_states, model_type = _fit_hmm_or_fallback(
        feature_matrix=feature_matrix,
        sequence_lengths=sequence_lengths,
        effective_states=effective_states,
        config=config,
    )

    if model_type == "fallback_gmm_state_model":
        final_states = _smooth_states_by_sequence(sorted_df, predicted_states)
        posterior_for_outputs = _posterior_from_state_ids(final_states, effective_states)
    else:
        final_states = predicted_states
        posterior_for_outputs = posterior_matrix

    sequence_df = _build_state_sequence(
        sorted_df=sorted_df,
        posterior_matrix=posterior_for_outputs,
        state_ids=final_states,
        output_state_columns=output_state_columns,
    )
    sequence_df["state_run_length"] = _compute_state_run_lengths(sequence_df)

    summary_df = _build_state_summary(sequence_df)
    summary_df = _add_state_risk_components(summary_df, config)
    summary_df = _assign_state_labels(summary_df)
    state_label_map = dict(zip(summary_df["state_id"], summary_df["state_label"]))
    risk_weights = dict(zip(summary_df["state_id"], summary_df["state_risk_weight"]))

    sequence_df["state_label"] = sequence_df["state_id"].map(state_label_map).fillna("unlabeled_state")
    sequence_df["welfare_risk_score"] = _compute_row_risk_scores(posterior_for_outputs, output_state_columns, risk_weights)
    sequence_df["risk_level"] = sequence_df["welfare_risk_score"].apply(_risk_level)
    sequence_df["sustained_risk_flag"] = _compute_sustained_risk_flags(sequence_df, config.sustained_risk_windows)
    sequence_df["high_risk_flag"] = sequence_df["risk_level"].eq("high")
    occupied_states = int(sequence_df["state_id"].nunique())

    artifact = {
        "model": model_object,
        "model_type": model_type,
        "effective_states": effective_states,
        "occupied_states": occupied_states,
        "configured_state_setting": config.hmm_n_states,
        "feature_columns": model_feature_columns,
        "preprocessing": preprocessing,
        "state_labels": state_label_map,
        "state_risk_weights": risk_weights,
        "risk_component_weights": _risk_component_weights(config),
        "config": config_as_dict(config),
    }
    joblib.dump(artifact, model_path)
    return StateModelResult(
        state_sequence_df=sequence_df,
        state_summary_df=summary_df,
        model_path=model_path,
        model_type=model_type,
        effective_states=effective_states,
        occupied_states=occupied_states,
        configured_state_setting=config.hmm_n_states,
        feature_columns=model_feature_columns,
        risk_component_weights=_risk_component_weights(config),
    )


def _select_model_feature_columns(biomarker_df: pd.DataFrame) -> list[str]:
    candidates = [
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "normalized_activity",
    ]
    if "resilience_pressure_index" in biomarker_df.columns and biomarker_df["resilience_pressure_index"].notna().any():
        candidates.append("resilience_pressure_index")
    available = [column for column in candidates if column in biomarker_df.columns]
    if not available:
        raise ValueError("No model features were available for the latent state model.")
    return available


def _determine_effective_state_count(n_windows: int, config: BiomarkerTwinConfig) -> int:
    if n_windows <= 1:
        return 1
    if config.hmm_n_states == "auto":
        if n_windows < 12:
            return min(2, n_windows)
        if n_windows < config.min_windows_for_hmm:
            return min(3, n_windows)
        return min(config.max_hmm_states, n_windows)
    return max(1, min(int(config.hmm_n_states), n_windows, config.max_hmm_states))


def _fit_hmm_or_fallback(
    feature_matrix: np.ndarray,
    sequence_lengths: list[int],
    effective_states: int,
    config: BiomarkerTwinConfig,
) -> tuple[object, np.ndarray, np.ndarray, str]:
    try:
        from hmmlearn.hmm import GaussianHMM

        hmm_model = GaussianHMM(
            n_components=effective_states,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            n_iter=250,
        )
        hmm_model.fit(feature_matrix, lengths=sequence_lengths)
        posterior_matrix = hmm_model.predict_proba(feature_matrix, lengths=sequence_lengths)
        predicted_states = hmm_model.predict(feature_matrix, lengths=sequence_lengths)
        return hmm_model, posterior_matrix, predicted_states, "gaussian_hmm"
    except Exception as exc:  # pragma: no cover - runtime-dependent
        LOGGER.warning("Gaussian HMM unavailable or failed; using fallback GMM state model. Reason: %s", exc)

    gmm_model = GaussianMixture(
        n_components=effective_states,
        covariance_type=config.covariance_type,
        random_state=config.random_state,
        reg_covar=1e-6,
    )
    gmm_model.fit(feature_matrix)
    posterior_matrix = gmm_model.predict_proba(feature_matrix)
    predicted_states = posterior_matrix.argmax(axis=1)
    return gmm_model, posterior_matrix, predicted_states, "fallback_gmm_state_model"


def _smooth_states_by_sequence(sorted_df: pd.DataFrame, state_ids: np.ndarray) -> np.ndarray:
    smoothed = np.asarray(state_ids, dtype=int).copy()
    for _, group_df in sorted_df.groupby(_sequence_group_columns(sorted_df), sort=False, dropna=False):
        positions = group_df.index.tolist()
        for position_index in range(1, len(positions) - 1):
            left_position = positions[position_index - 1]
            center_position = positions[position_index]
            right_position = positions[position_index + 1]
            left_state = smoothed[left_position]
            center_state = smoothed[center_position]
            right_state = smoothed[right_position]
            if left_state == right_state and center_state != left_state:
                smoothed[center_position] = left_state
    return smoothed


def _posterior_from_state_ids(state_ids: np.ndarray, effective_states: int) -> np.ndarray:
    posterior_matrix = np.zeros((len(state_ids), effective_states), dtype=float)
    for row_index, state_id in enumerate(state_ids):
        posterior_matrix[row_index, int(state_id)] = 1.0
    return posterior_matrix


def _build_state_sequence(
    sorted_df: pd.DataFrame,
    posterior_matrix: np.ndarray,
    state_ids: np.ndarray,
    output_state_columns: int,
) -> pd.DataFrame:
    base_columns = [
        "window_id",
        "media_id",
        "room_id",
        "session_id",
        "start_time",
        "end_time",
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "activity_mean",
        "normalized_activity",
        "resilience_pressure_index",
    ]
    optional_columns = [
        "event_phase",
        "event_id",
        "event_type",
        "seconds_from_event_start",
        "seconds_from_event_end",
        "overlaps_event",
        "event_start_time",
        "event_end_time",
    ]
    sequence_columns = [column for column in base_columns + optional_columns if column in sorted_df.columns]
    sequence_df = sorted_df[sequence_columns].copy()
    sequence_df["state_id"] = state_ids
    sequence_df["state_probability_max"] = posterior_matrix.max(axis=1)
    for state_index in range(output_state_columns):
        column_name = f"posterior_state_{state_index}"
        if state_index < posterior_matrix.shape[1]:
            sequence_df[column_name] = posterior_matrix[:, state_index]
        else:
            sequence_df[column_name] = 0.0
    sequence_df["state_label"] = "pending_label"
    return sequence_df


def _compute_state_run_lengths(sequence_df: pd.DataFrame) -> pd.Series:
    run_lengths = pd.Series([1] * len(sequence_df), index=sequence_df.index, dtype=int)
    for _, group_df in sequence_df.groupby(_sequence_group_columns(sequence_df), sort=False, dropna=False):
        positions = group_df.index.tolist()
        run_start = 0
        while run_start < len(positions):
            run_end = run_start + 1
            while run_end < len(positions) and sequence_df.loc[positions[run_end], "state_id"] == sequence_df.loc[positions[run_start], "state_id"]:
                run_end += 1
            run_length = run_end - run_start
            for position in positions[run_start:run_end]:
                run_lengths.loc[position] = run_length
            run_start = run_end
    return run_lengths


def _build_state_summary(sequence_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        sequence_df.groupby("state_id", sort=True)
        .agg(
            window_count=("window_id", "count"),
            mobility_index_mean=("mobility_index", "mean"),
            spatial_freedom_index_mean=("spatial_freedom_index", "mean"),
            occupancy_imbalance_index_mean=("occupancy_imbalance_index", "mean"),
            activity_mean_mean=("activity_mean", "mean"),
            normalized_activity_mean=("normalized_activity", "mean"),
            resilience_pressure_index_mean=("resilience_pressure_index", "mean"),
            state_probability_max_mean=("state_probability_max", "mean"),
            state_run_length_mean=("state_run_length", "mean"),
            state_run_length_max=("state_run_length", "max"),
        )
        .reset_index()
    )
    total_windows = max(len(sequence_df), 1)
    summary_df["window_fraction"] = summary_df["window_count"] / total_windows
    return summary_df.sort_values("state_id", kind="stable").reset_index(drop=True)


def _add_state_risk_components(summary_df: pd.DataFrame, config: BiomarkerTwinConfig) -> pd.DataFrame:
    enriched = summary_df.copy()
    enriched["component_activity_level"] = pd.to_numeric(enriched["normalized_activity_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["component_mobility_level"] = _series_min_max(enriched["mobility_index_mean"])
    enriched["component_activity_mobility"] = (
        0.5 * enriched["component_activity_level"] + 0.5 * enriched["component_mobility_level"]
    ).clip(0.0, 1.0)
    enriched["component_imbalance"] = pd.to_numeric(enriched["occupancy_imbalance_index_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["component_low_spatial_freedom"] = (
        1.0 - pd.to_numeric(enriched["spatial_freedom_index_mean"], errors="coerce").fillna(0.0)
    ).clip(0.0, 1.0)
    enriched["component_resilience_pressure"] = _series_min_max(enriched["resilience_pressure_index_mean"])
    enriched["component_persistence"] = _series_min_max(enriched["state_run_length_mean"])

    enriched["state_risk_weight"] = (
        config.risk_activity_mobility_weight * enriched["component_activity_mobility"]
        + config.risk_imbalance_weight * enriched["component_imbalance"]
        + config.risk_low_spatial_freedom_weight * enriched["component_low_spatial_freedom"]
        + config.risk_resilience_pressure_weight * enriched["component_resilience_pressure"]
        + config.risk_persistence_weight * enriched["component_persistence"]
    ).clip(0.0, 1.0)
    return enriched


def _assign_state_labels(summary_df: pd.DataFrame) -> pd.DataFrame:
    labeled = summary_df.copy()
    if labeled.empty:
        labeled["state_label"] = []
        return labeled

    seen_labels: dict[str, int] = {}
    labels: list[str] = []
    for _, row in labeled.iterrows():
        activity = _safe_float(row.get("component_activity_level"))
        mobility = _safe_float(row.get("component_mobility_level"))
        activity_mobility = _safe_float(row.get("component_activity_mobility"))
        spatial = _safe_float(row.get("spatial_freedom_index_mean"))
        low_spatial = _safe_float(row.get("component_low_spatial_freedom"))
        imbalance = _safe_float(row.get("component_imbalance"))
        resilience = _safe_float(row.get("component_resilience_pressure"))
        persistence = _safe_float(row.get("component_persistence"))
        risk = _safe_float(row.get("state_risk_weight"))

        if risk >= 0.70 and persistence >= 0.55 and (activity_mobility >= 0.50 or imbalance >= 0.35 or low_spatial >= 0.35):
            base_label = "sustained_deviation"
        elif activity_mobility >= 0.60 and resilience >= 0.45:
            base_label = "disturbance_or_transition"
        elif activity <= 0.18 and mobility <= 0.25 and imbalance <= 0.20:
            base_label = "stable_low_activity"
        elif activity <= 0.22 and imbalance >= 0.30 and (low_spatial >= 0.25 or spatial <= 0.80):
            base_label = "localized_low_activity_concentration"
        elif activity >= 0.45 and imbalance >= 0.30 and low_spatial >= 0.25:
            base_label = "concentrated_high_activity"
        elif activity >= 0.28 and spatial >= 0.65 and imbalance <= 0.35:
            base_label = "normal_distributed_activity"
        elif activity <= 0.25 and spatial >= 0.65:
            base_label = "distributed_low_activity"
        elif low_spatial >= 0.35:
            base_label = "localized_activity_shift"
        else:
            base_label = "mixed_activity_pattern"

        seen_labels[base_label] = seen_labels.get(base_label, 0) + 1
        if seen_labels[base_label] > 1:
            base_label = f"{base_label}_{int(row['state_id'])}"
        labels.append(base_label)
    labeled["state_label"] = labels
    return labeled


def _compute_row_risk_scores(
    posterior_matrix: np.ndarray,
    output_state_columns: int,
    risk_weights: dict[int, float],
) -> pd.Series:
    padded_matrix = np.zeros((posterior_matrix.shape[0], output_state_columns), dtype=float)
    padded_matrix[:, : posterior_matrix.shape[1]] = posterior_matrix
    weight_vector = np.array([risk_weights.get(state_id, 0.0) for state_id in range(output_state_columns)], dtype=float)
    return pd.Series(np.clip(padded_matrix @ weight_vector, 0.0, 1.0), dtype=float)


def _compute_sustained_risk_flags(sequence_df: pd.DataFrame, sustained_windows: int) -> pd.Series:
    flags = pd.Series([False] * len(sequence_df), index=sequence_df.index)
    for _, group_df in sequence_df.groupby(_sequence_group_columns(sequence_df), sort=False, dropna=False):
        consecutive_count = 0
        for index, risk_level in group_df["risk_level"].items():
            if risk_level in {"medium", "high"}:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count >= sustained_windows:
                flags.loc[index] = True
    return flags


def _risk_level(risk_score: float) -> str:
    if risk_score < 0.33:
        return "low"
    if risk_score <= 0.66:
        return "medium"
    return "high"


def _sequence_group_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = ["room_id"]
    if "session_id" in dataframe.columns:
        columns.append("session_id")
    return columns


def _sequence_sort_columns(dataframe: pd.DataFrame) -> list[str]:
    return [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in dataframe.columns]


def _series_min_max(series: pd.Series) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(numeric_series.min())
    maximum = float(numeric_series.max())
    if minimum == maximum:
        return pd.Series([0.5] * len(numeric_series), index=numeric_series.index, dtype=float)
    return (numeric_series - minimum) / (maximum - minimum)


def _risk_component_weights(config: BiomarkerTwinConfig) -> dict[str, float]:
    return {
        "activity_mobility": config.risk_activity_mobility_weight,
        "imbalance": config.risk_imbalance_weight,
        "low_spatial_freedom": config.risk_low_spatial_freedom_weight,
        "resilience_pressure": config.risk_resilience_pressure_weight,
        "persistence": config.risk_persistence_weight,
    }


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
