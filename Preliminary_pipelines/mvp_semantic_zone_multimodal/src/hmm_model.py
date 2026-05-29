from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import SemanticZoneConfig
from .utils import minmax_scale, prepare_time_columns


@dataclass(frozen=True)
class SemanticHmmResult:
    sequence_df: pd.DataFrame
    summary_df: pd.DataFrame
    metrics_df: pd.DataFrame
    sequence_path: Path
    summary_path: Path
    metrics_path: Path
    meeting_bullet: str


MODEL_FEATURES = {
    "semantic_video_only": [
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "normalized_activity",
        "drinking_activity_fraction",
        "feeding_activity_fraction",
        "general_activity_fraction",
        "feeding_plus_drinking_activity_fraction",
        "semantic_transition_proxy",
    ],
    "semantic_multimodal": [
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "normalized_activity",
        "drinking_activity_fraction",
        "feeding_activity_fraction",
        "general_activity_fraction",
        "feeding_plus_drinking_activity_fraction",
        "semantic_transition_proxy",
        "audio_rms",
        "audio_short_time_energy",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_spectral_rolloff",
        "temp_context",
        "rh_context",
    ],
}


def fit_semantic_hmm_models(
    config: SemanticZoneConfig,
    labelled_df: pd.DataFrame,
) -> SemanticHmmResult:
    if labelled_df.empty:
        raise ValueError("Semantic labelled window table is empty; cannot fit semantic HMM models.")

    working_df = prepare_time_columns(labelled_df)
    working_df = (
        working_df.drop_duplicates(subset=["window_id"], keep="first")
        .sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable")
        .reset_index(drop=True)
    )
    n_states = _determine_state_count(len(working_df), config)

    sequence_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for model_name, feature_columns in MODEL_FEATURES.items():
        available_columns = [column for column in feature_columns if column in working_df.columns]
        if not available_columns:
            continue
        model_result = _fit_single_model(
            config=config,
            working_df=working_df,
            model_name=model_name,
            feature_columns=available_columns,
            n_states=n_states,
        )
        sequence_frames.append(model_result["sequence_df"])
        summary_frames.append(model_result["summary_df"])
        metric_rows.extend(model_result["metrics"])

    sequence_df = pd.concat(sequence_frames, ignore_index=True) if sequence_frames else pd.DataFrame()
    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)

    sequence_path = config.features_dir / "semantic_hmm_state_sequence.csv"
    summary_path = config.features_dir / "semantic_hmm_state_summary.csv"
    metrics_path = config.features_dir / "semantic_hmm_ablation_metrics.csv"
    sequence_df.to_csv(sequence_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    meeting_bullet = _build_meeting_bullet(metrics_df)
    return SemanticHmmResult(
        sequence_df=sequence_df,
        summary_df=summary_df,
        metrics_df=metrics_df,
        sequence_path=sequence_path,
        summary_path=summary_path,
        metrics_path=metrics_path,
        meeting_bullet=meeting_bullet,
    )


def _fit_single_model(
    config: SemanticZoneConfig,
    working_df: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    n_states: int,
) -> dict[str, object]:
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    feature_matrix = preprocessing.fit_transform(working_df[feature_columns])
    sequence_lengths = working_df.groupby(["room_id", "session_id"], sort=False, dropna=False).size().tolist()
    model, posterior_matrix, predicted_states, model_type = _fit_gaussian_hmm_or_fallback(
        feature_matrix=feature_matrix,
        sequence_lengths=sequence_lengths,
        n_states=n_states,
        config=config,
    )
    if model_type == "fallback_gmm_state_model":
        predicted_states = _smooth_states_by_sequence(working_df, predicted_states)
        posterior_matrix = _posterior_from_state_ids(predicted_states, n_states)

    sequence_df = _build_sequence_df(working_df, predicted_states, posterior_matrix, model_name, model_type, n_states)
    summary_df = _build_state_summary(sequence_df, model_name=model_name, model_type=model_type)
    summary_df = _add_risk_components(summary_df)
    summary_df = _assign_state_labels(summary_df)

    label_map = dict(zip(summary_df["state_id"], summary_df["state_label"]))
    risk_map = dict(zip(summary_df["state_id"], summary_df["state_risk_weight"]))
    sequence_df["state_label"] = sequence_df["state_id"].map(label_map).fillna("unlabeled_state")
    sequence_df["risk_score"] = _compute_row_risk_scores(posterior_matrix, n_states, risk_map)
    sequence_df["risk_level"] = sequence_df["risk_score"].apply(_risk_level)
    sequence_df["sustained_risk_flag"] = _compute_sustained_risk_flags(sequence_df, sustained_windows=3)
    sequence_df["high_risk_flag"] = sequence_df["risk_level"].eq("high")

    metrics = _build_model_metrics(sequence_df, feature_matrix, model, model_name, model_type, n_states)
    return {
        "sequence_df": sequence_df,
        "summary_df": summary_df,
        "metrics": metrics,
    }


def _fit_gaussian_hmm_or_fallback(
    feature_matrix: np.ndarray,
    sequence_lengths: list[int],
    n_states: int,
    config: SemanticZoneConfig,
) -> tuple[object, np.ndarray, np.ndarray, str]:
    try:
        from hmmlearn.hmm import GaussianHMM

        hmm_model = GaussianHMM(
            n_components=n_states,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            n_iter=250,
        )
        hmm_model.fit(feature_matrix, lengths=sequence_lengths)
        posterior_matrix = hmm_model.predict_proba(feature_matrix, lengths=sequence_lengths)
        predicted_states = hmm_model.predict(feature_matrix, lengths=sequence_lengths)
        return hmm_model, posterior_matrix, predicted_states, "gaussian_hmm"
    except Exception:
        gmm_model = GaussianMixture(
            n_components=n_states,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            reg_covar=1e-6,
        )
        gmm_model.fit(feature_matrix)
        posterior_matrix = gmm_model.predict_proba(feature_matrix)
        predicted_states = posterior_matrix.argmax(axis=1)
        return gmm_model, posterior_matrix, predicted_states, "fallback_gmm_state_model"


def _smooth_states_by_sequence(working_df: pd.DataFrame, state_ids: np.ndarray) -> np.ndarray:
    smoothed = np.asarray(state_ids, dtype=int).copy()
    for _, group_df in working_df.groupby(["room_id", "session_id"], sort=False, dropna=False):
        positions = group_df.index.tolist()
        for position_index in range(1, len(positions) - 1):
            left_position = positions[position_index - 1]
            center_position = positions[position_index]
            right_position = positions[position_index + 1]
            if smoothed[left_position] == smoothed[right_position] and smoothed[center_position] != smoothed[left_position]:
                smoothed[center_position] = smoothed[left_position]
    return smoothed


def _posterior_from_state_ids(state_ids: np.ndarray, n_states: int) -> np.ndarray:
    posterior_matrix = np.zeros((len(state_ids), n_states), dtype=float)
    for row_index, state_id in enumerate(state_ids):
        posterior_matrix[row_index, int(state_id)] = 1.0
    return posterior_matrix


def _build_sequence_df(
    working_df: pd.DataFrame,
    predicted_states: np.ndarray,
    posterior_matrix: np.ndarray,
    model_name: str,
    model_type: str,
    n_states: int,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "media_id",
        "system_id",
        "room_id",
        "session_id",
        "start_time",
        "end_time",
        "start_time_dt",
        "end_time_dt",
        "event_id",
        "event_type",
        "event_phase",
        "seconds_from_event_start",
        "seconds_from_event_end",
        "event_start_time",
        "event_end_time",
        "overlaps_event",
        "activity_mean",
        "normalized_activity",
        "mobility_index",
        "spatial_freedom_index",
        "occupancy_imbalance_index",
        "drinking_activity_fraction",
        "feeding_activity_fraction",
        "general_activity_fraction",
        "feeding_plus_drinking_activity_fraction",
        "functional_area_activity_fraction",
        "semantic_transition_proxy",
        "audio_rms",
        "audio_short_time_energy",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_spectral_rolloff",
        "temp_context",
        "rh_context",
    ]
    available_columns = [column for column in columns if column in working_df.columns]
    sequence_df = working_df[available_columns].copy()
    sequence_df["model_name"] = model_name
    sequence_df["model_type"] = model_type
    sequence_df["state_id"] = predicted_states
    sequence_df["state_probability_max"] = posterior_matrix.max(axis=1)
    for state_index in range(n_states):
        sequence_df[f"posterior_state_{state_index}"] = posterior_matrix[:, state_index]
    sequence_df["state_label"] = "pending_label"
    return sequence_df


def _build_state_summary(sequence_df: pd.DataFrame, model_name: str, model_type: str) -> pd.DataFrame:
    summary_df = (
        sequence_df.groupby("state_id", sort=True)
        .agg(
            window_count=("window_id", "count"),
            activity_mean_mean=("activity_mean", "mean"),
            normalized_activity_mean=("normalized_activity", "mean"),
            mobility_index_mean=("mobility_index", "mean"),
            spatial_freedom_index_mean=("spatial_freedom_index", "mean"),
            occupancy_imbalance_index_mean=("occupancy_imbalance_index", "mean"),
            drinking_activity_fraction_mean=("drinking_activity_fraction", "mean"),
            feeding_activity_fraction_mean=("feeding_activity_fraction", "mean"),
            general_activity_fraction_mean=("general_activity_fraction", "mean"),
            feeding_plus_drinking_activity_fraction_mean=("feeding_plus_drinking_activity_fraction", "mean"),
            semantic_transition_proxy_mean=("semantic_transition_proxy", "mean"),
            audio_rms_mean=("audio_rms", "mean"),
            audio_spectral_centroid_mean=("audio_spectral_centroid", "mean"),
            temp_context_mean=("temp_context", "mean"),
            rh_context_mean=("rh_context", "mean"),
            state_probability_max_mean=("state_probability_max", "mean"),
        )
        .reset_index()
    )
    summary_df["model_name"] = model_name
    summary_df["model_type"] = model_type
    total_windows = max(len(sequence_df), 1)
    summary_df["window_fraction"] = summary_df["window_count"] / total_windows
    return summary_df


def _add_risk_components(summary_df: pd.DataFrame) -> pd.DataFrame:
    enriched = summary_df.copy()
    enriched["component_activity"] = minmax_scale(enriched["normalized_activity_mean"])
    enriched["component_mobility"] = minmax_scale(enriched["mobility_index_mean"])
    enriched["component_imbalance"] = pd.to_numeric(enriched["occupancy_imbalance_index_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["component_low_spatial"] = (1.0 - pd.to_numeric(enriched["spatial_freedom_index_mean"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    enriched["component_transition"] = minmax_scale(enriched["semantic_transition_proxy_mean"])
    enriched["component_functional_shift"] = pd.to_numeric(enriched["feeding_plus_drinking_activity_fraction_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["state_risk_weight"] = (
        0.22 * enriched["component_activity"]
        + 0.24 * enriched["component_mobility"]
        + 0.20 * enriched["component_imbalance"]
        + 0.16 * enriched["component_low_spatial"]
        + 0.13 * enriched["component_transition"]
        + 0.05 * enriched["component_functional_shift"]
    ).clip(0.0, 1.0)
    return enriched


def _assign_state_labels(summary_df: pd.DataFrame) -> pd.DataFrame:
    labelled = summary_df.copy()
    labels: list[str] = []
    seen: dict[str, int] = {}
    for _, row in labelled.iterrows():
        activity = _safe(row.get("normalized_activity_mean"))
        mobility = _safe(row.get("mobility_index_mean"))
        spatial = _safe(row.get("spatial_freedom_index_mean"))
        imbalance = _safe(row.get("occupancy_imbalance_index_mean"))
        transition = _safe(row.get("semantic_transition_proxy_mean"))
        drinking = _safe(row.get("drinking_activity_fraction_mean"))
        feeding = _safe(row.get("feeding_activity_fraction_mean"))
        general = _safe(row.get("general_activity_fraction_mean"))
        functional = _safe(row.get("feeding_plus_drinking_activity_fraction_mean"))
        risk = _safe(row.get("state_risk_weight"))

        if risk >= 0.68 and (mobility >= 0.45 or transition >= 0.40):
            base_label = "disturbance_or_transition"
        elif activity <= 0.18 and mobility <= 0.22 and imbalance <= 0.22:
            base_label = "distributed_low_activity"
        elif activity <= 0.22 and (imbalance >= 0.30 or spatial <= 0.72):
            base_label = "localized_low_activity_concentration"
        elif drinking >= max(feeding, general) and drinking >= 0.42:
            base_label = "drinker_concentrated_activity"
        elif feeding >= max(drinking, general) and feeding >= 0.42:
            base_label = "feeder_concentrated_activity"
        elif functional >= 0.60 and activity >= 0.24:
            base_label = "functional_area_activity"
        elif general >= 0.55 and activity >= 0.28:
            base_label = "general_area_activity"
        else:
            base_label = "mixed_semantic_activity_pattern"

        seen[base_label] = seen.get(base_label, 0) + 1
        if seen[base_label] > 1:
            base_label = f"{base_label}_{int(row['state_id'])}"
        labels.append(base_label)
    labelled["state_label"] = labels
    return labelled


def _compute_row_risk_scores(posterior_matrix: np.ndarray, n_states: int, risk_map: dict[int, float]) -> pd.Series:
    padded = np.zeros((posterior_matrix.shape[0], n_states), dtype=float)
    padded[:, : posterior_matrix.shape[1]] = posterior_matrix
    weights = np.array([risk_map.get(state_id, 0.0) for state_id in range(n_states)], dtype=float)
    return pd.Series(np.clip(padded @ weights, 0.0, 1.0), dtype=float)


def _compute_sustained_risk_flags(sequence_df: pd.DataFrame, sustained_windows: int) -> pd.Series:
    flags = pd.Series([False] * len(sequence_df), index=sequence_df.index)
    for _, group_df in sequence_df.groupby(["room_id", "session_id"], sort=False, dropna=False):
        consecutive = 0
        for index, risk_level in group_df["risk_level"].items():
            if risk_level in {"medium", "high"}:
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= sustained_windows:
                flags.loc[index] = True
    return flags


def _build_model_metrics(
    sequence_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    model: object,
    model_name: str,
    model_type: str,
    n_states: int,
) -> list[dict]:
    baseline_df = sequence_df[sequence_df["event_phase"] == "pre_entry_baseline"]
    during_df = sequence_df[sequence_df["event_phase"] == "during_entry"]
    recovery_df = sequence_df[sequence_df["event_phase"] == "post_entry_recovery"]
    outside_df = sequence_df[sequence_df["event_phase"] == "outside_event_window"]

    baseline_mean = float(baseline_df["risk_score"].mean()) if not baseline_df.empty else np.nan
    baseline_std = float(baseline_df["risk_score"].std()) if not baseline_df.empty else np.nan
    threshold = baseline_mean + baseline_std if pd.notna(baseline_mean) and pd.notna(baseline_std) else np.nan
    time_to_detect = np.nan
    if pd.notna(threshold) and not during_df.empty:
        hit_df = during_df[during_df["risk_score"] > threshold]
        if not hit_df.empty:
            time_to_detect = float(hit_df["seconds_from_event_start"].iloc[0] / 60.0)

    purity_scores: list[float] = []
    for phase in ("pre_entry_baseline", "during_entry", "post_entry_recovery"):
        phase_df = sequence_df[sequence_df["event_phase"] == phase]
        if not phase_df.empty:
            purity_scores.append(float(phase_df["state_id"].value_counts(normalize=True).iloc[0]))

    log_likelihood = np.nan
    if hasattr(model, "score"):
        try:
            log_likelihood = float(model.score(feature_matrix) / max(len(feature_matrix), 1))
        except Exception:
            log_likelihood = np.nan

    metrics = [
        ("event_detection_contrast", float(during_df["risk_score"].mean() - baseline_mean) if not during_df.empty and pd.notna(baseline_mean) else np.nan),
        ("state_purity_by_event_phase", float(np.mean(purity_scores)) if purity_scores else np.nan),
        ("time_to_detect_minutes", time_to_detect),
        ("false_positive_rate_outside_event_window", float((outside_df["risk_score"] > threshold).mean()) if pd.notna(threshold) and not outside_df.empty else np.nan),
        ("log_likelihood_per_sample", log_likelihood),
        ("posterior_confidence_mean", float(sequence_df["state_probability_max"].mean())),
        ("posterior_confidence_during_event", float(during_df["state_probability_max"].mean()) if not during_df.empty else np.nan),
        ("risk_mean_baseline", baseline_mean),
        ("risk_mean_during_event", float(during_df["risk_score"].mean()) if not during_df.empty else np.nan),
        ("risk_mean_recovery", float(recovery_df["risk_score"].mean()) if not recovery_df.empty else np.nan),
        ("occupied_states", float(sequence_df["state_id"].nunique())),
    ]
    return [
        {
            "model_name": model_name,
            "model_type": model_type,
            "n_states": n_states,
            "metric_name": metric_name,
            "value": value,
        }
        for metric_name, value in metrics
    ]


def _build_meeting_bullet(metrics_df: pd.DataFrame) -> str:
    if metrics_df.empty:
        return "- Semantic HMM models did not produce usable outputs."
    def _metric(model_name: str, metric_name: str) -> float:
        subset = metrics_df[(metrics_df["model_name"] == model_name) & (metrics_df["metric_name"] == metric_name)]
        if subset.empty:
            return np.nan
        value = subset.iloc[0]["value"]
        return float(value) if pd.notna(value) else np.nan

    video_contrast = _metric("semantic_video_only", "event_detection_contrast")
    multi_contrast = _metric("semantic_multimodal", "event_detection_contrast")
    if pd.notna(video_contrast) and pd.notna(multi_contrast):
        if multi_contrast > video_contrast:
            return "- The semantic multimodal HMM produced stronger caretaker-event contrast than the semantic video-only HMM."
        if multi_contrast < video_contrast:
            return "- The semantic video-only HMM retained stronger caretaker-event contrast than the semantic multimodal HMM."
        return "- The semantic video-only and semantic multimodal HMMs produced similar caretaker-event contrast."
    return "- Semantic HMM models ran, but the event-contrast comparison was inconclusive."


def _determine_state_count(n_windows: int, config: SemanticZoneConfig) -> int:
    if config.hmm_n_states == "auto":
        if n_windows < 12:
            return min(2, n_windows)
        if n_windows < 30:
            return min(3, n_windows)
        return min(config.max_hmm_states, n_windows)
    return max(1, min(int(config.hmm_n_states), config.max_hmm_states, n_windows))


def _risk_level(value: float) -> str:
    if value < 0.33:
        return "low"
    if value <= 0.66:
        return "medium"
    return "high"


def _safe(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
