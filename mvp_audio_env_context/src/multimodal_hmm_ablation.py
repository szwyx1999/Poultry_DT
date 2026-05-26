from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import AudioEnvContextConfig
from .plotting import plot_hmm_ablation_event_alignment, plot_video_only_vs_multimodal_risk


@dataclass(frozen=True)
class HmmAblationResult:
    sequence_df: pd.DataFrame
    metrics_df: pd.DataFrame
    sequences_path: Path
    metrics_path: Path
    report_path: Path
    meeting_bullet: str


VIDEO_FEATURE_COLUMNS = [
    "mobility_index",
    "spatial_freedom_index",
    "occupancy_imbalance_index",
    "activity_mean",
]

MULTIMODAL_FEATURE_COLUMNS = VIDEO_FEATURE_COLUMNS + [
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "temp_context",
    "rh_context",
]


def run_multimodal_hmm_ablation(
    config: AudioEnvContextConfig,
    multimodal_df: pd.DataFrame,
) -> HmmAblationResult:
    if multimodal_df.empty:
        raise ValueError("Multimodal window table is empty; cannot run HMM ablation.")

    working_df = multimodal_df.copy()
    working_df["start_time_dt"] = pd.to_datetime(working_df["start_time"], errors="coerce", utc=False)
    working_df["minutes_from_event_start"] = pd.to_numeric(working_df["seconds_from_event_start"], errors="coerce") / 60.0
    usable_df = working_df[
        working_df["audio_available"].fillna(False).astype(bool)
        & working_df["temp_context"].notna()
        & working_df["rh_context"].notna()
    ].copy()
    if usable_df.empty:
        raise ValueError("No multimodal windows with audio and environment context were available for HMM ablation.")

    n_states = _determine_state_count(len(usable_df), config)
    sequence_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    model_summaries: list[str] = []

    for model_name, feature_columns in (
        ("video_only", VIDEO_FEATURE_COLUMNS),
        ("video_audio_env", MULTIMODAL_FEATURE_COLUMNS),
    ):
        model_result = _fit_hmm_model(
            usable_df=usable_df,
            feature_columns=feature_columns,
            model_name=model_name,
            n_states=n_states,
            config=config,
        )
        sequence_frames.append(model_result["sequence_df"])
        metric_rows.extend(model_result["metric_rows"])
        model_summaries.append(model_result["summary_text"])

    sequence_df = pd.concat(sequence_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    sequences_path = config.features_dir / "hmm_ablation_state_sequences.csv"
    metrics_path = config.features_dir / "hmm_ablation_metrics.csv"
    report_path = config.reports_dir / "multimodal_hmm_ablation_report.md"
    sequence_df.to_csv(sequences_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    plot_hmm_ablation_event_alignment(sequence_df, config.plots_dir / "hmm_ablation_event_alignment.png")
    plot_video_only_vs_multimodal_risk(sequence_df, config.plots_dir / "video_only_vs_multimodal_risk.png")

    report_text, meeting_bullet = _build_ablation_report(metrics_df, model_summaries, n_states)
    report_path.write_text(report_text, encoding="utf-8")
    return HmmAblationResult(
        sequence_df=sequence_df,
        metrics_df=metrics_df,
        sequences_path=sequences_path,
        metrics_path=metrics_path,
        report_path=report_path,
        meeting_bullet=meeting_bullet,
    )


def _fit_hmm_model(
    usable_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    n_states: int,
    config: AudioEnvContextConfig,
) -> dict[str, object]:
    sorted_df = usable_df.sort_values(
        [column for column in ("room_id", "session_id", "start_time_dt", "window_id") if column in usable_df.columns],
        kind="stable",
    ).reset_index(drop=True)
    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    feature_matrix = preprocessing.fit_transform(sorted_df[feature_columns])
    sequence_lengths = sorted_df.groupby(
        [column for column in ("room_id", "session_id") if column in sorted_df.columns],
        sort=False,
        dropna=False,
    ).size().tolist()

    model, posterior_matrix, state_ids, model_type = _fit_gaussian_hmm_or_fallback(
        feature_matrix=feature_matrix,
        sequence_lengths=sequence_lengths,
        n_states=n_states,
        config=config,
    )

    state_summary_df = _build_state_summary(sorted_df, state_ids)
    state_risk_weights = _compute_state_risk_weights(state_summary_df)
    risk_vector = np.array([state_risk_weights.get(state_id, 0.0) for state_id in range(n_states)], dtype=float)
    padded_posteriors = np.zeros((posterior_matrix.shape[0], n_states), dtype=float)
    padded_posteriors[:, : posterior_matrix.shape[1]] = posterior_matrix
    risk_scores = np.clip(padded_posteriors @ risk_vector, 0.0, 1.0)

    sequence_df = sorted_df[
        [
            "window_id",
            "room_id",
            "session_id",
            "start_time",
            "end_time",
            "event_id",
            "event_phase",
            "minutes_from_event_start",
            "event_start_time",
            "event_end_time",
        ]
    ].copy()
    sequence_df["model_name"] = model_name
    sequence_df["model_type"] = model_type
    sequence_df["state_id"] = state_ids
    sequence_df["state_probability_max"] = posterior_matrix.max(axis=1)
    sequence_df["risk_score"] = risk_scores
    sequence_df["risk_level"] = pd.cut(
        risk_scores,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    metric_rows = _build_model_metrics(sequence_df, feature_matrix, model, model_name, n_states)
    summary_text = (
        f"- `{model_name}` used {n_states} states on {len(sorted_df)} windows and produced mean posterior confidence "
        f"`{sequence_df['state_probability_max'].mean():.3f}`."
    )
    return {
        "sequence_df": sequence_df,
        "metric_rows": metric_rows,
        "summary_text": summary_text,
    }


def _fit_gaussian_hmm_or_fallback(
    feature_matrix: np.ndarray,
    sequence_lengths: list[int],
    n_states: int,
    config: AudioEnvContextConfig,
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
        state_ids = hmm_model.predict(feature_matrix, lengths=sequence_lengths)
        return hmm_model, posterior_matrix, state_ids, "gaussian_hmm"
    except Exception:
        gmm_model = GaussianMixture(
            n_components=n_states,
            covariance_type=config.covariance_type,
            random_state=config.random_state,
            reg_covar=1e-6,
        )
        gmm_model.fit(feature_matrix)
        posterior_matrix = gmm_model.predict_proba(feature_matrix)
        state_ids = posterior_matrix.argmax(axis=1)
        return gmm_model, posterior_matrix, state_ids, "fallback_gmm_state_model"


def _build_state_summary(sorted_df: pd.DataFrame, state_ids: np.ndarray) -> pd.DataFrame:
    summary_df = sorted_df.copy()
    summary_df["state_id"] = state_ids
    return (
        summary_df.groupby("state_id", sort=True)
        .agg(
            mobility_index_mean=("mobility_index", "mean"),
            spatial_freedom_index_mean=("spatial_freedom_index", "mean"),
            occupancy_imbalance_index_mean=("occupancy_imbalance_index", "mean"),
            activity_mean_mean=("activity_mean", "mean"),
        )
        .reset_index()
    )


def _compute_state_risk_weights(summary_df: pd.DataFrame) -> dict[int, float]:
    working_df = summary_df.copy()
    working_df["activity_component"] = _minmax(working_df["activity_mean_mean"])
    working_df["mobility_component"] = _minmax(working_df["mobility_index_mean"])
    working_df["low_spatial_component"] = (1.0 - pd.to_numeric(working_df["spatial_freedom_index_mean"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    working_df["imbalance_component"] = pd.to_numeric(working_df["occupancy_imbalance_index_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    working_df["risk_weight"] = (
        0.35 * working_df["activity_component"]
        + 0.35 * working_df["mobility_component"]
        + 0.20 * working_df["imbalance_component"]
        + 0.10 * working_df["low_spatial_component"]
    ).clip(0.0, 1.0)
    return dict(zip(working_df["state_id"], working_df["risk_weight"]))


def _build_model_metrics(
    sequence_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    model: object,
    model_name: str,
    n_states: int,
) -> list[dict]:
    baseline_df = sequence_df[sequence_df["event_phase"] == "pre_entry_baseline"]
    during_df = sequence_df[sequence_df["event_phase"] == "during_entry"]
    recovery_df = sequence_df[sequence_df["event_phase"] == "post_entry_recovery"]
    outside_df = sequence_df[sequence_df["event_phase"] == "outside_event_window"]

    baseline_mean = baseline_df["risk_score"].mean() if not baseline_df.empty else np.nan
    baseline_std = baseline_df["risk_score"].std() if not baseline_df.empty else np.nan
    threshold = baseline_mean + baseline_std if pd.notna(baseline_mean) and pd.notna(baseline_std) else np.nan
    detection_minutes = np.nan
    if pd.notna(threshold) and not during_df.empty:
        hit_df = during_df[during_df["risk_score"] > threshold]
        if not hit_df.empty:
            detection_minutes = float(hit_df["minutes_from_event_start"].iloc[0])

    purity_scores = []
    for phase in ("pre_entry_baseline", "during_entry", "post_entry_recovery"):
        phase_df = sequence_df[sequence_df["event_phase"] == phase]
        if phase_df.empty:
            continue
        purity_scores.append(float(phase_df["state_id"].value_counts(normalize=True).iloc[0]))
    outside_false_positive = np.nan
    if pd.notna(threshold) and not outside_df.empty:
        outside_false_positive = float((outside_df["risk_score"] > threshold).mean())

    log_likelihood_per_sample = np.nan
    if hasattr(model, "score"):
        try:
            log_likelihood_per_sample = float(model.score(feature_matrix) / max(len(feature_matrix), 1))
        except Exception:
            log_likelihood_per_sample = np.nan

    metrics = [
        ("event_detection_contrast", float(during_df["risk_score"].mean() - baseline_mean) if not during_df.empty and pd.notna(baseline_mean) else np.nan),
        ("state_purity_by_event_phase", float(np.mean(purity_scores)) if purity_scores else np.nan),
        ("time_to_detect_minutes", detection_minutes),
        ("false_positive_rate_outside_event_window", outside_false_positive),
        ("log_likelihood_per_sample", log_likelihood_per_sample),
        ("posterior_confidence_mean", float(sequence_df["state_probability_max"].mean())),
        ("posterior_confidence_during_event", float(during_df["state_probability_max"].mean()) if not during_df.empty else np.nan),
        ("risk_mean_baseline", float(baseline_mean) if pd.notna(baseline_mean) else np.nan),
        ("risk_mean_during_event", float(during_df["risk_score"].mean()) if not during_df.empty else np.nan),
        ("risk_mean_recovery", float(recovery_df["risk_score"].mean()) if not recovery_df.empty else np.nan),
    ]
    return [
        {
            "model_name": model_name,
            "n_states": n_states,
            "metric_name": metric_name,
            "value": value,
        }
        for metric_name, value in metrics
    ]


def _build_ablation_report(metrics_df: pd.DataFrame, model_summaries: list[str], n_states: int) -> tuple[str, str]:
    def metric(model_name: str, metric_name: str) -> float:
        subset_df = metrics_df[
            (metrics_df["model_name"] == model_name)
            & (metrics_df["metric_name"] == metric_name)
        ]
        if subset_df.empty:
            return np.nan
        value = subset_df.iloc[0]["value"]
        return float(value) if pd.notna(value) else np.nan

    video_contrast = metric("video_only", "event_detection_contrast")
    multi_contrast = metric("video_audio_env", "event_detection_contrast")
    if pd.notna(video_contrast) and pd.notna(multi_contrast):
        if multi_contrast > video_contrast:
            interpretation = "Multimodal HMM provided stronger event contrast than video-only."
        elif multi_contrast < video_contrast:
            interpretation = "Video-only HMM retained stronger event contrast than the multimodal variant."
        else:
            interpretation = "Video-only and multimodal HMM variants produced similar event contrast."
    else:
        interpretation = "HMM event-contrast comparison was inconclusive."

    lines = [
        "# Multimodal HMM Ablation Report",
        "",
        "This ablation compares a video-only latent-state model against a video+audio+environment feature set.",
        "",
        f"- Shared state count used: {n_states}",
        f"- Interpretation: {interpretation}",
        "",
        "## Model Notes",
        "",
    ]
    lines.extend(model_summaries)
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```text",
            metrics_df.to_string(index=False),
            "```",
            "",
            "## Caveat",
            "",
            "- Environment is daily/coarse, so it may add contextual structure without improving short event detection.",
            "- If multimodal does not improve event contrast, that should be reported honestly rather than forced into a welfare claim.",
        ]
    )
    meeting_bullet = f"- Multimodal HMM ablation result: {interpretation}"
    return "\n".join(lines) + "\n", meeting_bullet


def _determine_state_count(n_windows: int, config: AudioEnvContextConfig) -> int:
    if config.hmm_n_states == "auto":
        if n_windows < 12:
            return min(2, n_windows)
        if n_windows < 30:
            return min(3, n_windows)
        return min(config.max_hmm_states, n_windows)
    return max(1, min(int(config.hmm_n_states), config.max_hmm_states, n_windows))


def _minmax(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if minimum == maximum:
        return pd.Series([0.5] * len(numeric), index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)
