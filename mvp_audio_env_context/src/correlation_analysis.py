from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import AudioEnvContextConfig
from .plotting import (
    plot_audio_env_correlation_heatmap,
    plot_audio_features_over_time,
    plot_env_context_over_time,
)


ENV_COLUMNS = [
    "temp_context",
    "rh_context",
    "temp_daily_mean",
    "rh_daily_mean",
    "temp_daily_range",
]

AUDIO_COLUMNS = [
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
]


@dataclass(frozen=True)
class CorrelationAnalysisResult:
    correlation_df: pd.DataFrame
    output_path: Path
    report_path: Path
    meeting_bullet: str


def run_audio_environment_correlation(
    config: AudioEnvContextConfig,
    multimodal_df: pd.DataFrame,
    env_df: pd.DataFrame,
) -> CorrelationAnalysisResult:
    usable_df = multimodal_df.copy()
    audio_available_series = usable_df["audio_available"] if "audio_available" in usable_df.columns else pd.Series(False, index=usable_df.index)
    usable_df = usable_df[audio_available_series.fillna(False).astype(bool)].copy()
    daily_df = _build_daily_audio_environment_table(usable_df)

    rows: list[dict] = []
    for env_column in ENV_COLUMNS:
        for audio_column in AUDIO_COLUMNS:
            pair_df = daily_df[[env_column, audio_column]].dropna().copy() if not daily_df.empty else pd.DataFrame()
            if len(pair_df) < 3 or pair_df[env_column].nunique() < 2 or pair_df[audio_column].nunique() < 2:
                rows.append(
                    {
                        "env_feature": env_column,
                        "audio_feature": audio_column,
                        "n_samples": int(len(pair_df)),
                        "pearson_r": np.nan,
                        "pearson_p": np.nan,
                        "spearman_rho": np.nan,
                        "spearman_p": np.nan,
                    }
                )
                continue
            pearson_r, pearson_p = stats.pearsonr(pair_df[env_column], pair_df[audio_column])
            spearman_rho, spearman_p = stats.spearmanr(pair_df[env_column], pair_df[audio_column])
            rows.append(
                {
                    "env_feature": env_column,
                    "audio_feature": audio_column,
                    "n_samples": int(len(pair_df)),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_rho": float(spearman_rho),
                    "spearman_p": float(spearman_p),
                }
            )

    correlation_df = pd.DataFrame(rows)
    correlation_df["pearson_fdr_bh"] = _benjamini_hochberg(correlation_df["pearson_p"])
    correlation_df["spearman_fdr_bh"] = _benjamini_hochberg(correlation_df["spearman_p"])
    output_path = config.features_dir / "audio_env_correlation_table.csv"
    report_path = config.reports_dir / "audio_env_correlation_report.md"
    correlation_df.to_csv(output_path, index=False)

    plot_audio_env_correlation_heatmap(correlation_df, config.plots_dir / "audio_env_correlation_heatmap.png")
    plot_audio_features_over_time(multimodal_df, config.plots_dir / "audio_features_over_time.png")
    plot_env_context_over_time(env_df, config.plots_dir / "env_context_over_time.png")

    report_text, meeting_bullet = _build_correlation_report(correlation_df, daily_df)
    report_path.write_text(report_text, encoding="utf-8")
    return CorrelationAnalysisResult(
        correlation_df=correlation_df,
        output_path=output_path,
        report_path=report_path,
        meeting_bullet=meeting_bullet,
    )


def _build_daily_audio_environment_table(multimodal_df: pd.DataFrame) -> pd.DataFrame:
    if multimodal_df.empty or "local_date" not in multimodal_df.columns:
        return pd.DataFrame()
    available_columns = [column for column in AUDIO_COLUMNS + ENV_COLUMNS if column in multimodal_df.columns]
    if not available_columns:
        return pd.DataFrame()
    working_df = multimodal_df.copy()
    working_df["local_date"] = pd.to_datetime(working_df["local_date"], errors="coerce")
    aggregated_df = (
        working_df.groupby("local_date", dropna=False)[available_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    return aggregated_df.dropna(subset=["local_date"]).reset_index(drop=True)


def _benjamini_hochberg(p_value_series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_value_series, errors="coerce")
    result = pd.Series(np.nan, index=p_value_series.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return result
    ranked = valid.sort_values()
    count = float(len(ranked))
    adjusted = ranked * count / np.arange(1, len(ranked) + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted.iloc[::-1])[::-1].clip(0.0, 1.0)
    result.loc[adjusted.index] = adjusted
    return result


def _build_correlation_report(correlation_df: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[str, str]:
    usable_df = correlation_df.dropna(subset=["spearman_rho"]).copy()
    humidity_df = usable_df[usable_df["env_feature"].str.contains("rh", na=False)].copy()
    temperature_df = usable_df[usable_df["env_feature"].str.contains("temp", na=False)].copy()

    humidity_text = _top_association_text(humidity_df, "humidity")
    temperature_text = _top_association_text(temperature_df, "temperature")
    spectral_humidity_df = humidity_df[humidity_df["audio_feature"].str.contains("spectral", na=False)].copy()
    spectral_humidity_text = _top_association_text(spectral_humidity_df, "humidity spectral") if not spectral_humidity_df.empty else "No humidity-spectral associations were available."

    lines = [
        "# Audio-Environment Correlation Report",
        "",
        "This analysis treats environment as daily Room 1 context. It should be interpreted as contextual association, not causal proof.",
        "",
        "## Resolution Note",
        "",
        f"- Correlations were computed on date-level aggregated audio means, not raw window-level repetition.",
        f"- Number of overlapping local dates with audio + environment context: {len(daily_df)}",
        "- This avoids overstating significance from repeated daily context values across many windows.",
        "",
        "## Strongest Humidity Associations",
        "",
        humidity_text,
        "",
        "## Strongest Temperature Associations",
        "",
        temperature_text,
        "",
        "## RH And Spectral Feature Framing",
        "",
        spectral_humidity_text,
        "",
        "## Table",
        "",
        "```text",
        correlation_df.to_string(index=False),
        "```",
        "",
        "## Caveat",
        "",
        "- Environment is daily/coarse, so these correlations provide contextual framing only.",
        "- In line with the project framing, humidity may modulate acoustic features and should be modelled as context rather than treated as standalone welfare proof.",
    ]
    meeting_bullet = (
        f"- Audio-environment correlations were aggregated to the date level; overlapping day count was {len(daily_df)}, so this step is contextual rather than inferential."
    )
    return "\n".join(lines) + "\n", meeting_bullet


def _top_association_text(correlation_df: pd.DataFrame, label: str) -> str:
    if correlation_df.empty:
        return f"No {label} associations were available at the current date-level overlap."
    ranked = correlation_df.assign(abs_rho=correlation_df["spearman_rho"].abs()).sort_values("abs_rho", ascending=False)
    top_row = ranked.iloc[0]
    return (
        f"Top {label} association: `{top_row['env_feature']} -> {top_row['audio_feature']}` "
        f"with Spearman rho `{top_row['spearman_rho']:.3f}` and FDR-adjusted p `{top_row['spearman_fdr_bh']:.3f}`."
    )
