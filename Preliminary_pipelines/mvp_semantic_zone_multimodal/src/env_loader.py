from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SemanticZoneConfig


@dataclass(frozen=True)
class EnvironmentLoadResult:
    env_df: pd.DataFrame
    output_path: Path
    report_path: Path
    warnings: tuple[str, ...]


EXPECTED_COLUMNS = {
    "Date": "date",
    "Temp_AM_Min": "temp_am_min",
    "Temp_AM_Max": "temp_am_max",
    "Temp_PM": "temp_pm",
    "RH_AM": "rh_am",
    "RH_PM": "rh_pm",
}


def load_environment_daily(config: SemanticZoneConfig) -> EnvironmentLoadResult:
    if not config.env_xlsx.exists():
        raise FileNotFoundError(f"Environment workbook not found: {config.env_xlsx}")

    raw_df = pd.read_excel(config.env_xlsx, engine="openpyxl")
    warnings: list[str] = []
    normalized_df = raw_df.replace(
        {
            "NA": np.nan,
            "N/A": np.nan,
            "na": np.nan,
            "n/a": np.nan,
            "": np.nan,
        }
    ).copy()

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in normalized_df.columns]
    if missing_columns:
        raise ValueError("Environment workbook is missing expected columns: " + ", ".join(missing_columns))

    normalized_df = normalized_df[list(EXPECTED_COLUMNS)].rename(columns=EXPECTED_COLUMNS)
    normalized_df["date"] = _coerce_excel_date(normalized_df["date"])
    if normalized_df["date"].isna().any():
        warnings.append("some_dates_failed_to_parse")

    for column in ("temp_am_min", "temp_am_max", "temp_pm", "rh_am", "rh_pm"):
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    normalized_df["temp_am_mean"] = normalized_df[["temp_am_min", "temp_am_max"]].mean(axis=1, skipna=True)
    normalized_df["temp_daily_mean"] = normalized_df[["temp_am_mean", "temp_pm"]].mean(axis=1, skipna=True)
    normalized_df["rh_daily_mean"] = normalized_df[["rh_am", "rh_pm"]].mean(axis=1, skipna=True)
    normalized_df["temp_daily_range"] = normalized_df["temp_am_max"] - normalized_df["temp_am_min"]
    normalized_df.loc[
        normalized_df["temp_am_max"].isna() | normalized_df["temp_am_min"].isna(),
        "temp_daily_range",
    ] = np.nan
    normalized_df["env_quality_flag"] = normalized_df.apply(_environment_quality_flag, axis=1)
    normalized_df = normalized_df.sort_values("date", kind="stable").reset_index(drop=True)
    normalized_df["date"] = normalized_df["date"].dt.date

    output_path = config.features_dir / "env_room1_daily.csv"
    report_path = config.reports_dir / "env_data_report.md"
    normalized_df.to_csv(output_path, index=False)
    report_path.write_text(_build_env_report(normalized_df, warnings), encoding="utf-8")
    return EnvironmentLoadResult(
        env_df=normalized_df,
        output_path=output_path,
        report_path=report_path,
        warnings=tuple(warnings),
    )


def _coerce_excel_date(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    numeric_mask = pd.to_numeric(series, errors="coerce").notna()
    result = pd.to_datetime(series, errors="coerce")
    if numeric_mask.any():
        numeric_dates = pd.to_datetime(
            pd.to_numeric(series[numeric_mask], errors="coerce"),
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        result.loc[numeric_mask] = numeric_dates
    return result


def _environment_quality_flag(row: pd.Series) -> str:
    if pd.isna(row.get("date")):
        return "date_parse_failed"
    if pd.isna(row.get("temp_daily_mean")) and pd.isna(row.get("rh_daily_mean")):
        return "missing_temp_and_rh"
    if pd.isna(row.get("temp_daily_mean")) or pd.isna(row.get("rh_daily_mean")):
        return "partial"
    return "ok"


def _build_env_report(env_df: pd.DataFrame, warnings: list[str]) -> str:
    summary_columns = [
        "temp_am_min",
        "temp_am_max",
        "temp_pm",
        "temp_am_mean",
        "temp_daily_mean",
        "temp_daily_range",
        "rh_am",
        "rh_pm",
        "rh_daily_mean",
    ]
    stats_df = env_df[summary_columns].describe().transpose() if not env_df.empty else pd.DataFrame()
    lines = [
        "# Environment Data Report",
        "",
        "- Room 1 environment data are used as daily contextual covariates, not minute-level causal signals.",
        f"- Number of rows: {len(env_df)}",
        f"- Date range start: `{env_df['date'].min() if not env_df.empty else 'n/a'}`",
        f"- Date range end: `{env_df['date'].max() if not env_df.empty else 'n/a'}`",
        "",
        "## Missing Values",
        "",
    ]
    for column, missing_count in env_df.isna().sum().to_dict().items():
        lines.append(f"- `{column}`: {int(missing_count)}")
    lines.extend(
        [
            "",
            "## Summary Statistics",
            "",
            "```text",
            stats_df.to_string() if not stats_df.empty else "No environment rows available.",
            "```",
            "",
            "## Parsing Warnings",
            "",
        ]
    )
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"
