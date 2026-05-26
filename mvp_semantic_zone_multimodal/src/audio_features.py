from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .config import SemanticZoneConfig
from .utils import prepare_time_columns, read_csv_if_exists, resolve_video_path


AUDIO_FEATURE_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "audio_spectral_flatness",
    "audio_duration_sec",
    "audio_sample_rate",
    "audio_available",
    "audio_quality_flag",
    "audio_warning",
]


@dataclass(frozen=True)
class AudioExtractionResult:
    audio_df: pd.DataFrame
    source_windows_df: pd.DataFrame
    output_path: Path
    report_path: Path
    failed_window_ids: tuple[str, ...]
    reused_precomputed_count: int


def extract_audio_window_features(
    config: SemanticZoneConfig,
    semantic_biomarker_df: pd.DataFrame,
) -> AudioExtractionResult:
    source_windows_df = _prepare_audio_source_windows(config, semantic_biomarker_df)
    output_path = config.features_dir / "audio_window_features.csv"
    report_path = config.reports_dir / "audio_extraction_report.md"

    cached_df = _load_cached_audio_features(output_path, source_windows_df, config.audio_force_recompute)
    completed_window_ids = set(cached_df["window_id"].astype(str).tolist()) if not cached_df.empty else set()

    reused_precomputed_count = 0
    if not config.audio_force_recompute:
        warm_cache_df = _load_warm_cache(config.existing_audio_window_features_csv, source_windows_df, completed_window_ids)
        if not warm_cache_df.empty:
            reused_precomputed_count = len(warm_cache_df)
            cached_df = pd.concat([cached_df, warm_cache_df], ignore_index=True) if not cached_df.empty else warm_cache_df
            completed_window_ids = set(cached_df["window_id"].astype(str).tolist())

    pending_df = source_windows_df[~source_windows_df["window_id"].astype(str).isin(completed_window_ids)].copy()
    chunks: list[pd.DataFrame] = [cached_df] if not cached_df.empty else []
    failed_window_ids: list[str] = []

    if not pending_df.empty:
        grouped = list(pending_df.groupby("media_id", sort=False, dropna=False))
        for _, media_df in grouped:
            media_df = media_df.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
            media_rows = _extract_media_audio_window_features(media_df, config)
            failures = media_rows.loc[
                ~media_rows["audio_available"].astype(bool),
                "window_id",
            ].astype(str).tolist()
            failed_window_ids.extend(failures)
            chunks.append(media_rows)
            partial_df = _sort_audio_df(pd.concat(chunks, ignore_index=True))
            partial_df.to_csv(output_path, index=False)

    final_df = _sort_audio_df(pd.concat(chunks, ignore_index=True)) if chunks else pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    final_df.to_csv(output_path, index=False)
    report_path.write_text(
        _build_audio_report(
            source_windows_df=source_windows_df,
            audio_df=final_df,
            failed_window_ids=failed_window_ids,
            reused_precomputed_count=reused_precomputed_count,
            max_failures=config.audio_max_report_failures,
        ),
        encoding="utf-8",
    )
    return AudioExtractionResult(
        audio_df=final_df,
        source_windows_df=source_windows_df,
        output_path=output_path,
        report_path=report_path,
        failed_window_ids=tuple(failed_window_ids),
        reused_precomputed_count=reused_precomputed_count,
    )


def _prepare_audio_source_windows(config: SemanticZoneConfig, semantic_biomarker_df: pd.DataFrame) -> pd.DataFrame:
    if semantic_biomarker_df.empty:
        raise ValueError("Semantic biomarker window table is empty; cannot prepare audio windows.")

    selected_df = read_csv_if_exists(config.selected_windows_csv)
    if selected_df.empty:
        selected_df = read_csv_if_exists(config.video_window_index_csv)
    if selected_df.empty:
        raise ValueError("No selected windows or indexed windows were available for audio extraction.")

    working_df = selected_df.copy()
    if "resolved_video_path" not in working_df.columns:
        working_df["resolved_video_path"] = working_df["video_path"].astype(str).apply(
            lambda value: str(resolve_video_path(config.workspace_root, value))
        )
    if "has_audio" not in working_df.columns:
        working_df["has_audio"] = True

    working_df = working_df.merge(
        semantic_biomarker_df[["window_id", "room_id", "session_id", "start_time", "end_time"]].drop_duplicates(subset=["window_id"]),
        on="window_id",
        how="inner",
        suffixes=("", "_bio"),
    )
    for column in ("room_id", "session_id", "start_time", "end_time"):
        bio_column = f"{column}_bio"
        if bio_column in working_df.columns:
            working_df[column] = working_df[bio_column]
            working_df = working_df.drop(columns=[bio_column])

    return (
        working_df[
            [
                "window_id",
                "media_id",
                "room_id",
                "session_id",
                "start_time",
                "end_time",
                "duration_seconds",
                "video_path",
                "resolved_video_path",
                "video_start_offset_sec",
                "has_audio",
            ]
        ]
        .drop_duplicates(subset=["window_id"], keep="first")
        .sort_values(["room_id", "session_id", "start_time", "window_id"], kind="stable")
        .reset_index(drop=True)
    )


def _load_cached_audio_features(output_path: Path, source_windows_df: pd.DataFrame, force_recompute: bool) -> pd.DataFrame:
    if force_recompute or not output_path.exists():
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    existing_df = pd.read_csv(output_path)
    if existing_df.empty or "window_id" not in existing_df.columns:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    valid_window_ids = set(source_windows_df["window_id"].astype(str).tolist())
    return existing_df[existing_df["window_id"].astype(str).isin(valid_window_ids)].copy()


def _load_warm_cache(
    warm_cache_path: Path,
    source_windows_df: pd.DataFrame,
    completed_window_ids: set[str],
) -> pd.DataFrame:
    if not warm_cache_path.exists():
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    warm_df = pd.read_csv(warm_cache_path)
    if warm_df.empty or "window_id" not in warm_df.columns:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    valid_window_ids = set(source_windows_df["window_id"].astype(str).tolist())
    warm_df = warm_df[
        warm_df["window_id"].astype(str).isin(valid_window_ids)
        & ~warm_df["window_id"].astype(str).isin(completed_window_ids)
    ].copy()
    return warm_df


def _extract_media_audio_window_features(media_df: pd.DataFrame, config: SemanticZoneConfig) -> pd.DataFrame:
    if media_df.empty:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    if not bool(media_df["has_audio"].fillna(False).any()):
        return _unavailable_audio_rows(media_df, "manifest_has_audio_false", config.audio_sample_rate)

    resolved_path = Path(str(media_df.iloc[0]["resolved_video_path"]))
    if not resolved_path.exists():
        return _unavailable_audio_rows(media_df, "missing_video_file", config.audio_sample_rate)

    decode_result = _decode_media_audio(
        media_path=resolved_path,
        ffmpeg_path=config.ffmpeg_path,
        sample_rate=config.audio_sample_rate,
    )
    if decode_result["waveform"] is None:
        return _unavailable_audio_rows(
            media_df,
            str(decode_result["quality_flag"]),
            config.audio_sample_rate,
            warning=str(decode_result["warning"]),
        )

    frame_feature_df = _compute_audio_frame_features(
        waveform=np.asarray(decode_result["waveform"], dtype=np.float32),
        sample_rate=config.audio_sample_rate,
        frame_seconds=config.audio_frame_seconds,
    )
    rows = [
        _aggregate_window_audio_features(
            row=row,
            frame_feature_df=frame_feature_df,
            sample_rate=config.audio_sample_rate,
            frame_seconds=config.audio_frame_seconds,
        )
        for _, row in media_df.iterrows()
    ]
    return pd.DataFrame(rows, columns=AUDIO_FEATURE_COLUMNS)


def _decode_media_audio(media_path: Path, ffmpeg_path: str, sample_rate: int) -> dict[str, object]:
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        return {
            "waveform": None,
            "quality_flag": "ffmpeg_decode_failed",
            "warning": completed.stderr.decode("utf-8", errors="ignore").strip()[:400],
        }
    if not completed.stdout:
        return {
            "waveform": None,
            "quality_flag": "no_audio_samples",
            "warning": "ffmpeg returned zero decoded audio bytes.",
        }
    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if waveform.size == 0:
        return {
            "waveform": None,
            "quality_flag": "no_audio_samples",
            "warning": "Decoded waveform was empty after conversion.",
        }
    return {"waveform": waveform, "quality_flag": "ok", "warning": ""}


def _compute_audio_frame_features(
    waveform: np.ndarray,
    sample_rate: int,
    frame_seconds: float,
) -> pd.DataFrame:
    frame_size = max(1, int(round(sample_rate * frame_seconds)))
    frame_count = int(np.ceil(len(waveform) / frame_size))
    padded = np.pad(waveform, (0, max(0, frame_count * frame_size - len(waveform))), mode="constant")
    frames = padded.reshape(frame_count, frame_size)
    frame_starts = np.arange(frame_count, dtype=float) * frame_seconds
    frame_centers = frame_starts + (frame_seconds / 2.0)

    window = np.hanning(frame_size).astype(np.float32)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    epsilon = 1e-12

    rms_values = np.sqrt(np.mean(frames * frames, axis=1))
    energy_values = np.mean(frames * frames, axis=1)
    zcr_values = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)

    magnitude = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    magnitude_sum = magnitude.sum(axis=1) + epsilon
    centroid = (magnitude * frequencies).sum(axis=1) / magnitude_sum
    bandwidth = np.sqrt(((frequencies[None, :] - centroid[:, None]) ** 2 * magnitude).sum(axis=1) / magnitude_sum)
    cumulative_mag = np.cumsum(magnitude, axis=1)
    rolloff_threshold = 0.85 * magnitude_sum
    rolloff_indices = (cumulative_mag >= rolloff_threshold[:, None]).argmax(axis=1)
    flatness = np.exp(np.mean(np.log(magnitude + epsilon), axis=1)) / (np.mean(magnitude + epsilon, axis=1))

    return pd.DataFrame(
        {
            "frame_start_sec": frame_starts,
            "frame_center_sec": frame_centers,
            "audio_rms": rms_values,
            "audio_short_time_energy": energy_values,
            "audio_zero_crossing_rate": zcr_values,
            "audio_spectral_centroid": centroid,
            "audio_spectral_bandwidth": bandwidth,
            "audio_spectral_rolloff": frequencies[rolloff_indices],
            "audio_spectral_flatness": flatness,
        }
    )


def _aggregate_window_audio_features(
    row: pd.Series,
    frame_feature_df: pd.DataFrame,
    sample_rate: int,
    frame_seconds: float,
) -> dict:
    start_offset = float(pd.to_numeric(row.get("video_start_offset_sec"), errors="coerce") or 0.0)
    duration_seconds = float(pd.to_numeric(row.get("duration_seconds"), errors="coerce") or 0.0)
    end_offset = start_offset + duration_seconds
    mask = (
        (frame_feature_df["frame_center_sec"] >= max(0.0, start_offset))
        & (frame_feature_df["frame_center_sec"] < end_offset)
    )
    subset_df = frame_feature_df.loc[mask].copy()
    if subset_df.empty:
        return _unavailable_audio_row(
            row,
            quality_flag="insufficient_audio_frames",
            sample_rate=sample_rate,
            warning="No audio frames overlapped the requested window.",
        )

    return {
        "window_id": str(row.get("window_id", "")),
        "media_id": str(row.get("media_id", "")),
        "room_id": str(row.get("room_id", "")),
        "session_id": str(row.get("session_id", "")),
        "start_time": str(row.get("start_time", "")),
        "end_time": str(row.get("end_time", "")),
        "audio_rms": float(subset_df["audio_rms"].mean()),
        "audio_short_time_energy": float(subset_df["audio_short_time_energy"].mean()),
        "audio_zero_crossing_rate": float(subset_df["audio_zero_crossing_rate"].mean()),
        "audio_spectral_centroid": float(subset_df["audio_spectral_centroid"].mean()),
        "audio_spectral_bandwidth": float(subset_df["audio_spectral_bandwidth"].mean()),
        "audio_spectral_rolloff": float(subset_df["audio_spectral_rolloff"].mean()),
        "audio_spectral_flatness": float(subset_df["audio_spectral_flatness"].mean()),
        "audio_duration_sec": duration_seconds,
        "audio_sample_rate": sample_rate,
        "audio_available": True,
        "audio_quality_flag": "ok",
        "audio_warning": "",
    }


def _unavailable_audio_rows(
    media_df: pd.DataFrame,
    quality_flag: str,
    sample_rate: int,
    warning: str = "",
) -> pd.DataFrame:
    rows = [
        _unavailable_audio_row(row, quality_flag=quality_flag, sample_rate=sample_rate, warning=warning)
        for _, row in media_df.iterrows()
    ]
    return pd.DataFrame(rows, columns=AUDIO_FEATURE_COLUMNS)


def _unavailable_audio_row(
    row: pd.Series,
    quality_flag: str,
    sample_rate: int,
    warning: str = "",
) -> dict:
    duration_seconds = float(pd.to_numeric(row.get("duration_seconds"), errors="coerce") or 0.0)
    return {
        "window_id": str(row.get("window_id", "")),
        "media_id": str(row.get("media_id", "")),
        "room_id": str(row.get("room_id", "")),
        "session_id": str(row.get("session_id", "")),
        "start_time": str(row.get("start_time", "")),
        "end_time": str(row.get("end_time", "")),
        "audio_rms": np.nan,
        "audio_short_time_energy": np.nan,
        "audio_zero_crossing_rate": np.nan,
        "audio_spectral_centroid": np.nan,
        "audio_spectral_bandwidth": np.nan,
        "audio_spectral_rolloff": np.nan,
        "audio_spectral_flatness": np.nan,
        "audio_duration_sec": duration_seconds,
        "audio_sample_rate": sample_rate,
        "audio_available": False,
        "audio_quality_flag": quality_flag,
        "audio_warning": warning,
    }


def _sort_audio_df(audio_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    ordered = prepare_time_columns(audio_df).sort_values(
        ["room_id", "session_id", "start_time_dt", "window_id"],
        kind="stable",
    )
    ordered = ordered.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore")
    return ordered.reset_index(drop=True)


def _build_audio_report(
    source_windows_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    failed_window_ids: list[str],
    reused_precomputed_count: int,
    max_failures: int,
) -> str:
    summary_columns = [
        "audio_rms",
        "audio_short_time_energy",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_spectral_rolloff",
    ]
    available_df = audio_df[audio_df["audio_available"].fillna(False).astype(bool)].copy() if not audio_df.empty else pd.DataFrame()
    stats_df = available_df[summary_columns].describe().transpose() if not available_df.empty else pd.DataFrame()
    lines = [
        "# Audio Extraction Report",
        "",
        "- Audio comes from embedded MP4 audio only. No external WAV files were used.",
        f"- Number of windows attempted: {len(source_windows_df)}",
        f"- Number with audio available: {int(audio_df['audio_available'].fillna(False).astype(bool).sum()) if not audio_df.empty else 0}",
        f"- Number failed or unavailable: {len(failed_window_ids)}",
        f"- Reused precomputed embedded-audio rows from earlier pipeline: {reused_precomputed_count}",
        "",
        "## Feature Summary Statistics",
        "",
        "```text",
        stats_df.to_string() if not stats_df.empty else "No successful audio rows were available.",
        "```",
        "",
        "## Failed Window IDs",
        "",
    ]
    if failed_window_ids:
        for window_id in failed_window_ids[:max_failures]:
            lines.append(f"- `{window_id}`")
        if len(failed_window_ids) > max_failures:
            lines.append(f"- plus {len(failed_window_ids) - max_failures} more failures")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Quality Caveat",
            "",
            "- Embedded room audio is whole-room audio and should not be interpreted as isolated chicken vocalization.",
        ]
    )
    return "\n".join(lines) + "\n"
