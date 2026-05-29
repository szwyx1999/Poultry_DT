from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .config import AudioEnvContextConfig


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


def extract_audio_window_features(config: AudioEnvContextConfig) -> AudioExtractionResult:
    source_windows_df = prepare_audio_source_windows(config)
    output_path = config.features_dir / "audio_window_features.csv"
    report_path = config.reports_dir / "audio_extraction_report.md"

    cached_df = _load_cached_audio_features(output_path, source_windows_df, config)
    completed_window_ids = set(cached_df["window_id"].astype(str).tolist()) if not cached_df.empty else set()
    pending_df = source_windows_df[
        ~source_windows_df["window_id"].astype(str).isin(completed_window_ids)
    ].copy()

    chunks: list[pd.DataFrame] = [cached_df] if not cached_df.empty else []
    failed_window_ids: list[str] = []

    if not pending_df.empty:
        grouped = list(pending_df.groupby("media_id", sort=False, dropna=False))
        for media_index, (_, media_df) in enumerate(grouped, start=1):
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
    )


def prepare_audio_source_windows(config: AudioEnvContextConfig) -> pd.DataFrame:
    biomarker_df = _read_csv(config.biomarker_window_table_csv)
    selected_df = _read_csv(config.selected_windows_csv)
    window_index_df = _read_csv(config.video_window_index_csv)

    base_df = selected_df if not selected_df.empty else window_index_df
    if base_df.empty:
        raise ValueError("No selected or indexed windows were available for audio extraction.")

    if not biomarker_df.empty and "window_id" in biomarker_df.columns:
        allowed_window_ids = set(biomarker_df["window_id"].astype(str).tolist())
        base_df = base_df[base_df["window_id"].astype(str).isin(allowed_window_ids)].copy()

    required_columns = [
        "window_id",
        "media_id",
        "room_id",
        "session_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "video_path",
        "video_start_offset_sec",
    ]
    missing_columns = [column for column in required_columns if column not in base_df.columns]
    if missing_columns:
        raise ValueError(
            "Audio window source is missing required columns: " + ", ".join(missing_columns)
        )

    working_df = base_df.copy()
    if "has_audio" not in working_df.columns:
        working_df["has_audio"] = True
    if "resolved_video_path" not in working_df.columns:
        working_df["resolved_video_path"] = working_df["video_path"].astype(str).apply(
            lambda value: str(_resolve_video_path(config, value))
        )

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


def _extract_media_audio_window_features(media_df: pd.DataFrame, config: AudioEnvContextConfig) -> pd.DataFrame:
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
            decode_result["quality_flag"],
            config.audio_sample_rate,
            warning=decode_result["warning"],
        )

    frame_feature_df = _compute_audio_frame_features(
        waveform=decode_result["waveform"],
        sample_rate=config.audio_sample_rate,
        frame_seconds=config.audio_frame_seconds,
    )

    rows: list[dict] = []
    for _, row in media_df.iterrows():
        window_features = _aggregate_window_audio_features(
            row=row,
            frame_feature_df=frame_feature_df,
            sample_rate=config.audio_sample_rate,
            frame_seconds=config.audio_frame_seconds,
        )
        rows.append(window_features)
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

    rms_values = np.empty(frame_count, dtype=np.float32)
    energy_values = np.empty(frame_count, dtype=np.float32)
    zcr_values = np.empty(frame_count, dtype=np.float32)
    centroid_values = np.empty(frame_count, dtype=np.float32)
    bandwidth_values = np.empty(frame_count, dtype=np.float32)
    rolloff_values = np.empty(frame_count, dtype=np.float32)
    flatness_values = np.empty(frame_count, dtype=np.float32)

    batch_size = 64
    epsilon = 1e-12
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(frame_count, batch_start + batch_size)
        batch = frames[batch_start:batch_end].astype(np.float32)
        batch_windowed = batch * window
        magnitude = np.abs(np.fft.rfft(batch_windowed, axis=1)).astype(np.float32)
        magnitude_sum = magnitude.sum(axis=1) + epsilon
        power = magnitude * magnitude

        rms_values[batch_start:batch_end] = np.sqrt(np.mean(batch * batch, axis=1))
        energy_values[batch_start:batch_end] = np.mean(batch * batch, axis=1)
        zcr_values[batch_start:batch_end] = np.mean(
            np.abs(np.diff(np.signbit(batch), axis=1)),
            axis=1,
        )
        centroid_batch = (magnitude * frequencies).sum(axis=1) / magnitude_sum
        centroid_values[batch_start:batch_end] = centroid_batch
        bandwidth_values[batch_start:batch_end] = np.sqrt(
            ((frequencies[None, :] - centroid_batch[:, None]) ** 2 * magnitude).sum(axis=1) / magnitude_sum
        )
        cumulative_mag = np.cumsum(magnitude, axis=1)
        rolloff_threshold = 0.85 * magnitude_sum
        rolloff_indices = (cumulative_mag >= rolloff_threshold[:, None]).argmax(axis=1)
        rolloff_values[batch_start:batch_end] = frequencies[rolloff_indices]
        flatness_values[batch_start:batch_end] = np.exp(np.mean(np.log(magnitude + epsilon), axis=1)) / (
            np.mean(magnitude + epsilon, axis=1)
        )

    return pd.DataFrame(
        {
            "frame_start_sec": frame_starts,
            "frame_center_sec": frame_centers,
            "audio_rms": rms_values,
            "audio_short_time_energy": energy_values,
            "audio_zero_crossing_rate": zcr_values,
            "audio_spectral_centroid": centroid_values,
            "audio_spectral_bandwidth": bandwidth_values,
            "audio_spectral_rolloff": rolloff_values,
            "audio_spectral_flatness": flatness_values,
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
        return _unavailable_audio_row(row, "insufficient_audio_frames", sample_rate, warning="No audio frames overlapped the requested window.")

    coverage_seconds = float(len(subset_df) * frame_seconds)
    quality_flag = "ok"
    if coverage_seconds + 1e-6 < max(0.0, duration_seconds - frame_seconds):
        quality_flag = "partial_audio_window"

    return {
        "window_id": row.get("window_id", ""),
        "media_id": row.get("media_id", ""),
        "room_id": row.get("room_id", ""),
        "session_id": row.get("session_id", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "audio_rms": float(subset_df["audio_rms"].mean()),
        "audio_short_time_energy": float(subset_df["audio_short_time_energy"].mean()),
        "audio_zero_crossing_rate": float(subset_df["audio_zero_crossing_rate"].mean()),
        "audio_spectral_centroid": float(subset_df["audio_spectral_centroid"].mean()),
        "audio_spectral_bandwidth": float(subset_df["audio_spectral_bandwidth"].mean()),
        "audio_spectral_rolloff": float(subset_df["audio_spectral_rolloff"].mean()),
        "audio_spectral_flatness": float(subset_df["audio_spectral_flatness"].mean()),
        "audio_duration_sec": coverage_seconds,
        "audio_sample_rate": sample_rate,
        "audio_available": True,
        "audio_quality_flag": quality_flag,
        "audio_warning": "",
    }


def _unavailable_audio_rows(
    media_df: pd.DataFrame,
    quality_flag: str,
    sample_rate: int,
    warning: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _unavailable_audio_row(row, quality_flag, sample_rate, warning)
            for _, row in media_df.iterrows()
        ],
        columns=AUDIO_FEATURE_COLUMNS,
    )


def _unavailable_audio_row(
    row: pd.Series,
    quality_flag: str,
    sample_rate: int,
    warning: str = "",
) -> dict:
    return {
        "window_id": row.get("window_id", ""),
        "media_id": row.get("media_id", ""),
        "room_id": row.get("room_id", ""),
        "session_id": row.get("session_id", ""),
        "start_time": row.get("start_time", ""),
        "end_time": row.get("end_time", ""),
        "audio_rms": np.nan,
        "audio_short_time_energy": np.nan,
        "audio_zero_crossing_rate": np.nan,
        "audio_spectral_centroid": np.nan,
        "audio_spectral_bandwidth": np.nan,
        "audio_spectral_rolloff": np.nan,
        "audio_spectral_flatness": np.nan,
        "audio_duration_sec": 0.0,
        "audio_sample_rate": sample_rate,
        "audio_available": False,
        "audio_quality_flag": quality_flag,
        "audio_warning": warning,
    }


def _load_cached_audio_features(
    output_path: Path,
    source_windows_df: pd.DataFrame,
    config: AudioEnvContextConfig,
) -> pd.DataFrame:
    if config.audio_force_recompute or not config.audio_cache_enabled or not output_path.exists():
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    cached_df = pd.read_csv(output_path)
    if cached_df.empty or "window_id" not in cached_df.columns:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    allowed_ids = set(source_windows_df["window_id"].astype(str).tolist())
    return cached_df[cached_df["window_id"].astype(str).isin(allowed_ids)].copy()


def _sort_audio_df(audio_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=AUDIO_FEATURE_COLUMNS)
    sort_columns = [column for column in ("start_time", "window_id") if column in audio_df.columns]
    return audio_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _build_audio_report(
    source_windows_df: pd.DataFrame,
    audio_df: pd.DataFrame,
    failed_window_ids: list[str],
    max_failures: int,
) -> str:
    attempted = int(len(source_windows_df))
    available = int(audio_df["audio_available"].astype(bool).sum()) if not audio_df.empty else 0
    failed = attempted - available
    feature_columns = [
        "audio_rms",
        "audio_short_time_energy",
        "audio_zero_crossing_rate",
        "audio_spectral_centroid",
        "audio_spectral_bandwidth",
        "audio_spectral_rolloff",
        "audio_spectral_flatness",
    ]
    summary_df = audio_df.loc[audio_df["audio_available"].astype(bool), feature_columns].describe().transpose() if available else pd.DataFrame()
    lines = [
        "# Audio Extraction Report",
        "",
        "- Audio source: embedded MP4 audio only",
        f"- Number of windows attempted: {attempted}",
        f"- Number with audio features: {available}",
        f"- Number failed or unavailable: {failed}",
        "",
        "## Feature Summary",
        "",
        "```text",
        summary_df.to_string() if not summary_df.empty else "No valid embedded-audio feature rows were available.",
        "```",
        "",
        "## Failed Window IDs",
        "",
    ]
    if failed_window_ids:
        for window_id in failed_window_ids[:max_failures]:
            lines.append(f"- `{window_id}`")
        if len(failed_window_ids) > max_failures:
            lines.append(f"- ... plus {len(failed_window_ids) - max_failures} additional window(s)")
    else:
        lines.append("- none")

    warning_counts = audio_df["audio_quality_flag"].fillna("missing_flag").value_counts().to_dict() if not audio_df.empty else {}
    lines.extend(["", "## Audio Quality Flags", ""])
    if warning_counts:
        for flag, count in warning_counts.items():
            lines.append(f"- `{flag}`: {int(count)}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _read_csv(path_value: Path) -> pd.DataFrame:
    if not path_value.exists():
        return pd.DataFrame()
    return pd.read_csv(path_value)


def _resolve_video_path(config: AudioEnvContextConfig, value: object) -> Path:
    path_text = str(value or "").strip()
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return config.workspace_root / "first_week_data_cleaning" / candidate
