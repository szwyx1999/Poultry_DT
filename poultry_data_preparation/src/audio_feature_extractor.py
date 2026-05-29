from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import PreparationConfig
from .utils import prepare_time_columns, write_table


LOGGER = logging.getLogger(__name__)


AUDIO_COLUMNS = [
    "window_id",
    "media_id",
    "room_id",
    "session_id",
    "start_time",
    "end_time",
    "audio_available",
    "audio_duration_sec",
    "audio_sample_rate",
    "audio_rms",
    "audio_short_time_energy",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid",
    "audio_spectral_bandwidth",
    "audio_spectral_rolloff",
    "audio_spectral_flatness",
    "audio_quality_flag",
    "warnings",
]


@dataclass(frozen=True)
class AudioFeatureResult:
    audio_df: pd.DataFrame
    output_path: Path
    report_path: Path
    failed_df: pd.DataFrame


def extract_audio_features(
    config: PreparationConfig,
    media_df: pd.DataFrame,
    window_df: pd.DataFrame,
    dry_run: bool = False,
) -> AudioFeatureResult:
    output_path = config.features_output_dir / "audio_window_features.csv"
    report_path = config.reports_output_dir / "audio_feature_report.md"

    if dry_run or not config.audio_features_enabled:
        audio_df = pd.DataFrame(columns=AUDIO_COLUMNS)
        report_path.write_text(_build_dry_run_report(window_df, config.audio_features_enabled), encoding="utf-8")
        return AudioFeatureResult(audio_df, output_path, report_path, pd.DataFrame())

    if shutil.which(config.ffmpeg_path) is None:
        audio_df = pd.DataFrame(columns=AUDIO_COLUMNS)
        report_path.write_text("# Audio Feature Report\n\nffmpeg was not available, so audio extraction was skipped.\n", encoding="utf-8")
        return AudioFeatureResult(audio_df, output_path, report_path, pd.DataFrame())

    media_lookup = media_df.set_index("media_id", drop=False).to_dict(orient="index")
    all_rows: list[dict] = []
    failed_rows: list[dict] = []
    grouped = list(window_df.groupby("media_id", sort=False, dropna=False))
    processed_windows = 0

    for group_index, (media_id, media_windows) in enumerate(grouped, start=1):
        media_windows = media_windows.sort_values("video_start_offset_sec", kind="stable").reset_index(drop=True)
        LOGGER.info("Audio features media %s/%s: %s (%s windows)", group_index, len(grouped), media_id, len(media_windows))
        media_metadata = media_lookup.get(str(media_id))
        if media_metadata is None:
            for _, row in media_windows.iterrows():
                failed_rows.append(_failed_row(row, "missing_media_manifest_row"))
            continue

        if not bool(media_metadata.get("has_audio", False)):
            media_rows = _unavailable_audio_rows(media_windows, "manifest_has_audio_false", config.audio_sample_rate)
        else:
            cache_data = _load_or_compute_audio_cache(config, Path(str(media_metadata["absolute_path"])), media_metadata["media_id"], force_recompute=config.audio_force_recompute)
            media_rows = _aggregate_audio_windows(media_windows, cache_data, config.audio_sample_rate, config.audio_frame_seconds)
        all_rows.extend(media_rows.to_dict(orient="records"))
        failures = media_rows.loc[~media_rows["audio_available"].fillna(False).astype(bool), ["window_id", "media_id", "room_id"]]
        for _, failure_row in failures.iterrows():
            failed_rows.append({"stage": "audio_features", "window_id": failure_row["window_id"], "media_id": failure_row["media_id"], "room_id": failure_row["room_id"], "issue": "audio_unavailable"})
        partial_df = _sort_audio_df(pd.DataFrame(all_rows, columns=AUDIO_COLUMNS))
        write_table(partial_df, output_path, write_csv=config.write_csv, write_parquet=False)
        processed_windows += len(media_windows)
        LOGGER.info("Processed %s/%s audio windows", processed_windows, len(window_df))

    audio_df = _sort_audio_df(pd.DataFrame(all_rows, columns=AUDIO_COLUMNS))
    write_table(audio_df, output_path, write_csv=config.write_csv, write_parquet=config.write_parquet)
    failed_df = pd.DataFrame(failed_rows)
    report_path.write_text(_build_audio_report(window_df, audio_df, failed_df), encoding="utf-8")
    return AudioFeatureResult(audio_df, output_path, report_path, failed_df)


def _load_or_compute_audio_cache(
    config: PreparationConfig,
    absolute_path: Path,
    media_id: str,
    force_recompute: bool,
) -> dict[str, object]:
    cache_path = config.audio_feature_cache_dir / f"{media_id}.joblib"
    if cache_path.exists() and not force_recompute:
        LOGGER.info("Reusing cached audio features for %s", media_id)
        return joblib.load(cache_path)
    LOGGER.info("Computing audio features for %s", media_id)
    cache_data = _decode_media_audio_to_frames(absolute_path, config.ffmpeg_path, config.audio_sample_rate, config.audio_frame_seconds)
    if config.audio_cache_per_media:
        joblib.dump(cache_data, cache_path)
    return cache_data


def _decode_media_audio_to_frames(
    absolute_path: Path,
    ffmpeg_path: str,
    sample_rate: int,
    frame_seconds: float,
) -> dict[str, object]:
    command = [
        ffmpeg_path,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(absolute_path),
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
        return {"frame_df": pd.DataFrame(), "quality_flag": "ffmpeg_decode_failed", "warning": completed.stderr.decode("utf-8", errors="ignore")[:300]}
    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if waveform.size == 0:
        return {"frame_df": pd.DataFrame(), "quality_flag": "no_audio_samples", "warning": "No audio samples decoded."}
    return {"frame_df": _compute_audio_frame_features(waveform, sample_rate, frame_seconds), "quality_flag": "ok", "warning": ""}


def _compute_audio_frame_features(waveform: np.ndarray, sample_rate: int, frame_seconds: float) -> pd.DataFrame:
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


def _aggregate_audio_windows(
    media_windows: pd.DataFrame,
    cache_data: dict[str, object],
    sample_rate: int,
    frame_seconds: float,
) -> pd.DataFrame:
    frame_df = cache_data.get("frame_df", pd.DataFrame())
    quality_flag = str(cache_data.get("quality_flag", "unknown"))
    warning = str(cache_data.get("warning", ""))
    if frame_df is None or not isinstance(frame_df, pd.DataFrame) or frame_df.empty:
        return _unavailable_audio_rows(media_windows, quality_flag, sample_rate, warning)

    rows: list[dict] = []
    for _, row in media_windows.iterrows():
        start_offset = float(pd.to_numeric(row["video_start_offset_sec"], errors="coerce"))
        duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
        end_offset = start_offset + duration_seconds
        mask = (frame_df["frame_center_sec"] >= start_offset) & (frame_df["frame_center_sec"] < end_offset)
        subset_df = frame_df.loc[mask]
        if subset_df.empty:
            rows.append(_unavailable_audio_row(row, "insufficient_audio_frames", sample_rate, "No audio frames overlapped the window."))
            continue
        rows.append(
            {
                "window_id": str(row["window_id"]),
                "media_id": str(row["media_id"]),
                "room_id": str(row["room_id"]),
                "session_id": str(row["session_id"]),
                "start_time": str(row["start_time"]),
                "end_time": str(row["end_time"]),
                "audio_available": True,
                "audio_duration_sec": duration_seconds,
                "audio_sample_rate": sample_rate,
                "audio_rms": float(subset_df["audio_rms"].mean()),
                "audio_short_time_energy": float(subset_df["audio_short_time_energy"].mean()),
                "audio_zero_crossing_rate": float(subset_df["audio_zero_crossing_rate"].mean()),
                "audio_spectral_centroid": float(subset_df["audio_spectral_centroid"].mean()),
                "audio_spectral_bandwidth": float(subset_df["audio_spectral_bandwidth"].mean()),
                "audio_spectral_rolloff": float(subset_df["audio_spectral_rolloff"].mean()),
                "audio_spectral_flatness": float(subset_df["audio_spectral_flatness"].mean()),
                "audio_quality_flag": "ok",
                "warnings": "",
            }
        )
    return pd.DataFrame(rows, columns=AUDIO_COLUMNS)


def _unavailable_audio_rows(media_windows: pd.DataFrame, quality_flag: str, sample_rate: int, warning: str = "") -> pd.DataFrame:
    rows = [_unavailable_audio_row(row, quality_flag, sample_rate, warning) for _, row in media_windows.iterrows()]
    return pd.DataFrame(rows, columns=AUDIO_COLUMNS)


def _unavailable_audio_row(row: pd.Series, quality_flag: str, sample_rate: int, warning: str = "") -> dict:
    duration_seconds = float(pd.to_numeric(row["duration_seconds"], errors="coerce"))
    return {
        "window_id": str(row["window_id"]),
        "media_id": str(row["media_id"]),
        "room_id": str(row["room_id"]),
        "session_id": str(row["session_id"]),
        "start_time": str(row["start_time"]),
        "end_time": str(row["end_time"]),
        "audio_available": False,
        "audio_duration_sec": duration_seconds,
        "audio_sample_rate": sample_rate,
        "audio_rms": np.nan,
        "audio_short_time_energy": np.nan,
        "audio_zero_crossing_rate": np.nan,
        "audio_spectral_centroid": np.nan,
        "audio_spectral_bandwidth": np.nan,
        "audio_spectral_rolloff": np.nan,
        "audio_spectral_flatness": np.nan,
        "audio_quality_flag": quality_flag,
        "warnings": warning,
    }


def _sort_audio_df(audio_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=AUDIO_COLUMNS)
    ordered = prepare_time_columns(audio_df).sort_values(["room_id", "session_id", "start_time_dt", "window_id"], kind="stable")
    return ordered.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore").reset_index(drop=True)


def _failed_row(row: pd.Series, issue: str) -> dict:
    return {"stage": "audio_features", "window_id": str(row.get("window_id", "")), "media_id": str(row.get("media_id", "")), "room_id": str(row.get("room_id", "")), "issue": issue}


def _build_dry_run_report(window_df: pd.DataFrame, enabled: bool) -> str:
    return "\n".join(
        [
            "# Audio Feature Report",
            "",
            f"- Audio extraction enabled: `{enabled}`",
            "- Dry run mode skipped heavy embedded-audio extraction.",
            f"- Planned windows: {len(window_df)}",
        ]
    ) + "\n"


def _build_audio_report(window_df: pd.DataFrame, audio_df: pd.DataFrame, failed_df: pd.DataFrame) -> str:
    available_df = audio_df[audio_df["audio_available"].fillna(False).astype(bool)].copy() if not audio_df.empty else pd.DataFrame()
    lines = [
        "# Audio Feature Report",
        "",
        "- Audio comes from embedded MP4 audio only. No external WAV files are used.",
        f"- Windows attempted: {len(window_df)}",
        f"- Windows with audio available: {int(audio_df['audio_available'].fillna(False).astype(bool).sum()) if not audio_df.empty else 0}",
        f"- Failed or unavailable windows: {len(failed_df)}",
        "",
        "## Feature Summary",
        "",
        "```text",
        available_df[
            [
                "audio_rms",
                "audio_short_time_energy",
                "audio_zero_crossing_rate",
                "audio_spectral_centroid",
                "audio_spectral_bandwidth",
                "audio_spectral_rolloff",
            ]
        ].describe().to_string() if not available_df.empty else "No successful audio rows available.",
        "```",
    ]
    return "\n".join(lines) + "\n"
