from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.video_feature_extractor import (
    _aggregate_media_windows,
    _load_or_compute_media_cache,
    build_semantic_biomarker_table,
)


WORKSPACE_ROOT = PROJECT_ROOT.parent
HANDOFF_DIR = WORKSPACE_ROOT / "data" / "processed" / "data_preparation_outputs" / "handoff_for_mvp"
OUTPUT_ROOT = PROJECT_ROOT / "validate" / "outputs" / "room1_16_17_video_evidence"

SESSION_ID = "room_1_16_17_aug"
ROOM_ID = "room_1"
DAYLIGHT_START_HOUR = 7
DAYLIGHT_END_HOUR = 17
MIN_GAP_MINUTES = 20
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
FORCE_RECOMPUTE_VALIDATION_FEATURES = False
SELECTION_OVERRIDES = {
    "drinking_peak": {
        "window_id": "media_000145_w0019",
        "explanation": "High drinking fraction with multiple birds visibly clustered at the drinker and no caretaker entering the frame.",
    },
    "general_day_peak": {
        "window_id": "media_000165_w0224",
        "explanation": "Among readable daytime windows, this clip shows the strongest general-zone share while keeping the scene bright enough for visual inspection.",
    },
    "transition_peak": {
        "window_id": "media_000142_w0116",
        "explanation": "High daylight transition score with birds visibly split across the drinker, feeder, and open floor.",
    },
    "mobility_peak": {
        "window_id": "media_000164_w0028",
        "explanation": "Near-maximum mobility with strong whole-pen movement and no caretaker intruding into the frame.",
    },
}


@dataclass(frozen=True)
class Strategy:
    key: str
    title: str
    metric: str
    ascending: bool
    explanation: str
    daylight_only: bool = True
    min_activity_mean: float | None = None
    min_metric_value: float | None = None


STRATEGIES = [
    Strategy(
        key="feeding_peak",
        title="Feeding-Dominant Window",
        metric="feeding_activity_fraction",
        ascending=False,
        explanation="Feeding fraction is highest here, so hens should be visibly concentrated around the feeder.",
    ),
    Strategy(
        key="drinking_peak",
        title="Drinking-Dominant Window",
        metric="drinking_activity_fraction",
        ascending=False,
        explanation="Drinking fraction peaks here, making this the clearest clip for validating drinker-side activity.",
    ),
    Strategy(
        key="general_day_peak",
        title="Daylight General-Zone Leader",
        metric="general_activity_fraction",
        ascending=False,
        min_activity_mean=0.00014,
        explanation="Among bright daytime windows, activity in this clip leans most strongly toward the general zone.",
    ),
    Strategy(
        key="transition_peak",
        title="High Cross-Zone Switching",
        metric="semantic_transition_proxy",
        ascending=False,
        min_activity_mean=0.005,
        min_metric_value=0.12,
        explanation="This window has the strongest cross-zone switching signal, so movement between the drinker, feeder, and open floor should be easy to see.",
    ),
    Strategy(
        key="mobility_peak",
        title="High Overall Mobility",
        metric="mobility_index",
        ascending=False,
        min_activity_mean=0.01,
        explanation="Mobility is near the session maximum here, so this clip should show the most obvious whole-pen movement.",
    ),
]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    zone_config = load_zone_config()
    window_df = pd.read_csv(HANDOFF_DIR / "video_window_index.csv", low_memory=False)
    window_df = window_df[(window_df["session_id"] == SESSION_ID) & (window_df["room_id"] == ROOM_ID)].copy()
    window_df = window_df.sort_values(["media_id", "video_start_offset_sec"], kind="stable").reset_index(drop=True)

    zone_rows: list[dict] = []
    for media_id, media_windows in window_df.groupby("media_id", sort=False):
        media_windows = media_windows.reset_index(drop=True)
        source_video_path = resolve_video_path(str(media_windows.iloc[0]["video_path"]))
        cache_data = _load_or_compute_media_cache(
            config=config,
            zone_config=zone_config,
            absolute_path=source_video_path,
            media_windows=media_windows,
            force_recompute=FORCE_RECOMPUTE_VALIDATION_FEATURES,
        )
        media_rows, _ = _aggregate_media_windows(media_windows, zone_config, cache_data)
        zone_rows.extend(media_rows)

    zone_df = pd.DataFrame(zone_rows)
    biomarker_df = build_semantic_biomarker_table(zone_df)
    biomarker_df = biomarker_df.merge(
        window_df[["window_id", "video_path", "video_start_offset_sec", "duration_seconds"]],
        on="window_id",
        how="left",
        validate="one_to_one",
    )
    biomarker_df["start_dt"] = pd.to_datetime(biomarker_df["start_time"])
    biomarker_df["end_dt"] = pd.to_datetime(biomarker_df["end_time"])
    return biomarker_df, zone_df


def load_zone_config() -> dict:
    configs = json.loads((HANDOFF_DIR / "semantic_zone_configs.json").read_text(encoding="utf-8"))
    for config in configs:
        if config["room_id"] == ROOM_ID:
            return config
    raise ValueError(f"Could not find zone config for {ROOM_ID}.")


def resolve_video_path(video_path: str) -> Path:
    candidate = Path(video_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = WORKSPACE_ROOT / candidate
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Video path does not exist: {video_path}")


def select_candidates(df: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.Series] = []

    def strategy_filter(frame: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
        candidate = frame.copy()
        if strategy.daylight_only:
            hours = candidate["start_dt"].dt.hour
            candidate = candidate[(hours >= DAYLIGHT_START_HOUR) & (hours < DAYLIGHT_END_HOUR)]
        if strategy.min_activity_mean is not None:
            candidate = candidate[candidate["activity_mean"] >= strategy.min_activity_mean]
        if strategy.min_metric_value is not None:
            candidate = candidate[candidate[strategy.metric] >= strategy.min_metric_value]
        for selected in selections:
            gap = (candidate["start_dt"] - selected["start_dt"]).abs()
            candidate = candidate[gap >= pd.Timedelta(minutes=MIN_GAP_MINUTES)]
        return candidate

    for strategy in STRATEGIES:
        candidate = strategy_filter(df, strategy)
        if candidate.empty:
            raise ValueError(f"No candidate matched strategy: {strategy.key}")
        override = SELECTION_OVERRIDES.get(strategy.key)
        if override is not None:
            override_window_id = str(override["window_id"])
            override_match = candidate[candidate["window_id"] == override_window_id]
            if override_match.empty:
                raise ValueError(f"Selection override window not found for {strategy.key}: {override_window_id}")
            chosen = override_match.iloc[0].copy()
        else:
            chosen = candidate.sort_values(strategy.metric, ascending=strategy.ascending).iloc[0].copy()
        chosen["selection_key"] = strategy.key
        chosen["selection_title"] = strategy.title
        chosen["selection_metric"] = strategy.metric
        chosen["selection_explanation"] = str(override["explanation"]) if override is not None else strategy.explanation
        chosen["selection_rank_pct"] = df[strategy.metric].rank(method="min", pct=True)[chosen.name]
        selections.append(chosen)

    selected_df = pd.DataFrame(selections).reset_index(drop=True)
    return selected_df


def ensure_output_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_ROOT,
        "raw": OUTPUT_ROOT / "clips" / "raw",
        "annotated": OUTPUT_ROOT / "clips" / "annotated",
        "preview": OUTPUT_ROOT / "previews",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{stderr_tail}")


def extract_raw_clip(video_path: Path, start_offset_sec: float, duration_sec: float, output_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_offset_sec:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_sec:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )


def brighten_frame(frame: np.ndarray) -> np.ndarray:
    mean_brightness = float(frame.mean())
    if mean_brightness < 55:
        return cv2.convertScaleAbs(frame, alpha=1.55, beta=28)
    if mean_brightness < 75:
        return cv2.convertScaleAbs(frame, alpha=1.30, beta=18)
    return cv2.convertScaleAbs(frame, alpha=1.10, beta=8)


def rectangle_from_polygon(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return x, y, w, h


def draw_zones(frame: np.ndarray, zone_config: dict) -> np.ndarray:
    overlay = frame.copy()
    color_map = {
        "drinking": (0, 165, 255),
        "feeding": (0, 220, 0),
        "general": (255, 180, 0),
    }
    for zone in zone_config["zones"]:
        semantic_type = zone["semantic_type"]
        color = color_map[semantic_type]
        if "polygon" in zone:
            x, y, w, h = rectangle_from_polygon(zone["polygon"])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
            cv2.putText(
                frame,
                zone["display_name"],
                (x + 8, max(32, y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )
    frame = cv2.addWeighted(overlay, 0.10, frame, 0.90, 0.0)
    return frame


def metrics_line(row: pd.Series) -> str:
    return (
        f"D {row['drinking_activity_fraction']:.2f} | "
        f"F {row['feeding_activity_fraction']:.2f} | "
        f"G {row['general_activity_fraction']:.2f} | "
        f"M {row['mobility_index']:.2f} | "
        f"T {row['semantic_transition_proxy']:.2f}"
    )


def add_caption(frame: np.ndarray, row: pd.Series, clip_name: str) -> np.ndarray:
    box_top = 24
    box_left = 24
    box_width = 1320
    box_height = 148
    cv2.rectangle(frame, (box_left, box_top), (box_left + box_width, box_top + box_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (box_left, box_top), (box_left + box_width, box_top + box_height), (255, 255, 255), 2)
    lines = [
        f"{clip_name} | {row['selection_title']}",
        f"{row['start_time']} to {row['end_time']} | metric {row['selection_metric']} = {row[row['selection_metric']]:.3f} | percentile {row['selection_rank_pct'] * 100:.1f}",
        metrics_line(row),
    ]
    y = box_top + 42
    for line in lines:
        cv2.putText(frame, line, (box_left + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
        y += 40
    return frame


def annotate_clip(raw_clip_path: Path, preview_path: Path, output_path: Path, row: pd.Series, zone_config: dict, clip_name: str) -> None:
    cap = cv2.VideoCapture(str(raw_clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for annotation: {raw_clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or TARGET_WIDTH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or TARGET_HEIGHT)
    temp_video_path = output_path.with_name(output_path.stem + "__silent.mp4")
    writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for {temp_video_path}")

    preview_frame_index = max(0, frame_count // 2)
    frame_idx = 0
    preview_written = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = brighten_frame(frame)
        frame = draw_zones(frame, zone_config)
        frame = add_caption(frame, row, clip_name)
        writer.write(frame)
        if frame_idx == preview_frame_index:
            cv2.imwrite(str(preview_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            preview_written = True
        frame_idx += 1

    cap.release()
    writer.release()

    if not preview_written:
        raise RuntimeError(f"Could not write preview for {raw_clip_path}")

    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video_path),
            "-i",
            str(raw_clip_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    temp_video_path.unlink(missing_ok=True)


def make_contact_sheet(preview_paths: list[Path], output_path: Path) -> None:
    tiles: list[np.ndarray] = []
    for path in preview_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (640, 360))
        cv2.putText(image, path.stem, (18, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(image)
    if not tiles:
        raise RuntimeError("No preview images available for contact sheet.")
    if len(tiles) % 2 == 1:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    contact_sheet = np.vstack(rows)
    cv2.imwrite(str(output_path), contact_sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def clip_slug(index: int, row: pd.Series) -> str:
    start_stamp = str(row["start_time"]).replace(":", "-")
    return f"{index:02d}_{row['selection_key']}__{start_stamp}__{row['window_id']}"


def build_summary(manifest_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Room 1 (16, 17 Aug) Video Evidence",
        "",
        "These clips were re-selected for `room_1_16_17_aug` after recomputing semantic-zone biomarkers from the raw videos with the current source code.",
        "The handoff window index was used only for time-window definitions and source-video mapping.",
        "Annotated clips are slightly brightened for readability; matching raw clips are also included unchanged.",
        "",
        "## Selected windows",
        "",
    ]
    for _, row in manifest_df.iterrows():
        lines.extend(
            [
                f"### {row['clip_name']} - {row['selection_title']}",
                f"- Window: `{row['window_id']}`",
                f"- Time: `{row['start_time']}` to `{row['end_time']}`",
                f"- Source video: `{row['source_video_path']}`",
                f"- Why it was chosen: {row['selection_explanation']}",
                f"- Primary metric: `{row['selection_metric']}` = `{row[row['selection_metric']]:.3f}` ({row['selection_rank_pct'] * 100:.1f} percentile within `room_1_16_17_aug`)",
                f"- Supporting metrics: `drinking={row['drinking_activity_fraction']:.3f}`, `feeding={row['feeding_activity_fraction']:.3f}`, `general={row['general_activity_fraction']:.3f}`, `mobility={row['mobility_index']:.3f}`, `transition={row['semantic_transition_proxy']:.3f}`, `occupancy_imbalance={row['occupancy_imbalance_index']:.3f}`, `spatial_freedom={row['spatial_freedom_index']:.3f}`",
                f"- Raw clip: `{row['raw_clip_path']}`",
                f"- Annotated clip: `{row['annotated_clip_path']}`",
                f"- Preview: `{row['preview_path']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dirs = ensure_output_dirs()
    df, zone_df = load_tables()
    zone_config = load_zone_config()
    selected_df = select_candidates(df)
    manifest_rows: list[dict] = []
    preview_paths: list[Path] = []

    for index, row in selected_df.iterrows():
        source_video_path = resolve_video_path(str(row["video_path"]))
        clip_name = clip_slug(index + 1, row)
        raw_clip_path = dirs["raw"] / f"{clip_name}.mp4"
        annotated_clip_path = dirs["annotated"] / f"{clip_name}.mp4"
        preview_path = dirs["preview"] / f"{clip_name}.jpg"

        print(f"[{index + 1}/{len(selected_df)}] Exporting {clip_name}")
        extract_raw_clip(
            video_path=source_video_path,
            start_offset_sec=float(row["video_start_offset_sec"]),
            duration_sec=float(row["duration_seconds"]),
            output_path=raw_clip_path,
        )
        annotate_clip(
            raw_clip_path=raw_clip_path,
            preview_path=preview_path,
            output_path=annotated_clip_path,
            row=row,
            zone_config=zone_config,
            clip_name=clip_name,
        )
        preview_paths.append(preview_path)

        manifest_row = row.to_dict()
        manifest_row["clip_name"] = clip_name
        manifest_row["source_video_path"] = str(source_video_path)
        manifest_row["raw_clip_path"] = str(raw_clip_path)
        manifest_row["annotated_clip_path"] = str(annotated_clip_path)
        manifest_row["preview_path"] = str(preview_path)
        manifest_rows.append(manifest_row)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = dirs["root"] / "evidence_manifest.csv"
    summary_path = dirs["root"] / "evidence_summary.md"
    contact_sheet_path = dirs["root"] / "preview_contact_sheet.jpg"
    recomputed_biomarker_path = dirs["root"] / "recomputed_semantic_biomarker_window_table.csv"
    recomputed_zone_feature_path = dirs["root"] / "recomputed_semantic_zone_video_features.csv"

    manifest_df.to_csv(manifest_path, index=False)
    df.drop(columns=["start_dt", "end_dt"], errors="ignore").to_csv(recomputed_biomarker_path, index=False)
    zone_df.to_csv(recomputed_zone_feature_path, index=False)
    build_summary(manifest_df, summary_path)
    make_contact_sheet(preview_paths, contact_sheet_path)

    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Recomputed biomarker table: {recomputed_biomarker_path}")
    print(f"Recomputed zone feature table: {recomputed_zone_feature_path}")


if __name__ == "__main__":
    main()
