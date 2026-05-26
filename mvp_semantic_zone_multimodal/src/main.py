from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .audio_features import extract_audio_window_features
from .config import ensure_output_dirs, load_config
from .env_loader import load_environment_daily
from .event_validation import (
    build_semantic_event_validation,
    ensure_event_log,
    label_semantic_windows,
)
from .hmm_model import fit_semantic_hmm_models
from .merge_multimodal import merge_semantic_multimodal_table
from .semantic_biomarkers import compute_semantic_biomarkers
from .semantic_zone_builder import build_semantic_zone_config
from .semantic_zone_features import compute_semantic_zone_video_features


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the semantic-zone multimodal Room 1 Aug 16-17 experiment."
    )
    parser.add_argument(
        "--config",
        default="mvp_semantic_zone_multimodal/config/default.yaml",
        help="Path to the semantic-zone experiment YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation and allow continuation past invalid zone warnings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_pipeline(config_path=args.config, force=args.force)
    except Exception as exc:  # pragma: no cover - CLI guard
        LOGGER.error(str(exc))
        return 1
    return 0


def run_pipeline(config_path: str | Path, force: bool = False) -> dict[str, str]:
    config = load_config(config_path, force=force)
    ensure_output_dirs(config)

    LOGGER.info("Building semantic zone definition")
    zone_result = build_semantic_zone_config(config)

    LOGGER.info("Computing semantic-zone video activity features")
    semantic_feature_result = compute_semantic_zone_video_features(config, zone_result)

    LOGGER.info("Computing semantic biomarker table")
    semantic_biomarker_result = compute_semantic_biomarkers(
        config=config,
        semantic_feature_df=semantic_feature_result.feature_df,
        event_log_path=config.event_log_csv if config.event_log_csv.exists() else None,
    )

    LOGGER.info("Extracting embedded MP4 audio features")
    audio_result = extract_audio_window_features(config, semantic_biomarker_result.biomarker_df)

    LOGGER.info("Loading Room 1 daily environment context")
    env_result = load_environment_daily(config)

    LOGGER.info("Ensuring caretaker-entry event log")
    event_result = ensure_event_log(config)

    LOGGER.info("Merging semantic biomarkers, audio, environment, and event context")
    merge_result = merge_semantic_multimodal_table(
        config=config,
        semantic_biomarker_df=semantic_biomarker_result.biomarker_df,
        audio_df=audio_result.audio_df,
        env_df=env_result.env_df,
        event_log_df=event_result.event_log_df,
    )

    LOGGER.info("Applying semantic event labels")
    labelled_result = label_semantic_windows(
        config=config,
        multimodal_df=merge_result.multimodal_df,
        event_log_df=event_result.event_log_df,
    )

    LOGGER.info("Fitting semantic-zone HMM models")
    hmm_result = fit_semantic_hmm_models(
        config=config,
        labelled_df=labelled_result.labelled_df,
    )

    LOGGER.info("Running semantic event validation")
    validation_result = build_semantic_event_validation(
        config=config,
        labelled_df=labelled_result.labelled_df,
        hmm_result=hmm_result,
        previous_fourzone_event_summary_csv=config.previous_fourzone_event_summary_csv,
    )

    summary_path = config.reports_dir / "semantic_zone_progress_summary_for_meeting.md"
    summary_path.write_text(
        _build_progress_summary(
            zone_result=zone_result,
            feature_result=semantic_feature_result,
            biomarker_result=semantic_biomarker_result,
            audio_result=audio_result,
            env_result=env_result,
            labelled_result=labelled_result,
            hmm_result=hmm_result,
            validation_result=validation_result,
        ),
        encoding="utf-8",
    )

    return {
        "semantic_zone_config": str(zone_result.output_json_path),
        "semantic_zone_video_features": str(semantic_feature_result.output_path),
        "semantic_biomarker_window_table": str(semantic_biomarker_result.output_path),
        "audio_window_features": str(audio_result.output_path),
        "env_room1_daily": str(env_result.output_path),
        "semantic_multimodal_window_table": str(merge_result.output_path),
        "semantic_labelled_window_table": str(labelled_result.output_path),
        "semantic_hmm_state_sequence": str(hmm_result.sequence_path),
        "semantic_hmm_state_summary": str(hmm_result.summary_path),
        "semantic_event_validation_summary": str(validation_result.output_path),
        "semantic_zone_progress_summary_for_meeting": str(summary_path),
    }


def _build_progress_summary(
    zone_result,
    feature_result,
    biomarker_result,
    audio_result,
    env_result,
    labelled_result,
    hmm_result,
    validation_result,
) -> str:
    lines = [
        "# Semantic Zone Progress Summary For Meeting",
        "",
        "## 1. Why Semantic Zones Were Added",
        "",
        "- The previous four equal image-space zones were useful for technical validation but were not biologically meaningful.",
        "- This experiment replaces them with manually defined semantic zones: drinking, feeding, and general area.",
        "",
        "## 2. Data Used",
        "",
        "- Two days of Room 1 Aug 16-17 video windows",
        "- Embedded MP4 audio only",
        "- Daily Room 1 environment context from `Combined Room 1.xlsx`",
        "- Manual caretaker-entry event label",
        "",
        "## 3. What The Semantic Zones Mean",
        "",
        "- `drinking_zone`: red-box annotated drinking area",
        "- `feeding_zone`: green-box annotated feeding area",
        "- `general_zone`: full-frame remainder outside the drinking and feeding areas",
        "",
        "## 4. Outputs Produced",
        "",
        f"- Semantic-zone feature rows: {len(feature_result.feature_df)}",
        f"- Semantic biomarker windows: {len(biomarker_result.biomarker_df)}",
        f"- Embedded-audio windows available: {int(audio_result.audio_df['audio_available'].fillna(False).astype(bool).sum()) if not audio_result.audio_df.empty else 0}",
        f"- Event-labelled semantic windows: {int(labelled_result.labelled_df['event_phase'].isin(['pre_entry_baseline','during_entry','post_entry_recovery']).sum()) if not labelled_result.labelled_df.empty else 0}",
        "",
        "## 5. Main Findings",
        "",
        validation_result.meeting_bullet,
        hmm_result.meeting_bullet,
        "",
        "## 6. How This Helps The Next Stage",
        "",
        "- Semantic-zone structure makes the activity response easier to interpret in terms of functional areas.",
        "- This prepares the analytics backbone for later individual-behaviour or occupancy modelling.",
        "",
        "## 7. Limitations",
        "",
        "- Semantic zones are manually defined from one reference view.",
        "- Activity is not the same as true occupancy.",
        "- Audio is whole-room embedded MP4 audio rather than isolated chicken vocalization.",
        "- Environment is daily Room 1 context.",
        "- Risk score remains a prototype heuristic and not a validated welfare diagnosis.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
