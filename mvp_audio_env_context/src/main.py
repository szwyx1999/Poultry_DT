from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .audio_features import extract_audio_window_features
from .config import ensure_output_dirs, load_config
from .correlation_analysis import run_audio_environment_correlation
from .env_loader import load_environment_daily
from .event_multimodal_validation import run_event_multimodal_validation
from .merge_features import build_multimodal_window_table
from .multimodal_hmm_ablation import run_multimodal_hmm_ablation


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend the poultry digital twin analytics backbone with embedded audio and Room 1 environment context."
    )
    parser.add_argument(
        "--config",
        default="mvp_audio_env_context/config/default.yaml",
        help="Path to the multimodal analytics YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_pipeline(config_path=args.config)
    except Exception as exc:  # pragma: no cover - CLI guard
        LOGGER.error(str(exc))
        return 1
    return 0


def run_pipeline(config_path: str | Path) -> dict[str, str]:
    config = load_config(config_path)
    ensure_output_dirs(config)

    LOGGER.info("Loading daily Room 1 environment context")
    env_result = load_environment_daily(config)

    LOGGER.info("Extracting embedded MP4 audio features")
    audio_result = extract_audio_window_features(config)

    LOGGER.info("Merging video, audio, environment, and event context")
    merge_result = build_multimodal_window_table(config, env_result.env_df, audio_result.audio_df)

    LOGGER.info("Running audio-environment correlation analysis")
    correlation_result = run_audio_environment_correlation(config, merge_result.multimodal_df, env_result.env_df)

    LOGGER.info("Running event-centered multimodal validation")
    event_result = run_event_multimodal_validation(config, merge_result.multimodal_df)

    ablation_result = None
    if config.enable_hmm_ablation:
        LOGGER.info("Running optional multimodal HMM ablation")
        ablation_result = run_multimodal_hmm_ablation(config, merge_result.multimodal_df)

    progress_summary_path = config.reports_dir / "progress_summary_for_meeting.md"
    progress_summary_path.write_text(
        _build_progress_summary(
            merge_result=merge_result,
            correlation_result=correlation_result,
            event_result=event_result,
            ablation_result=ablation_result,
        ),
        encoding="utf-8",
    )

    return {
        "env_room1_daily": str(env_result.output_path),
        "audio_window_features": str(audio_result.output_path),
        "multimodal_window_table": str(merge_result.output_path),
        "audio_env_correlation_table": str(correlation_result.output_path),
        "event_multimodal_summary": str(event_result.output_path),
        "progress_summary_for_meeting": str(progress_summary_path),
        "hmm_ablation_metrics": str(ablation_result.metrics_path) if ablation_result else "",
    }


def _build_progress_summary(merge_result, correlation_result, event_result, ablation_result) -> str:
    lines = [
        "# Progress Summary For Meeting",
        "",
        "## 1. Existing Video-Only Pipeline Achievements",
        "",
        "- Preprocessing indexes MP4 video windows with embedded audio metadata retained.",
        "- MVP1 converts indexed MP4 windows into zone-level activity features.",
        "- The biomarker/HMM layer already detects an interpretable response around the labelled caretaker-entry event.",
        "",
        "## 2. New Audio + Environment Extension",
        "",
        "- Embedded MP4 audio was added as a new per-window modality.",
        "- Daily Room 1 temperature and relative humidity were added as coarse contextual covariates.",
        f"- Multimodal merge coverage: {merge_result.audio_matched_windows} windows with audio features and {merge_result.environment_matched_windows} windows with environment context.",
        "",
        "## 3. What The Multimodal Analysis Shows",
        "",
        correlation_result.meeting_bullet,
        event_result.meeting_bullet,
    ]

    if ablation_result is not None:
        lines.extend(
            [
                "",
                "## 4. Multimodal HMM Ablation",
                "",
                ablation_result.meeting_bullet,
            ]
        )

    lines.extend(
        [
            "",
            "## 5. Why This Helps While Unity Is Delayed",
            "",
            "- The analytics backbone continues to mature without requiring Unity runtime validation.",
            "- We now have a stronger multimodal story for internal review: video response, embedded audio response, and environmental context.",
            "",
            "## 6. Recommended Next Step",
            "",
            "- Use this multimodal layer to compare more labelled management events or developmental periods before treating environment or audio as welfare-proxy signals.",
            "- If more event labels become available, promote the ablation analysis into a more formal model-selection benchmark.",
            "",
            "All outputs remain exploratory and are not validated welfare diagnosis.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
