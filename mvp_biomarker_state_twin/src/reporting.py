from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BiomarkerTwinConfig
from .state_model import StateModelResult


def write_risk_score_explanation_report(
    state_model_result: StateModelResult,
    config: BiomarkerTwinConfig,
    output_path: Path,
) -> None:
    summary_df = state_model_result.state_summary_df.copy()
    display_columns = [
        column
        for column in (
            "state_id",
            "state_label",
            "window_count",
            "window_fraction",
            "normalized_activity_mean",
            "mobility_index_mean",
            "spatial_freedom_index_mean",
            "occupancy_imbalance_index_mean",
            "resilience_pressure_index_mean",
            "state_run_length_mean",
            "component_activity_mobility",
            "component_imbalance",
            "component_low_spatial_freedom",
            "component_resilience_pressure",
            "component_persistence",
            "state_risk_weight",
        )
        if column in summary_df.columns
    ]

    lines = [
        "# Risk Score Explanation",
        "",
        "The current welfare risk score is a prototype heuristic derived from the latent-state summary. It is intended for exploratory digital-twin demonstration work only.",
        "",
        "## Warning",
        "",
        "Prototype risk score, not validated welfare diagnosis.",
        "",
        "## Formula",
        "",
        "State-level risk weight is computed as a weighted combination of:",
        "",
        "- high activity plus high mobility",
        "- high occupancy imbalance",
        "- low spatial freedom",
        "- resilience pressure",
        "- persistence of the state across consecutive windows",
        "",
        "For each state:",
        "",
        "`state_risk_weight = (w_activity_mobility * component_activity_mobility)`",
        "` + (w_imbalance * component_imbalance)`",
        "` + (w_low_spatial_freedom * component_low_spatial_freedom)`",
        "` + (w_resilience_pressure * component_resilience_pressure)`",
        "` + (w_persistence * component_persistence)`",
        "",
        "Window-level risk score is then the posterior expectation over the active latent state probabilities.",
        "",
        "## Feature Weights",
        "",
        f"- `w_activity_mobility`: {config.risk_activity_mobility_weight:.2f}",
        f"- `w_imbalance`: {config.risk_imbalance_weight:.2f}",
        f"- `w_low_spatial_freedom`: {config.risk_low_spatial_freedom_weight:.2f}",
        f"- `w_resilience_pressure`: {config.risk_resilience_pressure_weight:.2f}",
        f"- `w_persistence`: {config.risk_persistence_weight:.2f}",
        "",
        "## Model Context",
        "",
        f"- Model type: `{state_model_result.model_type}`",
        f"- HMM state setting: `{state_model_result.configured_state_setting}`",
        f"- Fitted HMM components used: {state_model_result.effective_states}",
        f"- Occupied latent states observed in the sequence: {state_model_result.occupied_states}",
        f"- Feature columns used by the state model: {', '.join(state_model_result.feature_columns)}",
        "",
        "## State Summary Table",
        "",
        "Only occupied states appear in the table below. If the model was fit with more components than were actually used by the current run, unoccupied components are not listed here.",
        "",
        "```text",
        summary_df[display_columns].to_string(index=False),
        "```",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
