from __future__ import annotations

import logging

import pandas as pd

from .biomarkers import build_canonical_zone_feature_table, compute_room_window_biomarkers
from .config import BiomarkerTwinConfig
from .data_management import apply_window_selection, ensure_sufficient_mvp1_features, write_data_coverage_report
from .diagnostics import (
    analyze_hmm_sequence,
    validate_hmm_sequence_for_plotting,
    validate_window_level_hmm_input,
    write_hmm_sequence_diagnostics_report,
)
from .event_validation import (
    build_event_validation_result,
    ensure_caretaker_event_log,
    generate_event_validation_plots,
    label_windows_relative_to_events,
)
from .io_utils import ensure_output_dirs, load_json, read_csv_with_datetimes, write_markdown
from .plotting import (
    plot_biomarker_timeseries,
    plot_biomarker_timeseries_by_room,
    plot_hmm_state_outputs,
    plot_resilience_curves,
    plot_risk_score_distribution,
    plot_state_risk_heatmap,
    plot_state_summary_profile,
)
from .reporting import write_risk_score_explanation_report
from .resilience import compute_resilience
from .state_model import fit_state_model
from .unity_export import export_unity_json


LOGGER = logging.getLogger(__name__)


def run_pipeline(config: BiomarkerTwinConfig) -> dict[str, str]:
    ensure_output_dirs(config)
    source_summary = ensure_sufficient_mvp1_features(config)

    LOGGER.info("Loading MVP1 feature tables from %s", config.zone_features_csv)
    zone_features_df = read_csv_with_datetimes(config.zone_features_csv)
    selected_windows_df = read_csv_with_datetimes(config.selected_windows_csv)
    window_index_df = read_csv_with_datetimes(config.video_window_index_csv)
    media_manifest_df = read_csv_with_datetimes(config.media_manifest_csv)

    canonical_result = build_canonical_zone_feature_table(
        zone_features_df=zone_features_df,
        selected_windows_df=selected_windows_df,
        window_index_df=window_index_df,
        media_manifest_df=media_manifest_df,
    )
    canonical_path = config.features_dir / "canonical_zone_feature_table.csv"
    canonical_result.canonical_df.drop(columns=["start_time_dt", "end_time_dt"], errors="ignore").to_csv(canonical_path, index=False)
    schema_path = config.reports_dir / "schema_mapping.md"
    write_markdown(schema_path, canonical_result.schema_mapping_markdown)

    LOGGER.info("Computing room/window biomarker tables")
    biomarker_full_df = compute_room_window_biomarkers(canonical_result.canonical_df, config)
    biomarker_df = apply_window_selection(biomarker_full_df, config)
    biomarker_df.attrs["max_windows_setting"] = "null" if config.max_windows is None else config.max_windows
    biomarker_df.attrs["max_windows_per_room_setting"] = "null" if config.max_windows_per_room is None else config.max_windows_per_room
    biomarker_df.attrs["max_windows_per_session_setting"] = "null" if config.max_windows_per_session is None else config.max_windows_per_session
    biomarker_df.attrs["min_windows_for_hmm_setting"] = config.min_windows_for_hmm
    validate_window_level_hmm_input(biomarker_df, table_name="biomarker_window_table.csv")

    caretaker_report_path = config.reports_dir / "caretaker_event_timestamp_report.md"
    event_log_df = ensure_caretaker_event_log(media_manifest_df, config, caretaker_report_path)
    event_log_df = read_csv_with_datetimes(config.generated_event_log_csv)
    LOGGER.info("Using event log from %s", config.generated_event_log_csv)

    resilience_result = compute_resilience(biomarker_df, event_log_df, config)
    validate_window_level_hmm_input(resilience_result.biomarker_window_df, table_name="biomarker_window_table.csv after resilience")

    labelled_biomarker_df = label_windows_relative_to_events(
        resilience_result.biomarker_window_df,
        event_log_df,
        config,
    )
    validate_window_level_hmm_input(labelled_biomarker_df, table_name="labelled_biomarker_window_table.csv")

    biomarker_path = config.features_dir / "biomarker_window_table.csv"
    labelled_biomarker_path = config.features_dir / "labelled_biomarker_window_table.csv"
    resilience_path = config.features_dir / "resilience_event_table.csv"
    resilience_result.biomarker_window_df.drop(columns=["distribution_vector"], errors="ignore").to_csv(biomarker_path, index=False)
    labelled_biomarker_df.drop(columns=["distribution_vector"], errors="ignore").to_csv(labelled_biomarker_path, index=False)
    resilience_result.resilience_event_df.to_csv(resilience_path, index=False)

    LOGGER.info("Rendering biomarker and resilience plots")
    plot_biomarker_timeseries(labelled_biomarker_df, str(config.plots_dir / "biomarker_timeseries.png"))
    plot_biomarker_timeseries_by_room(labelled_biomarker_df, str(config.plots_dir / "biomarker_timeseries_by_room.png"))
    plot_resilience_curves(
        labelled_biomarker_df,
        resilience_result.resilience_event_df,
        str(config.plots_dir / "resilience_curves.png"),
        config,
    )

    LOGGER.info("Fitting latent behavioural state model")
    state_model_result = fit_state_model(
        labelled_biomarker_df,
        config,
        model_path=str(config.model_dir / "hmm_model.joblib"),
    )
    state_sequence_path = config.features_dir / "hmm_state_sequence.csv"
    state_summary_path = config.features_dir / "hmm_state_summary.csv"
    state_model_result.state_sequence_df.to_csv(state_sequence_path, index=False)
    state_model_result.state_summary_df.to_csv(state_summary_path, index=False)

    event_labelled_window_count = int(
        labelled_biomarker_df[
            labelled_biomarker_df["event_phase"].astype(str).isin(
                ["pre_entry_baseline", "during_entry", "post_entry_recovery"]
            )
        ]["window_id"].nunique()
    )
    caretaker_event_covered = event_labelled_window_count > 0
    hmm_diagnostics = analyze_hmm_sequence(
        canonical_df=canonical_result.canonical_df,
        biomarker_df=labelled_biomarker_df,
        sequence_df=state_model_result.state_sequence_df,
        model_type=state_model_result.model_type,
        requested_state_setting=state_model_result.configured_state_setting,
        fitted_state_count=state_model_result.effective_states,
        occupied_state_count=state_model_result.occupied_states,
        event_labelled_window_count=event_labelled_window_count,
        caretaker_event_covered=caretaker_event_covered,
    )
    diagnostics_path = config.reports_dir / "hmm_sequence_diagnostics.md"
    write_hmm_sequence_diagnostics_report(hmm_diagnostics, str(diagnostics_path))
    validate_hmm_sequence_for_plotting(hmm_diagnostics)

    plot_hmm_state_outputs(
        state_model_result.state_sequence_df,
        plots_dir=config.plots_dir,
        hmm_by_sequence_dir=config.plots_hmm_by_sequence_dir,
    )
    plot_state_risk_heatmap(state_model_result.state_sequence_df, str(config.plots_dir / "state_risk_heatmap.png"))
    plot_state_summary_profile(state_model_result.state_summary_df, str(config.plots_dir / "state_summary_radar_or_bar.png"))
    plot_risk_score_distribution(state_model_result.state_sequence_df, str(config.plots_dir / "risk_score_distribution.png"))

    event_validation_result = build_event_validation_result(
        labelled_window_df=labelled_biomarker_df,
        state_sequence_df=state_model_result.state_sequence_df,
        event_log_df=event_log_df,
        config=config,
    )
    event_validation_summary_path = config.features_dir / "event_validation_summary.csv"
    event_validation_report_path = config.reports_dir / "event_validation_report.md"
    event_validation_result.summary_df.to_csv(event_validation_summary_path, index=False)
    write_markdown(event_validation_report_path, event_validation_result.report_markdown)
    generate_event_validation_plots(event_validation_result, event_log_df, config)

    data_coverage_report_path = config.reports_dir / "data_coverage_report.md"
    write_data_coverage_report(
        source_summary=source_summary,
        canonical_df=canonical_result.canonical_df,
        biomarker_df=labelled_biomarker_df,
        state_summary_df=state_model_result.state_summary_df,
        configured_state_setting=state_model_result.configured_state_setting,
        effective_states=state_model_result.effective_states,
        occupied_states=state_model_result.occupied_states,
        max_windows_setting=config.max_windows,
        max_windows_per_room_setting=config.max_windows_per_room,
        max_windows_per_session_setting=config.max_windows_per_session,
        min_windows_for_hmm_setting=config.min_windows_for_hmm,
        output_path=data_coverage_report_path,
    )
    risk_explanation_path = config.reports_dir / "risk_score_explanation.md"
    write_risk_score_explanation_report(state_model_result, config, risk_explanation_path)

    LOGGER.info("Exporting Unity timeline JSON")
    zone_config = _load_zone_config(config)
    unity_json_path, copied_paths = export_unity_json(
        canonical_df=canonical_result.canonical_df,
        biomarker_df=labelled_biomarker_df,
        state_df=state_model_result.state_sequence_df,
        zone_config=zone_config,
        config=config,
        model_type=state_model_result.model_type,
    )

    return {
        "canonical_zone_feature_table": str(canonical_path),
        "schema_mapping": str(schema_path),
        "biomarker_window_table": str(biomarker_path),
        "labelled_biomarker_window_table": str(labelled_biomarker_path),
        "resilience_event_table": str(resilience_path),
        "hmm_model": str(config.model_dir / "hmm_model.joblib"),
        "hmm_state_sequence": str(state_sequence_path),
        "hmm_state_summary": str(state_summary_path),
        "event_validation_summary": str(event_validation_summary_path),
        "caretaker_event_timestamp_report": str(caretaker_report_path),
        "data_coverage_report": str(data_coverage_report_path),
        "hmm_sequence_diagnostics": str(diagnostics_path),
        "risk_score_explanation": str(risk_explanation_path),
        "event_validation_report": str(event_validation_report_path),
        "unity_json": str(unity_json_path),
        "unity_json_copies": ", ".join(str(path_value) for path_value in copied_paths),
    }


def _load_zone_config(config: BiomarkerTwinConfig) -> dict:
    if config.zone_config_json.exists():
        return load_json(config.zone_config_json)
    if config.mvp1_unity_json.exists():
        payload = load_json(config.mvp1_unity_json)
        return {
            "room_id": "room_1",
            "zones": payload.get("zones", []),
        }
    return {"room_id": "room_1", "zones": []}
