from __future__ import annotations

import numpy as np

from mvp_audio_env_context.src.audio_features import _compute_audio_frame_features


def test_compute_audio_frame_features_detects_energy() -> None:
    sample_rate = 22050
    duration_seconds = 2.0
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    feature_df = _compute_audio_frame_features(waveform, sample_rate=sample_rate, frame_seconds=1.0)

    assert len(feature_df) == 2
    assert float(feature_df["audio_rms"].mean()) > 0.1
    assert float(feature_df["audio_spectral_centroid"].mean()) > 100.0
    assert float(feature_df["audio_zero_crossing_rate"].mean()) > 0.0
