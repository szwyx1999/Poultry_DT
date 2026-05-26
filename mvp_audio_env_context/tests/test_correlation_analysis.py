from __future__ import annotations

import pandas as pd

from mvp_audio_env_context.src.correlation_analysis import _benjamini_hochberg


def test_benjamini_hochberg_returns_ordered_adjusted_values() -> None:
    series = pd.Series([0.01, 0.03, 0.20, 0.04])
    adjusted = _benjamini_hochberg(series)
    assert adjusted.notna().all()
    assert float(adjusted.min()) >= 0.0
    assert float(adjusted.max()) <= 1.0
