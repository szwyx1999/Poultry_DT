from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_zone_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        zone_config = json.load(handle)
    _validate_zone_config(zone_config)
    return zone_config


def parse_layout(layout: str) -> tuple[int, int]:
    normalized = layout.strip().lower()
    if normalized not in {"2x2", "3x3"}:
        raise ValueError(f"Unsupported zone layout: {layout}")
    rows_text, cols_text = normalized.split("x", maxsplit=1)
    return int(rows_text), int(cols_text)


def compute_zone_boundaries(
    frame_height: int,
    frame_width: int,
    zone_config: dict,
    layout_override: str | None = None,
) -> list[dict]:
    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("Frame dimensions must be positive integers.")

    layout = layout_override or zone_config.get("layout", "")
    rows, cols = parse_layout(layout)
    y_edges = np.linspace(0, frame_height, num=rows + 1, dtype=int)
    x_edges = np.linspace(0, frame_width, num=cols + 1, dtype=int)

    ordered_zones = sorted(zone_config["zones"], key=lambda item: (item["row"], item["col"]))
    boundaries: list[dict] = []
    for zone in ordered_zones:
        row = int(zone["row"])
        col = int(zone["col"])
        boundaries.append(
            {
                "zone_id": zone["zone_id"],
                "row": row,
                "col": col,
                "y_start": int(y_edges[row]),
                "y_end": int(y_edges[row + 1]),
                "x_start": int(x_edges[col]),
                "x_end": int(x_edges[col + 1]),
            }
        )
    return boundaries


def _validate_zone_config(zone_config: dict) -> None:
    if "layout" not in zone_config:
        raise ValueError("Zone config must define a layout.")
    rows, cols = parse_layout(str(zone_config["layout"]))
    zones = zone_config.get("zones", [])
    if len(zones) != rows * cols:
        raise ValueError(
            f"Zone config layout {rows}x{cols} requires {rows * cols} zones, found {len(zones)}."
        )

    seen_positions: set[tuple[int, int]] = set()
    for zone in zones:
        if "zone_id" not in zone or "row" not in zone or "col" not in zone:
            raise ValueError("Each zone must define zone_id, row, and col.")
        row = int(zone["row"])
        col = int(zone["col"])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(
                f"Zone {zone['zone_id']} is outside the declared {rows}x{cols} layout."
            )
        position = (row, col)
        if position in seen_positions:
            raise ValueError("Zone config contains duplicate row/col positions.")
        seen_positions.add(position)
