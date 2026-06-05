from __future__ import annotations

from pathlib import Path
import sys
import uuid

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_zone_loader import build_zone_config_from_images


def test_build_zone_config_from_synthetic_images() -> None:
    scratch_root = Path("tmp_tests")
    scratch_root.mkdir(parents=True, exist_ok=True)
    tmp_path = scratch_root / f"zone_test_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    width, height = 200, 120
    reference = Image.new("RGB", (width, height), color=(120, 120, 120))
    annotated = reference.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([20, 20, 70, 60], outline=(255, 0, 0), width=3)
    draw.rectangle([90, 30, 150, 90], outline=(0, 255, 0), width=3)

    reference_path = tmp_path / "room1_test_reference.png"
    annotated_path = tmp_path / "room1_test_reference_with_notes.png"
    reference.save(reference_path)
    annotated.save(annotated_path)

    zone_config = build_zone_config_from_images("room_1", reference_path, annotated_path)
    assert zone_config["room_id"] == "room_1"
    assert zone_config["zone_config_id"] == "room_1_semantic_v1"
    zone_ids = [zone["zone_id"] for zone in zone_config["zones"]]
    assert zone_ids == ["drinking_zone", "feeding_zone", "general_zone"]
    assert zone_config["zones"][0]["polygon"] is not None
    assert zone_config["zones"][1]["polygon"] is not None
