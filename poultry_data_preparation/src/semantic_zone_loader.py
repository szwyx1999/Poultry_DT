from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image

from .config import PreparationConfig
from .room_parser import parse_room_identifier_from_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - depends on local environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticZoneLoadResult:
    zone_configs: list[dict]
    combined_output_path: Path
    report_path: Path
    valid_rooms: set[str]


def load_semantic_zone_configs(config: PreparationConfig, dry_run: bool = False) -> SemanticZoneLoadResult:
    _ensure_manual_fallback_template(config)
    zone_pairs = discover_zone_reference_pairs(config.semantic_zone_ref_dir)
    manual_fallbacks = _load_manual_fallbacks(config)
    zone_configs: list[dict] = []
    warnings: list[str] = []

    for room_id, pair in zone_pairs.items():
        LOGGER.info("Preparing semantic zones for %s", room_id)
        built_config = None
        try:
            built_config = build_zone_config_from_images(
                room_id=room_id,
                reference_image=pair["reference_image"],
                annotated_image=pair["annotated_image"],
            )
        except Exception as exc:
            warnings.append(f"{room_id}:automatic_detection_failed:{exc}")

        if built_config is not None and not _validate_zone_config(built_config)["valid"]:
            warnings.append(f"{room_id}:automatic_detection_invalid")
            built_config = None

        if built_config is None and config.semantic_allow_manual_fallback:
            manual_config = manual_fallbacks.get(room_id)
            if manual_config is not None:
                built_config = manual_config
                built_config["detection_mode"] = "manual_fallback"
                warnings.append(f"{room_id}:manual_fallback_used")

        if built_config is None:
            warnings.append(f"{room_id}:no_valid_zone_config")
            continue

        zone_configs.append(built_config)
        _write_zone_outputs(config, built_config)

    combined_output_path = config.zones_output_dir / "semantic_zone_configs.json"
    combined_output_path.write_text(json.dumps(zone_configs, indent=2), encoding="utf-8")
    report_path = config.reports_output_dir / "semantic_zone_report.md"
    report_path.write_text(_build_zone_report(zone_configs, warnings, dry_run), encoding="utf-8")
    return SemanticZoneLoadResult(
        zone_configs=zone_configs,
        combined_output_path=combined_output_path,
        report_path=report_path,
        valid_rooms={item["room_id"] for item in zone_configs},
    )


def discover_zone_reference_pairs(reference_dir: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for annotated_path in sorted(reference_dir.glob("*_reference_with_notes.png"), key=lambda path: path.as_posix().lower()):
        base_name = annotated_path.name.replace("_reference_with_notes.png", "")
        reference_path = annotated_path.with_name(base_name + "_reference.png")
        if not reference_path.exists():
            continue
        room_result = parse_room_identifier_from_path(annotated_path.name)
        pairs.setdefault(
            room_result.room_id,
            {
                "reference_image": reference_path,
                "annotated_image": annotated_path,
            },
        )
    return pairs


def build_zone_config_from_images(room_id: str, reference_image: Path, annotated_image: Path) -> dict:
    reference_rgb = np.asarray(Image.open(reference_image).convert("RGB"))
    annotated_rgb = np.asarray(Image.open(annotated_image).convert("RGB"))
    if reference_rgb.shape != annotated_rgb.shape:
        raise ValueError("Reference and annotated images must have the same dimensions.")

    red_box = _shrink_box(_detect_colored_box(reference_rgb, annotated_rgb, "red"), margin=2)
    green_box = _shrink_box(_detect_colored_box(reference_rgb, annotated_rgb, "green"), margin=2)
    image_height, image_width = reference_rgb.shape[:2]
    config = {
        "room_id": room_id,
        "zone_config_id": f"{room_id}_semantic_v1",
        "reference_image": str(reference_image),
        "annotated_image": str(annotated_image),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "detection_mode": "automatic",
        "zones": [
            _box_zone("drinking_zone", "Drinking Area", "drinking", red_box),
            _box_zone("feeding_zone", "Feeding Area", "feeding", green_box),
            {
                "zone_id": "general_zone",
                "display_name": "General Area",
                "semantic_type": "general",
                "definition": "full_frame_minus_drinking_and_feeding",
            },
        ],
    }
    return config


def build_zone_masks(zone_config: dict, image_width: int, image_height: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    occupied = np.zeros((image_height, image_width), dtype=bool)
    for zone in zone_config.get("zones", []):
        polygon = zone.get("polygon")
        if polygon:
            mask = polygon_to_mask(np.asarray(polygon, dtype=float), image_width, image_height)
            masks[str(zone["zone_id"])] = mask
            occupied |= mask
    masks["general_zone"] = ~occupied
    return masks


def polygon_to_mask(polygon: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    if cv2 is None:
        x_values = polygon[:, 0]
        y_values = polygon[:, 1]
        x_min = max(0, int(np.floor(x_values.min())))
        x_max = min(image_width, int(np.ceil(x_values.max())))
        y_min = max(0, int(np.floor(y_values.min())))
        y_max = min(image_height, int(np.ceil(y_values.max())))
        mask = np.zeros((image_height, image_width), dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        return mask
    points = np.round(polygon).astype(np.int32)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask.astype(bool)


def scale_polygon(polygon, source_width: int, source_height: int, target_width: int, target_height: int) -> np.ndarray:
    polygon_array = np.asarray(polygon, dtype=float)
    scaled = polygon_array.copy()
    scaled[:, 0] *= float(target_width) / float(source_width)
    scaled[:, 1] *= float(target_height) / float(source_height)
    return scaled


def _ensure_manual_fallback_template(config: PreparationConfig) -> None:
    manual_path = config.config_dir / "manual_semantic_zones.yaml"
    if manual_path.exists():
        return
    template_path = Path(__file__).resolve().parents[1] / "config" / "manual_semantic_zones.yaml"
    manual_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")


def _load_manual_fallbacks(config: PreparationConfig) -> dict[str, dict]:
    manual_path = config.config_dir / "manual_semantic_zones.yaml"
    if not manual_path.exists():
        return {}
    with manual_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return {str(room_id): dict(item) for room_id, item in payload.items() if isinstance(item, dict)}


def _detect_colored_box(reference_rgb: np.ndarray, annotated_rgb: np.ndarray, color_name: str) -> tuple[int, int, int, int]:
    diff = np.abs(annotated_rgb.astype(np.int16) - reference_rgb.astype(np.int16))
    if color_name == "red":
        mask = (
            (annotated_rgb[:, :, 0] > 150)
            & ((annotated_rgb[:, :, 0].astype(np.int16) - annotated_rgb[:, :, 1].astype(np.int16)) > 60)
            & ((annotated_rgb[:, :, 0].astype(np.int16) - annotated_rgb[:, :, 2].astype(np.int16)) > 60)
            & (diff.sum(axis=2) > 25)
        )
    else:
        mask = (
            (annotated_rgb[:, :, 1] > 120)
            & ((annotated_rgb[:, :, 1].astype(np.int16) - annotated_rgb[:, :, 0].astype(np.int16)) > 35)
            & ((annotated_rgb[:, :, 1].astype(np.int16) - annotated_rgb[:, :, 2].astype(np.int16)) > 10)
            & (diff.sum(axis=2) > 25)
        )

    mask_uint8 = (mask.astype(np.uint8) * 255)
    if cv2 is not None:
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        best_index = None
        best_area = -1
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_index = component_index
        if best_index is None:
            raise ValueError(f"No {color_name} annotation component was detected.")
        return (
            int(stats[best_index, cv2.CC_STAT_LEFT]),
            int(stats[best_index, cv2.CC_STAT_TOP]),
            int(stats[best_index, cv2.CC_STAT_WIDTH]),
            int(stats[best_index, cv2.CC_STAT_HEIGHT]),
        )

    coordinates = np.argwhere(mask_uint8 > 0)
    if coordinates.size == 0:
        raise ValueError(f"No {color_name} annotation component was detected.")
    y_values = coordinates[:, 0]
    x_values = coordinates[:, 1]
    return (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max() - x_values.min() + 1),
        int(y_values.max() - y_values.min() + 1),
    )


def _shrink_box(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x, y, width, height = box
    return (x + margin, y + margin, max(1, width - (2 * margin)), max(1, height - (2 * margin)))


def _box_zone(zone_id: str, display_name: str, semantic_type: str, box: tuple[int, int, int, int]) -> dict:
    x, y, width, height = box
    return {
        "zone_id": zone_id,
        "display_name": display_name,
        "semantic_type": semantic_type,
        "polygon": [
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ],
    }


def _validate_zone_config(zone_config: dict) -> dict[str, object]:
    masks = build_zone_masks(zone_config, int(zone_config["image_width"]), int(zone_config["image_height"]))
    drinking = masks.get("drinking_zone")
    feeding = masks.get("feeding_zone")
    overlap_pixels = int(np.logical_and(drinking, feeding).sum()) if drinking is not None and feeding is not None else 0
    general_pixels = int(masks["general_zone"].sum()) if "general_zone" in masks else 0
    valid = overlap_pixels == 0 and general_pixels > 0
    return {"valid": valid, "overlap_pixels": overlap_pixels, "general_pixels": general_pixels}


def _write_zone_outputs(config: PreparationConfig, zone_config: dict) -> None:
    room_id = zone_config["room_id"]
    output_path = config.zones_output_dir / f"semantic_zone_config_{room_id}.json"
    output_path.write_text(json.dumps(zone_config, indent=2), encoding="utf-8")

    reference_image = np.asarray(Image.open(zone_config["reference_image"]).convert("RGB"))
    overlay_path = config.zones_output_dir / f"semantic_zone_overlay_{room_id}.png"
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(reference_image)
    axis.set_title(f"Semantic Zone Overlay: {room_id}")
    axis.set_axis_off()
    colors = {"drinking_zone": "#d62728", "feeding_zone": "#2ca02c", "general_zone": "#1f77b4"}
    for zone in zone_config["zones"]:
        polygon = zone.get("polygon")
        if not polygon:
            axis.text(80, 70, "General Area = full frame minus drinking and feeding", color=colors["general_zone"], fontsize=10, bbox={"facecolor": "white", "alpha": 0.75})
            continue
        polygon_array = np.asarray(polygon, dtype=float)
        closed = np.vstack([polygon_array, polygon_array[0]])
        color = colors.get(zone["zone_id"], "#444444")
        axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.2)
        axis.text(float(polygon_array[:, 0].mean()), float(polygon_array[:, 1].mean()), zone["display_name"], color=color, fontsize=10, bbox={"facecolor": "white", "alpha": 0.75})
    figure.tight_layout()
    figure.savefig(overlay_path, dpi=200)
    plt.close(figure)


def _build_zone_report(zone_configs: list[dict], warnings: list[str], dry_run: bool) -> str:
    lines = [
        "# Semantic Zone Report",
        "",
        f"- Dry run: `{dry_run}`",
        f"- Rooms with valid semantic zones: {len(zone_configs)}",
        "",
        "## Zone Configs",
        "",
    ]
    for zone_config in zone_configs:
        validation = _validate_zone_config(zone_config)
        lines.append(
            f"- `{zone_config['room_id']}` -> `{zone_config['zone_config_id']}` "
            f"(mode `{zone_config.get('detection_mode', 'unknown')}`, overlap `{validation['overlap_pixels']}` pixels)"
        )
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Manual Fallback",
            "",
            "- If automatic detection fails for a room, define polygons in `poultry_data_preparation/config/manual_semantic_zones.yaml` and rerun the `zones` stage.",
        ]
    )
    return "\n".join(lines) + "\n"
