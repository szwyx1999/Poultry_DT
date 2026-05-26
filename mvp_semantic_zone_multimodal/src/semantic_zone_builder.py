from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from PIL import Image

from .config import SemanticZoneConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - availability depends on local environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class SemanticZoneBuildResult:
    zone_config: dict
    output_json_path: Path
    overlay_path: Path
    report_path: Path
    warnings: tuple[str, ...]
    valid: bool


def build_semantic_zone_config(config: SemanticZoneConfig) -> SemanticZoneBuildResult:
    reference_rgb = np.asarray(Image.open(config.reference_image).convert("RGB"))
    annotated_rgb = np.asarray(Image.open(config.annotated_image).convert("RGB"))
    if reference_rgb.shape != annotated_rgb.shape:
        raise ValueError(
            "Semantic zone reference images must have the same dimensions: "
            f"{reference_rgb.shape} vs {annotated_rgb.shape}"
        )

    image_height, image_width = reference_rgb.shape[:2]
    warnings: list[str] = []
    zone_config: dict | None = None
    detection_mode = "automatic"

    try:
        red_box = _detect_colored_box(reference_rgb, annotated_rgb, color_name="red")
        green_box = _detect_colored_box(reference_rgb, annotated_rgb, color_name="green")
        zone_config = _build_zone_config_from_boxes(
            reference_image=config.reference_image,
            annotated_image=config.annotated_image,
            image_width=image_width,
            image_height=image_height,
            red_box=red_box,
            green_box=green_box,
            red_source="red_box",
            green_source="green_box",
        )
    except Exception as exc:
        warnings.append(f"automatic_zone_detection_failed:{exc}")

    if zone_config is None or not _zone_config_is_valid(zone_config):
        manual_config, manual_warning = _load_manual_zone_config(config.manual_zone_yaml)
        if manual_warning:
            warnings.append(manual_warning)
        if manual_config is not None:
            zone_config = manual_config
            detection_mode = "manual_fallback"

    if zone_config is None:
        raise ValueError(
            "Semantic zone detection failed and no valid manual fallback config was available."
        )

    validation = _validate_zone_config(zone_config)
    warnings.extend(validation["warnings"])
    is_valid = bool(validation["valid"])
    if not is_valid and not config.force_invalid_zones:
        raise ValueError(
            "Semantic zone detection produced invalid geometry. "
            "Use --force only if you intentionally want to continue with the fallback geometry."
        )

    output_json_path = config.zones_dir / "semantic_zone_config_room1_aug16_17.json"
    overlay_path = config.zones_dir / "semantic_zone_overlay.png"
    report_path = config.reports_dir / "semantic_zone_definition_report.md"

    zone_config["detection_mode"] = detection_mode
    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(zone_config, handle, indent=2)

    _save_zone_overlay(reference_rgb, zone_config, overlay_path)
    report_path.write_text(
        _build_zone_report(zone_config=zone_config, validation=validation, warnings=warnings),
        encoding="utf-8",
    )
    return SemanticZoneBuildResult(
        zone_config=zone_config,
        output_json_path=output_json_path,
        overlay_path=overlay_path,
        report_path=report_path,
        warnings=tuple(warnings),
        valid=is_valid,
    )


def build_zone_masks(zone_config: dict, image_width: int, image_height: int) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    occupancy = np.zeros((image_height, image_width), dtype=bool)

    for zone in zone_config.get("zones", []):
        zone_id = str(zone.get("zone_id"))
        polygon = zone.get("polygon")
        if polygon:
            mask = polygon_to_mask(
                polygon=np.asarray(polygon, dtype=float),
                image_width=image_width,
                image_height=image_height,
            )
            masks[zone_id] = mask
            occupancy |= mask

    if "general_zone" not in masks:
        masks["general_zone"] = ~occupancy
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


def scale_polygon(
    polygon: list[list[float]] | np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    polygon_array = np.asarray(polygon, dtype=float)
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    scaled = polygon_array.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def _detect_colored_box(reference_rgb: np.ndarray, annotated_rgb: np.ndarray, color_name: str) -> tuple[int, int, int, int]:
    if color_name not in {"red", "green"}:
        raise ValueError(f"Unsupported color name: {color_name}")

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
        connected = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        component_count, labels, stats, _ = connected
        best_index = None
        best_area = -1
        for component_index in range(1, component_count):
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_index = component_index
        if best_index is None:
            raise ValueError(f"No {color_name} annotation component was detected.")
        x = int(stats[best_index, cv2.CC_STAT_LEFT])
        y = int(stats[best_index, cv2.CC_STAT_TOP])
        w = int(stats[best_index, cv2.CC_STAT_WIDTH])
        h = int(stats[best_index, cv2.CC_STAT_HEIGHT])
        return x, y, w, h

    coordinates = np.argwhere(mask_uint8 > 0)
    if coordinates.size == 0:
        raise ValueError(f"No {color_name} annotation component was detected.")
    y_values = coordinates[:, 0]
    x_values = coordinates[:, 1]
    x = int(x_values.min())
    y = int(y_values.min())
    w = int(x_values.max() - x + 1)
    h = int(y_values.max() - y + 1)
    return x, y, w, h


def _build_zone_config_from_boxes(
    reference_image: Path,
    annotated_image: Path,
    image_width: int,
    image_height: int,
    red_box: tuple[int, int, int, int],
    green_box: tuple[int, int, int, int],
    red_source: str,
    green_source: str,
) -> dict:
    rx, ry, rw, rh = red_box
    gx, gy, gw, gh = green_box
    return {
        "zone_config_id": "room1_aug16_17_semantic_v1",
        "reference_image": str(reference_image.as_posix()),
        "annotated_image": str(annotated_image.as_posix()),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "zones": [
            {
                "zone_id": "drinking_zone",
                "display_name": "Drinking Area",
                "semantic_type": "drinking",
                "source_annotation": red_source,
                "polygon": [
                    [rx, ry],
                    [rx + rw, ry],
                    [rx + rw, ry + rh],
                    [rx, ry + rh],
                ],
            },
            {
                "zone_id": "feeding_zone",
                "display_name": "Feeding Area",
                "semantic_type": "feeding",
                "source_annotation": green_source,
                "polygon": [
                    [gx, gy],
                    [gx + gw, gy],
                    [gx + gw, gy + gh],
                    [gx, gy + gh],
                ],
            },
            {
                "zone_id": "general_zone",
                "display_name": "General Area",
                "semantic_type": "general",
                "polygon": None,
                "definition": "full_frame_minus_drinking_and_feeding",
            },
        ],
    }


def _load_manual_zone_config(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, "manual_zone_yaml_missing"
    with path.open("r", encoding="utf-8") as handle:
        manual_config = yaml.safe_load(handle) or {}
    if not isinstance(manual_config, dict) or "zones" not in manual_config:
        return None, "manual_zone_yaml_invalid"
    return manual_config, None


def _zone_config_is_valid(zone_config: dict) -> bool:
    return bool(_validate_zone_config(zone_config)["valid"])


def _validate_zone_config(zone_config: dict) -> dict[str, object]:
    image_width = int(zone_config.get("image_width", 0) or 0)
    image_height = int(zone_config.get("image_height", 0) or 0)
    warnings: list[str] = []
    if image_width <= 0 or image_height <= 0:
        warnings.append("non_positive_image_dimensions")
        return {"valid": False, "warnings": warnings, "area_summary": {}}

    masks = build_zone_masks(zone_config, image_width=image_width, image_height=image_height)
    total_pixels = image_width * image_height
    area_summary: dict[str, dict[str, float]] = {}

    for zone in zone_config.get("zones", []):
        zone_id = str(zone.get("zone_id"))
        mask = masks.get(zone_id)
        if mask is None:
            continue
        area_pixels = int(mask.sum())
        area_fraction = float(area_pixels / total_pixels)
        area_summary[zone_id] = {
            "area_pixels": area_pixels,
            "area_fraction": area_fraction,
        }
        if zone_id != "general_zone" and area_fraction < 0.001:
            warnings.append(f"{zone_id}_too_small")
        if zone_id != "general_zone" and area_fraction > 0.4:
            warnings.append(f"{zone_id}_too_large")

    drinking = masks.get("drinking_zone")
    feeding = masks.get("feeding_zone")
    overlap_pixels = int(np.logical_and(drinking, feeding).sum()) if drinking is not None and feeding is not None else 0
    if overlap_pixels > 0:
        warnings.append("drinking_feeding_overlap")
    general_fraction = area_summary.get("general_zone", {}).get("area_fraction", 0.0)
    if general_fraction <= 0:
        warnings.append("general_zone_empty")

    is_valid = not any(
        warning in {"non_positive_image_dimensions", "drinking_feeding_overlap", "general_zone_empty"}
        for warning in warnings
    )
    return {
        "valid": is_valid,
        "warnings": warnings,
        "area_summary": area_summary,
        "overlap_pixels": overlap_pixels,
        "remainder_definition": True,
    }


def _save_zone_overlay(reference_rgb: np.ndarray, zone_config: dict, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(reference_rgb)
    axis.set_title("Semantic Zone Overlay")
    axis.set_axis_off()

    color_map = {
        "drinking_zone": ("#d62728", "Drinking Area"),
        "feeding_zone": ("#2ca02c", "Feeding Area"),
        "general_zone": ("#1f77b4", "General Area"),
    }

    for zone in zone_config.get("zones", []):
        zone_id = str(zone.get("zone_id"))
        polygon = zone.get("polygon")
        color, label = color_map.get(zone_id, ("#444444", zone_id))
        if polygon:
            polygon_array = np.asarray(polygon, dtype=float)
            closed = np.vstack([polygon_array, polygon_array[0]])
            axis.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2.2, label=label)
            centroid_x = float(polygon_array[:, 0].mean())
            centroid_y = float(polygon_array[:, 1].mean())
            axis.text(centroid_x, centroid_y, label, color=color, fontsize=10, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": color})
        else:
            axis.text(
                80,
                70,
                "General Area = full frame minus drinking and feeding boxes",
                color=color,
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": color},
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _build_zone_report(zone_config: dict, validation: dict[str, object], warnings: list[str]) -> str:
    image_width = int(zone_config.get("image_width", 0) or 0)
    image_height = int(zone_config.get("image_height", 0) or 0)
    area_summary: dict[str, dict[str, float]] = validation.get("area_summary", {})  # type: ignore[assignment]
    zones = {zone["zone_id"]: zone for zone in zone_config.get("zones", [])}

    lines = [
        "# Semantic Zone Definition Report",
        "",
        "- This is a new semantic-zone experiment. Previous code and previous outputs were not modified.",
        f"- Image dimensions: `{image_width} x {image_height}`",
        f"- Detection mode: `{zone_config.get('detection_mode', 'unknown')}`",
        f"- Drinking polygon: `{zones.get('drinking_zone', {}).get('polygon')}`",
        f"- Feeding polygon: `{zones.get('feeding_zone', {}).get('polygon')}`",
        f"- General zone computed as remainder: `{bool(validation.get('remainder_definition', False))}`",
        f"- Drinking/feeding overlap pixels: `{int(validation.get('overlap_pixels', 0) or 0)}`",
        "",
        "## Zone Areas",
        "",
    ]
    for zone_id in ("drinking_zone", "feeding_zone", "general_zone"):
        summary = area_summary.get(zone_id, {})
        lines.append(
            f"- `{zone_id}`: `{int(summary.get('area_pixels', 0))}` pixels "
            f"({float(summary.get('area_fraction', 0.0)):.4f} of frame)"
        )

    lines.extend(["", "## Validation Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- `{warning}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"

