from __future__ import annotations

import unittest

import numpy as np

from mvp_video_zone_heatmap.src.mvp1.zone_utils import compute_zone_boundaries


ZONE_CONFIG_2X2 = {
    "room_id": "room1",
    "layout": "2x2",
    "zones": [
        {"zone_id": "zone_A", "row": 0, "col": 0},
        {"zone_id": "zone_B", "row": 0, "col": 1},
        {"zone_id": "zone_C", "row": 1, "col": 0},
        {"zone_id": "zone_D", "row": 1, "col": 1},
    ],
}

ZONE_CONFIG_3X3 = {
    "room_id": "room1",
    "layout": "3x3",
    "zones": [
        {"zone_id": f"zone_{row}_{col}", "row": row, "col": col}
        for row in range(3)
        for col in range(3)
    ],
}


class ZoneUtilsTests(unittest.TestCase):
    def test_2x2_boundaries_cover_full_frame(self) -> None:
        boundaries = compute_zone_boundaries(360, 640, ZONE_CONFIG_2X2)

        coverage = np.zeros((360, 640), dtype=int)
        for index, zone in enumerate(boundaries, start=1):
            coverage[zone["y_start"] : zone["y_end"], zone["x_start"] : zone["x_end"]] += index
        self.assertEqual(len(boundaries), 4)
        self.assertTrue(np.all(coverage > 0))

    def test_3x3_boundaries_cover_full_frame(self) -> None:
        boundaries = compute_zone_boundaries(300, 600, ZONE_CONFIG_3X3)

        coverage = np.zeros((300, 600), dtype=int)
        for index, zone in enumerate(boundaries, start=1):
            coverage[zone["y_start"] : zone["y_end"], zone["x_start"] : zone["x_end"]] += index
        self.assertEqual(len(boundaries), 9)
        self.assertTrue(np.all(coverage > 0))


if __name__ == "__main__":
    unittest.main()
