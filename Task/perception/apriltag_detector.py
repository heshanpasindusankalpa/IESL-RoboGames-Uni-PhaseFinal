"""AprilTag detection and airport-code decoding.

Parses tag36h11 payloads as three-digit airport identifiers:
- country_code (digit 1),
- airport_status (digit 2; 1=landable),
- reachable airport count (digit 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from pupil_apriltags import Detector as _Detector
except ImportError:
    raise ImportError(
        "pupil-apriltags is required. Install with:  pip install pupil-apriltags"
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TagResult:
    """Decoded information from one AprilTag detection."""

    tag_id:        int          # raw tag ID number
    country_code:  int          # digit 1
    airport_status: int         # digit 2  (1 = safe, 0 = unsafe)
    reachable:     int          # digit 3  (number of reachable airports)
    center:        Tuple[float, float]   # (cx, cy) pixel centre of detection
    corners:       np.ndarray           # 4×2 corner points

    @property
    def is_landable(self) -> bool:
        return self.airport_status == 1

    def __repr__(self) -> str:
        return (f"Tag(id={self.tag_id}, country={self.country_code}, "
                f"status={'SAFE' if self.is_landable else 'UNSAFE'}, "
                f"reachable={self.reachable}, center=({self.center[0]:.0f},{self.center[1]:.0f}))")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AprilTagDetector:
    """
    Wraps pupil_apriltags to detect tag36h11 tags and decode the
    three-digit airport information.

    Usage:
        det = AprilTagDetector()
        tags = det.detect(gray_frame)   # list[TagResult]
    """

    def __init__(self, *, quad_decimate: float = 1.0, quad_sigma: float = 0.0,
                 decode_sharpening: float = 0.25, nthreads: int = 2):
        self._detector = _Detector(
            families="tag36h11",
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            decode_sharpening=decode_sharpening,
            nthreads=nthreads,
        )

    def detect(self, gray: np.ndarray) -> List[TagResult]:
        """
        Run AprilTag detection on a grayscale frame.

        Args:
            gray: uint8 grayscale image (H×W)

        Returns:
            List of TagResult, one per detected tag.
        """
        if gray.ndim != 2:
            raise ValueError("AprilTagDetector expects a 2D grayscale image")

        detections = self._detector.detect(gray)
        results: List[TagResult] = []

        for det in detections:
            tag_id = det.tag_id
            # Decode as three-digit number: e.g. tag_id=112 → 1,1,2
            d1 = tag_id // 100          # country code
            d2 = (tag_id // 10) % 10    # airport status
            d3 = tag_id % 10            # reachable airports

            center = (float(det.center[0]), float(det.center[1]))
            corners = det.corners  # 4×2 ndarray

            results.append(TagResult(
                tag_id=tag_id,
                country_code=d1,
                airport_status=d2,
                reachable=d3,
                center=center,
                corners=corners,
            ))

        return results

    def detect_best(self, gray: np.ndarray) -> TagResult | None:
        """
        Detect tags and return the largest (closest) one, or None.
        Largest = biggest bounding area in the image.
        """
        tags = self.detect(gray)
        if not tags:
            return None
        # Pick the tag with the largest bounding area (closest tag)
        def _area(t: TagResult) -> float:
            c = t.corners  # 4×2
            # Shoelace formula
            x = c[:, 0]
            y = c[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return max(tags, key=_area)
