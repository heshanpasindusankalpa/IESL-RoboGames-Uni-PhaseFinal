"""
perception/line_detector.py

Takes the binary mask from Preprocessor and returns the line's
centroid position, heading angle, and a confidence score.

Two detection strategies are available and can be selected via config:

  Strategy.SLIDING_WINDOW  — best for curved lines (your arena)
      Divides the mask into N horizontal slices.
      Finds the centroid of white pixels in each slice.
      Fits a line through those centroids.

  Strategy.HOUGH           — best for straight segments, fast
      Runs Canny edge detection on the mask.
      Applies HoughLinesP to find line segments.
      Averages segment midpoints for centroid.

Output: LineResult dataclass
    centroid_x  : float  — x position of line center in ROI pixel coords
    angle_deg   : float  — heading angle of line (-90 to +90 deg, 0 = vertical)
    confidence  : float  — 0.0 (no line) to 1.0 (strong detection)
    slice_points: list   — [(x, y), ...] centroids per slice (debug / visualizer)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Strategy(Enum):
    SLIDING_WINDOW = "sliding_window"
    HOUGH = "hough"


@dataclass
class LineResult:
    """Output of one detection pass. Consumed by PID controller."""

    centroid_x:   float        # x pixel in ROI coords
    angle_deg:    float        # degrees, 0 = straight ahead, + = leaning right
    confidence:   float        # 0.0 – 1.0
    # [(x,y),...] for visualizer
    slice_points: list = field(default_factory=list)

    @property
    def is_detected(self) -> bool:
        return self.confidence > 0.0

    # Error relative to frame center — this is what the PID receives
    def lateral_error(self, frame_width: int) -> float:
        """
        Positive  → line is to the RIGHT of center → drone should move right.
        Negative  → line is to the LEFT  of center → drone should move left.
        """
        return self.centroid_x - (frame_width / 2)


@dataclass
class LineDetectorConfig:
    strategy:        Strategy = Strategy.SLIDING_WINDOW

    # Sliding window
    num_slices:      int = 6      # horizontal scan lines
    min_pixels:      int = 10     # min white pixels to count a slice as valid

    # Hough
    canny_low:       int = 50
    canny_high:      int = 150
    hough_threshold: int = 20
    hough_min_len:   int = 30
    hough_max_gap:   int = 20

    # Confidence: fraction of slices (or segments) that must fire
    min_confidence:  float = 0.3


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class LineDetector:
    """
    Detects the line in a binary mask and returns a LineResult.

    Usage:
        detector = LineDetector()
        result   = detector.detect(mask, roi_width)

        error = result.lateral_error(roi_width)   # → PID
    """

    def __init__(self, config: LineDetectorConfig = None):
        self.cfg = config or LineDetectorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, mask: np.ndarray) -> LineResult:
        """
        Run line detection on a binary mask.

        Args:
            mask: uint8 binary image (0/255) from Preprocessor.process()

        Returns:
            LineResult — always returned, check .is_detected or .confidence
        """
        if self.cfg.strategy == Strategy.SLIDING_WINDOW:
            return self._sliding_window(mask)
        else:
            return self._hough(mask)

    # ------------------------------------------------------------------
    # Strategy A: Sliding window
    # ------------------------------------------------------------------

    def _sliding_window(self, mask: np.ndarray) -> LineResult:
        """
        Scan the mask in horizontal slices from bottom to top.
        Each slice votes with its centroid x if it has enough white pixels.
        """
        h, w = mask.shape
        slice_h = h // self.cfg.num_slices

        valid_points = []   # (x, y) centroids of valid slices

        for i in range(self.cfg.num_slices):
            # Work bottom-up: slice 0 = bottom (closest to drone)
            y_bot = h - i * slice_h
            y_top = y_bot - slice_h
            y_top = max(y_top, 0)

            strip = mask[y_top:y_bot, :]
            cx = self._centroid_x(strip)

            if cx is not None:
                y_mid = (y_top + y_bot) // 2
                valid_points.append((cx, y_mid))

        if len(valid_points) < 2:
            # Not enough slices fired — no confident detection
            confidence = len(valid_points) / self.cfg.num_slices
            if len(valid_points) == 1:
                # If only one slice fired (often means horizontal line!),
                # compute angle strictly pointing towards that centroid from the bottom-center of ROI
                dx = valid_points[0][0] - (w / 2)
                dy = valid_points[0][1] - h  # negative since y_mid < h

                # To match np.polyfit, we use dx/dy.
                # If the line is to the right (dx > 0), dx/dy < 0, angle should be NEGATIVE.
                angle_deg = float(np.degrees(
                    np.arctan(dx / dy))) if dy != 0 else 0.0

                return LineResult(
                    centroid_x=valid_points[0][0],
                    angle_deg=angle_deg,
                    confidence=confidence,
                    slice_points=valid_points,
                )
            return LineResult(centroid_x=w / 2, angle_deg=0.0,
                              confidence=0.0, slice_points=[])

        # Fit a line through the slice centroids
        xs = np.array([p[0] for p in valid_points], dtype=np.float32)
        ys = np.array([p[1] for p in valid_points], dtype=np.float32)

        # Use a weighted average of slice centroids for positioning.
        # Bottom slices (closer to drone) get more weight, but including top
        # slices helps the drone "look ahead" on curves to reduce tracking lag.
        weights = []
        for _, y in valid_points:
            # y is height in ROI (bottom is larger y)
            # Give pixels in the bottom half 2x weight vs top half
            w_val = 2.0 if y > h / 2 else 1.0
            weights.append(w_val)

        weights = np.array(weights)
        centroid_x = float(np.sum(xs * weights) / np.sum(weights))

        # Angle: fit line, measure deviation from vertical
        angle_deg = self._fit_angle(xs, ys)

        confidence = len(valid_points) / self.cfg.num_slices

        return LineResult(
            centroid_x=centroid_x,
            angle_deg=angle_deg,
            confidence=confidence,
            slice_points=valid_points,
        )

    # ------------------------------------------------------------------
    # Strategy B: Hough lines
    # ------------------------------------------------------------------

    def _hough(self, mask: np.ndarray) -> LineResult:
        """
        Detect line segments via Canny + HoughLinesP.
        Average their midpoints for centroid, average angles for heading.
        """
        h, w = mask.shape

        edges = cv2.Canny(mask, self.cfg.canny_low, self.cfg.canny_high)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.hough_min_len,
            maxLineGap=self.cfg.hough_max_gap,
        )

        if lines is None:
            return LineResult(centroid_x=w / 2, angle_deg=0.0,
                              confidence=0.0, slice_points=[])

        midpoints = []
        angles = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            midpoints.append(((x1 + x2) / 2, (y1 + y2) / 2))
            angle = np.degrees(np.arctan2(x2 - x1, y2 - y1)
                               )  # deviation from vertical
            angles.append(angle)

        centroid_x = float(np.mean([p[0] for p in midpoints]))
        angle_deg = float(np.mean(angles))

        # Confidence: normalise by a "good" segment count (10 segments = 1.0)
        confidence = min(len(lines) / 10.0, 1.0)

        return LineResult(
            centroid_x=centroid_x,
            angle_deg=angle_deg,
            confidence=confidence,
            slice_points=[(int(p[0]), int(p[1])) for p in midpoints],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _centroid_x(self, strip: np.ndarray) -> float | None:
        """
        Return the x centroid of white pixels in a horizontal strip.
        Returns None if fewer than min_pixels are white.
        """
        white_cols = np.where(strip > 0)

        if len(white_cols[1]) < self.cfg.min_pixels:
            return None
        return float(np.mean(white_cols[1]))

    @staticmethod
    def _fit_angle(xs: np.ndarray, ys: np.ndarray) -> float:
        """
        Fit a line through (xs, ys) and return its angle from vertical (degrees).
        Positive = leaning right, negative = leaning left.
        """
        if len(xs) < 2:
            return 0.0
        # np.polyfit: fit y = m*x + b, but we want x = m*y + b (line is mostly vertical)
        coeffs = np.polyfit(ys, xs, 1)   # x as function of y
        # slope in (x / y) space → convert to degrees from vertical
        angle = float(np.degrees(np.arctan(coeffs[0])))
        return angle
