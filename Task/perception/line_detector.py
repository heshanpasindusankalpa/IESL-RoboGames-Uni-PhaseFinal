"""
perception/line_detector.py

Takes the binary mask from Preprocessor and returns the line's
centroid position, heading angle, and a confidence score.

Two detection strategies are available and can be selected via config:

  Strategy.SLIDING_WINDOW  — best for curved lines (your arena)
      Divides the mask into N horizontal slices.
      Finds the centroid of white pixels in each slice.
      Utilizes 1D spatial clustering to prevent averaging at junctions.
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
    slice_points: list = field(default_factory=list)
    
    # --- NEW: Graph & Junction Data ---
    junction_detected: bool = False
    branch_centroids: list[float] = field(default_factory=list) # X-coords of all branches
    junction_y: int = 0                                         # Y-coord where the split happens

    @property
    def is_detected(self) -> bool:
        return self.confidence > 0.0

    def lateral_error(self, frame_width: int) -> float:
        return self.centroid_x - (frame_width / 2)


@dataclass
class LineDetectorConfig:
    strategy:        Strategy = Strategy.SLIDING_WINDOW

    # Sliding window
    num_slices:      int = 6      # horizontal scan lines
    min_pixels:      int = 10     # min white pixels to count a slice as valid
    cluster_gap:     int = 10     # NEW: pixels gap to separate left/right branches at junctions

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

    def detect(self, mask: np.ndarray, turn_bias: str = "straight") -> LineResult:
        """
        Run line detection. turn_bias allows the graph algorithm to steer the drone 
        ("left", "right", "straight") when multiple branches are detected.
        """
        if self.cfg.strategy == Strategy.SLIDING_WINDOW:
            return self._sliding_window(mask, turn_bias)
        else:
            return self._hough(mask)

    # ------------------------------------------------------------------
    # Strategy A: Sliding window (with Branch Lock)
    # ------------------------------------------------------------------

    def _sliding_window(self, mask: np.ndarray, turn_bias: str) -> LineResult:
        h, w = mask.shape
        slice_h = h // self.cfg.num_slices

        valid_points = []
        current_target_x = w / 2.0 
        
        # Junction tracking state
        j_detected = False
        j_branches = []
        j_y = 0

        for i in range(self.cfg.num_slices):
            y_bot = h - i * slice_h
            y_top = max(y_bot - slice_h, 0)
            strip = mask[y_top:y_bot, :]
            
            # Pass the bias down to the centroid calculator
            cx, all_centers = self._centroid_x(strip, target_x=current_target_x, turn_bias=turn_bias)
            
            # If we see multiple distinct lines in this horizontal slice, it's a junction!
            if len(all_centers) > 1:
                j_detected = True
                j_branches = all_centers
                j_y = (y_top + y_bot) // 2

            if cx is not None:
                y_mid = (y_top + y_bot) // 2
                valid_points.append((cx, y_mid))
                current_target_x = cx

        if len(valid_points) < 2:
            confidence = len(valid_points) / self.cfg.num_slices
            if len(valid_points) == 1:
                dx = valid_points[0][0] - (w / 2)
                dy = valid_points[0][1] - h
                angle_deg = float(np.degrees(np.arctan(dx / dy))) if dy != 0 else 0.0
                return LineResult(centroid_x=valid_points[0][0], angle_deg=angle_deg, confidence=confidence, 
                                  slice_points=valid_points, junction_detected=j_detected, 
                                  branch_centroids=j_branches, junction_y=j_y)
            return LineResult(centroid_x=w / 2, angle_deg=0.0, confidence=0.0, slice_points=[])

        xs = np.array([p[0] for p in valid_points], dtype=np.float32)
        ys = np.array([p[1] for p in valid_points], dtype=np.float32)

        weights = []
        for _, y in valid_points:
            w_val = 2.0 if y > h / 2 else 1.0
            weights.append(w_val)
        weights = np.array(weights)
        
        centroid_x = float(np.sum(xs * weights) / np.sum(weights))
        angle_deg = self._fit_angle(xs, ys)
        confidence = len(valid_points) / self.cfg.num_slices

        return LineResult(
            centroid_x=centroid_x, angle_deg=angle_deg, confidence=confidence, slice_points=valid_points,
            junction_detected=j_detected, branch_centroids=j_branches, junction_y=j_y
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

    def _centroid_x(self, strip: np.ndarray, target_x: float, turn_bias: str) -> tuple[float | None, list[float]]:
        """Returns the chosen centroid AND a list of all branch centroids found."""
        white_cols = np.where(strip > 0)[1]

        if len(white_cols) < self.cfg.min_pixels:
            return None, []
            
        jumps = np.where(np.diff(white_cols) > self.cfg.cluster_gap)[0]
        branches = np.split(white_cols, jumps + 1)
        branch_centers = [np.mean(branch) for branch in branches if len(branch) >= self.cfg.min_pixels]
        
        if not branch_centers:
            return None, []
            
        if len(branch_centers) == 1:
            return float(branch_centers[0]), branch_centers
            
        # --- GRAPH NAVIGATION DECISION EXECUTION ---
        # We are at a split. We physically bias the tracking X based on the graph planner's command.
        if turn_bias == "left":
            best_center = min(branch_centers) # Pick the leftmost line
        elif turn_bias == "right":
            best_center = max(branch_centers) # Pick the rightmost line
        else:
            # Default straight continuity
            best_center = min(branch_centers, key=lambda cx: abs(cx - target_x))
            
        return float(best_center), branch_centers

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