"""

Converts a raw BGR frame from the camera into a clean binary mask
that isolates the line (yellow) for the line detector.

Pipeline per frame:
    BGR frame
        → resize to working resolution
        → crop to ROI (bottom portion of frame — the look-ahead zone)
        → convert BGR → HSV
        → apply color threshold  (isolate line color)
        → morphological clean-up (remove noise, fill gaps)
        → binary mask (0 / 255)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PreprocessorConfig:
    """
    All tunable parameters in one place.
    Override via params.yaml and pass as kwargs, e.g.:
        cfg = PreprocessorConfig(roi_top_frac=0.5)
    """

    # --- Working resolution (resize before all processing) ---
    frame_width:  int = 320
    frame_height: int = 240

    # --- ROI: fraction from the TOP of the (resized) frame to start the crop ---
    # 0.6 means: ignore top 60%, keep bottom 40%
    roi_top_frac: float = 0.6

    # --- HSV thresholds for YELLOW line ---
    # Tune with a helper script or the live tuner (see bottom of file)
    hsv_lower: list = field(default_factory=lambda: [18, 80, 80])
    hsv_upper: list = field(default_factory=lambda: [35, 255, 255])

    # --- Morphological kernel size (removes speckle noise) ---
    morph_kernel_size: int = 5


class Preprocessor:
    """
    Converts a raw BGR frame into a binary mask ready for line detection.

    Usage:
        pre = Preprocessor()                      # defaults
        pre = Preprocessor(PreprocessorConfig(roi_top_frac=0.5))

        mask, roi = pre.process(frame)
        # mask : binary np.ndarray  (0/255), shape = ROI crop size
        # roi  : the colour ROI crop (BGR), useful for debug display
    """

    def __init__(self, config: PreprocessorConfig = None):
        self.cfg = config or PreprocessorConfig()
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Args:
            frame: raw BGR frame from Camera.get_frame()

        Returns:
            mask : binary np.ndarray (uint8, 0/255) — line pixels are 255
            roi  : BGR crop of the ROI — pass to visualizer for debug overlay
        """
        resized = self._resize(frame)
        roi = self._crop_roi(resized)
        mask = self._color_mask(roi)
        mask = self._clean(mask)
        return mask, roi

    def update_hsv(self, lower: list, upper: list):
        """Hot-update HSV thresholds at runtime (e.g. from a tuner slider)."""
        self.cfg.hsv_lower = lower
        self.cfg.hsv_upper = upper

    def roi_offset_y(self) -> int:
        """
        Pixel y-offset of the ROI top edge within the resized frame.
        Used by line_detector to map ROI coordinates back to full-frame coords.
        """
        return int(self.cfg.frame_height * self.cfg.roi_top_frac)

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize to fixed working resolution for consistent processing speed."""
        return cv2.resize(
            frame,
            (self.cfg.frame_width, self.cfg.frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop to the look-ahead zone (bottom portion of frame).
        Removes sky, horizon, and distant clutter the drone doesn't need.
        """
        y_start = self.roi_offset_y()
        return frame[y_start:, :]

    def _color_mask(self, roi: np.ndarray) -> np.ndarray:
        """
        Convert to HSV and threshold to isolate the line color.
        Returns a binary mask: 255 where the line color is detected, 0 elsewhere.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array(self.cfg.hsv_lower, dtype=np.uint8)
        upper = np.array(self.cfg.hsv_upper, dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        """
        Morphological open  → removes small noise specks.
        Morphological close → fills small gaps/holes in the line.
        """
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)
        return closed


# ---------------------------------------------------------------------------
# Live HSV tuner — run this file directly to tune thresholds interactively
#
#   python perception/preprocessor.py
#
# Adjust the sliders until the line is clean white and background is black.
# Copy the printed values into PreprocessorConfig or params.yaml.
# ---------------------------------------------------------------------------

def _run_tuner():
    """Interactive HSV tuner using trackbars. Press Q to quit."""
    import sys
    from perception.camera import Camera  # adjust import path if running standalone

    cam = Camera()
    cam.start()

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    for name, val in [("H low", 18), ("H high", 35),
                      ("S low", 80), ("S high", 255),
                      ("V low", 80), ("V high", 255)]:
        cv2.createTrackbar(name, "HSV Tuner", val, 255, lambda x: None)

    pre = Preprocessor()

    print("[Tuner] Adjust sliders. Press Q to quit and print final values.")
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        h_lo = cv2.getTrackbarPos("H low",  "HSV Tuner")
        h_hi = cv2.getTrackbarPos("H high", "HSV Tuner")
        s_lo = cv2.getTrackbarPos("S low",  "HSV Tuner")
        s_hi = cv2.getTrackbarPos("S high", "HSV Tuner")
        v_lo = cv2.getTrackbarPos("V low",  "HSV Tuner")
        v_hi = cv2.getTrackbarPos("V high", "HSV Tuner")

        pre.update_hsv([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi])
        mask, roi = pre.process(frame)

        cv2.imshow("HSV Tuner", np.hstack(
            [roi, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[Tuner] Final HSV values:")
            print(f"  hsv_lower: [{h_lo}, {s_lo}, {v_lo}]")
            print(f"  hsv_upper: [{h_hi}, {s_hi}, {v_hi}]")
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_tuner()
