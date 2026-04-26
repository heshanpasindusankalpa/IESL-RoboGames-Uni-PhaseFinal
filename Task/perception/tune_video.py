import cv2
import numpy as np
import sys
import os

try:
    from perception.preprocessor import Preprocessor, PreprocessorConfig
    from perception.apriltag_detector import AprilTagDetector
except ImportError:
    from preprocessor import Preprocessor, PreprocessorConfig
    from apriltag_detector import AprilTagDetector

def draw_tags_on_frame(vis, tags):
    """Helper function to draw bounding boxes and decoded text over AprilTags."""
    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(vis, tuple(corners[i]), tuple(
                corners[(i + 1) % 4]), (0, 255, 255), 2)
        cx, cy = int(tag.center[0]), int(tag.center[1])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        
        label = f"ID:{tag.tag_id} C:{tag.country_code} S:{tag.airport_status}"
        cv2.putText(vis, label, (cx - 40, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)

def run_video_tuner(video_path: str):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    cv2.namedWindow("HSV Tuner (Dual Mask)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Full Frame + Tags", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Full Frame + Tags", 640, 480)
    
    # --- UPDATED DUAL RED MASK SLIDERS ---
    # H1 Max controls the 0-15 side. H2 Min controls the 160-179 side.
    for name, val in [("H1 Max (Low Red)", 15), ("H2 Min (High Red)", 160),
                      ("S Min", 60), ("S Max", 255),
                      ("V Min", 60), ("V Max", 255)]:
        cv2.createTrackbar(name, "HSV Tuner (Dual Mask)", val, 255, lambda x: None)

    pre = Preprocessor(PreprocessorConfig())
    
    tag_detector = AprilTagDetector(
        quad_decimate=2.0,
        quad_sigma=0.2,
        decode_sharpening=0.3,
        nthreads=2,
    )

    print(f"[Tuner] Playing {video_path}. Adjust sliders.")
    print("[Tuner] SPACE to pause/play. Q to quit and print final values.")
    
    paused = False
    raw_frame = None

    while True:
        if not paused:
            ret, new_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            raw_frame = new_frame

        if raw_frame is None:
            continue

        frame = cv2.resize(raw_frame, (640, 480))
        display_frame = frame.copy()

        # 1. Tag Detection
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        tags = tag_detector.detect(gray)
        draw_tags_on_frame(display_frame, tags)

        # 2. Get Tuner Values
        h1_max = cv2.getTrackbarPos("H1 Max (Low Red)",  "HSV Tuner (Dual Mask)")
        h2_min = cv2.getTrackbarPos("H2 Min (High Red)", "HSV Tuner (Dual Mask)")
        s_min  = cv2.getTrackbarPos("S Min",  "HSV Tuner (Dual Mask)")
        s_max  = cv2.getTrackbarPos("S Max",  "HSV Tuner (Dual Mask)")
        v_min  = cv2.getTrackbarPos("V Min",  "HSV Tuner (Dual Mask)")
        v_max  = cv2.getTrackbarPos("V Max",  "HSV Tuner (Dual Mask)")

        # 3. Exact Dual Mask Logic (Matches flight.py)
        # Use preprocessor to safely crop the ROI
        resized_for_roi = pre._resize(display_frame)
        roi = pre._crop_roi(resized_for_roi)
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Mask 1 (Low Reds)
        lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
        upper_red1 = np.array([h1_max, s_max, v_max], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Mask 2 (High Reds)
        lower_red2 = np.array([h2_min, s_min, v_min], dtype=np.uint8)
        upper_red2 = np.array([179, s_max, v_max], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine
        m = cv2.bitwise_or(mask1, mask2)
        
        # Clean (Morphology)
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 21))
        mask = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_bridge)

        # 4. Display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack([roi, mask_bgr])
        
        cv2.imshow("HSV Tuner (Dual Mask)", combined_view)
        cv2.imshow("Full Frame + Tags", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print(f"\n[Tuner] Final Dual Mask HSV values to put in flight.py:")
            print(f"  Mask 1 (Low Red) : [0, {s_min}, {v_min}] to [{h1_max}, {s_max}, {v_max}]")
            print(f"  Mask 2 (High Red): [{h2_min}, {s_min}, {v_min}] to [179, {s_max}, {v_max}]")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    raw_path = "~/IESL-RoboGames-Uni-Finals/samples/video/new_sample.mp4" 
    absolute_path = os.path.expanduser(raw_path)
    run_video_tuner(absolute_path)