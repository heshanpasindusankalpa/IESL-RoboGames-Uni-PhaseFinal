import cv2
import numpy as np
import sys
import os

from preprocessor import Preprocessor, PreprocessorConfig

def run_video_tuner(video_path: str):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
    
    # Starting values (roughly yellow)
    for name, val in [("H low", 18), ("H high", 35),
                      ("S low", 80), ("S high", 255),
                      ("V low", 80), ("V high", 255)]:
        cv2.createTrackbar(name, "HSV Tuner", val, 255, lambda x: None)

    pre = Preprocessor(PreprocessorConfig())

    print(f"[Tuner] Playing {video_path}. Adjust sliders.")
    print("[Tuner] SPACE to pause/play. Q to quit and print final values.")
    
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            
            # Loop the video continuously if it reaches the end
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        h_lo = cv2.getTrackbarPos("H low",  "HSV Tuner")
        h_hi = cv2.getTrackbarPos("H high", "HSV Tuner")
        s_lo = cv2.getTrackbarPos("S low",  "HSV Tuner")
        s_hi = cv2.getTrackbarPos("S high", "HSV Tuner")
        v_lo = cv2.getTrackbarPos("V low",  "HSV Tuner")
        v_hi = cv2.getTrackbarPos("V high", "HSV Tuner")

        pre.update_hsv([h_lo, s_lo, v_lo], [h_hi, s_hi, v_hi])
        
        # Process the frame through your actual pipeline
        mask, roi = pre.process(frame)

        # Display the color crop and the binary mask side-by-side
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack([roi, mask_bgr])
        
        cv2.imshow("HSV Tuner", combined_view)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print(f"\n[Tuner] Final HSV values to put in your PreprocessorConfig:")
            print(f"  hsv_lower: [{h_lo}, {s_lo}, {v_lo}]")
            print(f"  hsv_upper: [{h_hi}, {s_hi}, {v_hi}]")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    raw_path = "~/IESL-RoboGames-Uni-Finals/samples/video/sample_1.mp4"
    absolute_path = os.path.expanduser(raw_path)
    run_video_tuner(absolute_path)