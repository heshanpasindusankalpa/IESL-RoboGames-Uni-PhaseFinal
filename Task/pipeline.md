# Autonomous Drone Navigation Pipeline

## 1. System Architecture

The system consists of an Iris quadcopter running within the **Webots** simulator, driven by **ArduPilot SITL** (Software In The Loop). Navigation and visual processing are handled by a custom Python control script (`line_follow_test.py`) that communicates via two main interfaces:

- **Flight Control (PyMAVLink):** Uses MAVLink over UDP to command drone velocities (Body-Frame NED), modes, and read telemetry (altitude).
- **Camera Stream (TCP Socket):** Receives raw 3-byte RGB frames from the Webots camera payload, interpreting them into OpenCV BGR arrays.

---

## 2. Vision & Perception Pipeline

The drone relies purely on computer vision to navigate the environment. The frame processing is divided into two distinct parallel tracks: **Line Detection** and **Tag Detection**.

### A. Yellow Line Detection

To track the guiding path, the pipeline extracts the shape of the path while ignoring distractors:

1. **HSV Color Filtering:** The BGR frame is converted to HSV. A strict `cv2.inRange()` mask filters exclusively for yellow `[20, 100, 100]` to `[40, 255, 255]`. (This makes it immune to lighting artifacts or white/gray environmental objects).
2. **Morphological Cleanup:** The script applies Open/Close operations to remove noise, and an elliptical bridge close to heal tiny gaps.
3. **Smart Tag Masking:** White AprilTags often sit directly on top of the yellow line. To prevent the tag's high contrast from confusing the line tracker, the script extracts the AprilTag polygon and blacks it out on the HSV mask using `cv2.fillConvexPoly()`.
4. **Contour Prioritization:** If multiple yellow lines branch out, `keep_line_contour()` isolates the true path by heavily penalizing boundaries (fence reflections) and rewarding spatial continuity from the previous frame.
5. **Error Calculation:** A sliding window strategy determines the path's centroid and angular deviation, producing an `angle_error` and `lateral_error`.

### B. AprilTag Detection & Decoding

The project uses `pupil_apriltags` to scan the ground for 36h11 family AprilTags:

1. **Grayscale Conversion:** The BGR frame is pushed to grayscale for tag reading.
2. **Periodic Scanning:** To save CPU cycles, the tag detector runs every $N$ frames, unless the drone is actively hovering/landing.
3. **Data Decoding:** The detected numeric `tag_id` encodes critical mission parameters:
   - **Digit 1:** Country Code (Must match target array)
   - **Digit 2:** Airport Status (`1` = Safe to land, `0` = Unsafe)
   - **Digit 3:** Outbound connectivity (Unused for landing triggers)

---

## 3. Flight Control Algorithms

The drone translates visual errors into physical velocities (`vx, vy, yaw_rate`).

### PID Controllers (Line Following)

Smooth tracking is achieved using customized PD (Proportional-Derivative) controllers:

- **Time-Normalized Derivatives:** The script divides the error change by $dt$ to handle variable framerates smoothly.
- **EMA Smoothing:** Exponential Moving Averages filter noise out of the raw camera data.
- **Adaptive Sharp Bends:** If the line curves beyond $28^\circ$, dynamic limits expand to boost yaw authority and slew rates, prioritizing turning aggressively over lateral correction.

### Visual Servoing (Tag Alignment)

When moving to land, the drone uses a purely proportional Visual Servoing controller. It translates pixel disparities directly into body-frame velocities:
$$v_x = -error_y \times K_{Fwd}$$
$$v_y = error_x \times K_{Lat}$$

---

## 4. Autonomous State Machine (`NavState`)

The heart of the system is an automated State Machine that cycles through distinct flight phases:

1. **`LINE_FOLLOW`**: The main travel state. Auto-starts when the yellow line is identified. Applies PID targeting to follow curves and straightaways.
2. **`TAG_HOVER`**: Triggered the moment a new AprilTag is detected. The drone full-stops and hovers over the tag for 2 seconds to guarantee an accurate, stable read.
3. **`DECIDING`**: Parses the tag's decoded ID against the mission targets (`Airports = [1, 2]`) and the safety digit.
   - If Match $\rightarrow$ `PRE_LAND_ALIGN`
   - If Mismatch $\rightarrow$ `TAG_REACQUIRE`
4. **`PRE_LAND_ALIGN`**: Visual servoing locks onto the specific tag. The drone maneuvers until the tag center rests within a designated pixel deadzone for multiple consecutive frames.
5. **`LANDING`**: The drone switches PyMAVLink into `LAND` mode and descends. Telemetry confirms touchdown. If targets remain, it automatically takes off, regains altitude, and returns to `LINE_FOLLOW`.
6. **`TAG_REACQUIRE`**: Used when skipping an invalid pad. The drone creeps forward slowly, ignoring the rejected tag until it picks up the plain yellow line again.
7. **`DONE`**: Arrives when the target list is entirely empty. The script triggers a final landing sequence and halts.
