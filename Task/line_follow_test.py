"""
Task/line_follow_test.py

Autonomous drone line follower with AprilTag-based airport navigation.

The drone follows a yellow guiding line and detects AprilTags (36h11) on
landing pads. Each tag encodes a 3-digit number:
    Digit 1 → Country Code
    Digit 2 → Airport Status (1=safe / 0=unsafe)
    Digit 3 → Number of reachable airports

The drone is given target country codes and must land at matching airports.

Oscillation fixes vs original:
    • PD controller for yaw (derivative term dampens overshoot)
    • EMA smoothing on angle and lateral error (filters noise)
    • Dead-zone for tiny errors (prevents jitter)

Controls:
    F   — toggle following ON/OFF
    C   — creep forward (find the line)
    T   — threshold slider
    0   — stop
    Q   — land + quit
"""

from perception.line_detector import LineDetector, LineDetectorConfig, Strategy
from perception.apriltag_detector import AprilTagDetector, TagResult
import math
import socket
import struct
import sys
import time
from enum import Enum, auto

import cv2
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, ".")


# ─────────────────────────────────────────────────────────────────────────────
# MISSION CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Target country codes — drone must land at airports matching these
TARGET_COUNTRIES = [1, 2]   # ← set these to the required country codes


# ─────────────────────────────────────────────────────────────────────────────
# TUNING  — buttery-smooth line following
# ─────────────────────────────────────────────────────────────────────────────
#
#   D terms are TIME-NORMALIZED (divided by dt), so gains must be small.
#   At 10 FPS (dt≈0.1s), a 3° angle change → d_angle/dt = 30 °/s.
#   D contribution = KD * 30 — keep this well below MAX_YAW!
#

FORWARD_SPEED = 0.15        # m/s — slow and steady
SLOW_SPEED = 0.08        # m/s — creep speed (tag reacquire)

# PD Controller — Yaw
#   At 20°: P = 0.50.  D (3°/frame) = 0.004*30 = 0.12.  Total ≈ 0.62 → clips 0.50
#   At 15°: P = 0.375. D (2°/frame) = 0.004*20 = 0.08.  Total ≈ 0.45 ✓ sustained curve
#   At 10°: P = 0.25.  D (1.5°/frame)= 0.004*15= 0.06.  Total ≈ 0.31 ✓
#   At  5°: P = 0.125. D (1°/frame) = 0.004*10 = 0.04.  Total ≈ 0.16 ✓ gentle
KP_YAW = -0.025      # proportional — stronger for sustained curves
KD_YAW = -0.004      # derivative (time-normalized, must be SMALL)

# PD Controller — Lateral centering
#   At 200px: P = 0.080.  D (30px/frame) = 0.0001*300 = 0.03.  Total ≈ 0.11 → clips
#   At  50px: P = 0.020.  D (10px/frame) = 0.0001*100 = 0.01.  Total ≈ 0.03 ✓ gentle
KP_LAT = 0.0004      # m/s per pixel — strong enough to correct 200px offset
KD_LAT = 0.0001      # m/s per (pixel/s) — dampens lateral oscillation

THRESHOLD = 155         # binary threshold for line detection
# ignore top 40% → see 60% of frame (more look-ahead on curves)
ROI_TOP_FRAC = 0.40
ALTITUDE = 1.5       # m — cruise altitude

MAX_YAW = 0.50        # rad/s — enough for 90° bends
MAX_LAT = 0.12        # m/s  — enough for lateral centering

# Slew rate — limits how fast yaw can change per frame
YAW_SLEW = 0.20        # rad/s per frame — smooth but not too restrictive

# Hard-bend boost (adaptive yaw authority)
HIGH_BEND_ANGLE = 28.0    # degrees — beyond this, prioritize turn-in
# rad/s — extra yaw authority for sharp curves (was 0.75)
HIGH_BEND_MAX_YAW = 1.50
# rad/s per frame — faster yaw buildup on bends (was 0.35)
HIGH_BEND_SLEW = 1.00

# Tag centering for landing alignment
TAG_ALIGN_P_LAT = 0.0025  # m/s per pixel error (lateral)
TAG_ALIGN_P_FWD = 0.0025  # m/s per pixel error (forward)
TAG_ALIGN_MAX_VEL = 0.15    # m/s max velocity for alignment
TAG_ALIGN_DEADZONE = 15     # pixels alignment tolerance

# Pre-landing precision gate (land only when tag is centered and stable)
PRE_LAND_OFFSET_X = 0       # pixels: +right / -left target offset from image center
# pixels: +down / -up target offset (camera geometry bias)
PRE_LAND_OFFSET_Y = 10
PRE_LAND_DEADZONE_X = 10    # pixels
PRE_LAND_DEADZONE_Y = 10    # pixels
PRE_LAND_LOCK_FRAMES = 8    # require this many centered frames before LAND
PRE_LAND_MAX_VEL = 0.10     # m/s while fine-aligning before land
PRE_LAND_MAX_ALIGN_TIME = 6.0  # seconds before retrying hover-read cycle


# Dead-zone — suppresses micro-corrections on straights
YAW_DEAD_ZONE = 2.0         # degrees — ignore tiny angle noise
LAT_DEAD_ZONE = 5.0         # pixels  — ignore tiny lateral noise

# EMA smoothing (0.0 = very smooth/laggy, 1.0 = no smoothing/noisy)
#   Heavy smoothing creates LAG on curves → drone always under-steers.
#   The slew rate limiter already prevents oscillation, so EMA can be lighter.
EMA_ALPHA = 0.40        # less lag → tracks curves in real-time

# Align phase — yaw-only until angle < this
ALIGN_THRESHOLD_DEG = 8.0

# Dynamic speed — automatically slow down at sharp bends
BEND_SLOW_ANGLE = 15.0      # degrees — start slowing early

# Line-loss recovery — keep yawing instead of stopping at bends
LINE_LOSS_GRACE = 15        # frames to keep searching before giving up

# AprilTag detection — run every N frames to save CPU
TAG_DETECT_INTERVAL = 3

# Tag hover
TAG_MIN_AREA = 800
TAG_HOVER_TIME = 2.0
TAG_APPROACH_SLOW = 0.08
# pixels around detected tag polygon (smaller to avoid over-cutting the line)
TAG_MASK_PAD = 8
# keep masking last seen tag briefly when detector misses a frame
TAG_MASK_HOLD_FRAMES = 5

# Filter side-fence false positives
BORDER_MARGIN_PX = 10
EDGE_PENALTY_PX = 28


# ─────────────────────────────────────────────────────────────────────────────
# Navigation State Machine
# ─────────────────────────────────────────────────────────────────────────────

class NavState(Enum):
    LINE_FOLLOW = auto()   # Following the yellow line
    TAG_HOVER = auto()   # Tag detected — hovering in place, reading tag
    DECIDING = auto()   # Decoded tag, deciding action
    PRE_LAND_ALIGN = auto()  # Fine centering over target tag before LAND
    TAG_REACQUIRE = auto()   # Skipped tag — creeping forward to re-find line
    LANDING = auto()   # Landing at target airport
    DONE = auto()   # Mission complete


# ─────────────────────────────────────────────────────────────────────────────
# MAVLink
# ─────────────────────────────────────────────────────────────────────────────

def connect(port=14550):
    m = mavutil.mavlink_connection(f"udp:0.0.0.0:{port}")
    print("[MAVLink] Waiting for heartbeat ...")
    m.wait_heartbeat()
    print(f"[MAVLink] Connected — system {m.target_system}")
    return m


def set_mode(m, mode):
    mid = m.mode_mapping()[mode]
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid)


def is_armed(m):
    msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    return bool(msg and msg.base_mode &
                mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def arm_and_takeoff(m, alt):
    set_mode(m, "GUIDED")
    print("[MAVLink] Arming ...")
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0)

    deadline = time.time() + 15
    while time.time() < deadline:
        if is_armed(m):
            print("[MAVLink] Armed ✅")
            break
        time.sleep(0.3)

    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)
    print(f"[MAVLink] Taking off to {alt} m ...")

    last_log = 0
    deadline = time.time() + 35
    while time.time() < deadline:
        msg = m.recv_match(type="GLOBAL_POSITION_INT",
                           blocking=True, timeout=2.0)
        if msg is None:
            continue
        cur = msg.relative_alt / 1000.0
        if time.time() - last_log > 0.5:
            print(f"  alt: {cur:.2f} m")
            last_log = time.time()
        if cur >= alt - 0.35:
            print(f"[MAVLink] Reached {cur:.2f} m ✅")
            return
    print("[MAVLink] Takeoff timeout — continuing")


def send_velocity(m, vx=0, vy=0, vz=0, yaw_rate=0):
    """Body-frame velocity command."""
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b010111000111,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate)


def get_alt(m):
    msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    return (msg.relative_alt / 1000.0) if msg else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def connect_camera(host="127.0.0.1", port=5599):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.settimeout(0.2)
    print(f"[Camera] Connected to {host}:{port}")
    return s


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        try:
            chunk = s.recv(n - len(buf))
        except socket.timeout:
            raise TimeoutError
        if not chunk:
            raise ConnectionError
        buf += chunk
    return buf


def get_frame(s):
    w, h = struct.unpack("<HH", recv_exact(s, 4))
    return np.frombuffer(recv_exact(s, w * h),
                         dtype=np.uint8).reshape((h, w))


# ─────────────────────────────────────────────────────────────────────────────
# Mask + contour filter
# ─────────────────────────────────────────────────────────────────────────────

def make_mask(gray, thresh):
    _, m = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # Bridge vertical gaps (like AprilTag stand occlusions) without
    # fattening the line too much horizontally
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 21))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_bridge)
    return m


def heal_line_after_tag_mask(mask):
    """
    Re-bridge short gaps created when a tag region is cut out from the mask.
    Keep this conservative to avoid joining unrelated blobs.
    """
    k_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 15))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_v)
    return m


def mask_out_tags(mask, tag_detections, roi_y_offset, pad=TAG_MASK_PAD):
    """
    Black out regions around detected AprilTags in the binary mask.
    This prevents tag white squares from being detected as the line.

    Args:
        mask:            binary mask (ROI coordinates)
        tag_detections:  list of TagResult (full-frame coordinates)
        roi_y_offset:    how many pixels the ROI is offset from the top of the frame
        pad:             extra pixels to black out around each tag
    """
    h, w = mask.shape[:2]
    for tag in tag_detections:
        # Convert tag corners from full-frame coords to ROI coords
        poly = tag.corners.copy()
        poly[:, 1] -= roi_y_offset

        # Skip if polygon is entirely outside ROI vertically
        if np.all(poly[:, 1] < 0) or np.all(poly[:, 1] >= h):
            continue

        # Mask polygon + a small dilation margin.
        # Using the polygon (instead of a big bbox) avoids cutting long
        # chunks of the yellow line at tag/line joinings.
        poly_i = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        tmp = np.zeros_like(mask)
        cv2.fillConvexPoly(tmp, poly_i, 255)

        if pad > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * pad + 1, 2 * pad + 1)
            )
            tmp = cv2.dilate(tmp, k, iterations=1)

        mask[tmp > 0] = 0

    return mask


def keep_line_contour(mask, prev_cx=None, min_area=120, min_thickness=12,
                      border_margin=BORDER_MARGIN_PX,
                      edge_penalty_margin=EDGE_PENALTY_PX):
    """
    Keep only the contour most likely to be the guiding line.

    The yellow line is physically WIDER than the fence/boundary.
    We simply reject any contour whose narrow side (from minAreaRect)
    is thinner than `min_thickness` pixels — that's the fence.

    Among remaining candidates, use spatial continuity (prev_cx)
    to avoid jumping between two visible lines at junctions.

    Returns (filtered_mask, chosen_centroid_x).
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, prev_cx

    def _filter_contours(cnts, thickness_limit):
        valid = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                narrow_side = min(rect[1])
                if narrow_side < thickness_limit:
                    continue
            valid.append(cnt)
        return valid

    # Try with standard thickness limit first
    candidates = _filter_contours(contours, min_thickness)

    # Fallback for curves: if no candidates, try with a lower thickness limit
    if not candidates:
        candidates = _filter_contours(contours, 8)

    clean = np.zeros_like(mask)

    if not candidates:
        return clean, prev_cx

    desired_x = prev_cx if prev_cx is not None else (w / 2)

    def _stats(cnt):
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else (w / 2)

        pts = cnt.reshape(-1, 2)
        xs = pts[:, 0]
        ys = pts[:, 1]

        # Does this contour reach near the drone (bottom of ROI)?
        touches_bottom = np.any(ys >= h - 3)

        # Bottom-most x location (more useful than global centroid at junctions)
        bot_xs = xs[ys >= h - 6]
        bottom_x = float(np.mean(bot_xs)) if len(bot_xs) else float(cx)

        near_side_edge = (
            bottom_x <= edge_penalty_margin or
            bottom_x >= (w - edge_penalty_margin)
        )

        x1, _, bw, _ = cv2.boundingRect(cnt)
        hugs_border = (x1 <= border_margin) or (
            (x1 + bw) >= (w - border_margin))

        return float(cx), touches_bottom, bottom_x, near_side_edge, hugs_border

    def _score(cnt):
        cx, touches_bottom, bottom_x, near_side_edge, hugs_border = _stats(cnt)

        # Continuity with previous track
        continuity = 0.75 * abs(cx - desired_x) + 0.25 * \
            abs(bottom_x - desired_x)

        # Penalize fence-like candidates (side-edge + border hugging)
        penalty = 0.0
        if not touches_bottom:
            penalty += 30.0
        if near_side_edge:
            penalty += 65.0
        if hugs_border:
            penalty += 35.0

        # Slight reward for larger (more stable) contour
        area_reward = min(cv2.contourArea(cnt) / 250.0, 20.0)

        return continuity + penalty - area_reward

    best = min(candidates, key=_score)

    # Compute centroid of the chosen contour for next-frame continuity
    M = cv2.moments(best)
    new_cx = M["m10"] / M["m00"] if M["m00"] > 0 else prev_cx

    cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)
    return clean, new_cx


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dead_zone(value, dead_zone):
    """Return 0 if |value| < dead_zone, else the value."""
    return 0.0 if abs(value) < dead_zone else value


def ema(prev, current, alpha):
    """Exponential moving average: alpha=1 → no smoothing, alpha→0 → heavy."""
    return alpha * current + (1.0 - alpha) * prev


# ─────────────────────────────────────────────────────────────────────────────
# Tag area helper
# ─────────────────────────────────────────────────────────────────────────────

def tag_corner_area(tag: TagResult) -> float:
    """Compute the area enclosed by the tag corners using the shoelace formula."""
    c = tag.corners
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def tag_align_command(tag: TagResult,
                      frame_shape,
                      target_offset_x=0.0,
                      target_offset_y=0.0,
                      deadzone_x=15.0,
                      deadzone_y=15.0,
                      max_vel=0.15):
    """
    Compute body-frame alignment velocity commands to center a detected tag.
    Returns: (vx, vy, err_x, err_y, centered)
    """
    h, w = frame_shape[:2]
    cx, cy = tag.center

    target_x = (w / 2) + target_offset_x
    target_y = (h / 2) + target_offset_y

    # +err_x means tag is right of target center -> move right (vy > 0)
    err_x = cx - target_x
    # +err_y means tag is lower in image -> move backward (vx < 0)
    err_y = cy - target_y

    vx = -err_y * TAG_ALIGN_P_FWD
    vy = err_x * TAG_ALIGN_P_LAT

    vx = float(np.clip(vx, -max_vel, max_vel))
    vy = float(np.clip(vy, -max_vel, max_vel))

    if abs(err_y) < deadzone_y:
        vx = 0.0
    if abs(err_x) < deadzone_x:
        vy = 0.0

    centered = abs(err_x) <= deadzone_x and abs(err_y) <= deadzone_y
    return vx, vy, float(err_x), float(err_y), centered


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def draw_ann(roi, mask, result, frame_w, error, vy, yr, nav_state,
             following, aligning, creep, thresh, alt, fps,
             tag_info_str, targets_remaining):
    h, w = roi.shape[:2]
    cx = w // 2
    vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Center dashed line
    for y in range(0, h, 12):
        cv2.line(vis, (cx, y), (cx, min(y+6, h)), (255, 255, 0), 1)

    if result.is_detected:
        # Blue slice dots
        for sx, sy in result.slice_points:
            cv2.circle(vis, (int(sx), int(sy)), 4, (255, 165, 0), -1)
        if len(result.slice_points) >= 2:
            pts = np.array(result.slice_points, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 165, 0), 1)

        # Green centroid line + dot
        dx = int(result.centroid_x)
        cv2.line(vis,   (dx, 0), (dx, h), (0, 255, 0), 2)
        cv2.circle(vis, (dx, h-10), 7, (0, 255, 0), -1)

        # Red lateral error arrow
        cv2.arrowedLine(vis, (cx, h//2), (dx, h//2),
                        (0, 0, 255), 2, tipLength=0.25)

        # Magenta angle line
        ap = int(math.tan(math.radians(result.angle_deg)) * h // 2)
        cv2.line(vis, (cx-ap, 0), (cx+ap, h), (255, 68, 255), 1)
    else:
        cv2.putText(vis, "NO LINE", (w//2-50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # HUD
    state_str = nav_state.name
    if not following:
        state_str = "[CREEP]" if creep else "[MANUAL]"
    elif aligning:
        state_str = "[ALIGNING]"

    lines = [
        f"Alt:{alt:.2f}m  FPS:{fps:.1f}  [{state_str}]",
        f"Thresh:{thresh}  Conf:{result.confidence:.2f}",
        f"LateralErr:{error:+.1f}px  Angle:{result.angle_deg:+.1f}deg",
        f"vy={vy:+.3f} m/s   yaw={yr:+.3f} r/s",
        f"Targets remaining: {targets_remaining}",
    ]
    if tag_info_str:
        lines.append(f"TAG: {tag_info_str}")
    lines.append("F=follow  C=creep  T=thresh  0=stop  Q=land")

    for i, line in enumerate(lines):
        y = 16 + i * 16
        cv2.putText(vis, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 2)
        cv2.putText(vis, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1)

    return vis


def draw_tags_on_frame(vis, tags):
    """Draw detected AprilTag outlines and IDs on an image."""
    for tag in tags:
        corners = tag.corners.astype(int)
        # Draw tag outline
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
        # Draw center dot
        cx, cy = int(tag.center[0]), int(tag.center[1])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        # Draw tag info
        label = f"ID:{tag.tag_id} C:{tag.country_code} S:{tag.airport_status}"
        cv2.putText(vis, label, (cx - 40, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    master = connect()
    arm_and_takeoff(master, ALTITUDE)
    print("[Test] Stabilizing 2 s ...")
    time.sleep(2.0)

    cam = connect_camera()
    last_frm = None

    # Perception modules
    detector = LineDetector(LineDetectorConfig(
        strategy=Strategy.SLIDING_WINDOW,
        num_slices=6,
        min_pixels=8,
    ))
    tag_detector = AprilTagDetector(quad_decimate=1.0, nthreads=2)

    cv2.namedWindow("Annotated ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary mask",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("Full frame",    cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotated ROI", 520, 340)
    cv2.resizeWindow("Binary mask",   520, 340)
    cv2.resizeWindow("Full frame",    520, 340)
    cv2.moveWindow("Annotated ROI",   0,   40)
    cv2.moveWindow("Binary mask",   540,   40)
    cv2.moveWindow("Full frame",      0,  420)

    # ── State ─────────────────────────────────────────────────────────────
    following = False
    creep = False
    tuner_on = False
    thresh = [THRESHOLD]
    vy_cmd = 0.0
    yr_cmd = 0.0
    prev_yr = 0.0            # for slew rate limiting
    prev_angle = 0.0         # for derivative term (yaw)
    prev_error = 0.0         # for derivative term (lateral)
    aligning = False
    fps = 0.0
    t_prev = time.time()
    t_ctrl = time.time()     # for time-normalized derivatives
    line_lost_count = 0      # frames since line was last seen
    frame_count = 0
    alt = ALTITUDE

    # Smoothed values
    smooth_angle = 0.0
    smooth_error = 0.0
    prev_line_cx = None   # for spatial continuity at junctions

    # Navigation state
    nav_state = NavState.LINE_FOLLOW
    tag_info_str = ""
    current_tag: TagResult | None = None
    tag_hover_start = 0.0
    pre_land_start = 0.0
    pre_land_lock_count = 0
    targets_remaining = list(TARGET_COUNTRIES)  # mutable copy
    visited_tags = set()    # tag IDs we've already processed
    last_tag_detections = []  # for display
    tag_mask_memory = []      # recent detections used only for short mask hold
    tag_mask_hold = 0         # remaining frames to keep using tag_mask_memory

    class _Empty:
        confidence = 0.0
        angle_deg = 0.0
        is_detected = False
        slice_points = []
        centroid_x = 160.0
        def lateral_error(self, w): return 0.0

    result = _Empty()
    error = 0.0
    blank = np.zeros((200, 320, 3), dtype=np.uint8)

    print("\n[Test] Ready.")
    print(f"  Target countries: {TARGET_COUNTRIES}")
    print("  1. Press C → creep until line is visible")
    print("  2. Press T → drag threshold until ONLY line is white")
    print("  3. Press 0 → hover over the line")
    print("  4. Press F → start following\n")

    while True:
        # ── Velocity output ───────────────────────────────────────────────
        if following and nav_state not in (NavState.LANDING, NavState.DONE):
            if result.is_detected:
                line_lost_count = 0  # reset loss counter
                # ── Apply dead-zone ───────────────────────────────────────
                angle_for_ctrl = apply_dead_zone(
                    result.angle_deg, YAW_DEAD_ZONE)
                lat_for_ctrl = apply_dead_zone(error, LAT_DEAD_ZONE)

                # ── EMA smoothing ─────────────────────────────────────────
                smooth_angle = ema(smooth_angle, angle_for_ctrl, EMA_ALPHA)
                smooth_error = ema(smooth_error, lat_for_ctrl, EMA_ALPHA)

                # ── Time delta for derivatives ────────────────────────────
                now = time.time()
                dt = max(now - t_ctrl, 0.01)  # clamp to avoid divide-by-zero
                t_ctrl = now

                # ── PD yaw controller (time-normalized D) ─────────────────
                p_yaw = KP_YAW * smooth_angle
                d_yaw = KD_YAW * (smooth_angle - prev_angle) / dt
                prev_angle = smooth_angle

                angle_abs = abs(smooth_angle)
                dynamic_max_yaw = HIGH_BEND_MAX_YAW if angle_abs > HIGH_BEND_ANGLE else MAX_YAW
                raw_yr = float(
                    np.clip(p_yaw + d_yaw, -dynamic_max_yaw, dynamic_max_yaw))

                # ── Slew rate limiter ─────────────────────────────────────
                dynamic_slew = HIGH_BEND_SLEW if angle_abs > HIGH_BEND_ANGLE else YAW_SLEW
                yr_cmd = float(np.clip(
                    raw_yr,
                    prev_yr - dynamic_slew,
                    prev_yr + dynamic_slew))
                prev_yr = yr_cmd

                # ── PD lateral controller (time-normalized D) ─────────────
                p_lat = KP_LAT * smooth_error
                d_lat = KD_LAT * (smooth_error - prev_error) / dt
                prev_error = smooth_error

                vy_cmd = float(np.clip(p_lat + d_lat, -MAX_LAT, MAX_LAT))

                # At sharp turns, prioritize heading over lateral correction
                if angle_abs > HIGH_BEND_ANGLE:
                    vy_cmd *= 0.4

                # ── Determine forward speed based on nav state ────────────
                # (TAG_HOVER and TAG_REACQUIRE handled separately below)
                # Slow down at sharp bends so drone has time to turn
                if angle_abs > BEND_SLOW_ANGLE:
                    # Scale from 100% at BEND_SLOW_ANGLE to 40% at 90°
                    scale = max(
                        0.4, 1.0 - (angle_abs - BEND_SLOW_ANGLE) / 90.0)
                    fwd_speed = FORWARD_SPEED * scale
                else:
                    fwd_speed = FORWARD_SPEED

                # ── Align phase ───────────────────────────────────────────
                if aligning:
                    if abs(result.angle_deg) < ALIGN_THRESHOLD_DEG:
                        aligning = False
                        print(
                            f"[Align] Done — angle={result.angle_deg:+.1f}° → moving forward")
                    else:
                        send_velocity(master, vx=0, vy=0, yaw_rate=yr_cmd)
                else:
                    send_velocity(master,
                                  vx=fwd_speed,
                                  vy=vy_cmd,
                                  yaw_rate=yr_cmd)
            else:
                # Lost line — keep searching by yawing in last direction
                line_lost_count += 1
                if line_lost_count <= LINE_LOSS_GRACE:
                    # Continue yawing in last known direction, stop fwd motion
                    # tiny forward creep helps pass brief occlusions at sharp bends
                    send_velocity(master, vx=0.03, vy=0, yaw_rate=yr_cmd)
                else:
                    # Too long without line — full stop
                    send_velocity(master)
                    vy_cmd = yr_cmd = prev_yr = prev_angle = prev_error = 0.0
                    smooth_angle = smooth_error = 0.0
                    t_ctrl = time.time()

        elif nav_state == NavState.TAG_HOVER:
            # Hovering — active visual servoing to center the tag
            if current_tag:
                shape_ref = frame.shape if 'frame' in locals(
                ) and frame is not None else last_frm.shape
                vx_align, vy_align, _, _, _ = tag_align_command(
                    current_tag,
                    shape_ref,
                    target_offset_x=0.0,
                    target_offset_y=0.0,
                    deadzone_x=TAG_ALIGN_DEADZONE,
                    deadzone_y=TAG_ALIGN_DEADZONE,
                    max_vel=TAG_ALIGN_MAX_VEL,
                )

                send_velocity(master, vx=vx_align, vy=vy_align, yaw_rate=0.0)
            else:
                send_velocity(master)

        elif nav_state == NavState.PRE_LAND_ALIGN:
            # Fine centering before LAND mode
            if current_tag:
                shape_ref = frame.shape if 'frame' in locals(
                ) and frame is not None else last_frm.shape
                vx_align, vy_align, err_x, err_y, centered = tag_align_command(
                    current_tag,
                    shape_ref,
                    target_offset_x=PRE_LAND_OFFSET_X,
                    target_offset_y=PRE_LAND_OFFSET_Y,
                    deadzone_x=PRE_LAND_DEADZONE_X,
                    deadzone_y=PRE_LAND_DEADZONE_Y,
                    max_vel=PRE_LAND_MAX_VEL,
                )

                if centered:
                    pre_land_lock_count += 1
                else:
                    pre_land_lock_count = max(0, pre_land_lock_count - 1)

                tag_info_str = (
                    f"Pre-land align ex={err_x:+.1f}px ey={err_y:+.1f}px "
                    f"lock={pre_land_lock_count}/{PRE_LAND_LOCK_FRAMES}"
                )
                send_velocity(master, vx=vx_align, vy=vy_align, yaw_rate=0.0)
            else:
                send_velocity(master)

        elif nav_state == NavState.TAG_REACQUIRE:
            # Creeping forward slowly to re-find the line after skipping a tag
            send_velocity(master, vx=SLOW_SPEED)

        elif creep:
            send_velocity(master, vx=0.15)

        # ── Read frame ────────────────────────────────────────────────────
        try:
            frame = get_frame(cam)
            last_frm = frame
        except TimeoutError:
            frame = last_frm
        except ConnectionError as e:
            print(f"[Camera] {e}")
            break

        if frame is None:
            time.sleep(0.01)
            continue

        # ── Perception: AprilTag detection (on full frame, BEFORE line) ────
        frame_w = frame.shape[1]
        roi_y = int(frame.shape[0] * ROI_TOP_FRAC)

        # Detect tags: periodically normally, or EVERY frame while aligning/hovering
        if (frame_count % TAG_DETECT_INTERVAL == 0) or (nav_state in (NavState.TAG_HOVER, NavState.PRE_LAND_ALIGN)):
            last_tag_detections = tag_detector.detect(frame)
            if last_tag_detections:
                tag_mask_memory = list(last_tag_detections)
                tag_mask_hold = TAG_MASK_HOLD_FRAMES
                best_tag = max(last_tag_detections, key=tag_corner_area)
                area = tag_corner_area(best_tag)

                # State machine: detect → TAG_HOVER immediately
                if following and nav_state == NavState.LINE_FOLLOW:
                    if best_tag.tag_id not in visited_tags:
                        nav_state = NavState.TAG_HOVER
                        current_tag = best_tag
                        tag_hover_start = time.time()
                        tag_info_str = str(best_tag)
                        # Stop all motion immediately
                        send_velocity(master)
                        prev_yr = 0.0
                        prev_angle = 0.0
                        prev_error = 0.0
                        smooth_angle = smooth_error = 0.0
                        t_ctrl = time.time()
                        print(f"\n[Nav] Tag detected! {best_tag}")
                        print(
                            f"[Nav] → TAG_HOVER (area={area:.0f}, hovering for {TAG_HOVER_TIME}s)")

                elif nav_state in (NavState.TAG_HOVER, NavState.PRE_LAND_ALIGN):
                    # Update tag detection while hovering / pre-landing align
                    current_tag = best_tag

        # ── Perception: Line detection (with tag regions masked out) ──────
        roi = frame[roi_y:, :]
        mask = make_mask(roi, thresh[0])

        # Black out tag regions so they don't confuse line detection.
        # If detector briefly misses, keep using recent tag corners for a few frames.
        tags_for_mask = last_tag_detections
        if not tags_for_mask and tag_mask_hold > 0 and tag_mask_memory:
            tags_for_mask = tag_mask_memory
            tag_mask_hold -= 1

        if tags_for_mask:
            mask = mask_out_tags(mask, tags_for_mask,
                                 roi_y, pad=TAG_MASK_PAD)
            mask = heal_line_after_tag_mask(mask)
        else:
            tag_mask_hold = 0
            tag_mask_memory = []

        mask, prev_line_cx = keep_line_contour(mask, prev_cx=prev_line_cx)

        result = detector.detect(mask)
        error = result.lateral_error(frame_w)
        alt = get_alt(master)

        # (AprilTag detection already ran above, before line detection)

        # ── State machine: TAG_HOVER → DECIDING ──────────────────────────
        if nav_state == NavState.TAG_HOVER:
            elapsed = time.time() - tag_hover_start
            if elapsed >= TAG_HOVER_TIME:
                # Re-read tag one more time for best accuracy
                final_tags = tag_detector.detect(frame)
                if final_tags:
                    current_tag = max(final_tags, key=tag_corner_area)

                if current_tag is not None:
                    nav_state = NavState.DECIDING
                    tag_info_str = str(current_tag)
                    print(f"[Nav] → DECIDING: {current_tag}")
                else:
                    # Lost the tag — go back to line following
                    nav_state = NavState.LINE_FOLLOW
                    print("[Nav] Lost tag during read — back to LINE_FOLLOW")

        # ── State machine: DECIDING ──────────────────────────────────────
        if nav_state == NavState.DECIDING and current_tag is not None:
            visited_tags.add(current_tag.tag_id)

            if (current_tag.country_code in targets_remaining
                    and current_tag.is_landable):
                # This is a target airport — first do precise centering, then land.
                print(
                    f"[Nav] ✅ MATCH! Country {current_tag.country_code} is landable!")
                nav_state = NavState.PRE_LAND_ALIGN
                pre_land_start = time.time()
                pre_land_lock_count = 0
                tag_info_str = f"Target {current_tag.tag_id}: fine-aligning before landing"
                print("[Nav] → PRE_LAND_ALIGN")
            else:
                # Not a match — skip and creep forward to re-find line
                reason = "wrong country" if current_tag.country_code not in targets_remaining else "unsafe"
                print(
                    f"[Nav] ❌ Skip airport — {reason} (tag {current_tag.tag_id})")
                nav_state = NavState.TAG_REACQUIRE
                tag_info_str = f"Skipped tag {current_tag.tag_id} ({reason}) — finding line..."
                current_tag = None
                print("[Nav] → TAG_REACQUIRE (creeping forward to find line)")

        # ── State machine: PRE_LAND_ALIGN ───────────────────────────────
        if nav_state == NavState.PRE_LAND_ALIGN:
            # If centered stably for several frames, commit to LAND.
            if pre_land_lock_count >= PRE_LAND_LOCK_FRAMES and current_tag is not None:
                print(
                    f"[Nav] ✅ Pre-land alignment locked ({pre_land_lock_count} frames)")
                print(f"[Nav] → LANDING")
                nav_state = NavState.LANDING
                send_velocity(master)
                time.sleep(0.4)
                set_mode(master, "LAND")

                # Wait for landing
                print("[Nav] Landing ...")
                deadline = time.time() + 30
                while time.time() < deadline:
                    if not is_armed(master):
                        break
                    time.sleep(0.5)

                # Guard against duplicate remove if mode/logic re-enters
                if current_tag.country_code in targets_remaining:
                    targets_remaining.remove(current_tag.country_code)
                print(f"[Nav] Landed! Targets remaining: {targets_remaining}")

                if not targets_remaining:
                    nav_state = NavState.DONE
                    print("[Nav] 🎉 ALL TARGETS REACHED — MISSION COMPLETE!")
                    tag_info_str = "MISSION COMPLETE!"
                else:
                    # Re-takeoff and continue
                    print("[Nav] Re-taking off to continue mission ...")
                    arm_and_takeoff(master, ALTITUDE)
                    time.sleep(2.0)
                    nav_state = NavState.LINE_FOLLOW
                    following = True
                    aligning = True
                    prev_yr = 0.0
                    prev_angle = 0.0
                    prev_error = 0.0
                    smooth_angle = smooth_error = 0.0
                    t_ctrl = time.time()
                    tag_info_str = f"Continuing — targets: {targets_remaining}"
                    print(f"[Nav] → LINE_FOLLOW, targets: {targets_remaining}")

            # If we can't lock alignment quickly, restart hover-read cycle.
            elif (time.time() - pre_land_start) > PRE_LAND_MAX_ALIGN_TIME:
                print("[Nav] Pre-land alignment timeout — retrying TAG_HOVER")
                nav_state = NavState.TAG_HOVER
                tag_hover_start = time.time()
                pre_land_lock_count = 0

        # ── State machine: TAG_REACQUIRE → LINE_FOLLOW ───────────────────
        if nav_state == NavState.TAG_REACQUIRE:
            if result.is_detected:
                # Line found again!
                nav_state = NavState.LINE_FOLLOW
                following = True
                aligning = True
                prev_yr = 0.0
                prev_angle = 0.0
                prev_error = 0.0
                smooth_angle = smooth_error = 0.0
                t_ctrl = time.time()
                prev_line_cx = None
                tag_info_str = f"Line re-acquired — targets: {targets_remaining}"
                print("[Nav] Line re-acquired! → LINE_FOLLOW")

        # ── FPS + log ─────────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            t_now = time.time()
            fps = 15 / (t_now - t_prev + 1e-6)
            t_prev = t_now
        if following and frame_count % 20 == 0:
            print(f"  [{nav_state.name}] conf={result.confidence:.2f}  "
                  f"err={error:+5.1f}px  "
                  f"ang={result.angle_deg:+5.1f}°  "
                  f"vy={vy_cmd:+.3f}  yr={yr_cmd:+.3f}")

        # ── Display ───────────────────────────────────────────────────────
        ann = draw_ann(roi, mask, result, frame_w, error,
                       vy_cmd, yr_cmd, nav_state,
                       following, aligning, creep,
                       thresh[0], alt, fps,
                       tag_info_str, len(targets_remaining))
        msk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Full frame with tag overlay
        full_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if last_tag_detections:
            draw_tags_on_frame(full_vis, last_tag_detections)

        cv2.imshow("Annotated ROI", ann)
        cv2.imshow("Binary mask",   msk)
        cv2.imshow("Full frame",    full_vis)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            send_velocity(master)
            time.sleep(0.3)
            set_mode(master, "LAND")
            print("[Test] Landing ...")
            break

        elif key == ord("f"):
            following = not following
            if following:
                creep = False
                aligning = True
                prev_yr = 0.0
                prev_angle = 0.0
                prev_error = 0.0
                smooth_angle = smooth_error = 0.0
                t_ctrl = time.time()
                nav_state = NavState.LINE_FOLLOW
                print(f"\n[Follow] ON — aligning to line first ...")
                print(f"  KP_YAW={KP_YAW}  KD_YAW={KD_YAW}  "
                      f"KP_LAT={KP_LAT}  speed={FORWARD_SPEED} m/s\n")
            else:
                send_velocity(master)
                aligning = False
                prev_yr = 0.0
                prev_angle = 0.0
                prev_error = 0.0
                smooth_angle = smooth_error = 0.0
                t_ctrl = time.time()
                print("[Follow] OFF")

        elif key == ord("c") and not following:
            creep = not creep
            if not creep:
                send_velocity(master)
            print(f"[Creep] {'ON' if creep else 'OFF'}")

        elif key == ord("t") and not tuner_on:
            cv2.createTrackbar("Threshold", "Binary mask",
                               thresh[0], 255,
                               lambda v: thresh.__setitem__(0, v))
            print("[Tuner] Threshold slider ON")
            tuner_on = True

        elif key == ord("0"):
            following = creep = False
            nav_state = NavState.LINE_FOLLOW
            send_velocity(master)
            print("[Manual] Stopped")

    cam.close()
    cv2.destroyAllWindows()
    print("[Test] Done.")


if __name__ == "__main__":
    run()
