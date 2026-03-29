"""
Task/flight.py

Autonomous drone line follower with AprilTag-based airport navigation.
Includes dynamic sensor hand-off, active search, anti-U-turn routing,
smart fence rejection, and dynamic junction stiffening.

Changes vs previous version:
  - Auto-starts line following after takeoff (no manual F key needed)
  - Uses full frame for line detection until the line is held for
    ROI_ACQUIRE_TIME seconds, then switches to the normal ROI crop
  - Increased forward/lateral speeds and gain for faster navigation
"""

from perception.line_detector import LineDetector, LineDetectorConfig, Strategy
from perception.apriltag_detector import AprilTagDetector, TagResult
from navigation.mission_planner import MissionPlanner
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

Airports = [1, 2]   # ← set these to the required country codes


# ─────────────────────────────────────────────────────────────────────────────
# TUNING
# ─────────────────────────────────────────────────────────────────────────────

FORWARD_SPEED   = 0.28      # was 0.15 — main cruise speed
SLOW_SPEED      = 0.15      # was 0.08 — used in TAG_REACQUIRE creep
CREEP_SPEED     = 0.20      # was 0.15 — manual creep key

KP_YAW = -0.020
KD_YAW = -0.008

KP_LAT = 0.0006             # was 0.0004 — more lateral authority at higher speed
KD_LAT = 0.0001

THRESHOLD    = 155
ROI_TOP_FRAC = 0.40
ALTITUDE     = 1.5

MAX_YAW = 0.50
MAX_LAT = 0.20              # was 0.12

YAW_SLEW = 0.15

HIGH_BEND_ANGLE   = 28.0
HIGH_BEND_MAX_YAW = 1.50
HIGH_BEND_SLEW    = 1.00

TAG_ALIGN_P_LAT   = 0.0015
TAG_ALIGN_P_FWD   = 0.0015
TAG_ALIGN_MAX_VEL = 0.12
TAG_ALIGN_DEADZONE = 25

PRE_LAND_OFFSET_X      = 0
PRE_LAND_OFFSET_Y      = 10
PRE_LAND_DEADZONE_X    = 40
PRE_LAND_DEADZONE_Y    = 40
PRE_LAND_LOCK_FRAMES   = 2
PRE_LAND_MAX_VEL       = 0.10
PRE_LAND_MAX_ALIGN_TIME = 20.0

YAW_DEAD_ZONE    = 2.0
LAT_DEAD_ZONE    = 5.0
EMA_ALPHA        = 0.40
ALIGN_THRESHOLD_DEG = 8.0
BEND_SLOW_ANGLE  = 15.0
LINE_LOSS_GRACE  = 15
TAG_DETECT_INTERVAL  = 3
TAG_MIN_AREA     = 800
TAG_HOVER_TIME   = 4.0
TAG_APPROACH_SLOW = 0.08
TAG_DETECT_CENTER_DEADZONE_X = 220
TAG_DETECT_CENTER_DEADZONE_Y = 170
TAG_MASK_PAD         = 8
TAG_MASK_HOLD_FRAMES = 5
BORDER_MARGIN_PX = 10
EDGE_PENALTY_PX  = 28

# ── ROI bootstrap ─────────────────────────────────────────────────────────────
# How many consecutive seconds of confident line following (confidence > this
# threshold) must pass before we narrow perception to the ROI crop.
ROI_ACQUIRE_TIME       = 4.0   # seconds
ROI_ACQUIRE_MIN_CONF   = 0.50  # minimum confidence to count toward the timer


# ─────────────────────────────────────────────────────────────────────────────
# Navigation State Machine
# ─────────────────────────────────────────────────────────────────────────────

class NavState(Enum):
    LINE_FOLLOW      = auto()
    TAG_HOVER        = auto()
    DECIDING         = auto()
    PRE_LAND_ALIGN   = auto()
    TAG_REACQUIRE    = auto()
    LANDING          = auto()
    POST_LAND_SEARCH = auto()
    DONE             = auto()


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
    return bool(msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


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
    else:
        raise RuntimeError("[MAVLink] Arming failed — aborting mission")

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
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b010111000111, 0, 0, 0,
        vx, vy, vz, 0, 0, 0, 0, yaw_rate)


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
    data = recv_exact(s, w * h * 3)
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Mask + contour filter
# ─────────────────────────────────────────────────────────────────────────────

def make_mask(bgr, _thresh_unused):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    m = cv2.inRange(hsv, lower_yellow, upper_yellow)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 21))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_bridge)
    return m

def keep_valid_contours(mask, min_area=120, min_thickness=12):
    """
    Filters out noise but keeps ALL valid line branches 
    so the graph algorithm can see junctions.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return mask

    def _filter_contours(cnts, thickness_limit):
        valid = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_area:
                continue
            if len(cnt) >= 5:
                if min(cv2.minAreaRect(cnt)[1]) < thickness_limit:
                    continue
            valid.append(cnt)
        return valid

    candidates = _filter_contours(contours, min_thickness)
    if not candidates:
        # Fallback to looser thickness constraint if needed
        candidates = _filter_contours(contours, 8)

    clean = np.zeros_like(mask)
    if candidates:
        # Draw ALL valid branches, not just the "best" one
        cv2.drawContours(clean, candidates, -1, 255, thickness=cv2.FILLED)
        
    return clean

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing & Tag helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_dead_zone(value, dead_zone):
    return 0.0 if abs(value) < dead_zone else value


def ema(prev, current, alpha):
    return alpha * current + (1.0 - alpha) * prev


def tag_corner_area(tag: TagResult) -> float:
    c = tag.corners
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def tag_align_command(tag: TagResult, frame_shape, target_offset_x=0.0,
                      target_offset_y=0.0, deadzone_x=15.0, deadzone_y=15.0, max_vel=0.15):
    h, w = frame_shape[:2]
    cx, cy = tag.center
    target_x = (w / 2) + target_offset_x
    target_y = (h / 2) + target_offset_y

    err_x = cx - target_x
    err_y = cy - target_y

    vx = float(np.clip(-err_y * TAG_ALIGN_P_FWD, -max_vel, max_vel))
    vy = float(np.clip(err_x * TAG_ALIGN_P_LAT, -max_vel, max_vel))

    if abs(err_y) < deadzone_y:
        vx = 0.0
    if abs(err_x) < deadzone_x:
        vy = 0.0

    centered = abs(err_x) <= deadzone_x and abs(err_y) <= deadzone_y
    return vx, vy, float(err_x), float(err_y), centered


def tag_is_centered(tag: TagResult, frame_shape, target_offset_x=0.0,
                    target_offset_y=0.0, deadzone_x=80.0, deadzone_y=80.0):
    h, w = frame_shape[:2]
    cx, cy = tag.center
    err_x = float(cx - ((w / 2) + target_offset_x))
    err_y = float(cy - ((h / 2) + target_offset_y))
    return abs(err_x) <= deadzone_x and abs(err_y) <= deadzone_y, err_x, err_y


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def draw_ann(roi, mask, result, frame_w, error, vy, yr, nav_state,
             following, aligning, creep, thresh, alt, fps,
             tag_info_str, targets_remaining, roi_active):
    h, w = roi.shape[:2]
    cx = w // 2
    vis = roi.copy()

    for y in range(0, h, 12):
        cv2.line(vis, (cx, y), (cx, min(y+6, h)), (255, 255, 0), 1)

    if result.junction_detected:
        cv2.putText(vis, "JUNCTION DETECTED", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for bx in result.branch_centroids:
            # Draw bright cyan circles at the split points to prove it sees all lines
            cv2.circle(vis, (int(bx), result.junction_y), 8, (255, 255, 0), 2)

    if result.is_detected:
        for sx, sy in result.slice_points:
            cv2.circle(vis, (int(sx), int(sy)), 4, (255, 165, 0), -1)
        if len(result.slice_points) >= 2:
            pts = np.array(result.slice_points, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 165, 0), 1)

        dx = int(result.centroid_x)
        cv2.line(vis,   (dx, 0), (dx, h), (0, 255, 0), 2)
        cv2.circle(vis, (dx, h-10), 7, (0, 255, 0), -1)
        cv2.arrowedLine(vis, (cx, h//2), (dx, h//2),
                        (0, 0, 255), 2, tipLength=0.25)
        ap = int(math.tan(math.radians(result.angle_deg)) * h // 2)
        cv2.line(vis, (cx-ap, 0), (cx+ap, h), (255, 68, 255), 1)
    else:
        cv2.putText(vis, "NO LINE", (w//2-50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    state_str = nav_state.name
    if not following:
        state_str = "[CREEP]" if creep else "[MANUAL]"
    elif aligning:
        state_str = "[ALIGNING]"

    roi_str = "ROI:ON" if roi_active else "ROI:SEARCH"
    lines = [
        f"Alt:{alt:.2f}m  FPS:{fps:.1f}  [{state_str}]  {roi_str}",
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
    for tag in tags:
        corners = tag.corners.astype(int)
        for i in range(4):
            cv2.line(vis, tuple(corners[i]), tuple(
                corners[(i + 1) % 4]), (0, 255, 255), 2)
        cx, cy = int(tag.center[0]), int(tag.center[1])
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
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

    detector = LineDetector(LineDetectorConfig(
        strategy=Strategy.SLIDING_WINDOW, num_slices=6, min_pixels=8))
    tag_detector = AprilTagDetector(quad_decimate=1.0, nthreads=2)

    planner = MissionPlanner(Airports)
    current_turn_bias = "straight"
    junction_cooldown = 0

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
    # Auto-start: drone begins searching for line immediately after takeoff.
    following  = True
    aligning   = True
    creep      = False
    tuner_on   = False
    thresh     = [THRESHOLD]
    vy_cmd     = 0.0
    yr_cmd     = 0.0
    prev_yr    = prev_angle = prev_error = 0.0
    smooth_angle = smooth_error = 0.0
    fps        = 0.0
    t_prev     = t_ctrl = time.time()
    line_lost_count = frame_count = 0
    alt        = ALTITUDE
    current_heading  = 0.0
    prev_line_cx     = None

    # ── ROI bootstrap state ───────────────────────────────────────────────
    # Start with full-frame line search. Once the line is held at sufficient
    # confidence for ROI_ACQUIRE_TIME seconds we narrow to the ROI crop.
    roi_active          = False
    line_acquired_since: float | None = None   # timestamp of first good detection

    nav_state        = NavState.LINE_FOLLOW
    tag_info_str     = ""
    current_tag: TagResult | None = None
    tag_hover_start  = pre_land_start = tag_reacquire_start = 0.0
    pre_land_lock_count = 0
    targets_remaining   = [c for c in Airports if c != 0]
    visited_tags        = set()
    last_tag_detections = []
    tag_mask_memory     = []
    tag_mask_hold       = 0
    arrival_heading     = 0.0

    # Sentinel used for the very first loop iteration before a real frame
    # arrives — keeps the velocity output section safe.
    class _Empty:
        confidence   = 0.0
        angle_deg    = 0.0
        is_detected  = False
        slice_points = []
        centroid_x   = 160.0
        def lateral_error(self, w): return 0.0

    result = _Empty()
    error  = 0.0
    roi    = np.zeros((200, 320, 3), dtype=np.uint8)  # safe initial value
    mask   = np.zeros((200, 320),    dtype=np.uint8)

    print("\n[Test] Ready — auto-starting line search.")
    print(f"  Target countries: {[c for c in Airports if c != 0]}")
    print("  F=follow  C=creep  T=thresh  0=stop  Q=land\n")

    while True:
        # ── Drain Telemetry ───────────────────────────────────────────────
        msg = master.recv_match(blocking=False)
        while msg:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                alt             = msg.relative_alt / 1000.0
                current_heading = msg.hdg / 100.0
            msg = master.recv_match(blocking=False)

        # ── Velocity & State Output ───────────────────────────────────────
        if following and nav_state not in (NavState.LANDING, NavState.DONE):

            if nav_state == NavState.LINE_FOLLOW:
                # Require at least 2 slices (0.33 conf) to trust the heading. 
                # Otherwise, treat it as noise/lost line.
                if result.is_detected and result.confidence >= 0.33:
                    line_lost_count = 0
                    angle_for_ctrl = apply_dead_zone(result.angle_deg, YAW_DEAD_ZONE)
                    lat_for_ctrl = apply_dead_zone(error, LAT_DEAD_ZONE)

                    smooth_angle = ema(smooth_angle, angle_for_ctrl, EMA_ALPHA)
                    smooth_error = ema(smooth_error, lat_for_ctrl, EMA_ALPHA)

                    now  = time.time()
                    dt   = max(now - t_ctrl, 0.01)
                    t_ctrl = now

                    p_yaw  = KP_YAW * smooth_angle
                    d_yaw  = KD_YAW * (smooth_angle - prev_angle) / dt
                    err_yaw = 0.005 * smooth_error
                    prev_angle = smooth_angle

                    angle_abs = abs(smooth_angle)

                    is_approaching_tag = False
                    if last_tag_detections:
                        best_tag_check = max(last_tag_detections, key=tag_corner_area)
                        if best_tag_check.center[1] < (frame.shape[0] * 0.70):
                            is_approaching_tag = True

                    if is_approaching_tag:
                        dynamic_max_yaw = 0.15
                    else:
                        dynamic_max_yaw = HIGH_BEND_MAX_YAW if (
                            angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80) else MAX_YAW

                    raw_yr = float(np.clip(p_yaw + d_yaw + err_yaw, -dynamic_max_yaw, dynamic_max_yaw))

                    if is_approaching_tag:
                        dynamic_slew = 0.05
                    else:
                        dynamic_slew = HIGH_BEND_SLEW if (
                            angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80) else YAW_SLEW

                    yr_cmd = float(np.clip(raw_yr, prev_yr - dynamic_slew, prev_yr + dynamic_slew))
                    prev_yr = yr_cmd

                    p_lat = KP_LAT * smooth_error
                    d_lat = KD_LAT * (smooth_error - prev_error) / dt
                    prev_error = smooth_error

                    vy_cmd = float(np.clip(p_lat + d_lat, -MAX_LAT, MAX_LAT))
                    if angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80:
                        vy_cmd *= 0.4

                    if angle_abs > BEND_SLOW_ANGLE or abs(smooth_error) > 80:
                        scale_ang = max(0.4, 1.0 - (angle_abs - BEND_SLOW_ANGLE) / 90.0) if angle_abs > BEND_SLOW_ANGLE else 1.0
                        scale_lat = max(0.2, 1.0 - (abs(smooth_error) - 80) / 150.0) if abs(smooth_error) > 80 else 1.0
                        fwd_speed = FORWARD_SPEED * min(scale_ang, scale_lat)
                    else:
                        fwd_speed = FORWARD_SPEED

                    if aligning:
                        if abs(result.angle_deg) < ALIGN_THRESHOLD_DEG:
                            aligning = False
                            print(f"[Align] Done — angle={result.angle_deg:+.1f}° → moving forward")
                        else:
                            send_velocity(master, vx=0, vy=0, yaw_rate=yr_cmd)
                    else:
                        send_velocity(master, vx=fwd_speed, vy=vy_cmd, yaw_rate=yr_cmd)
                
                else:
                    # --- LINE LOST LOGIC ---
                    line_lost_count += 1
                    if last_tag_detections:
                        best_tag = max(last_tag_detections, key=tag_corner_area)
                        tag_cx = best_tag.center[0]
                        err_x  = tag_cx - (frame_w / 2)
                        send_velocity(master, vx=0.15, vy=0, yaw_rate=float(err_x * 0.0015))
                        tag_info_str = "Gap bridge: tracking tag"
                    elif line_lost_count <= LINE_LOSS_GRACE:
                        # HARDBRAKE: Drift slightly forward, but KILL yaw and lateral velocity immediately
                        send_velocity(master, vx=0.03, vy=0.0, yaw_rate=0.0)
                    else:
                        # FULL STOP & EXPAND ROI
                        send_velocity(master, vx=0.0, vy=0.0, yaw_rate=0.0)
                        
                        # Reset tracking state to prevent sudden jerks when line is found
                        vy_cmd = yr_cmd = prev_yr = prev_angle = prev_error = 0.0
                        smooth_angle = smooth_error = 0.0
                        t_ctrl = time.time()
                        
                        if roi_active:
                            print("[Nav] Line lost! Expanding to FULL FRAME search and halting.")
                            roi_active = False    # Expand vision
                            aligning = True       # Force alignment when found again

            elif nav_state == NavState.TAG_HOVER:
                if current_tag:
                    shape_ref = frame.shape if 'frame' in locals(
                    ) and frame is not None else last_frm.shape
                    vx_align, vy_align, _, _, _ = tag_align_command(
                        current_tag, shape_ref, deadzone_x=TAG_ALIGN_DEADZONE,
                        deadzone_y=TAG_ALIGN_DEADZONE, max_vel=TAG_ALIGN_MAX_VEL)
                    send_velocity(master, vx=vx_align,
                                  vy=vy_align, yaw_rate=0.0)
                else:
                    send_velocity(master)

            elif nav_state == NavState.PRE_LAND_ALIGN:
                if current_tag:
                    shape_ref = frame.shape if 'frame' in locals(
                    ) and frame is not None else last_frm.shape
                    vx_align, vy_align, err_x, err_y, centered = tag_align_command(
                        current_tag, shape_ref, target_offset_x=PRE_LAND_OFFSET_X,
                        target_offset_y=PRE_LAND_OFFSET_Y, deadzone_x=PRE_LAND_DEADZONE_X,
                        deadzone_y=PRE_LAND_DEADZONE_Y, max_vel=PRE_LAND_MAX_VEL)

                    if centered:
                        pre_land_lock_count += 1
                    else:
                        pre_land_lock_count = max(0, pre_land_lock_count - 1)

                    if frame_count % 10 == 0:
                        print(
                            f"  [PRE_LAND_ALIGN] err_x={err_x:+.1f}px err_y={err_y:+.1f}px "
                            f"vx={vx_align:+.3f} vy={vy_align:+.3f} lock={pre_land_lock_count}/{PRE_LAND_LOCK_FRAMES}")

                    tag_info_str = (f"Pre-land align ex={err_x:+.1f}px ey={err_y:+.1f}px "
                                    f"lock={pre_land_lock_count}/{PRE_LAND_LOCK_FRAMES}")
                    send_velocity(master, vx=vx_align,
                                  vy=vy_align, yaw_rate=0.0)
                else:
                    send_velocity(master)
                    pre_land_lock_count = max(0, pre_land_lock_count - 1)

            elif nav_state == NavState.TAG_REACQUIRE:
                if time.time() - tag_reacquire_start < 2.5:
                    send_velocity(master, vx=SLOW_SPEED, yaw_rate=0.0)
                    tag_info_str = "Creeping past airport..."
                else:
                    send_velocity(master, vx=0.0, yaw_rate=0.4)
                    tag_info_str = "Active sweep for line..."

            elif nav_state == NavState.POST_LAND_SEARCH:
                send_velocity(master, vx=0.0, yaw_rate=0.4)
                tag_info_str = "Sweeping for new outgoing line..."

        elif creep:
            send_velocity(master, vx=CREEP_SPEED)

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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_w    = frame.shape[1]
        roi_y      = int(frame.shape[0] * ROI_TOP_FRAC)

        # ── Perception: AprilTag ──────────────────────────────────────────
        if (frame_count % TAG_DETECT_INTERVAL == 0) or (nav_state in (NavState.TAG_HOVER, NavState.PRE_LAND_ALIGN)):
            last_tag_detections = tag_detector.detect(gray_frame)

            if not last_tag_detections and tag_mask_hold > 0:
                last_tag_detections = tag_mask_memory
                tag_mask_hold -= 1

            if last_tag_detections:
                tag_mask_memory = list(last_tag_detections)
                tag_mask_hold   = TAG_MASK_HOLD_FRAMES
                best_tag        = max(last_tag_detections, key=tag_corner_area)
                area            = tag_corner_area(best_tag)

                if following and nav_state == NavState.LINE_FOLLOW:
                    centered_for_capture, cap_ex, cap_ey = tag_is_centered(
                        best_tag, frame.shape, deadzone_x=TAG_DETECT_CENTER_DEADZONE_X,
                        deadzone_y=TAG_DETECT_CENTER_DEADZONE_Y)

                    if (best_tag.tag_id not in visited_tags and area > TAG_MIN_AREA and centered_for_capture):
                        nav_state       = NavState.TAG_HOVER
                        arrival_heading = current_heading
                        current_tag     = best_tag
                        tag_hover_start = time.time()
                        tag_info_str    = str(best_tag)
                        roi_active = False

                        send_velocity(master)
                        prev_yr = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                        t_ctrl  = time.time()
                        print(f"\n[Nav] Tag detected! {best_tag}")
                        print(
                            f"[Nav] → TAG_HOVER (area={area:.0f}, heading saved: {arrival_heading:.1f}°)")
                    elif best_tag.tag_id not in visited_tags and area > TAG_MIN_AREA:
                        tag_info_str = f"Tag seen but off-center ex={cap_ex:+.0f}px ey={cap_ey:+.0f}px"

                elif nav_state in (NavState.TAG_HOVER, NavState.PRE_LAND_ALIGN):
                    current_tag = best_tag
            else:
                if nav_state == NavState.TAG_HOVER:
                    current_tag = None

        # ── Perception: Line ──────────────────────────────────────────────
        # Before ROI is active use the full frame so the drone can find the
        # line from any position after takeoff. Once it has tracked
        # confidently for ROI_ACQUIRE_TIME seconds, narrow to the ROI crop.
        if roi_active:
            roi = frame[roi_y:, :]
        else:
            roi = frame           # full frame search

        mask = keep_valid_contours(make_mask(roi, thresh[0]))
        if junction_cooldown > 0:
            junction_cooldown -= 1
        else:
            current_turn_bias = "straight"

        result = detector.detect(mask, turn_bias=current_turn_bias)
        
        if result.junction_detected and following:
            current_turn_bias = planner.get_junction_decision(len(result.branch_centroids))
            result = detector.detect(mask, turn_bias=current_turn_bias) # Re-run with the forced bias
            junction_cooldown = 10 # Hold the turn bias for 10 frames to ensure it commits to the branch
            
        error  = result.lateral_error(frame_w)

        # ── ROI acquisition timer ─────────────────────────────────────────
        if not roi_active:
            if (nav_state == NavState.LINE_FOLLOW
                    and result.confidence >= ROI_ACQUIRE_MIN_CONF
                    and not aligning):
                if line_acquired_since is None:
                    line_acquired_since = time.time()
                    print("[Nav] Line acquired — starting ROI acquisition timer ...")
                elif time.time() - line_acquired_since >= ROI_ACQUIRE_TIME:
                    roi_active          = True
                    line_acquired_since = None
                    prev_line_cx        = None   # x-coords stay the same but reset for safety
                    print("[Nav] ✅ ROI activated — switching to ROI-based line following")
            else:
                # Reset timer if we lose the line or leave LINE_FOLLOW
                if line_acquired_since is not None:
                    line_acquired_since = None

        # ── State transitions logic ───────────────────────────────────────
        if nav_state == NavState.TAG_HOVER:
            if time.time() - tag_hover_start >= TAG_HOVER_TIME:
                final_tags = tag_detector.detect(gray_frame)
                if final_tags:
                    current_tag = max(final_tags, key=tag_corner_area)

                if current_tag is not None:
                    nav_state    = NavState.DECIDING
                    tag_info_str = str(current_tag)
                    print(f"[Nav] → DECIDING: {current_tag}")
                else:
                    nav_state = NavState.LINE_FOLLOW
                    print("[Nav] Lost tag during read — back to LINE_FOLLOW")

        if nav_state == NavState.DECIDING and current_tag is not None:
            should_land = planner.on_tag_reached(current_tag.tag_id, current_tag.country_code, current_tag.is_landable)
            
            if should_land:
                centered_now, ex_now, ey_now = tag_is_centered(
                    current_tag, frame.shape, target_offset_x=PRE_LAND_OFFSET_X,
                    target_offset_y=PRE_LAND_OFFSET_Y, deadzone_x=PRE_LAND_DEADZONE_X,
                    deadzone_y=PRE_LAND_DEADZONE_Y)
                print(f"[Nav] ✅ MATCH! Country {current_tag.country_code} is landable!")
                nav_state           = NavState.PRE_LAND_ALIGN
                pre_land_start      = time.time()
                pre_land_lock_count = PRE_LAND_LOCK_FRAMES if centered_now else 0
                tag_info_str        = f"Target {current_tag.tag_id}: fine-aligning before landing"
                print("[Nav] → PRE_LAND_ALIGN")
            else:
                reason = "wrong country" if current_tag.country_code not in targets_remaining else "unsafe"
                print(
                    f"[Nav] ❌ Skip airport — {reason} (tag {current_tag.tag_id})")
                nav_state           = NavState.TAG_REACQUIRE
                tag_reacquire_start = time.time()
                tag_info_str        = f"Skipped tag {current_tag.tag_id} ({reason}) — finding line..."
                current_tag         = None
                print("[Nav] → TAG_REACQUIRE (creeping forward to find line)")

        if nav_state == NavState.PRE_LAND_ALIGN:
            if pre_land_lock_count >= PRE_LAND_LOCK_FRAMES and current_tag is not None:
                print("[Nav] ✅ Pre-land alignment locked. → LANDING")
                nav_state = NavState.LANDING
                send_velocity(master)
                time.sleep(0.4)
                set_mode(master, "LAND")

                print("[Nav] Landing ...")
                deadline = time.time() + 30
                touchdown_time = None
                
                while time.time() < deadline:
                    # 1. Keep draining telemetry to get fresh altitude
                    msg = master.recv_match(blocking=False)
                    while msg:
                        if msg.get_type() == "GLOBAL_POSITION_INT":
                            alt = msg.relative_alt / 1000.0
                        msg = master.recv_match(blocking=False)

                    # 2. Keep camera buffer empty and display alive so it doesn't freeze
                    try:
                        cam_frame = get_frame(cam)
                        if cam_frame is not None:
                            last_frm = cam_frame
                            cv2.imshow("Full frame", cam_frame)
                            cv2.waitKey(1)
                    except:
                        pass
                    
                    # 3. Detect touchdown (altitude near ground) or auto-disarm
                    if (alt < 0.2 or not is_armed(master)) and touchdown_time is None:
                        print("[Nav] Touchdown detected! Waiting 7 seconds on the pad...")
                        touchdown_time = time.time()
                        
                    # 4. Break loop after 7 seconds on the ground
                    if touchdown_time is not None and (time.time() - touchdown_time) >= 7.0:
                        break
                        
                    time.sleep(0.05)

                if current_tag.country_code in targets_remaining:
                    targets_remaining.remove(current_tag.country_code)
                print(f"[Nav] Landed! Targets remaining: {targets_remaining}")
        
        # ── Line Re-acquisition Logic ────────────────────────────────────
        if following and result.is_detected:
            if nav_state == NavState.TAG_REACQUIRE:
                nav_state    = NavState.LINE_FOLLOW
                aligning     = True
                prev_yr      = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                t_ctrl       = time.time()                
                tag_info_str = f"Line re-acquired — targets: {targets_remaining}"
                print("[Nav] Line re-acquired! → LINE_FOLLOW")

            elif nav_state == NavState.POST_LAND_SEARCH:
                if result.confidence > 0.3:
                    return_hdg = (arrival_heading + 180.0) % 360.0
                    diff       = abs(current_heading - return_hdg)
                    if diff > 180:
                        diff = 360 - diff

                    if diff > 60:
                        nav_state    = NavState.LINE_FOLLOW
                        aligning     = True
                        prev_yr      = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                        t_ctrl       = time.time()
                        print(
                            f"[Nav] Valid outgoing line found at {current_heading:.0f}°! → LINE_FOLLOW")
                    else:
                        tag_info_str = f"Line found but ignoring return path ({current_heading:.0f}°)"

        # ── FPS & Display ─────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            t_now  = time.time()
            fps    = 15 / (t_now - t_prev + 1e-6)
            t_prev = t_now

        if following and frame_count % 20 == 0:
            if nav_state in (NavState.LINE_FOLLOW, NavState.TAG_REACQUIRE, NavState.POST_LAND_SEARCH):
                print(f"  [{nav_state.name}] hdg={current_heading:03.0f}° conf={result.confidence:.2f}  "
                      f"err={error:+5.1f}px  ang={result.angle_deg:+5.1f}°  vy={vy_cmd:+.3f}  yr={yr_cmd:+.3f}  "
                      f"{'[FULL]' if not roi_active else '[ROI]'}")

        ann = draw_ann(roi, mask, result, frame_w, error, vy_cmd, yr_cmd, nav_state,
                       following, aligning, creep, thresh[0], alt, fps,
                       tag_info_str, len(targets_remaining), roi_active)
        msk      = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        full_vis = frame.copy()
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
                creep        = False
                aligning     = True
                prev_yr      = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                t_ctrl       = time.time()
                nav_state    = NavState.LINE_FOLLOW
                print("\n[Follow] ON — aligning to line first ...")
            else:
                send_velocity(master)
                aligning = False
                print("[Follow] OFF")
        elif key == ord("c") and not following:
            creep = not creep
            if not creep:
                send_velocity(master)
            print(f"[Creep] {'ON' if creep else 'OFF'}")
        elif key == ord("t") and not tuner_on:
            cv2.createTrackbar("Threshold", "Binary mask",
                               thresh[0], 255, lambda v: thresh.__setitem__(0, v))
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