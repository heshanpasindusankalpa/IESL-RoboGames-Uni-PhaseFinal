"""
Task/line_follow_test.py

Autonomous drone line follower with AprilTag-based airport navigation.
Includes dynamic sensor hand-off, active search, anti-U-turn routing,
smart fence rejection, and dynamic junction stiffening.
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

Airports = [1, 2]   # ← set these to the required country codes


# ─────────────────────────────────────────────────────────────────────────────
# TUNING
# ─────────────────────────────────────────────────────────────────────────────

FORWARD_SPEED = 0.15
SLOW_SPEED = 0.08

KP_YAW = -0.025
KD_YAW = -0.004

KP_LAT = 0.0004
KD_LAT = 0.0001

THRESHOLD = 155
ROI_TOP_FRAC = 0.40
ALTITUDE = 1.5

MAX_YAW = 0.50
MAX_LAT = 0.12

YAW_SLEW = 0.20

HIGH_BEND_ANGLE = 28.0
HIGH_BEND_MAX_YAW = 1.50
HIGH_BEND_SLEW = 1.00

TAG_ALIGN_P_LAT = 0.0015
TAG_ALIGN_P_FWD = 0.0015
TAG_ALIGN_MAX_VEL = 0.12
TAG_ALIGN_DEADZONE = 25

PRE_LAND_OFFSET_X = 0
PRE_LAND_OFFSET_Y = 10
PRE_LAND_DEADZONE_X = 80
PRE_LAND_DEADZONE_Y = 80
PRE_LAND_LOCK_FRAMES = 2
PRE_LAND_MAX_VEL = 0.10
PRE_LAND_MAX_ALIGN_TIME = 20.0

YAW_DEAD_ZONE = 2.0
LAT_DEAD_ZONE = 5.0
EMA_ALPHA = 0.40
ALIGN_THRESHOLD_DEG = 8.0
BEND_SLOW_ANGLE = 15.0
LINE_LOSS_GRACE = 15
TAG_DETECT_INTERVAL = 3
TAG_MIN_AREA = 800
TAG_HOVER_TIME = 2.0
TAG_APPROACH_SLOW = 0.08
TAG_DETECT_CENTER_DEADZONE_X = 220
TAG_DETECT_CENTER_DEADZONE_Y = 170
TAG_MASK_PAD = 8
TAG_MASK_HOLD_FRAMES = 5
BORDER_MARGIN_PX = 10
EDGE_PENALTY_PX = 28


# ─────────────────────────────────────────────────────────────────────────────
# Navigation State Machine
# ─────────────────────────────────────────────────────────────────────────────

class NavState(Enum):
    LINE_FOLLOW = auto()
    TAG_HOVER = auto()
    DECIDING = auto()
    PRE_LAND_ALIGN = auto()
    TAG_REACQUIRE = auto()
    LANDING = auto()
    POST_LAND_SEARCH = auto()
    DONE = auto()


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


def keep_line_contour(mask, prev_cx=None, min_area=120, min_thickness=12,
                      border_margin=BORDER_MARGIN_PX, edge_penalty_margin=EDGE_PENALTY_PX):
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, prev_cx

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
        candidates = _filter_contours(contours, 8)

    clean = np.zeros_like(mask)
    if not candidates:
        return clean, prev_cx

    desired_x = prev_cx if prev_cx is not None else (w / 2)

    def _score(cnt):
        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else (w / 2)
        pts = cnt.reshape(-1, 2)
        ys = pts[:, 1]
        bot_xs = pts[:, 0][ys >= h - 6]
        bottom_x = float(np.mean(bot_xs)) if len(bot_xs) else float(cx)

        continuity = 0.75 * abs(cx - desired_x) + 0.25 * \
            abs(bottom_x - desired_x)
        area_reward = min(cv2.contourArea(cnt) / 250.0, 20.0)

        # ── FIX 1: Smarter Fence Penalty ──
        penalty = 0.0
        x1, y1, bw, bh = cv2.boundingRect(cnt)

        # Check if contour touches edges
        hugs_border = (x1 <= 15) or ((x1 + bw) >= (w - 15))
        touches_bottom = np.any(ys >= (h - 15))

        # If it touches the side but DOES NOT come from the bottom, it's probably the fence.
        # If it touches both, it's a sharp 90-degree outgoing turn.
        if hugs_border and not touches_bottom:
            penalty += 500.0

        return continuity + penalty - area_reward

    best = min(candidates, key=_score)
    M = cv2.moments(best)
    new_cx = M["m10"] / M["m00"] if M["m00"] > 0 else prev_cx

    cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)
    return clean, new_cx


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
             tag_info_str, targets_remaining):
    h, w = roi.shape[:2]
    cx = w // 2
    vis = roi.copy()

    for y in range(0, h, 12):
        cv2.line(vis, (cx, y), (cx, min(y+6, h)), (255, 255, 0), 1)

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
    prev_yr = prev_angle = prev_error = 0.0
    smooth_angle = smooth_error = 0.0
    aligning = False
    fps = 0.0
    t_prev = t_ctrl = time.time()
    line_lost_count = frame_count = 0
    alt = ALTITUDE
    current_heading = 0.0
    prev_line_cx = None

    nav_state = NavState.LINE_FOLLOW
    tag_info_str = ""
    current_tag: TagResult | None = None
    tag_hover_start = pre_land_start = tag_reacquire_start = 0.0
    pre_land_lock_count = 0
    targets_remaining = list(Airports)
    visited_tags = set()
    last_tag_detections = []
    tag_mask_memory = []
    tag_mask_hold = 0
    arrival_heading = 0.0

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
    print(f"  Target countries: {Airports}")
    print("  F=follow  C=creep  T=thresh  0=stop  Q=land\n")

    while True:
        # ── Drain Telemetry ───────────────────────────────────────────────
        msg = master.recv_match(blocking=False)
        while msg:
            if msg.get_type() == "GLOBAL_POSITION_INT":
                alt = msg.relative_alt / 1000.0
                current_heading = msg.hdg / 100.0
            msg = master.recv_match(blocking=False)

        # ── Velocity & State Output ───────────────────────────────────────
        if following and nav_state not in (NavState.LANDING, NavState.DONE):

            if nav_state == NavState.LINE_FOLLOW:
                if result.is_detected:
                    line_lost_count = 0
                    angle_for_ctrl = apply_dead_zone(
                        result.angle_deg, YAW_DEAD_ZONE)
                    lat_for_ctrl = apply_dead_zone(error, LAT_DEAD_ZONE)

                    smooth_angle = ema(smooth_angle, angle_for_ctrl, EMA_ALPHA)
                    smooth_error = ema(smooth_error, lat_for_ctrl, EMA_ALPHA)

                    now = time.time()
                    dt = max(now - t_ctrl, 0.01)
                    t_ctrl = now

                    p_yaw = KP_YAW * smooth_angle
                    d_yaw = KD_YAW * (smooth_angle - prev_angle) / dt
                    err_yaw = 0.005 * smooth_error
                    prev_angle = smooth_angle

                    angle_abs = abs(smooth_angle)

                    # ── FIX 2: Dynamic Tag Stiffening Release ──
                    is_approaching_tag = False
                    if last_tag_detections:
                        best_tag_check = max(
                            last_tag_detections, key=tag_corner_area)
                        # Release the steering lock if the tag is in the bottom 30% of the frame
                        # (Meaning the drone is flying over it and needs to aggressively turn)
                        if best_tag_check.center[1] < (frame.shape[0] * 0.70):
                            is_approaching_tag = True

                    if is_approaching_tag:
                        # Clamp max yaw heavily to prevent swerving into junction branches
                        dynamic_max_yaw = 0.15
                    else:
                        dynamic_max_yaw = HIGH_BEND_MAX_YAW if (
                            angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80) else MAX_YAW

                    raw_yr = float(
                        np.clip(p_yaw + d_yaw + err_yaw, -dynamic_max_yaw, dynamic_max_yaw))

                    # Limit slew rate to prevent sudden jerks
                    if is_approaching_tag:
                        dynamic_slew = 0.05
                    else:
                        dynamic_slew = HIGH_BEND_SLEW if (
                            angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80) else YAW_SLEW

                    yr_cmd = float(np.clip(raw_yr, prev_yr -
                                   dynamic_slew, prev_yr + dynamic_slew))
                    prev_yr = yr_cmd

                    p_lat = KP_LAT * smooth_error
                    d_lat = KD_LAT * (smooth_error - prev_error) / dt
                    prev_error = smooth_error

                    vy_cmd = float(np.clip(p_lat + d_lat, -MAX_LAT, MAX_LAT))
                    if angle_abs > HIGH_BEND_ANGLE or abs(smooth_error) > 80:
                        vy_cmd *= 0.4

                    if angle_abs > BEND_SLOW_ANGLE or abs(smooth_error) > 80:
                        scale_ang = max(0.4, 1.0 - (angle_abs - BEND_SLOW_ANGLE) /
                                        90.0) if angle_abs > BEND_SLOW_ANGLE else 1.0
                        scale_lat = max(
                            0.2, 1.0 - (abs(smooth_error) - 80) / 150.0) if abs(smooth_error) > 80 else 1.0
                        fwd_speed = FORWARD_SPEED * min(scale_ang, scale_lat)
                    else:
                        fwd_speed = FORWARD_SPEED

                    if aligning:
                        if abs(result.angle_deg) < ALIGN_THRESHOLD_DEG:
                            aligning = False
                            print(
                                f"[Align] Done — angle={result.angle_deg:+.1f}° → moving forward")
                        else:
                            send_velocity(master, vx=0, vy=0, yaw_rate=yr_cmd)
                    else:
                        send_velocity(master, vx=fwd_speed,
                                      vy=vy_cmd, yaw_rate=yr_cmd)
                else:
                    line_lost_count += 1
                    if last_tag_detections:
                        best_tag = max(last_tag_detections,
                                       key=tag_corner_area)
                        tag_cx = best_tag.center[0]
                        err_x = tag_cx - (frame_w / 2)
                        send_velocity(master, vx=0.15, vy=0,
                                      yaw_rate=float(err_x * 0.0015))
                        tag_info_str = "Gap bridge: tracking tag"
                    elif line_lost_count <= LINE_LOSS_GRACE:
                        send_velocity(master, vx=0.03, vy=0, yaw_rate=yr_cmd)
                    else:
                        send_velocity(master)
                        vy_cmd = yr_cmd = prev_yr = prev_angle = prev_error = 0.0
                        smooth_angle = smooth_error = 0.0
                        t_ctrl = time.time()

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
                            f"  [PRE_LAND_ALIGN] err_x={err_x:+.1f}px err_y={err_y:+.1f}px vx={vx_align:+.3f} vy={vy_align:+.3f} lock={pre_land_lock_count}/{PRE_LAND_LOCK_FRAMES}")

                    tag_info_str = f"Pre-land align ex={err_x:+.1f}px ey={err_y:+.1f}px lock={pre_land_lock_count}/{PRE_LAND_LOCK_FRAMES}"
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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_w = frame.shape[1]
        roi_y = int(frame.shape[0] * ROI_TOP_FRAC)

        # ── Perception: AprilTag ──────────────────────────────────────────
        if (frame_count % TAG_DETECT_INTERVAL == 0) or (nav_state in (NavState.TAG_HOVER, NavState.PRE_LAND_ALIGN)):
            last_tag_detections = tag_detector.detect(gray_frame)

            if not last_tag_detections and tag_mask_hold > 0:
                last_tag_detections = tag_mask_memory
                tag_mask_hold -= 1

            if last_tag_detections:
                tag_mask_memory = list(last_tag_detections)
                tag_mask_hold = TAG_MASK_HOLD_FRAMES
                best_tag = max(last_tag_detections, key=tag_corner_area)
                area = tag_corner_area(best_tag)

                if following and nav_state == NavState.LINE_FOLLOW:
                    centered_for_capture, cap_ex, cap_ey = tag_is_centered(
                        best_tag, frame.shape, deadzone_x=TAG_DETECT_CENTER_DEADZONE_X,
                        deadzone_y=TAG_DETECT_CENTER_DEADZONE_Y)

                    if (best_tag.tag_id not in visited_tags and area > TAG_MIN_AREA and centered_for_capture):
                        nav_state = NavState.TAG_HOVER
                        arrival_heading = current_heading
                        current_tag = best_tag
                        tag_hover_start = time.time()
                        tag_info_str = str(best_tag)

                        send_velocity(master)
                        prev_yr = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                        t_ctrl = time.time()
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
        roi = frame[roi_y:, :]
        mask = make_mask(roi, thresh[0])
        mask, prev_line_cx = keep_line_contour(mask, prev_cx=prev_line_cx)
        result = detector.detect(mask)
        error = result.lateral_error(frame_w)

        # ── State transitions logic ───────────────────────────────────────
        if nav_state == NavState.TAG_HOVER:
            if time.time() - tag_hover_start >= TAG_HOVER_TIME:
                final_tags = tag_detector.detect(gray_frame)
                if final_tags:
                    current_tag = max(final_tags, key=tag_corner_area)

                if current_tag is not None:
                    nav_state = NavState.DECIDING
                    tag_info_str = str(current_tag)
                    print(f"[Nav] → DECIDING: {current_tag}")
                else:
                    nav_state = NavState.LINE_FOLLOW
                    print("[Nav] Lost tag during read — back to LINE_FOLLOW")

        if nav_state == NavState.DECIDING and current_tag is not None:
            visited_tags.add(current_tag.tag_id)
            if (current_tag.country_code in targets_remaining and current_tag.is_landable):
                centered_now, ex_now, ey_now = tag_is_centered(
                    current_tag, frame.shape, target_offset_x=PRE_LAND_OFFSET_X,
                    target_offset_y=PRE_LAND_OFFSET_Y, deadzone_x=PRE_LAND_DEADZONE_X,
                    deadzone_y=PRE_LAND_DEADZONE_Y)
                print(
                    f"[Nav] ✅ MATCH! Country {current_tag.country_code} is landable!")
                nav_state = NavState.PRE_LAND_ALIGN
                pre_land_start = time.time()
                pre_land_lock_count = PRE_LAND_LOCK_FRAMES if centered_now else 0
                tag_info_str = f"Target {current_tag.tag_id}: fine-aligning before landing"
                print("[Nav] → PRE_LAND_ALIGN")
            else:
                reason = "wrong country" if current_tag.country_code not in targets_remaining else "unsafe"
                print(
                    f"[Nav] ❌ Skip airport — {reason} (tag {current_tag.tag_id})")
                nav_state = NavState.TAG_REACQUIRE
                tag_reacquire_start = time.time()
                tag_info_str = f"Skipped tag {current_tag.tag_id} ({reason}) — finding line..."
                current_tag = None
                print("[Nav] → TAG_REACQUIRE (creeping forward to find line)")

        if nav_state == NavState.PRE_LAND_ALIGN:
            if pre_land_lock_count >= PRE_LAND_LOCK_FRAMES and current_tag is not None:
                print(f"[Nav] ✅ Pre-land alignment locked. → LANDING")
                nav_state = NavState.LANDING
                send_velocity(master)
                time.sleep(0.4)
                set_mode(master, "LAND")

                print("[Nav] Landing ...")
                deadline = time.time() + 30
                while time.time() < deadline:
                    if not is_armed(master):
                        break
                    time.sleep(0.5)

                if current_tag.country_code in targets_remaining:
                    targets_remaining.remove(current_tag.country_code)
                print(f"[Nav] Landed! Targets remaining: {targets_remaining}")

                if not targets_remaining:
                    nav_state = NavState.DONE
                    print("[Nav] 🎉 ALL TARGETS REACHED — MISSION COMPLETE!")
                    tag_info_str = "MISSION COMPLETE!"
                else:
                    print("[Nav] Re-taking off to continue mission ...")
                    arm_and_takeoff(master, ALTITUDE)
                    time.sleep(2.0)

                    nav_state = NavState.POST_LAND_SEARCH
                    following = True
                    aligning = False
                    print(
                        f"[Nav] → POST_LAND_SEARCH (avoiding {(arrival_heading+180) % 360:.0f}°)")

            elif (time.time() - pre_land_start) > PRE_LAND_MAX_ALIGN_TIME:
                coarse_centered = False
                if current_tag is not None:
                    coarse_centered, _, _ = tag_is_centered(
                        current_tag, frame.shape, target_offset_x=PRE_LAND_OFFSET_X,
                        target_offset_y=PRE_LAND_OFFSET_Y, deadzone_x=PRE_LAND_DEADZONE_X * 1.5,
                        deadzone_y=PRE_LAND_DEADZONE_Y * 1.5)
                if coarse_centered and current_tag is not None:
                    print("[Nav] Pre-land timeout, but tag is near center → landing")
                    pre_land_lock_count = PRE_LAND_LOCK_FRAMES
                else:
                    print("[Nav] Pre-land alignment timeout — retrying TAG_HOVER")
                    nav_state = NavState.TAG_HOVER
                    tag_hover_start = time.time()
                    pre_land_lock_count = 0

        # ── Line Re-acquisition Logic ────────────────────────────────────
        if following and result.is_detected:
            if nav_state == NavState.TAG_REACQUIRE:
                nav_state = NavState.LINE_FOLLOW
                aligning = True
                prev_yr = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                t_ctrl = time.time()
                prev_line_cx = None
                tag_info_str = f"Line re-acquired — targets: {targets_remaining}"
                print("[Nav] Line re-acquired! → LINE_FOLLOW")

            elif nav_state == NavState.POST_LAND_SEARCH:
                if result.confidence > 0.3:
                    return_hdg = (arrival_heading + 180.0) % 360.0
                    diff = abs(current_heading - return_hdg)
                    if diff > 180:
                        diff = 360 - diff

                    if diff > 60:
                        nav_state = NavState.LINE_FOLLOW
                        aligning = True
                        prev_yr = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                        t_ctrl = time.time()
                        prev_line_cx = None
                        print(
                            f"[Nav] Valid outgoing line found at {current_heading:.0f}°! → LINE_FOLLOW")
                    else:
                        tag_info_str = f"Line found but ignoring return path ({current_heading:.0f}°)"

        # ── FPS & Display ─────────────────────────────────────────────────
        frame_count += 1
        if frame_count % 15 == 0:
            t_now = time.time()
            fps = 15 / (t_now - t_prev + 1e-6)
            t_prev = t_now

        if following and frame_count % 20 == 0:
            if nav_state in (NavState.LINE_FOLLOW, NavState.TAG_REACQUIRE, NavState.POST_LAND_SEARCH):
                print(f"  [{nav_state.name}] hdg={current_heading:03.0f}° conf={result.confidence:.2f}  "
                      f"err={error:+5.1f}px  ang={result.angle_deg:+5.1f}°  vy={vy_cmd:+.3f}  yr={yr_cmd:+.3f}")

        ann = draw_ann(roi, mask, result, frame_w, error, vy_cmd, yr_cmd, nav_state,
                       following, aligning, creep, thresh[0], alt, fps, tag_info_str, len(targets_remaining))
        msk = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
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
                creep = False
                aligning = True
                prev_yr = prev_angle = prev_error = smooth_angle = smooth_error = 0.0
                t_ctrl = time.time()
                nav_state = NavState.LINE_FOLLOW
                print(f"\n[Follow] ON — aligning to line first ...")
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
