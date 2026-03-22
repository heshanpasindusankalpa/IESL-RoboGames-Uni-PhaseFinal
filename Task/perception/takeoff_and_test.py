"""
perception/takeoff_and_test.py

Takeoff + live perception pipeline test.
Matches the arming/camera patterns from the working flight.py.

Usage:
    python3 -m perception.takeoff_and_test --altitude 2.5
    python3 -m perception.takeoff_and_test --altitude 2.5 --color  # if color cam

Controls:
    Q      — land and quit
    S      — toggle SLIDING_WINDOW / HOUGH
    T      — threshold slider (gray) or HSV sliders (color)
    SPACE  — pause / resume
"""

from perception.line_detector import LineDetector, LineDetectorConfig, Strategy
from perception.preprocessor import Preprocessor, PreprocessorConfig
import argparse
import socket
import struct
import sys
import time

import cv2
import numpy as np
from pymavlink import mavutil

sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# MAVLink helpers  (matches existing controls.py patterns)
# ---------------------------------------------------------------------------

def connect_mavlink(port: int = 14550):
    master = mavutil.mavlink_connection(f"udp:0.0.0.0:{port}")
    print(f"[MAVLink] Waiting for heartbeat on udp:0.0.0.0:{port} ...")
    master.wait_heartbeat()
    print(f"[MAVLink] Heartbeat — system {master.target_system}, "
          f"component {master.target_component}")
    return master


def set_mode(master, mode_name: str):
    modes = master.mode_mapping()
    mode_id = modes.get(mode_name)
    if mode_id is None:
        raise RuntimeError(f"Mode '{mode_name}' not in {list(modes)}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    print(f"[MAVLink] Mode → {mode_name}")


def is_armed(master) -> bool:
    """Poll heartbeat to check arm state — avoids motors_armed_wait() issues."""
    msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
    if msg is None:
        return False
    return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)


def force_arm(master):
    """Force-arm using ArduPilot magic number — same pattern as controls.py."""
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,      # 1 = arm
        21196,  # force-arm magic number
        0, 0, 0, 0, 0,
    )


def arm_with_retry(master, timeout: float = 15.0):
    """Force-arm then poll heartbeat until armed."""
    set_mode(master, "GUIDED")
    deadline = time.time() + timeout
    print("[MAVLink] Arming ...")
    force_arm(master)
    last_retry = time.time()

    while time.time() < deadline:
        if is_armed(master):
            print("[MAVLink] Armed ✅")
            return
        # Retry arm request every 5 s
        if time.time() - last_retry > 5.0:
            force_arm(master)
            last_retry = time.time()
        time.sleep(0.3)

    raise TimeoutError("[MAVLink] Arming timed out.")


def do_takeoff(master, altitude: float):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0,
        altitude,
    )
    print(f"[MAVLink] Takeoff command sent — target {altitude} m")


def wait_for_altitude(master, target: float, tol: float = 0.35,
                      timeout: float = 35.0):
    deadline = time.time() + timeout
    min_alt = target - tol
    last_log = 0.0

    while time.time() < deadline:
        msg = master.recv_match(type="GLOBAL_POSITION_INT",
                                blocking=True, timeout=2.0)
        if msg is None:
            continue
        alt = msg.relative_alt / 1000.0
        now = time.time()
        if now - last_log >= 0.5:
            print(f"  alt: {alt:.2f} m")
            last_log = now
        if alt >= min_alt:
            print(f"[MAVLink] Altitude reached: {alt:.2f} m ✅")
            return alt

    msg = master.recv_match(type="GLOBAL_POSITION_INT",
                            blocking=True, timeout=1.0)
    alt = (msg.relative_alt / 1000.0) if msg else 0.0
    if alt >= min_alt * 0.8:
        print(f"[MAVLink] Takeoff timeout — altitude {alt:.2f} m (acceptable)")
        return alt
    raise TimeoutError(f"[MAVLink] Takeoff timeout — only reached {alt:.2f} m")


def get_altitude(master) -> float:
    msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    return (msg.relative_alt / 1000.0) if msg else 0.0


def land(master):
    set_mode(master, "LAND")
    print("[MAVLink] Landing ...")


# ---------------------------------------------------------------------------
# Camera helpers  (matches existing flight.py socket protocol exactly)
# ---------------------------------------------------------------------------

def connect_camera(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.settimeout(0.2)
    print(f"[Camera] Connected to {host}:{port}")
    return sock


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except socket.timeout:
            raise TimeoutError("Camera recv timeout")
        if not chunk:
            raise ConnectionError("Camera stream closed")
        buf += chunk
    return buf


def read_frame(sock: socket.socket, color: bool = False):
    """
    Read one frame from the TCP stream.
      color=False → grayscale: width * height bytes        (sim default)
      color=True  → BGR:       width * height * 3 bytes    (--color flag)
    Uses little-endian header (<HH) matching the sim protocol.
    """
    header = recv_exact(sock, 4)
    width, height = struct.unpack("<HH", header)
    n_bytes = width * height * (3 if color else 1)
    payload = recv_exact(sock, n_bytes)

    if color:
        return np.frombuffer(payload, dtype=np.uint8).reshape((height, width, 3))
    else:
        return np.frombuffer(payload, dtype=np.uint8).reshape((height, width))


# ---------------------------------------------------------------------------
# Grayscale mask  (mirrors vision.detect_line threshold logic)
# ---------------------------------------------------------------------------

def gray_to_mask(gray: np.ndarray, thresh: int = 155) -> np.ndarray:
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

CLR_CENTROID = (0, 255, 0)
CLR_SLICE = (255, 165, 0)
CLR_ERROR_BAR = (0, 0, 255)
CLR_CENTER = (255, 255, 0)
CLR_ANGLE = (180, 0, 255)
CLR_TEXT = (255, 255, 255)


def to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()


def draw_center_line(img):
    h, w = img.shape[:2]
    cx = w // 2
    for y in range(0, h, 12):
        cv2.line(img, (cx, y), (cx, min(y + 6, h)), CLR_CENTER, 1)


def draw_result(roi, result, frame_width):
    vis = to_bgr(roi)
    h, w = vis.shape[:2]
    cx = w // 2
    draw_center_line(vis)

    if not result.is_detected:
        cv2.putText(vis, "NO LINE", (w // 2 - 50, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis

    for sx, sy in result.slice_points:
        cv2.circle(vis, (int(sx), int(sy)), 4, CLR_SLICE, -1)
    if len(result.slice_points) >= 2:
        pts = np.array(result.slice_points, dtype=np.int32)
        cv2.polylines(vis, [pts], False, CLR_SLICE, 1)

    det_x = int(result.centroid_x)
    cv2.circle(vis, (det_x, h - 10), 6, CLR_CENTROID, -1)
    cv2.line(vis,   (det_x, 0), (det_x, h), CLR_CENTROID, 2)

    error = result.lateral_error(frame_width)
    cv2.arrowedLine(vis, (cx, h // 2), (det_x, h // 2),
                    CLR_ERROR_BAR, 2, tipLength=0.2)
    angle_px = int(np.tan(np.radians(result.angle_deg)) * (h // 2))
    cv2.line(vis, (cx - angle_px, 0), (cx + angle_px, h), CLR_ANGLE, 1)
    return vis


def draw_hud(frame, result, error, strategy, fps, alt, paused, creep):
    vis = to_bgr(frame)
    creep_str = "  [CREEP ON]" if creep else ""
    lines = [
        f"FPS:{fps:.1f}  Alt:{alt:.2f}m{'  [PAUSED]' if paused else ''}{creep_str}",
        f"Strategy  : {strategy.value}",
        f"Confidence: {result.confidence:.2f}",
        f"Error     : {error:+.1f}px",
        f"Angle     : {result.angle_deg:+.1f}deg",
        "",
        "W/S=fwd/back  A/D=strafe  Arrows=yaw/alt  0=stop",
        "C=creep fwd  Q=land  S=strategy  T=tuner  SPACE=pause",
    ]
    y = 20
    for line in lines:
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3)
        cv2.putText(vis, line, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1)
        y += 18
    return vis


# ---------------------------------------------------------------------------
# Manual velocity control  (body frame, matches controls.py pattern)
# ---------------------------------------------------------------------------

def send_body_velocity(master, vx: float, vy: float, vz: float, yaw_rate: float):
    """
    Send SET_POSITION_TARGET_LOCAL_NED in body frame.
    vx = forward/back,  vy = right/left,  vz = down (negative = up)
    yaw_rate = rad/s clockwise
    """
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b010111000111,   # ignore pos + accel, use velocity + yaw_rate
        0, 0, 0,          # x y z position (ignored)
        vx, vy, vz,       # velocities m/s
        0, 0, 0,          # accelerations (ignored)
        0,                # yaw (ignored)
        yaw_rate,         # yaw rate rad/s
    )


MANUAL_SPEED = 0.6   # m/s linear
MANUAL_YAW_RATE = 0.5   # rad/s
CREEP_SPEED = 0.15  # m/s slow forward creep


def keys_to_velocity(key: int) -> tuple[float, float, float, float] | None:
    """
    Map keypress to (vx, vy, vz, yaw_rate).
    Returns None if key is not a movement key.

    W / S          forward / back
    A / D          left / right (strafe)
    Up / Down      altitude up / down
    Left / Right   yaw left / right
    0              stop / hover
    """
    mapping = {
        ord("w"): (MANUAL_SPEED,  0,             0,  0),
        ord("s"): (-MANUAL_SPEED,  0,             0,  0),
        ord("a"): (0,            -MANUAL_SPEED,  0,  0),
        ord("d"): (0,             MANUAL_SPEED,  0,  0),
        82:       (0,             0,            -0.4, 0),  # Up arrow   → climb
        # Down arrow → descend
        84:       (0,             0,             0.4, 0),
        # Left arrow
        81:       (0,             0,             0,  -MANUAL_YAW_RATE),
        # Right arrow
        83:       (0,             0,             0,   MANUAL_YAW_RATE),
        ord("0"): (0,             0,             0,  0),  # stop
    }
    return mapping.get(key)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    # 1 — Connect MAVLink, arm, take off
    master = connect_mavlink(args.mavport)
    arm_with_retry(master)
    do_takeoff(master, args.altitude)
    wait_for_altitude(master, args.altitude)
    print(f"[Test] Stabilizing 2 s ...")
    time.sleep(2.0)

    # 2 — Camera socket
    cam_sock = connect_camera(args.host, args.camport)
    last_frame = None

    # 3 — Perception modules
    pre = Preprocessor(PreprocessorConfig())
    detector = LineDetector(LineDetectorConfig(
        strategy=Strategy.SLIDING_WINDOW))
    gray_thresh = [155]   # mutable so slider callback can update it

    # 4 — Windows
    WIN_RAW = "1 - Raw frame"
    WIN_MASK = "2 - Binary mask"
    WIN_ANN = "3 - Annotated ROI"
    for win in [WIN_RAW, WIN_MASK, WIN_ANN]:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 400, 300)
    cv2.moveWindow(WIN_RAW,   0,  50)
    cv2.moveWindow(WIN_MASK, 420, 50)
    cv2.moveWindow(WIN_ANN,  840, 50)

    tuner_active = False
    paused = False
    creep_active = False   # C key toggles slow forward creep
    fps = 0.0
    alt = args.altitude
    t_prev = time.time()
    frame_count = 0
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    raw_display = mask_display = ann_display = blank

    print("\n[Test] Running. Q=land+quit  S=strategy  T=tuner  C=creep  SPACE=pause\n")

    while True:
        # Send creep velocity every iteration when active
        if creep_active and not paused:
            send_body_velocity(master, CREEP_SPEED, 0, 0, 0)

        if not paused:
            try:
                frame = read_frame(cam_sock, color=args.color)
                last_frame = frame
            except TimeoutError:
                frame = last_frame
            except ConnectionError as e:
                print(f"[Camera] {e}")
                break

            if frame is None:
                time.sleep(0.01)
                continue

            # Build mask
            if args.color:
                mask, roi = pre.process(frame)
                frame_w = pre.cfg.frame_width
            else:
                frame_w = frame.shape[1]
                roi_y = int(frame.shape[0] * pre.cfg.roi_top_frac)
                roi = frame[roi_y:, :]
                mask = gray_to_mask(roi, gray_thresh[0])

            result = detector.detect(mask)
            error = result.lateral_error(frame_w)
            alt = get_altitude(master)

            frame_count += 1
            if frame_count % 15 == 0:
                t_now = time.time()
                fps = 15 / (t_now - t_prev + 1e-6)
                t_prev = t_now
            if frame_count % 30 == 0:
                print(f"  alt={alt:.2f}m  conf={result.confidence:.2f}  "
                      f"error={error:+6.1f}px  "
                      f"angle={result.angle_deg:+5.1f}°  fps={fps:.1f}")

            raw_display = draw_hud(
                cv2.resize(to_bgr(frame), (frame_w, pre.cfg.frame_height)),
                result, error, detector.cfg.strategy, fps, alt, paused, creep_active)
            mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            ann_display = draw_result(roi, result, frame_w)

        cv2.imshow(WIN_RAW,  raw_display)
        cv2.imshow(WIN_MASK, mask_display)
        cv2.imshow(WIN_ANN,  ann_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            land(master)
            break
        elif key == ord("s"):
            if detector.cfg.strategy == Strategy.SLIDING_WINDOW:
                detector.cfg.strategy = Strategy.HOUGH
            else:
                detector.cfg.strategy = Strategy.SLIDING_WINDOW
            print(f"[Test] Strategy → {detector.cfg.strategy.value}")
        elif key == ord("t") and not tuner_active:
            if args.color:
                lo, hi = list(pre.cfg.hsv_lower), list(pre.cfg.hsv_upper)

                def upd(idx, is_lo):
                    def fn(v):
                        if is_lo:
                            lo[idx] = v
                        else:
                            hi[idx] = v
                        pre.update_hsv(lo, hi)
                    return fn
                cv2.createTrackbar("H lo", WIN_MASK, lo[0], 179, upd(0, True))
                cv2.createTrackbar("H hi", WIN_MASK, hi[0], 179, upd(0, False))
                cv2.createTrackbar("S lo", WIN_MASK, lo[1], 255, upd(1, True))
                cv2.createTrackbar("S hi", WIN_MASK, hi[1], 255, upd(1, False))
                cv2.createTrackbar("V lo", WIN_MASK, lo[2], 255, upd(2, True))
                cv2.createTrackbar("V hi", WIN_MASK, hi[2], 255, upd(2, False))
            else:
                cv2.createTrackbar("Threshold", WIN_MASK, gray_thresh[0], 255,
                                   lambda v: gray_thresh.__setitem__(0, v))
                print(
                    f"[Tuner] Threshold slider active (current: {gray_thresh[0]})")
            tuner_active = True
        elif key == ord("c"):
            creep_active = not creep_active
            if creep_active:
                print(
                    f"[Creep] ON — {CREEP_SPEED} m/s forward. Press C again to stop.")
            else:
                send_body_velocity(master, 0, 0, 0, 0)
                print("[Creep] OFF — hovering.")
        elif key == ord(" "):
            paused = not paused
            print(f"[Test] {'Paused' if paused else 'Resumed'}")
        else:
            vel = keys_to_velocity(key)
            if vel is not None:
                vx, vy, vz, yr = vel
                send_body_velocity(master, vx, vy, vz, yr)
                if key == ord("0"):
                    print("[Manual] Stop / hover")
                else:
                    print(f"[Manual] vx={vx:+.1f}  vy={vy:+.1f}  "
                          f"vz={vz:+.1f}  yaw={yr:+.2f}")

    cam_sock.close()
    cv2.destroyAllWindows()
    print("[Test] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--mavport",  default=14550, type=int)
    parser.add_argument("--camport",  default=5599,  type=int)
    parser.add_argument("--altitude", default=2.5,   type=float)
    parser.add_argument("--color",    action="store_true",
                        help="Camera sends BGR (3 bytes/px). Omit for grayscale.")
    run(parser.parse_args())
