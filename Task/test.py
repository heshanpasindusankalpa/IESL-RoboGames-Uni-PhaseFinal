import time
from pymavlink import mavutil

def connect(port=14550):
    print(f"[MAVLink] Connecting to udp:127.0.0.1:{port} ...")
    # Using 127.0.0.1 to properly bind to the local MAVProxy router on Debian
    m = mavutil.mavlink_connection(f"udp:127.0.0.1:{port}")
    m.wait_heartbeat()
    print(f"[MAVLink] Connected to system {m.target_system}")
    return m

def set_mode(m, mode):
    mid = m.mode_mapping()[mode]
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid)
    print(f"[MAVLink] Mode set to {mode}")

def arm(m):
    print("[MAVLink] Arming ...")
    # Normal Arm: performs all pre-arm safety checks
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    
    # Wait until armed flag is confirmed
    while True:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            print("[MAVLink] Armed ✅")
            break
        time.sleep(0.5)

def send_velocity(m, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
    """Sends offboard velocity commands to ArduPilot."""
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b010111000111, 0, 0, 0,
        vx, vy, vz, 0, 0, 0, 0, yaw_rate)

def takeoff(m, alt):
    print(f"[MAVLink] Taking off to {alt}m ...")
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)
    
    # Block until altitude is reached
    while True:
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if msg:
            current_alt = msg.relative_alt / 1000.0
            print(f"  Alt: {current_alt:.2f} m")
            # Account for barometer/sensor drift
            if current_alt >= (alt - 0.2): 
                print(f"[MAVLink] Reached target altitude of {alt}m ✅")
                break

def hold_and_land(m):
    print("[MAVLink] Scrubbing residual momentum before descent...")
    # Send a strict 0 m/s velocity command in all directions to prevent lateral drift
    send_velocity(m, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
    time.sleep(2.0)  # Allow physical airframe to settle
    
    print("[MAVLink] Configuring descent speeds...")
    # Cap upper descent speed (WPNAV_SPEED_DN)
    m.mav.param_set_send(
        m.target_system, m.target_component,
        b'WPNAV_SPEED_DN', 30.0,  # 30 cm/s
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(0.5)
    
    # Cap final touchdown speed (LAND_SPEED)
    m.mav.param_set_send(
        m.target_system, m.target_component,
        b'LAND_SPEED', 15.0,      # 15 cm/s
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    time.sleep(1.0) # Let the parameters register in EEPROM
    
    print("[MAVLink] Initiating native LAND mode ...")
    set_mode(m, "LAND")

def wait_for_landing(m, timeout=60):
    print("[MAVLink] Waiting for touchdown and auto-disarm...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if msg:
            armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            if not armed:
                print("[MAVLink] Disarmed — landed safely ✅")
                return True
        time.sleep(0.5)
    print("[MAVLink] ⚠️ Timeout waiting for landing disarm.")
    return False

def main():
    TARGET_ALTITUDE = 1.2
    HOLD_TIME = 10
    
    master = connect()
    
    # 1. Set to GUIDED mode for offboard MAVLink control
    set_mode(master, "GUIDED")
    time.sleep(1)
    
    # 2. Arm the motors
    arm(master)
    
    # 3. Execute takeoff
    takeoff(master, TARGET_ALTITUDE)
    
    # 4. Hold position
    print(f"[Test] Holding altitude at {TARGET_ALTITUDE}m for {HOLD_TIME} seconds...")
    time.sleep(HOLD_TIME)
    
    # 5. Stabilize and execute the dual-stage landing sequence
    hold_and_land(master)
    
    # 6. Block the script from exiting until ArduPilot confirms the motors are off
    wait_for_landing(master)
    print("[Test] Flight sequence complete.")

if __name__ == "__main__":
    main()