"""
perception/tcp_camera.py

Threaded TCP Camera client for physical drone streams.
Decodes standard compressed video streams (MJPEG/H.264) via OpenCV backend
instead of expecting raw uncompressed socket bytes.
"""

import cv2
import threading
import time
import sys

class ThreadedTCPCamera:
    """Reads standard encoded TCP video streams safely in a background thread."""
    def __init__(self, ip="127.0.0.1", port=9000):
        self.stream_url = f"tcp://{ip}:{port}"
        self.cap = cv2.VideoCapture(self.stream_url)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError(f"[Camera] Could not open {self.stream_url}. Is the port 9000 server running?")

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

        print(f"[Camera] Connected to {self.stream_url}! Buffering first frame...")
        
        # Wait for the first frame to arrive before returning control to the main script
        timeout = time.time() + 5.0
        while self.frame is None and self.running:
            if time.time() > timeout:
                self.stop()
                raise RuntimeError("[Camera] Connected to port but no frames arriving. Is the stream active?")
            time.sleep(0.01)
            
        print("[Camera] Stream active and frames are flowing!")

    def _update(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[Camera] 🚨 Stream error: {e} — stopping.")
                self.running = False
                break

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()

# ─────────────────────────────────────────────────────────────────────────────
# Standalone Testing Block
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Allow testing via laptop (e.g. python3 tcp_camera.py 192.168.1.71 9000)
    # Defaults to localhost if run directly on the drone
    test_ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    test_port = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

    print(f"--- Testing ThreadedTCPCamera on {test_ip}:{test_port} ---")
    cam = ThreadedTCPCamera(ip=test_ip, port=test_port)
    cam.start()

    if not cam.running:
        print("Test failed: Could not establish stream.")
        sys.exit(1)

    print("Press 'q' in the video window to quit, or Ctrl+C in terminal.")
    
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("TCP Camera Test", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("Camera released. Test complete.")