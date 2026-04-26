"""TCP camera client with asynchronous frame plumbing.

The client pulls raw BGR frames from a remote server, stores the
latest frame under lock, and provides a non-blocking getter.
"""
import socket
import struct
import threading
import cv2
import numpy as np


class Camera:
    """
    TCP camera client.

    Usage:
        cam = Camera(host='127.0.0.1', port=9000)
        cam.start(callback=my_fn)   # my_fn(frame: np.ndarray) called per frame
        ...
        cam.stop()
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self.host = host
        self.port = port

        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

        # Latest frame + lock for safe cross-thread reads
        self._latest_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, callback=None):
        """
        Connect to the camera server and begin receiving frames.

        Args:
            callback: optional fn(frame: np.ndarray) called on every new frame.
                      If None, frames are only stored and retrievable via get_frame().
        """
        if self.is_running():
            return

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            args=(callback,),
            daemon=True,
            name="CameraThread",
        )
        self._thread.start()

    def stop(self):
        """Disconnect and stop the camera thread."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._stop_event = None
        self._thread = None
        # cv2.destroyAllWindows()

    def get_frame(self) -> np.ndarray | None:
        """
        Return the most recently received frame (non-blocking).
        Returns None if no frame has arrived yet.
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_running(self) -> bool:
        """True if the camera thread is active."""
        return self._stop_event is not None and not self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, callback):
        """Main camera thread: connects and pulls frames in a loop."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.host, self.port))
                print(f"[Camera] Connected to {self.host}:{self.port}")

                while not self._stop_event.is_set():
                    frame = self._recv_frame(sock)
                    if frame is None:
                        print("[Camera] Connection lost or bad frame — stopping.")
                        break

                    with self._frame_lock:
                        self._latest_frame = frame

                    if callback is not None:
                        callback(frame)

        except ConnectionRefusedError:
            print(f"[Camera] Could not connect to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Camera] Unexpected error: {e}")

    def _recv_frame(self, sock: socket.socket) -> np.ndarray | None:
        """
        Read one frame from the socket.

        Header: 4 bytes → (width: uint16, height: uint16)
        Payload: width * height * 3 bytes → raw BGR
        """
        header = self._recv_exact(sock, 4)
        if header is None:
            return None

        width, height = struct.unpack("=HH", header)

        payload = self._recv_exact(sock, width * height * 3)
        if payload is None:
            return None

        frame = np.frombuffer(payload, dtype=np.uint8).reshape(
            (height, width, 3))
        return frame

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        """Receive exactly n bytes, returning None on disconnect."""
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def __del__(self):
        self.stop()
