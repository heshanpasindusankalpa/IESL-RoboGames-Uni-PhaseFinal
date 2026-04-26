"""
Lightweight MJPEG HTTP Streamer using pure Python standard libraries.
Broadcasts OpenCV frames to a web browser without needing Flask.
"""
import cv2
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    frame = self.server.streamer.get_jpeg()
                    if frame is not None:
                        self.wfile.write(b'--FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.05) # Cap at ~20 FPS to save Wi-Fi bandwidth
            except Exception:
                # Client disconnected
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass # Suppress standard HTTP logging to keep your terminal clean

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

class MJPEGStreamer:
    def __init__(self, port=5000):
        self.port = port
        self._jpeg_bytes = None
        self._lock = threading.Lock()
        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamHandler)
        self.server.streamer = self # Give handler access to the streamer instance
        
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[Streamer] Live view available at http://<drone-ip>:{self.port}/")

    def update_frame(self, frame):
        """Pass your BGR OpenCV frame here."""
        if frame is None: return
        # Compress to JPEG to save bandwidth
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ret:
            with self._lock:
                self._jpeg_bytes = jpeg.tobytes()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg_bytes