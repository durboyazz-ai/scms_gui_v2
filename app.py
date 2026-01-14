
from __future__ import annotations

import atexit
import platform
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
from flask import Flask, Response, jsonify, render_template, send_file

from scms.config import SCMSConfig
from scms.attention import SCMSEngine
from scms.logger import SessionLogger

app = Flask(__name__, template_folder="templates", static_folder="static")

cfg = SCMSConfig()

# Initialize engine (may raise if no landmark model is found)
# To allow the UI to still run without landmarks, we catch exceptions and run a limited mode.
_engine: Optional[SCMSEngine] = None
_engine_error: Optional[str] = None
try:
    _engine = SCMSEngine(cfg)
except Exception as e:
    _engine_error = str(e)

logger = SessionLogger(cfg.log_dir)

_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_latest_metrics: Dict[str, Any] = {"label": "Initializing...", "concentration": None}
_running = True

def _open_camera() -> cv2.VideoCapture:
    is_windows = platform.system().lower().startswith("win")
    if is_windows and cfg.use_directshow_on_windows and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(cfg.camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.camera_index)

    if cfg.frame_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.frame_width))
    if cfg.frame_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.frame_height))

    return cap

def _camera_loop() -> None:
    global _latest_jpeg, _latest_metrics, _running

    cap = _open_camera()
    if not cap.isOpened():
        with _lock:
            _latest_metrics = {"label": "Camera Error", "concentration": None, "error": "Failed to open camera."}
        return

    frame_count = 0
    while _running:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1

        if _engine is None:
            # Limited mode: just show raw camera
            ok, jpg = cv2.imencode(".jpg", frame)
            if ok:
                with _lock:
                    _latest_jpeg = jpg.tobytes()
                    _latest_metrics = {
                        "label": "Face Only",
                        "concentration": None,
                        "error": _engine_error,
                        "logging": {"active": logger.state.active, "file_path": logger.state.file_path},
                    }
            time.sleep(0.01)
            continue

        annotated, metrics_obj = _engine.process(frame)
        metrics = _engine.metrics_to_dict(metrics_obj)
        metrics["logging"] = {"active": logger.state.active, "file_path": logger.state.file_path}

        if logger.state.active and (frame_count % cfg.log_every_n_frames == 0):
            logger.log(metrics)

        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with _lock:
                _latest_jpeg = jpg.tobytes()
                _latest_metrics = metrics

        time.sleep(0.001)

    cap.release()

_thread = threading.Thread(target=_camera_loop, daemon=True)
_thread.start()

@atexit.register
def _cleanup():
    global _running
    _running = False
    try:
        logger.stop()
    except Exception:
        pass

@app.route("/")
def index():
    return render_template("index.html")

def _frame_generator():
    while True:
        with _lock:
            frame = _latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(_frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/metrics")
def metrics():
    with _lock:
        data = dict(_latest_metrics)
    if _engine_error and "error" not in data:
        data["error"] = _engine_error
    # Always include logging status
    if "logging" not in data:
        data["logging"] = {"active": logger.state.active, "file_path": logger.state.file_path}
    return jsonify(data)

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Collect a fresh baseline (yaw/pitch/pupil ratio) for the current user/camera."""
    if _engine is None:
        return jsonify({"ok": False, "error": _engine_error or "Engine unavailable"}), 400
    _engine.request_calibration()
    return jsonify({"ok": True})

@app.route("/start_session", methods=["POST"])
def start_session():
    path = logger.start()
    return jsonify({"ok": True, "file_path": path})

@app.route("/stop_session", methods=["POST"])
def stop_session():
    logger.stop()
    return jsonify({"ok": True})

@app.route("/download_log")
def download_log():
    if not logger.state.file_path:
        return jsonify({"ok": False, "error": "No log file yet."}), 404
    path = Path(logger.state.file_path)
    if not path.exists():
        return jsonify({"ok": False, "error": "Log file missing on disk."}), 404
    return send_file(str(path), as_attachment=True, download_name=path.name)

if __name__ == "__main__":
    # Use: python app.py
    # Open: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
