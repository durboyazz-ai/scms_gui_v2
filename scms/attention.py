from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Deque, Tuple

import cv2
import numpy as np

from .config import SCMSConfig
from .vision import (
    HaarFaceDetector,
    LandmarkDetector,
    HeadPoseEstimator,
    eye_aspect_ratio,
    gaze_direction_from_landmarks,
    FaceBox,
)


@dataclass
class SCMSMetrics:
    timestamp: float
    label: str
    concentration: int
    face_detected: bool
    fatigue: bool
    gaze: str
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    ear: Optional[float]
    pupil_ratio: Optional[float]
    fps: float

    # Calibration / baseline
    calibrating: bool
    calib_progress: int
    calib_required: int
    baseline_yaw: Optional[float]
    baseline_pitch: Optional[float]
    baseline_pupil_ratio: Optional[float]
    yaw_delta: Optional[float]
    pitch_delta: Optional[float]
    pupil_delta: Optional[float]


class SCMSEngine:
    """Core real-time processing engine.

    Pipeline:
      - Face detection
      - Landmark detection
      - EAR (fatigue)
      - Gaze estimation
      - Head pose estimation
      - Heuristic attention labeling + concentration score

    Notes on accuracy:
      - Head-pose angles can wrap near ±180°. vision.HeadPoseEstimator normalizes them.
      - Different users/cameras have different baselines. We therefore calibrate a baseline
        (yaw, pitch, pupil ratio) and then threshold on *deviation* from that baseline.
    """

    def __init__(self, cfg: Optional[SCMSConfig] = None):
        self.cfg = cfg or SCMSConfig()

        self.face_detector = HaarFaceDetector(
            scale_factor=self.cfg.haar_scale_factor,
            min_neighbors=self.cfg.haar_min_neighbors,
        )

        self.landmark_detector = LandmarkDetector(
            method=self.cfg.landmark_method,
            opencv_lbf_model_path=self.cfg.opencv_lbf_model_path,
            dlib_68_model_path=self.cfg.dlib_68_model_path,
        )

        self.head_pose = HeadPoseEstimator()

        # State
        self._fatigue_counter = 0
        self._label_window: Deque[str] = deque(maxlen=self.cfg.label_smoothing_window)
        self._att_window: Deque[int] = deque(maxlen=self.cfg.concentration_window)

        # Face hold (reduce temporary 'No Face' drops)
        self._last_face: Optional[FaceBox] = None
        self._missed_face_frames = 0

        # FPS
        self._last_time = time.time()
        self._fps = 0.0

        # Calibration / baseline
        self.baseline_yaw: Optional[float] = None
        self.baseline_pitch: Optional[float] = None
        self.baseline_pupil_ratio: Optional[float] = None

        self._calibrate_requested = False
        self._calibrating = True  # auto-calibrate at startup
        self._calib_start_time = time.time()
        self._calib_yaw: list[float] = []
        self._calib_pitch: list[float] = []
        self._calib_pupil: list[float] = []

    # ----------------------------
    # Public control
    # ----------------------------
    def request_calibration(self) -> None:
        """Request recalibration (safe to call from another thread)."""
        self._calibrate_requested = True

    # ----------------------------
    # Internals
    # ----------------------------
    def _update_fps(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 1e-6:
            inst = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps > 0 else inst
        return self._fps

    @staticmethod
    def _expand_face(face: FaceBox, shape: Tuple[int, int], scale: float) -> FaceBox:
        h, w = shape
        cx = face.x + face.w / 2.0
        cy = face.y + face.h / 2.0
        nw = face.w * scale
        nh = face.h * scale
        x = int(max(0, cx - nw / 2.0))
        y = int(max(0, cy - nh / 2.0))
        x2 = int(min(w - 1, cx + nw / 2.0))
        y2 = int(min(h - 1, cy + nh / 2.0))
        return FaceBox(x=x, y=y, w=max(1, x2 - x), h=max(1, y2 - y))

    def _maybe_start_calibration(self) -> None:
        if not self._calibrate_requested:
            return
        self._calibrate_requested = False
        self._calibrating = True
        self._calib_start_time = time.time()
        self._calib_yaw.clear()
        self._calib_pitch.clear()
        self._calib_pupil.clear()

    def _maybe_collect_calibration_sample(
        self,
        yaw: Optional[float],
        pitch: Optional[float],
        pupil_ratio: Optional[float],
        gaze_label: str,
        fatigue: bool,
    ) -> None:
        if not self._calibrating:
            return

        # Timeout safety
        if (time.time() - self._calib_start_time) > self.cfg.calibration_timeout_sec:
            self._calibrating = False
            return

        if fatigue:
            return
        if yaw is None or pitch is None or pupil_ratio is None:
            return
        if gaze_label == "Unknown":
            return

        self._calib_yaw.append(float(yaw))
        self._calib_pitch.append(float(pitch))
        self._calib_pupil.append(float(pupil_ratio))

        if len(self._calib_yaw) >= self.cfg.calibration_frames:
            # Robust median baseline
            self.baseline_yaw = float(np.median(self._calib_yaw))
            self.baseline_pitch = float(np.median(self._calib_pitch))
            self.baseline_pupil_ratio = float(np.median(self._calib_pupil))
            self._calibrating = False

    def _calibrated_gaze_label(self, pupil_ratio: Optional[float], fallback_label: str) -> str:
        if pupil_ratio is None:
            return "Unknown"
        if self.baseline_pupil_ratio is None:
            return fallback_label
        base = self.baseline_pupil_ratio
        delta = self.cfg.gaze_delta
        if pupil_ratio < base - delta:
            return "Left"
        if pupil_ratio > base + delta:
            return "Right"
        return "Center"

    def _head_away(self, yaw: Optional[float], pitch: Optional[float]) -> bool:
        if yaw is None or pitch is None:
            return False

        if self.baseline_yaw is not None and self.baseline_pitch is not None:
            dy = abs(float(yaw) - float(self.baseline_yaw))
            dp = abs(float(pitch) - float(self.baseline_pitch))
            return (dy > self.cfg.yaw_threshold_deg) or (dp > self.cfg.pitch_threshold_deg)

        # Fallback if not calibrated
        return (abs(float(yaw)) > self.cfg.yaw_threshold_deg) or (abs(float(pitch)) > self.cfg.pitch_threshold_deg)

    def _maybe_update_baseline(self, label: str, fatigue: bool, gaze: str, yaw: Optional[float], pitch: Optional[float], pupil_ratio: Optional[float]) -> None:
        """Slowly adapt baseline (optional) during very confident attentive moments."""
        if self._calibrating:
            return
        if self.baseline_yaw is None or self.baseline_pitch is None or self.baseline_pupil_ratio is None:
            return
        if label != "Attentive":
            return
        if fatigue or gaze != "Center":
            return
        if yaw is None or pitch is None or pupil_ratio is None:
            return

        a = float(self.cfg.baseline_update_alpha)
        self.baseline_yaw = (1 - a) * self.baseline_yaw + a * float(yaw)
        self.baseline_pitch = (1 - a) * self.baseline_pitch + a * float(pitch)
        self.baseline_pupil_ratio = (1 - a) * self.baseline_pupil_ratio + a * float(pupil_ratio)

    # ----------------------------
    # Main processing
    # ----------------------------
    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, SCMSMetrics]:
        ts = time.time()
        fps = self._update_fps()

        # Apply pending recalibration request (safe flag)
        self._maybe_start_calibration()

        frame = frame_bgr.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detect(gray)

        # Face selection / hold
        face_detected = len(faces) > 0
        face: Optional[FaceBox] = None
        if face_detected:
            face = max(faces, key=lambda f: f.area)
            self._last_face = face
            self._missed_face_frames = 0
        else:
            if self._last_face is not None and self._missed_face_frames < self.cfg.face_hold_frames:
                self._missed_face_frames += 1
                face = self._expand_face(self._last_face, gray.shape[:2], self.cfg.face_hold_scale)
                face_detected = True
            else:
                self._last_face = None
                face_detected = False

        label = "No Face"
        fatigue = False
        gaze_label = "Unknown"
        pitch = yaw = roll = None
        ear = None
        pupil_ratio = None

        if face_detected and face is not None:
            # Draw face box
            cv2.rectangle(frame, (face.x, face.y), (face.x + face.w, face.y + face.h), (0, 255, 0), 2)

            landmarks = self.landmark_detector.detect(gray, face)
            if landmarks is not None:
                # Draw landmarks
                for (x, y) in landmarks.astype(int):
                    cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), -1)

                # EAR + fatigue
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if ear < self.cfg.ear_threshold:
                    self._fatigue_counter += 1
                else:
                    self._fatigue_counter = 0
                fatigue = self._fatigue_counter >= self.cfg.ear_consecutive_frames

                # Gaze (fallback)
                gaze_fallback, pupil_ratio = gaze_direction_from_landmarks(
                    gray,
                    landmarks,
                    left_thresh=self.cfg.gaze_left_threshold,
                    right_thresh=self.cfg.gaze_right_threshold,
                )

                # Head pose
                pose = self.head_pose.estimate(frame, landmarks)
                if pose is not None:
                    pitch = float(pose["pitch"])
                    yaw = float(pose["yaw"])
                    roll = float(pose["roll"])
                    p1, p2 = pose["nose_line"]
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)

                # Calibration collection (uses fallback gaze label)
                self._maybe_collect_calibration_sample(yaw, pitch, pupil_ratio, gaze_fallback, fatigue)

                # Calibrated gaze label (if baseline exists)
                gaze_label = self._calibrated_gaze_label(pupil_ratio, gaze_fallback)

                # --- Attention decision ---
                looking_away = gaze_label in ("Left", "Right")
                head_away = self._head_away(yaw, pitch)

                if fatigue:
                    label = "Drowsy"
                elif looking_away or head_away:
                    label = "Distracted"
                else:
                    label = "Attentive"

            else:
                label = "Face Only"

        # Smooth label for stability
        self._label_window.append(label)
        smooth_label = max(set(self._label_window), key=self._label_window.count)

        # Update baseline slowly (optional)
        self._maybe_update_baseline(smooth_label, fatigue, gaze_label, yaw, pitch, pupil_ratio)

        # Concentration score: % of Attentive frames in last N frames
        self._att_window.append(1 if smooth_label == "Attentive" else 0)
        concentration = int(round(100.0 * (sum(self._att_window) / max(1, len(self._att_window)))))

        # Deltas
        yaw_delta = pitch_delta = pupil_delta = None
        if self.baseline_yaw is not None and yaw is not None:
            yaw_delta = float(yaw - self.baseline_yaw)
        if self.baseline_pitch is not None and pitch is not None:
            pitch_delta = float(pitch - self.baseline_pitch)
        if self.baseline_pupil_ratio is not None and pupil_ratio is not None:
            pupil_delta = float(pupil_ratio - self.baseline_pupil_ratio)

        # Overlay UI text
        self._draw_overlay(
            frame,
            smooth_label,
            concentration,
            gaze_label,
            fatigue,
            pitch,
            yaw,
            roll,
            yaw_delta,
            pitch_delta,
            fps,
            calibrating=self._calibrating,
            calib_progress=len(self._calib_yaw),
        )

        metrics = SCMSMetrics(
            timestamp=ts,
            label=smooth_label,
            concentration=int(concentration),
            face_detected=bool(face_detected),
            fatigue=bool(fatigue),
            gaze=str(gaze_label),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            ear=ear,
            pupil_ratio=pupil_ratio,
            fps=float(fps),
            calibrating=bool(self._calibrating),
            calib_progress=int(len(self._calib_yaw)),
            calib_required=int(self.cfg.calibration_frames),
            baseline_yaw=self.baseline_yaw,
            baseline_pitch=self.baseline_pitch,
            baseline_pupil_ratio=self.baseline_pupil_ratio,
            yaw_delta=yaw_delta,
            pitch_delta=pitch_delta,
            pupil_delta=pupil_delta,
        )

        return frame, metrics

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        label: str,
        concentration: int,
        gaze: str,
        fatigue: bool,
        pitch: Optional[float],
        yaw: Optional[float],
        roll: Optional[float],
        yaw_delta: Optional[float],
        pitch_delta: Optional[float],
        fps: float,
        calibrating: bool,
        calib_progress: int,
    ) -> None:
        # Header
        cv2.putText(frame, f"State: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Concentration: {concentration:3d}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Gaze/Fatigue
        cv2.putText(frame, f"Gaze: {gaze}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if fatigue:
            cv2.putText(frame, "Fatigue: EYES CLOSED", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Calibration status / deltas
        if calibrating:
            cv2.putText(
                frame,
                f"Calibrating... ({calib_progress})",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        else:
            if yaw_delta is not None and pitch_delta is not None:
                cv2.putText(
                    frame,
                    f"YawΔ: {yaw_delta:+.1f}  PitchΔ: {pitch_delta:+.1f}",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

        # Pose
        if pitch is not None and yaw is not None and roll is not None:
            cv2.putText(frame, f"Pitch: {pitch:+.1f}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Yaw:   {yaw:+.1f}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll:  {roll:+.1f}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    def metrics_to_dict(self, m: SCMSMetrics) -> Dict[str, Any]:
        return {
            "timestamp": m.timestamp,
            "label": m.label,
            "concentration": m.concentration,
            "face_detected": m.face_detected,
            "fatigue": m.fatigue,
            "gaze": m.gaze,
            "pitch": m.pitch,
            "yaw": m.yaw,
            "roll": m.roll,
            "ear": m.ear,
            "pupil_ratio": m.pupil_ratio,
            "fps": m.fps,
            "calibrating": m.calibrating,
            "calib_progress": m.calib_progress,
            "calib_required": m.calib_required,
            "baseline_yaw": m.baseline_yaw,
            "baseline_pitch": m.baseline_pitch,
            "baseline_pupil_ratio": m.baseline_pupil_ratio,
            "yaw_delta": m.yaw_delta,
            "pitch_delta": m.pitch_delta,
            "pupil_delta": m.pupil_delta,
        }
