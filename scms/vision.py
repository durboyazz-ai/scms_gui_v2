
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict

import cv2
import numpy as np

try:
    import dlib  # type: ignore
except Exception:  # pragma: no cover
    dlib = None  # type: ignore


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return int(self.w * self.h)


class HaarFaceDetector:
    """Simple face detector using OpenCV Haar cascades."""
    def __init__(self, scale_factor: float = 1.2, min_neighbors: int = 5):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")

    def detect(self, gray: np.ndarray) -> list[FaceBox]:
        faces = self.cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors)
        return [FaceBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


class LandmarkDetector:
    """
    Landmark detector abstraction.

    Supported methods:
      - opencv_lbf: OpenCV FacemarkLBF (requires opencv-contrib + lbfmodel.yaml)
      - dlib_68: dlib shape_predictor_68_face_landmarks.dat
      - auto: try opencv_lbf, then dlib_68
    """
    def __init__(
        self,
        method: str = "auto",
        opencv_lbf_model_path: str = "lbfmodel.yaml",
        dlib_68_model_path: str = "shape_predictor_68_face_landmarks.dat",
    ):
        self.method = method.lower().strip()
        self.opencv_lbf_model_path = opencv_lbf_model_path
        self.dlib_68_model_path = dlib_68_model_path

        self._opencv_facemark = None
        self._dlib_predictor = None

        if self.method in ("opencv_lbf", "auto"):
            self._try_init_opencv_lbf()

        if self.method in ("dlib_68", "auto") and self._opencv_facemark is None:
            self._try_init_dlib()

        if self._opencv_facemark is None and self._dlib_predictor is None:
            raise RuntimeError(
                "No landmark detector could be initialized. "
                "Provide lbfmodel.yaml (OpenCV LBF) or shape_predictor_68_face_landmarks.dat (dlib)."
            )

    def _try_init_opencv_lbf(self) -> None:
        try:
            if not hasattr(cv2, "face"):
                return
            if not os.path.exists(self.opencv_lbf_model_path):
                return
            facemark = cv2.face.createFacemarkLBF()
            facemark.loadModel(self.opencv_lbf_model_path)
            self._opencv_facemark = facemark
        except Exception:
            self._opencv_facemark = None

    def _try_init_dlib(self) -> None:
        if dlib is None:
            return
        try:
            if not os.path.exists(self.dlib_68_model_path):
                return
            self._dlib_predictor = dlib.shape_predictor(self.dlib_68_model_path)
        except Exception:
            self._dlib_predictor = None

    def detect(self, gray: np.ndarray, face: FaceBox) -> Optional[np.ndarray]:
        """
        Returns:
          landmarks: np.ndarray shape (68, 2) of (x,y) points, or None.
        """
        if self._opencv_facemark is not None:
            faces = np.array([[face.x, face.y, face.w, face.h]], dtype=np.int32)
            try:
                success, landmarks = self._opencv_facemark.fit(gray, faces)
                if not success or len(landmarks) == 0:
                    return None
                pts = landmarks[0][0].astype(np.float32)  # (68,2)
                return pts
            except Exception:
                return None

        if self._dlib_predictor is not None and dlib is not None:
            try:
                rect = dlib.rectangle(face.x, face.y, face.x + face.w, face.y + face.h)
                shape = self._dlib_predictor(gray, rect)
                pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.float32)
                return pts
            except Exception:
                return None

        return None


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """Compute Eye Aspect Ratio (EAR) for blink / fatigue detection."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    if C < 1e-6:
        return 0.0
    return float((A + B) / (2.0 * C))


def _safe_crop(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return gray[y1:y2, x1:x2].copy()


def pupil_ratio_from_eye(gray: np.ndarray, eye_pts: np.ndarray, padding: int = 4) -> Optional[float]:
    """
    Estimates pupil position within an eye ROI using simple thresholding.
    Returns ratio in [0,1] where 0 is left edge of ROI and 1 is right edge.
    """
    x_min = int(np.min(eye_pts[:, 0])) - padding
    x_max = int(np.max(eye_pts[:, 0])) + padding
    y_min = int(np.min(eye_pts[:, 1])) - padding
    y_max = int(np.max(eye_pts[:, 1])) + padding

    roi = _safe_crop(gray, x_min, y_min, x_max, y_max)
    if roi is None or roi.size == 0:
        return None

    roi_blur = cv2.GaussianBlur(roi, (7, 7), 0)

    # pupil is dark -> invert to make it white
    _, thr = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 10:
        return None

    M = cv2.moments(largest)
    if abs(M.get("m00", 0.0)) < 1e-6:
        return None

    cx = float(M["m10"] / M["m00"])
    ratio = cx / float(roi.shape[1])
    return float(max(0.0, min(1.0, ratio)))


def gaze_direction_from_landmarks(
    gray: np.ndarray,
    landmarks: np.ndarray,
    left_thresh: float = 0.35,
    right_thresh: float = 0.65,
) -> tuple[str, Optional[float]]:
    """
    Returns (label, mean_ratio) where label in {'Left','Center','Right','Unknown'}.
    """
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    left_ratio = pupil_ratio_from_eye(gray, left_eye)
    right_ratio = pupil_ratio_from_eye(gray, right_eye)

    ratios = [r for r in (left_ratio, right_ratio) if r is not None]
    if not ratios:
        return "Unknown", None

    mean_ratio = float(sum(ratios) / len(ratios))
    if mean_ratio < left_thresh:
        return "Left", mean_ratio
    if mean_ratio > right_thresh:
        return "Right", mean_ratio
    return "Center", mean_ratio


def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """
    Converts rotation matrix to Euler angles (pitch, yaw, roll) in degrees.

    Conventions vary; this is a standard conversion suitable for thresholding.
    """
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])   # pitch
        y = math.atan2(-R[2, 0], sy)      # yaw
        z = math.atan2(R[1, 0], R[0, 0])  # roll
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return math.degrees(x), math.degrees(y), math.degrees(z)


def normalize_angle_deg_90(angle_deg: float) -> float:
    """Normalize an angle in degrees to a range close to [-90, 90].

    Many head-pose pipelines can output angles wrapped near ±180° for a frontal face.
    For SCMS thresholding we want a stable, human-readable range.

    Examples:
      -171°  ->  +9°
      +175°  ->  -5°
    """
    a = float(angle_deg)

    # Wrap to [-180, 180)
    a = (a + 180.0) % 360.0 - 180.0

    # Map to [-90, 90]
    if a < -90.0:
        a += 180.0
    elif a > 90.0:
        a -= 180.0

    return float(a)


class HeadPoseEstimator:
    """Head pose estimation via solvePnP using 6 facial landmark points."""
    def __init__(self):
        self.model_points = np.array(
            [
                (0.0, 0.0, 0.0),           # Nose tip
                (0.0, -330.0, -65.0),      # Chin
                (-225.0, 170.0, -135.0),   # Left eye left corner
                (225.0, 170.0, -135.0),    # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left mouth corner
                (150.0, -150.0, -125.0),   # Right mouth corner
            ],
            dtype=np.float64,
        )
        self.landmark_indices = [30, 8, 36, 45, 48, 54]

    def estimate(self, frame: np.ndarray, landmarks: np.ndarray) -> Optional[Dict[str, object]]:
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2.0, h / 2.0)

        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = np.array([landmarks[idx] for idx in self.landmark_indices], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        nose_end_3d = np.array([[0, 0, 1000.0]], dtype=np.float64)
        nose_end_2d, _ = cv2.projectPoints(nose_end_3d, rvec, tvec, camera_matrix, dist_coeffs)

        p1 = tuple(int(x) for x in image_points[0])
        p2 = tuple(int(x) for x in nose_end_2d[0][0])

        R, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = rotation_matrix_to_euler_angles(R)

        # Normalize angles to a stable range for UI + thresholding
        pitch = normalize_angle_deg_90(pitch)
        yaw = normalize_angle_deg_90(yaw)
        roll = normalize_angle_deg_90(roll)

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "nose_line": (p1, p2),
        }
