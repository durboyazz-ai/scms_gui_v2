from dataclasses import dataclass


@dataclass
class SCMSConfig:
    """Configuration for the Student Concentration Monitoring System (SCMS)."""

    # Video
    camera_index: int = 0
    frame_width: int | None = None   # set to e.g. 640 for speed, or None for default
    frame_height: int | None = None
    use_directshow_on_windows: bool = True

    # Face detection (OpenCV Haar)
    # Note: slightly more permissive defaults reduce 'No Face' drops on common webcams.
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 4

    # If the face detector drops for a few frames, keep the last face box briefly.
    face_hold_frames: int = 12
    face_hold_scale: float = 1.15

    # Landmark detectors (auto tries in order: mediapipe -> opencv_lbf -> dlib_68)
    landmark_method: str = "opencv_lbf"  # 'auto', 'opencv_lbf', 'dlib_68'
    opencv_lbf_model_path: str = "lbfmodel.yaml"  # OpenCV LBF facemark model file
    dlib_68_model_path: str = "shape_predictor_68_face_landmarks.dat"  # Dlib landmark predictor

    # Fatigue (EAR)
    ear_threshold: float = 0.20
    ear_consecutive_frames: int = 12

    # Head pose thresholds (DEVIATION thresholds if calibration is available)
    yaw_threshold_deg: float = 22.0
    pitch_threshold_deg: float = 18.0

    # Gaze thresholds (fallback when no calibration is set)
    gaze_left_threshold: float = 0.35
    gaze_right_threshold: float = 0.65

    # Gaze deviation threshold used AFTER calibration (baseline ± delta)
    gaze_delta: float = 0.08

    # Calibration
    # Collect this many valid frames to estimate a baseline (yaw, pitch, pupil ratio).
    calibration_frames: int = 35
    calibration_timeout_sec: float = 12.0

    # After calibration, slowly adapt baseline during "very confident" attentive moments
    baseline_update_alpha: float = 0.02

    # Smoothing
    label_smoothing_window: int = 12

    # Concentration score window: percentage of attentive frames in last N frames.
    concentration_window: int = 90

    # Logging
    log_dir: str = "logs"
    log_every_n_frames: int = 1
