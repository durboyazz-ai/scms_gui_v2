from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LogState:
    active: bool = False
    file_path: Optional[str] = None


class SessionLogger:
    """CSV session logger for SCMS.

    We keep the schema explicit to make the report/results reproducible.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._fp = None
        self._writer = None
        self.state = LogState(active=False, file_path=None)

    def start(self) -> str:
        if self.state.active:
            return self.state.file_path or ""

        fname = time.strftime("scms_session_%Y%m%d_%H%M%S.csv")
        path = self.log_dir / fname
        self._fp = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)

        header = [
            "timestamp",
            "label",
            "concentration",
            "face_detected",
            "fatigue",
            "gaze",
            "pitch",
            "yaw",
            "roll",
            "ear",
            "pupil_ratio",
            "fps",
            # Calibration/baseline diagnostics
            "calibrating",
            "calib_progress",
            "calib_required",
            "baseline_yaw",
            "baseline_pitch",
            "baseline_pupil_ratio",
            "yaw_delta",
            "pitch_delta",
            "pupil_delta",
        ]
        self._writer.writerow(header)
        self.state = LogState(active=True, file_path=str(path))
        return str(path)

    def stop(self) -> None:
        if self._fp:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                pass
        self._fp = None
        self._writer = None
        self.state = LogState(active=False, file_path=self.state.file_path)

    def log(self, metrics: Dict[str, Any]) -> None:
        if not self.state.active or self._writer is None:
            return

        row = [
            metrics.get("timestamp"),
            metrics.get("label"),
            metrics.get("concentration"),
            metrics.get("face_detected"),
            metrics.get("fatigue"),
            metrics.get("gaze"),
            metrics.get("pitch"),
            metrics.get("yaw"),
            metrics.get("roll"),
            metrics.get("ear"),
            metrics.get("pupil_ratio"),
            metrics.get("fps"),
            metrics.get("calibrating"),
            metrics.get("calib_progress"),
            metrics.get("calib_required"),
            metrics.get("baseline_yaw"),
            metrics.get("baseline_pitch"),
            metrics.get("baseline_pupil_ratio"),
            metrics.get("yaw_delta"),
            metrics.get("pitch_delta"),
            metrics.get("pupil_delta"),
        ]
        self._writer.writerow(row)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
