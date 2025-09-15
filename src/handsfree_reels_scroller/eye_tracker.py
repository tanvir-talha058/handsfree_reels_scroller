"""Eye tracking placeholder module.

Future implementation plan:
- Use MediaPipe FaceMesh to extract iris landmarks
- Compute gaze vector via relative position of iris center inside eye region
- Smooth with temporal filter (e.g., exponential moving average)
- Map to screen coordinates using calibration (collect min/max per axis)
- Dwell detection: if cursor remains in region for N ms -> trigger click/scroll

For now we provide a stub class so main can import without errors.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

@dataclass
class GazePoint:
    x: float
    y: float
    confidence: float

class EyeTracker:
    def __init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe required for eye tracking. Install via pip.")
        # Placeholder: would initialize FaceMesh here
        # self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self._ok = True

    def process_frame(self, frame_bgr) -> Optional[GazePoint]:
        # Placeholder algorithm; returns center point with low confidence
        h, w, _ = frame_bgr.shape
        return GazePoint(x=0.5, y=0.5, confidence=0.0)

    def close(self) -> None:
        pass
