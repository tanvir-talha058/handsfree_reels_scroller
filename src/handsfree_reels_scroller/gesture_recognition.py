"""Gesture recognition module using MediaPipe Hands.

Focus: Detect simple horizontal or vertical swipe gestures mapped to NEXT/PREV actions.
The pure logic (detect_swipe) is separated for unit testing without needing camera frames.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Deque
from collections import deque
import time

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional at test time
    mp = None  # type: ignore

import numpy as np  # type: ignore

from .actions import Action

# Indexes for landmarks (MediaPipe Hands) we might use
# 0: wrist, 8: index fingertip
INDEX_TIP = 8

@dataclass
class SwipeConfig:
    min_displacement: float = 0.15  # normalized (0-1) proportion of frame
    max_duration: float = 0.6       # seconds within which movement must complete
    axis: str = "vertical"          # 'vertical' for reels (default), or 'horizontal'
    history: int = 5                # frames of smoothing
    cooldown: float = 0.8           # seconds between accepted swipes

class SwipeDetector:
    def __init__(self, config: SwipeConfig | None = None) -> None:
        self.config = config or SwipeConfig()
        self.points: Deque[Tuple[float, float, float]] = deque(maxlen=self.config.history)
        self.last_action_time: float = 0.0
        self.last_direction: Optional[str] = None

    def update(self, norm_x: float, norm_y: float) -> Optional[Action]:
        now = time.time()
        self.points.append((now, norm_x, norm_y))
        if len(self.points) < 2:
            return None
        # Use oldest and newest for displacement
        t0, x0, y0 = self.points[0]
        t1, x1, y1 = self.points[-1]
        dt = t1 - t0
        if dt > self.config.max_duration:
            return None
        dx = x1 - x0
        dy = y1 - y0
        axis = self.config.axis
        # Choose primary displacement
        primary = dy if axis == "vertical" else dx
        secondary = dx if axis == "vertical" else dy
        # Basic direction check & threshold
        if abs(primary) < self.config.min_displacement or abs(primary) < abs(secondary):
            return None
        direction = "forward" if primary > 0 else "backward"
        # Cooldown
        if (now - self.last_action_time) < self.config.cooldown:
            return None
        self.last_action_time = now
        self.last_direction = direction
        if axis == "vertical":
            # y increases downward in image coordinates; decide mapping:
            # Swipe down (primary>0) -> NEXT (scroll down)
            return Action.NEXT if primary > 0 else Action.PREV
        else:
            # horizontal axis: right is NEXT
            return Action.NEXT if primary > 0 else Action.PREV

# Pure function for tests

def detect_swipe(points: list[Tuple[float, float, float]], config: SwipeConfig | None = None) -> Optional[Action]:
    """Given a chronological list of (t,x,y) normalized points, decide if they form a swipe.
    Returns an Action or None.
    """
    cfg = config or SwipeConfig()
    if len(points) < 2:
        return None
    t0, x0, y0 = points[0]
    t1, x1, y1 = points[-1]
    dt = t1 - t0
    if dt > cfg.max_duration:
        return None
    dx = x1 - x0
    dy = y1 - y0
    axis = cfg.axis
    primary = dy if axis == "vertical" else dx
    secondary = dx if axis == "vertical" else dy
    if abs(primary) < cfg.min_displacement or abs(primary) < abs(secondary):
        return None
    if axis == "vertical":
        return Action.NEXT if primary > 0 else Action.PREV
    else:
        return Action.NEXT if primary > 0 else Action.PREV

class GestureRecognizer:
    """Higher-level class managing MediaPipe and producing Actions from frames."""
    def __init__(self, swipe_config: SwipeConfig | None = None) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for gesture mode. Install via pip.")
        self.swipe_detector = SwipeDetector(swipe_config)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    def process_frame(self, frame_bgr) -> Optional[Action]:
        # Convert BGR to RGB for mediapipe
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[INDEX_TIP]
        # Landmarks provide normalized coordinates already (0..1)
        action = self.swipe_detector.update(lm.x, lm.y)
        return action

    def close(self) -> None:
        self.hands.close()
