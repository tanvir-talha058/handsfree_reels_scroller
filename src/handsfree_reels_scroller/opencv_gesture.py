"""Simple OpenCV-based gesture detection as fallback when MediaPipe isn't available.

Uses basic motion detection and optical flow to detect swipe-like movements.
Less accurate than MediaPipe but works with any Python version.
"""
from __future__ import annotations
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, Tuple, Deque
from collections import deque
import time

from .actions import Action
from .gesture_recognition import SwipeConfig

class OpenCVGestureDetector:
    """Fallback gesture detector using OpenCV optical flow."""
    
    def __init__(self, config: SwipeConfig | None = None) -> None:
        self.config = config or SwipeConfig()
        self.prev_gray = None
        self.tracks = deque(maxlen=self.config.history)
        self.last_action_time = 0.0
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    def process_frame(self, frame_bgr) -> Optional[Action]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
            
        # Detect corners in previous frame
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        
        if p0 is None or len(p0) < 5:
            self.prev_gray = gray
            return None
            
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
        
        # Select good tracks
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            if len(good_new) < 5:
                self.prev_gray = gray
                return None
                
            # Calculate average motion
            motion_vectors = good_new - good_old
            avg_motion = np.mean(motion_vectors, axis=0)
            
            now = time.time()
            h, w = gray.shape
            
            # Normalize motion to frame size
            norm_dx = avg_motion[0] / w
            norm_dy = avg_motion[1] / h
            
            # Store motion data
            self.tracks.append((now, norm_dx, norm_dy))
            
            # Detect swipe
            action = self._detect_swipe_from_tracks()
            
        self.prev_gray = gray
        return action
    
    def _detect_swipe_from_tracks(self) -> Optional[Action]:
        if len(self.tracks) < 3:
            return None
            
        now = time.time()
        
        # Check cooldown
        if (now - self.last_action_time) < self.config.cooldown:
            return None
            
        # Analyze recent motion
        recent_tracks = list(self.tracks)[-5:]  # Last 5 frames
        
        if len(recent_tracks) < 3:
            return None
            
        # Calculate cumulative motion
        total_dx = sum(dx for _, dx, dy in recent_tracks)
        total_dy = sum(dy for _, dx, dy in recent_tracks)
        
        # Determine primary axis
        if self.config.axis == "vertical":
            primary = total_dy
            secondary = total_dx
        else:
            primary = total_dx
            secondary = total_dy
            
        # Check if motion is significant and primarily in desired axis
        if abs(primary) < self.config.min_displacement:
            return None
            
        if abs(secondary) > abs(primary) * 0.7:  # Too much cross-axis motion
            return None
            
        # Determine action
        self.last_action_time = now
        
        if self.config.axis == "vertical":
            return Action.NEXT if primary > 0 else Action.PREV
        else:
            return Action.NEXT if primary > 0 else Action.PREV

    def close(self) -> None:
        pass