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
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            if self.prev_gray is None:
                self.prev_gray = gray
                return None
                
            # Simple motion detection using frame difference
            frame_diff = cv2.absdiff(self.prev_gray, gray)
            
            # Threshold to get motion regions
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find contours of motion
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.prev_gray = gray
                return None
            
            # Find the largest motion area
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only process significant motion
            if area < 1000:  # Minimum motion area
                self.prev_gray = gray
                return None
            
            # Get center of motion
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                self.prev_gray = gray
                return None
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Normalize coordinates
            h, w = gray.shape
            norm_x = cx / w
            norm_y = cy / h
            
            now = time.time()
            self.tracks.append((now, norm_x, norm_y))
            
            # Detect swipe from position changes
            action = self._detect_swipe_from_positions()
            
            self.prev_gray = gray
            return action
            
        except Exception as e:
            print(f"OpenCV processing error: {e}")
            self.prev_gray = gray if 'gray' in locals() else None
            return None
    
    def _detect_swipe_from_positions(self) -> Optional[Action]:
        if len(self.tracks) < 3:
            return None
            
        now = time.time()
        
        # Check cooldown
        if (now - self.last_action_time) < self.config.cooldown:
            return None
            
        # Get first and last positions from recent tracks
        recent_tracks = list(self.tracks)[-10:]  # Last 10 frames
        
        if len(recent_tracks) < 5:
            return None
            
        # Calculate displacement from start to end
        start_time, start_x, start_y = recent_tracks[0]
        end_time, end_x, end_y = recent_tracks[-1]
        
        # Check if motion happened within time limit
        dt = end_time - start_time
        if dt > self.config.max_duration:
            return None
            
        # Calculate displacement
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Determine primary axis movement
        if self.config.axis == "vertical":
            primary = dy
            secondary = dx
        else:
            primary = dx
            secondary = dy
            
        # Check if movement is significant and primarily in desired axis
        if abs(primary) < self.config.min_displacement:
            return None
            
        if abs(secondary) > abs(primary) * 0.8:  # Too much cross-axis motion
            return None
            
        # Clear tracks after detection to prevent repeated triggers
        self.tracks.clear()
        self.last_action_time = now
        print(f"🎯 Motion detected: {self.config.axis} movement = {primary:.3f} (threshold: {self.config.min_displacement})")
        
        # Determine action based on direction
        if self.config.axis == "vertical":
            return Action.NEXT if primary > 0 else Action.PREV
        else:
            return Action.NEXT if primary > 0 else Action.PREV

    def close(self) -> None:
        pass