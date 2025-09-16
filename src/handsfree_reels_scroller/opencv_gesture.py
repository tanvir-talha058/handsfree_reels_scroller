"""Enhanced OpenCV-based gesture detection for hand swipe recognition.

Uses multiple OpenCV techniques:
1. Background subtraction for motion detection
2. Contour analysis for hand shape detection
3. Centroid tracking for movement direction
4. Temporal smoothing for gesture recognition

This provides reliable gesture detection without requiring MediaPipe.
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
    """Enhanced OpenCV-based gesture detector using multiple techniques."""
    
    def __init__(self, config: SwipeConfig | None = None) -> None:
        self.config = config or SwipeConfig()
        self.last_action_time = 0.0
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        # Hand position tracking
        self.hand_positions = deque(maxlen=15)  # Store recent hand positions
        self.gesture_buffer = deque(maxlen=5)   # Buffer for gesture smoothing
        
        # Morphological operations kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Gesture detection parameters
        self.min_contour_area = 2000  # Minimum area for hand detection
        self.max_contour_area = 50000  # Maximum area to filter out full-body motion
        
        print("🤖 Enhanced OpenCV gesture detector initialized")
        print(f"📐 Detection area: {self.min_contour_area}-{self.max_contour_area} pixels")
        print(f"⚡ Sensitivity: {self.config.min_displacement}, Cooldown: {self.config.cooldown}s")

    def process_frame(self, frame_bgr) -> Optional[Action]:
        """Process frame using enhanced OpenCV techniques for robust hand detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(blurred)
            
            # Remove shadows (they appear as gray pixels)
            fg_mask[fg_mask == 127] = 0
            
            # Morphological operations to clean up the mask
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area (hand-sized objects)
            valid_contours = [
                c for c in contours 
                if self.min_contour_area <= cv2.contourArea(c) <= self.max_contour_area
            ]
            
            if not valid_contours:
                return None
            
            # Find the largest valid contour (most likely the hand)
            hand_contour = max(valid_contours, key=cv2.contourArea)
            
            # Get hand center
            M = cv2.moments(hand_contour)
            if M["m00"] == 0:
                return None
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Normalize coordinates to 0-1 range
            h, w = frame_bgr.shape[:2]
            norm_x = cx / w
            norm_y = cy / h
            
            # Store position with timestamp
            now = time.time()
            self.hand_positions.append((now, norm_x, norm_y))
            
            # Analyze gesture
            action = self._analyze_gesture()
            
            # Optional: Draw debug info on frame
            self._draw_debug_info(frame_bgr, hand_contour, (cx, cy), action)
            
            return action
            
        except Exception as e:
            print(f"🚨 OpenCV processing error: {e}")
            return None
    
    def _analyze_gesture(self) -> Optional[Action]:
        """Analyze hand positions to detect swipe gestures."""
        if len(self.hand_positions) < 5:
            return None
            
        now = time.time()
        
        # Check cooldown
        if (now - self.last_action_time) < self.config.cooldown:
            return None
            
        # Get recent positions for analysis
        recent_positions = list(self.hand_positions)[-10:]
        
        if len(recent_positions) < 5:
            return None
            
        # Calculate movement vector from first to last position
        start_time, start_x, start_y = recent_positions[0]
        end_time, end_x, end_y = recent_positions[-1]
        
        # Check if movement happened within time limit
        dt = end_time - start_time
        if dt > self.config.max_duration:
            return None
            
        # Calculate displacement
        dx = end_x - start_x
        dy = end_y - start_y
        
        # Calculate movement consistency (how straight the gesture is)
        consistency = self._calculate_movement_consistency(recent_positions)
        
        # Determine primary axis movement
        if self.config.axis == "vertical":
            primary = dy
            secondary = abs(dx)
        else:
            primary = dx
            secondary = abs(dy)
            
        # Check if movement is significant
        if abs(primary) < self.config.min_displacement:
            return None
            
        # Check if movement is primarily in desired axis (80% threshold)
        if secondary > abs(primary) * 0.8:
            return None
            
        # Check movement consistency (straight line vs erratic)
        if consistency < 0.6:  # Too erratic
            return None
            
        # Valid gesture detected
        self.last_action_time = now
        
        # Add to gesture buffer for smoothing
        if self.config.axis == "vertical":
            gesture = Action.NEXT if primary > 0 else Action.PREV
        else:
            gesture = Action.NEXT if primary > 0 else Action.PREV
            
        self.gesture_buffer.append(gesture)
        
        # Require consistent gesture detection
        if len(self.gesture_buffer) >= 3:
            # Check if last 3 gestures are the same
            if all(g == gesture for g in list(self.gesture_buffer)[-3:]):
                self.gesture_buffer.clear()  # Clear buffer
                self.hand_positions.clear()  # Clear positions to prevent repeat
                
                print(f"✅ Gesture detected: {gesture.name}")
                print(f"📊 Movement: {primary:.3f} ({self.config.axis}), Consistency: {consistency:.2f}")
                
                return gesture
        
        return None
    
    def _calculate_movement_consistency(self, positions) -> float:
        """Calculate how consistent (straight-line) the movement is."""
        if len(positions) < 3:
            return 0.0
        
        # Extract coordinates
        coords = [(x, y) for _, x, y in positions]
        
        # Calculate total distance traveled (sum of all segments)
        total_distance = 0
        for i in range(1, len(coords)):
            dx = coords[i][0] - coords[i-1][0]
            dy = coords[i][1] - coords[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        if total_distance == 0:
            return 0.0
        
        # Calculate straight-line distance (start to end)
        start_x, start_y = coords[0]
        end_x, end_y = coords[-1]
        straight_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Consistency is ratio of straight distance to total distance
        # 1.0 = perfect straight line, lower values = more erratic movement
        return straight_distance / total_distance if total_distance > 0 else 0.0
    
    def _draw_debug_info(self, frame, hand_contour, center, action):
        """Draw debug information on the frame."""
        # Draw hand contour
        cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
        
        # Draw hand center
        cv2.circle(frame, center, 8, (255, 0, 0), -1)
        
        # Draw recent positions trail
        if len(self.hand_positions) > 1:
            positions = [(int(x * frame.shape[1]), int(y * frame.shape[0])) 
                        for _, x, y in list(self.hand_positions)[-5:]]
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i-1], positions[i], (255, 255, 0), 2)
        
        # Draw action text
        if action:
            cv2.putText(frame, f"Action: {action.name}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def close(self) -> None:
        pass