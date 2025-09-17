"""Entry point for Handsfree Reels Scroller.

Usage:
    python -m src.handsfree_reels_scroller.main --mode gesture
"""
from __future__ import annotations
import argparse
import sys
import time
from typing import Optional

import cv2  # type: ignore

try:
    import pyautogui  # type: ignore
except ImportError:
    pyautogui = None  # type: ignore

# Handle both relative and absolute imports
try:
    from .actions import Action
    from .gesture_recognition import GestureRecognizer, SwipeConfig
    from .opencv_gesture import OpenCVGestureDetector
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from actions import Action
    from gesture_recognition import GestureRecognizer, SwipeConfig
    from opencv_gesture import OpenCVGestureDetector

# Key mapping for actions (can be adapted per platform)
KEY_NEXT = "down"
KEY_PREV = "up"

# Alternative keys for different platforms
ALT_KEYS = {
    Action.NEXT: ["j", "s", "space"],  # YouTube, TikTok alternatives
    Action.PREV: ["k", "w"]            # YouTube, TikTok alternatives
}


def send_action(action: Action, key_mode: str = "arrows") -> None:
    if pyautogui is None:
        print(f"[DRY RUN] Would send: {action}")
        return
    
    # Determine which keys to use
    if key_mode == "arrows":
        next_key, prev_key = "down", "up"
    elif key_mode == "wasd":
        next_key, prev_key = "s", "w"
    elif key_mode == "jk":
        next_key, prev_key = "j", "k"
    elif key_mode == "space":
        next_key, prev_key = "space", "shift+space"
    else:
        next_key, prev_key = "down", "up"
    
    try:
        if action == Action.NEXT:
            print(f"🔽 Sending '{next_key}' for NEXT reel...")
            if "+" in next_key:
                keys = next_key.split("+")
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(next_key)
        elif action == Action.PREV:
            print(f"🔼 Sending '{prev_key}' for PREV reel...")
            if "+" in prev_key:
                keys = prev_key.split("+")
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(prev_key)
        else:
            print(f"Unhandled action: {action}")
    except Exception as e:
        print(f"Error sending key: {e}")


def run_gesture_mode(args) -> int:
    # Use enhanced OpenCV detector as primary method
    recognizer = None
    if OpenCVGestureDetector is not None:
        recognizer = OpenCVGestureDetector(
            SwipeConfig(
                min_displacement=args.min_displacement,
                max_duration=args.max_duration,
                axis=args.axis,
                cooldown=args.cooldown,
            )
        )
        print("🚀 Using Enhanced OpenCV gesture detection (Primary Method)")
        print("💡 This provides reliable hand gesture detection without MediaPipe dependency")
    else:
        # Try MediaPipe as fallback
        try:
            recognizer = GestureRecognizer(
                SwipeConfig(
                    min_displacement=args.min_displacement,
                    max_duration=args.max_duration,
                    axis=args.axis,
                    cooldown=args.cooldown,
                )
            )
            print("Using MediaPipe hand tracking for gesture detection.")
        except ImportError:
            print("Error: Neither OpenCV nor MediaPipe gesture detection is available.")
            print("\nSolutions:")
            print("1. Ensure OpenCV is installed: pip install opencv-python")
            print("2. Use Python 3.8-3.11 for MediaPipe: conda create -n reels python=3.11")
            return 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return 1

    print("Gesture mode started. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break
            action = recognizer.process_frame(frame)
            if action:
                print(f"Detected action: {action.name}")
                send_action(action, args.keys)
            # Show window for feedback with enhanced info
            cv2.putText(frame, "Enhanced OpenCV Gesture Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Keys: {args.keys.upper()} | Axis: {args.axis}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Handsfree Reels - Enhanced OpenCV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


def run_gaze_mode(args) -> int:
    print("Gaze mode not yet implemented. Placeholder.")
    return 0


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="Handsfree Reels Scroller")
    p.add_argument("--mode", choices=["gesture", "gaze", "test"], default="gesture")
    p.add_argument("--min-displacement", type=float, default=0.15)
    p.add_argument("--max-duration", type=float, default=0.6)
    p.add_argument("--axis", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--cooldown", type=float, default=0.8)
    p.add_argument("--keys", choices=["arrows", "wasd", "jk", "space"], default="arrows", 
                   help="Key mapping: arrows=up/down, wasd=w/s, jk=k/j, space=space/shift+space")
    return p.parse_args(argv)


def run_test_mode(args) -> int:
    """Test key sending without camera."""
    print(f"🧪 Test mode - Using '{args.keys}' key mapping")
    print("Commands:")
    print("  'n' or 'j' - Send NEXT action")
    print("  'p' or 'k' - Send PREV action") 
    print("  'q' - Quit")
    
    while True:
        try:
            key = input("\nPress key (n/p/q): ").lower().strip()
            if key in ['q', 'quit']:
                break
            elif key in ['n', 'j', 'next']:
                print("Testing NEXT action...")
                send_action(Action.NEXT, args.keys)
            elif key in ['p', 'k', 'prev', 'previous']:
                print("Testing PREV action...")
                send_action(Action.PREV, args.keys)
            else:
                print("Unknown command. Use n/p/q")
        except KeyboardInterrupt:
            break
    
    print("Test mode ended.")
    return 0

def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "gesture":
        return run_gesture_mode(args)
    elif args.mode == "test":
        return run_test_mode(args)
    else:
        return run_gaze_mode(args)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
