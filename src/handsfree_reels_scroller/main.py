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

from .actions import Action
from .gesture_recognition import GestureRecognizer, SwipeConfig

# Key mapping for actions (can be adapted per platform)
KEY_NEXT = "down"
KEY_PREV = "up"


def send_action(action: Action) -> None:
    if pyautogui is None:
        print(f"[DRY RUN] Would send: {action}")
        return
    if action == Action.NEXT:
        pyautogui.press(KEY_NEXT)
    elif action == Action.PREV:
        pyautogui.press(KEY_PREV)
    else:
        print(f"Unhandled action: {action}")


def run_gesture_mode(args) -> int:
    try:
        recognizer = GestureRecognizer(
            SwipeConfig(
                min_displacement=args.min_displacement,
                max_duration=args.max_duration,
                axis=args.axis,
                cooldown=args.cooldown,
            )
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nSolutions:")
        print("1. Use Python 3.8-3.11: conda create -n reels python=3.11")
        print("2. Install from fallback requirements: pip install -r requirements-fallback.txt")
        print("3. Or manually install compatible versions")
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
                send_action(action)
            # Show window for feedback
            cv2.imshow("Handsfree Reels (Gesture)", frame)
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
    p.add_argument("--mode", choices=["gesture", "gaze"], default="gesture")
    p.add_argument("--min-displacement", type=float, default=0.15)
    p.add_argument("--max-duration", type=float, default=0.6)
    p.add_argument("--axis", choices=["vertical", "horizontal"], default="vertical")
    p.add_argument("--cooldown", type=float, default=0.8)
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "gesture":
        return run_gesture_mode(args)
    else:
        return run_gaze_mode(args)

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
