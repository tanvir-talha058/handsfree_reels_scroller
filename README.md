<<<<<<< HEAD
# Handsfree Reels Scroller

Control vertical short-video / reels feeds (e.g., Instagram, YouTube Shorts, TikTok in a browser) using camera-based hand swipe gestures (and future eye-gaze) instead of keyboard/mouse.

## Features (Initial MVP)
- Real-time webcam capture
- Basic left/right (or up/down) swipe gesture detection using hand landmarks (Mediapipe Hands)
- Maps gestures to keyboard presses (default: Up / Down arrow)
- Adjustable sensitivity & debounce
- Modular design to later plug in eye tracking for dwell-based click/scroll

## Roadmap
- [ ] Eye tracking prototype (gaze-based pointer)
- [ ] Calibration routine
- [ ] Config file & CLI arguments for sensitivity
- [ ] Multi-hand support & pause gesture

## Installation

**Important:** MediaPipe requires Python 3.8-3.11. If you have Python 3.12+, see troubleshooting below.

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Troubleshooting Python 3.12+
If MediaPipe installation fails:

**Option 1: Use Python 3.11 (Recommended)**
```powershell
# Install Python 3.11 via conda or pyenv
conda create -n reels python=3.11
conda activate reels
pip install -r requirements.txt
```

**Option 2: Fallback dependencies**
```powershell
pip install -r requirements-fallback.txt
# Note: Gesture detection won't work without MediaPipe
```

## Run Gesture Mode
```bash
python -m src.handsfree_reels_scroller.main --mode gesture
```
Then focus your browser window with the reels feed.

### Quick Run Summary
1. Create & activate venv: (PowerShell)
	```powershell
	python -m venv .venv
	. .venv/ScriptS/Activate.ps1
	```
2. Install deps:
	```powershell
	pip install -r requirements.txt
	```
3. Run app (gesture mode):
	```powershell
	python -m src.handsfree_reels_scroller.main --mode gesture --min-displacement 0.15 --max-duration 0.6 --cooldown 0.8 --axis vertical
	```
4. Perform clear downward swipe (for next) or upward swipe (for previous).
5. Press `q` in the video window to exit.

Adjust thresholds to reduce accidental triggers (increase `--min-displacement` or `--cooldown`).

## Usage Tips
- Ensure good lighting and plain background behind hands.
- Keep hand inside camera frame.
- Perform a clean horizontal swipe (left->right or right->left) or vertical depending on configured axis.

## Safety & Terms
Automating input may violate some platforms' terms of service. Use only for personal accessibility / experimentation.

## License
MIT (add a LICENSE file if distributing).
=======
# handsfree_reels_scroller
>>>>>>> 6024bed12fdb6ee515c3e2c73dfd45946a0e1c02
