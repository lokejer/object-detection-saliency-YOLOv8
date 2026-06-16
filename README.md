# cifar-cv

A learning-oriented computer vision repo built to explore real-time object detection
on a local GPU, using YOLOv8, ByteTrack, and OpenCV.

---

## Tech Stack

| Tool | Role |
|---|---|
| PyTorch (nightly, cu132) | Tensor ops, GPU inference |
| Ultralytics YOLOv8 | Real-time object detection + ByteTrack |
| OpenCV (`cv2`) | Webcam capture, frame rendering, trackbars |

---

## Project Structure

```
cifar-cv/
├── camera.py            # entry point: device setup, capture loop, window/trackbar creation, cleanup
├── config.py            # all constants and tunable values — one place to change settings
├── detector.py          # Detector class + Detection dataclass; owns YOLO, ByteTrack, label voting
├── display.py           # draw_detections() and draw_hud(); pure rendering, no model/camera deps
├── yolov8n.pt           # yolo nano weights (auto-downloaded, fastest)
└── yolov8l.pt           # yolo large weights (used by default)
```

---

## Prerequisites

- Python 3.13
- NVIDIA GPU with CUDA support (project targets RTX 5060 Laptop, Blackwell CC 12.0)
- A webcam (built-in or external)

### PyTorch — Blackwell / CUDA 13.2

The RTX 5060 (Blackwell) requires PyTorch nightly with CUDA 13.2 support:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

> For older GPUs with stable CUDA support, use the standard install from https://pytorch.org/get-started/locally/

### Other dependencies

```bash
pip install ultralytics opencv-python
```

---

## Setup

1. Clone or download this repo into your dev directory.
2. Install PyTorch nightly (see above).
3. Install remaining dependencies.
4. YOLOv8 weights (`yolov8l.pt` etc.) auto-download from Ultralytics on first run if not present.

---

## Key Commands

```bash
# real-time yolov8 object detection
python camera.py
```

Press **Q** in the OpenCV window to quit.

---

## Workflow Overview

The detection pipeline is split into four focused modules following the Single
Responsibility Principle — each file does one thing.

### `config.py` — constants and tunable values

Single source of truth for every magic number: model path, camera index, capture resolution,
window size, trackbar names, colours, blur kernel size, hysteresis gap, label history length,
and the FPS smoothing factor. Change a setting once here; every other module reads from
`config` instead of hardcoding values.

### `detector.py` — `Detector` class + `Detection` dataclass

Owns all stateful inference logic. `Detector.__init__()` loads the YOLO model and moves it
to the target device. `Detector.track()` runs inference each frame and returns a list of
`Detection` dataclass instances (typed coordinates, label, confidence) — a clean contract
that `display.py` consumes without needing to know anything about YOLO internals.

Key techniques inside `Detector`:

- **Hysteresis confidence thresholding** — a box activates at `conf_high` (slider) but only
  deactivates when confidence drops below `conf_low = conf_high - 0.25`. Prevents rapid
  on/off flicker when confidence oscillates around a single threshold.
- **Temporal label voting** — keeps a rolling deque of the last 10 class predictions per
  track ID and displays the mode (most frequent), preventing class label flicker between
  visually similar categories.
- **Stale-track pruning** — ByteTrack IDs increase monotonically across a session, so retired
  IDs are removed from the state dicts each frame to prevent unbounded memory growth.

### `display.py` — `draw_detections()` and `draw_hud()`

Pure rendering functions with no model or camera dependencies. Accepts a frame and a list
of `Detection` objects, draws bounding boxes and labels, and overlays the semi-transparent
HUD bar showing brightness offset, active/deactivate confidence thresholds, live FPS, and
compute device. Keeping rendering isolated makes it easy to swap the UI without touching
inference logic.

### `camera.py` — entry point

Thin orchestrator: resolves the compute device, instantiates `Detector`, opens `VideoCapture`,
creates the named window and trackbars, then runs the main loop. Each iteration reads trackbar
values, applies brightness correction (`convertScaleAbs`) and Gaussian blur preprocessing,
calls `detector.track()`, measures end-to-end FPS, calls both display functions, and checks
for the quit key. Cleanup (`cap.release()`, `destroyAllWindows()`) always runs on exit.

Webcam is forced to 720p (1280×720) before YOLO downsamples frames to its internal 640×640,
giving the model more signal to work with.

The FPS counter measures the full loop time (capture + preprocess + inference + draw) and
smooths it with an exponential moving average, so the HUD reading is a true end-to-end
throughput number — useful for benchmarking model variants on your GPU.

---

## Concepts Covered

- **Inference vs. training** — using pretrained weights without gradient updates
- **YOLO + ByteTrack** — single-stage detection with persistent cross-frame object IDs
- **BGR vs. RGB** — OpenCV's historical BGR ordering and when/why to convert
- **Hysteresis thresholding** — two-threshold approach to eliminate oscillation artifacts
- **Temporal voting** — stabilizing noisy per-frame predictions with a rolling majority vote
- **Dataclass as typed contract** — structured output between inference and rendering layers
- **Stale-state pruning** — preventing unbounded dict growth in long-running tracker sessions
- **Exponential moving average** — smoothing a jittery per-frame FPS signal without a history buffer

---

## Contributing

This is a personal learning project. Feel free to fork and experiment. Suggested extensions:

- Swap `yolov8l.pt` for `yolov8x.pt` for maximum accuracy, or `yolov8n.pt` for speed testing,
  and read the steady-state FPS off the HUD to compare.
- Adjust `CONF_HYSTERESIS_GAP` or `LABEL_HISTORY_LEN` in `config.py` to tune stability vs. responsiveness.
- Tune `FPS_SMOOTHING` in `config.py` for a smoother (lower) or more responsive (higher) FPS readout.
