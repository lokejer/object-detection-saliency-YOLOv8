# cifar-cv

A learning-oriented computer vision repo built to explore real-time object detection,
image classification, and the Hugging Face Transformers workflow on a local GPU.

---

## Tech Stack

| Tool | Role |
|---|---|
| PyTorch (nightly, cu132) | Tensor ops, GPU inference |
| Hugging Face Transformers | ViT models + image processors |
| Ultralytics YOLOv8 | Real-time object detection + ByteTrack |
| OpenCV (`cv2`) | Webcam capture, frame rendering, trackbars |
| Hugging Face `datasets` | CIFAR-10 dataset loading |
| Matplotlib | Notebook visualization |

---

## Project Structure

```
cifar-cv/
├── camera.py            # entry point: device setup, capture loop, window/trackbar creation, cleanup
├── config.py            # all constants and tunable values — one place to change settings
├── detector.py          # Detector class + Detection dataclass; owns YOLO, ByteTrack, label voting
├── display.py           # draw_detections() and draw_hud(); pure rendering, no model/camera deps
├── imagenet-camera.py   # real-time imagenet vit classification on webcam
├── cifar10.ipynb        # cifar-10 inference walkthrough using a pretrained vit
├── model.py             # scratch file / early experiments (wip)
├── yolov8n.pt           # yolo nano weights (auto-downloaded, fastest)
├── yolov8m.pt           # yolo medium weights
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
pip install transformers ultralytics opencv-python pillow datasets matplotlib
```

---

## Setup

1. Clone or download this repo into your dev directory.
2. Install PyTorch nightly (see above).
3. Install remaining dependencies.
4. YOLOv8 weights (`yolov8l.pt` etc.) auto-download from Ultralytics on first run if not present.
5. The Hugging Face models (`google/vit-base-patch16-224`, `aaraki/vit-base-patch16-224-in21k-finetuned-cifar10`) are cached locally on first run.

---

## Key Commands

```bash
# real-time yolov8 object detection (main demo)
python camera.py

# real-time imagenet vit classification on webcam
python imagenet-camera.py

# open the cifar-10 notebook
jupyter notebook cifar10.ipynb
```

Press **Q** in any OpenCV window to quit.

---

## Workflow Overview

### YOLOv8 Detection — module breakdown

The detection pipeline was refactored from a single script into four focused modules.

#### `config.py` — constants and tunable values

Single source of truth for every magic number: model path, camera index, capture resolution,
window size, trackbar names, colours, blur kernel size, hysteresis gap, and label history length.
Change a setting once here; every other module reads from `config` instead of hardcoding values.

#### `detector.py` — `Detector` class + `Detection` dataclass

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

#### `display.py` — `draw_detections()` and `draw_hud()`

Pure rendering functions with no model or camera dependencies. Accepts a frame and a list
of `Detection` objects, draws bounding boxes and labels, and overlays the semi-transparent
HUD bar showing brightness offset, active/deactivate confidence thresholds, and compute device.
Keeping rendering isolated makes it easy to swap the UI without touching inference logic.

#### `camera.py` — entry point

Thin orchestrator: resolves the compute device, instantiates `Detector`, opens `VideoCapture`,
creates the named window and trackbars, then runs the main loop. Each iteration reads trackbar
values, applies brightness correction (`convertScaleAbs`) and Gaussian blur preprocessing,
calls `detector.track()`, calls both display functions, and checks for the quit key.
Cleanup (`cap.release()`, `destroyAllWindows()`) always runs on exit.

Webcam is forced to 720p (1280×720) before YOLO downsamples frames to its internal 640×640,
giving the model more signal to work with.

---

### `imagenet-camera.py` — ViT ImageNet Classification

Runs `google/vit-base-patch16-224` (1000-class ImageNet ViT) on each webcam frame:

- Converts BGR (OpenCV default) to RGB before passing to the Hugging Face processor.
- `model.eval()` disables dropout layers used only during training, making predictions
  deterministic and consistent across identical inputs.
- Preprocessing (resize to 224×224, normalize) is handled by `AutoImageProcessor`.
- Inference runs under `torch.no_grad()` — no gradient graph is built, saving memory.
- The predicted label and softmax confidence are overlaid on the live frame.

Good for experimenting with what a general-purpose ImageNet classifier sees in everyday scenes.

---

### `cifar10.ipynb` — CIFAR-10 Inference Notebook

Walks through the end-to-end Hugging Face inference pipeline on the CIFAR-10 dataset:

1. Load the `uoft-cs/cifar10` dataset via Hugging Face `datasets` (50k train / 10k test).
2. Visualize a sample image with Matplotlib.
3. Load `aaraki/vit-base-patch16-224-in21k-finetuned-cifar10` — a ViT pretrained on
   ImageNet-21k and fine-tuned on CIFAR-10's 10 classes.
4. Run inference: `processor` → `model(**inputs)` → `softmax` → predicted label.
5. Inspect raw logits and probability tensors.

This notebook is the conceptual foundation for the live camera scripts — the same
processor → model → softmax → `id2label` pattern is reused in both `.py` files.

---

## Concepts Covered

- **Inference vs. training** — using pretrained weights without gradient updates
- **ViT (Vision Transformer)** — patch-based image classification using attention
- **YOLO + ByteTrack** — single-stage detection with persistent cross-frame object IDs
- **BGR vs. RGB** — OpenCV's historical BGR ordering and when/why to convert
- **Softmax + logits** — converting raw model scores to probabilities
- **Hysteresis thresholding** — two-threshold approach to eliminate oscillation artifacts
- **Temporal voting** — stabilizing noisy per-frame predictions with a rolling majority vote
- **Dataclass as typed contract** — structured output between inference and rendering layers
- **Stale-state pruning** — preventing unbounded dict growth in long-running tracker sessions
- **`model.eval()` / `torch.no_grad()`** — inference-mode best practices

---

## Contributing

This is a personal learning project. Feel free to fork and experiment. Suggested extensions:

- Swap `yolov8l.pt` for `yolov8x.pt` for maximum accuracy, or `yolov8n.pt` for speed testing.
- Try a different backbone in `imagenet-camera.py` — any `AutoModelForImageClassification`
  model from the Hugging Face Hub (e.g. `microsoft/resnet-50`) will work with the same code.
- Add FPS counter overlay to benchmark model variants on your GPU.
- Extend the notebook to run batch inference and compute accuracy across the full test set.
- Adjust `CONF_HYSTERESIS_GAP` or `LABEL_HISTORY_LEN` in `config.py` to tune stability vs. responsiveness.
