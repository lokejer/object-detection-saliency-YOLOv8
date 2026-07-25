# Real-time CV with YOLOv8

Exploring real-time object detection, persistent multi-object tracking, and explainable AI
(XAI) on a local GPU, using YOLOv8, Ultralytics tracking, Eigen-CAM, and OpenCV.

<img width="540" height="360" alt="v2 YOLOv8 demo" src="https://github.com/user-attachments/assets/6258e14b-26c0-4571-9f5f-ba741175817b" />

<!-- suggested image: replace or pair the demo above with a screenshot of the XAI overlay
     active (press E) — a JET heatmap concentrated on a detected person shows the new
     headline feature at a glance. a side-by-side (detection only | detection + heatmap)
     works even better. -->

---

## Tech Stack

| Tool | Role |
|---|---|
| PyTorch (nightly, cu132) | Tensor ops, GPU inference |
| Ultralytics YOLOv8 | Real-time object detection and persistent multi-object tracking |
| OpenCV (`cv2`) | Webcam capture, frame rendering, trackbars, heatmap blending |
| NumPy | Power iteration + PCA for the Eigen-CAM saliency map |

---

## Project Structure

```
YOLOv8-realtime-cv/
├── camera.py            # webcam loop, preprocessing, inference, XAI toggle, FPS, and display
├── config.py            # shared model, camera, UI, tracking, XAI, and smoothing settings
├── detector.py          # YOLO tracking, confidence hysteresis, label voting, and stale-state pruning
├── display.py           # bounding-box, heatmap, and HUD rendering functions
├── explain.py           # Eigen-CAM from captured detector activations using power iteration
├── yolov8n.pt           # yolo nano weights (fastest but least accurate)
└── yolov8l.pt           # yolo large weights (slower but most accurate)
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

### Setup

1. Clone or download this repo into your dev directory.
2. Install PyTorch nightly (see above).
3. Install remaining dependencies.
4. YOLOv8 weights (`yolov8l.pt` etc.) auto-download from Ultralytics on first run if not present.

---

### Key Commands

```bash
# real-time yolov8 object detection with optional XAI overlay
python camera.py
```

Press **Q** in the OpenCV window to quit. Press **E** to toggle the Eigen-CAM saliency heatmap.

---

## Workflow

Each file focuses on one task.

### `config.py` — constants and tunable values

Single source of truth for shared settings: model path, camera index, Windows DirectShow
capture backend, capture and window sizes, trackbar names, colours, blur kernel, hysteresis
gap, label-history length, tracking input width, XAI target layer, heatmap appearance, and
FPS smoothing. The application modules read these values from `config` instead of repeating
them.

### `detector.py` — `Detector` class + `Detection` dataclass

Owns all stateful inference logic. `Detector.__init__()` loads the YOLO model and moves it
to the target device, then puts the underlying network in evaluation mode. `Detector.track()`
calls `model.track(..., persist=True)` so tracker state and object IDs continue between
frames. No tracker configuration is passed explicitly, so Ultralytics uses the default for
the installed version (BoT-SORT in the current environment). `Detection` exposes integer box
coordinates, the stabilized class label, and the current confidence to `display.py`; the
tracker ID remains internal and keys the temporal state.

Key techniques inside `Detector`:

- **Hysteresis confidence thresholding** — a box activates at `conf_high` (slider) but only
  deactivates when confidence drops below `conf_low = max(0, conf_high - 0.15)`. Prevents rapid
  on/off flicker when confidence oscillates around a single threshold.
- **Temporal label voting** — keeps a rolling deque of the last 10 class predictions per
  track ID and displays the mode (most frequent), preventing class label flicker between
  visually similar categories.
- **Stale-track pruning** — IDs absent from the current tracking result are removed from the
  label-history and active-state dictionaries so retired tracks do not accumulate in memory.

### `explain.py` — `Explainer` class

Produces a live, class-agnostic Eigen-CAM saliency map from a late detector layer. The output
is a normalized `float32` array with the same height and width as the displayed frame.

- **Activation reuse** — `enable()` attaches a forward hook to the configured target layer.
  The hook saves the latest detached activation produced by `Detector.track()`, so the
  explainer does not run the model a second time. `disable()` removes the hook and releases
  the stored activation.
- **Power iteration rather than SVD** — the activation tensor is reshaped so every spatial
  cell becomes one channel vector and is centered across space. Eight power-iteration steps
  estimate only the dominant principal component. The previous frame's component is reused
  as the next starting vector for faster convergence and better temporal stability.
- **Detection-guided sign selection** — PCA direction has an arbitrary sign. The explainer
  forms positive and negative ReLU maps, compares their mean response inside detected boxes
  against the background, and keeps the orientation that emphasizes the detections. When
  there is no usable box mask, it keeps the orientation with the stronger peak response.
- **Letterbox-aware projection** — the feature map is expanded to the configured tracking
  input width, vertical padding is removed for the webcam's aspect ratio, and the result is
  resized back to the displayed frame. Blank or unavailable activations produce an all-zero
  map instead of an error.

<!-- suggested image: a before/after pair here showing the inverted heatmap (background hot,
     person cold) next to the corrected one — it makes the sign-ambiguity explanation
     instantly clear and documents a real debugging story. -->

### `display.py` — `draw_detections()`, `draw_hud()`, and `draw_heatmap()`

Rendering functions with no model or camera-capture work. `draw_heatmap()` converts a
normalized saliency map to the configured OpenCV colormap and alpha-blends it into the frame.
`draw_detections()` then adds blue boxes with stabilized labels and live confidence values.
`draw_hud()` blends a dark strip across the bottom and displays brightness offset, activation
and deactivation confidence thresholds, XAI state, smoothed FPS, and compute device.

### `camera.py` — entry point

Thin orchestrator: resolves the compute device, instantiates `Detector` and `Explainer`,
opens the webcam through DirectShow, requests 1280×720 capture, creates the window and two
trackbars, then runs the frame loop. Each frame is mirrored, brightness-adjusted with
`convertScaleAbs`, denoised with a 3×3 Gaussian blur, and passed to `Detector.track()`. If
XAI is enabled, the latest captured activation becomes a heatmap that is blended beneath the
boxes and HUD. **E** attaches or removes the activation hook; **Q** exits the loop.

The tracker uses Ultralytics' default inference size, currently a width of 640 pixels, while
preserving the frame aspect ratio with letterboxing rather than stretching every frame to
640×640.

The FPS sample includes capture, mirroring, preprocessing, tracking, and optional heatmap
generation/blending. Its timestamp is taken before the final boxes, HUD, `imshow()`, and
keyboard handling, so those UI operations are not part of the measured interval. An
exponential moving average reduces frame-to-frame jitter in the displayed value.

---

## Tuning/Improvements Made

- **YOLO + persistent tracking** — single-stage detection with tracker state retained across frames
- **Hysteresis thresholding** — two-threshold approach to eliminate oscillation artifacts
- **Temporal voting** — stabilizing noisy per-frame predictions with a rolling majority vote
- **Dataclass as typed contract** — structured output between inference and rendering layers
- **Stale-state pruning** — preventing unbounded dict growth in long-running tracker sessions
- **Exponential moving average** — smoothing a jittery per-frame FPS signal without a history buffer
- **Live Eigen-CAM saliency (XAI)** — class-agnostic, gradient-free heatmap showing which regions
  drive the model, toggled live with **E**
- **Activation reuse via forward hooks** — reading intermediate activations from the pass that
  already happened, turning a second inference into free data
- **Power iteration** — finding the dominant principal component with cheap matrix-vector
  products instead of a full SVD, warm-started across frames for speed and stability
- **Sign disambiguation with detections** — PCA components have arbitrary sign; the detected
  boxes pick the orientation that lights up the objects

---

### Further Tuning/Improvements

- Swap `yolov8l.pt` for `yolov8x.pt` for maximum accuracy, or `yolov8n.pt` for speed testing,
  and read the steady-state FPS off the HUD to compare.
- Adjust `CONF_HYSTERESIS_GAP` or `LABEL_HISTORY_LEN` in `config.py` to tune stability vs. responsiveness.
- Tune `FPS_SMOOTHING` in `config.py` for a smoother (lower) or more responsive (higher) FPS readout.
- Tune `XAI_TARGET_LAYER_INDEX` in `config.py` to explore abstraction levels: earlier layers
  highlight textures and edges, later layers highlight whole objects.
- Per-detection explanation — freeze a frame, then run Grad-CAM++ or D-RISE on an individual
  box to explain why that specific object was detected (finer attribution, more compute).
- Per-box uncertainty via test-time augmentation — re-run a detection under small crops, flips,
  or brightness shifts and measure confidence variance, a trust signal without Bayesian inference.

<!-- suggested image: a short gif at the top of the README (or here) toggling the E key on a
     live scene — detection boxes stay put while the heatmap fades in over the objects.
     gifs autoplay on github and demo real-time behaviour better than any still. -->
