# Real-Time CV with YOLOv8

Live object detection, persistent multi-object tracking, and explainable AI using
Ultralytics YOLOv8, OpenCV, and Eigen-CAM. The application runs on a live webcam, while a
separate evaluation workflow compares ByteTrack and BoT-SORT on prerecorded video.

The live explanation is a class-agnostic feature-activation map. Per-detection pixel
attribution with D-RISE is currently WIP.

#### live detection demo:

<img width="540" height="360" alt="YOLOv8 live detection demo" src="https://github.com/user-attachments/assets/6258e14b-26c0-4571-9f5f-ba741175817b" />

#### simplified [MOTChallenge benchmark](https://arxiv.org/abs/2010.07548) demo:

![BoT-SORT tracking preview](evaluation/bot-sort%20video%20demo.gif)

---

## Tech Stack

| tool | role |
|---|---|
| PyTorch | tensor operations and CPU/CUDA inference |
| Ultralytics YOLOv8 | object detection and persistent tracking |
| ByteTrack / BoT-SORT | multi-object tracking algorithms |
| OpenCV | camera capture, preprocessing, controls, and rendering |
| NumPy | power iteration and PCA for the Eigen-CAM map |
| pandas / SciPy | tracker evaluation and metric calculation |

---

## (super) Quick Start

the project is tested with Python 3.13 on Windows. the live application uses DirectShow by
default; change `CAMERA_BACKEND` in [`config.py`](config.py) for another operating system.
CUDA is optional, although CPU inference is slower.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

the current RTX 5060 Laptop setup uses the PyTorch nightly CUDA 13.2 build. install it before
`requirements.txt` when reproducing that environment:

```powershell
python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

install the remaining dependencies and start the live application:

```powershell
python -m pip install -r requirements.txt
python camera.py
```

Ultralytics downloads the configured `yolov8l.pt` weights if they are not already in the
repository root. change `MODEL_PATH` to `yolov8n.pt` if you want faster inference, sacrificing some accuracy.

press **Q** to quit and **E** to toggle the class-agnostic Eigen-CAM overlay.

to run the tracker comparison, provide `evaluation/stationary_webcam_eval.mp4` and completed
MOTChallenge annotations at `evaluation/ground_truth.txt`, then run:

```powershell
python experiment.py
```

---

## How Is This Possible?

1. [`camera.py`](camera.py) captures and mirrors each frame, applies the brightness control
   and Gaussian blur, and passes the processed image to the detector.
2. [`detector.py`](detector.py) runs persistent YOLO tracking, applies confidence hysteresis,
   stabilizes labels with temporal voting, and removes stale track state.
3. when XAI is enabled, [`explain.py`](explain.py) reuses a hooked detector activation and
   estimates its dominant Eigen-CAM component without a second inference pass.
4. [`display.py`](display.py) blends the heatmap first, then draws detections and the
   FPS/device/settings HUD on top.

[`experiment.py`](experiment.py) is a separate prerecorded-video path. it runs ByteTrack and
BoT-SORT independently, matches their outputs to ground truth, and reports tracking accuracy
and latency metrics.

---

## Highlights

- persistent multi-object tracking with confidence hysteresis, label voting, and stale-state cleanup
- live Eigen-CAM generated from reused model activations and warm-started power iteration
- letterbox-aware heatmap projection with detection-guided PCA sign selection
- adjustable brightness and confidence controls with smoothed end-to-end FPS reporting
- controlled ByteTrack versus BoT-SORT evaluation using HOTA, IDF1, MOTA, precision, recall,
  identity switches, fragmentation, and latency

the Eigen-CAM overlay is scene-wide and class-agnostic: it visualizes a dominant internal
feature direction, not causal pixel importance for an individual detection.

---

## Documentation

- [`improvements.md`](improvements.md) — implementation history, design rationale, and open gaps
- [`x.md`](x.md) — tracker experiment inputs, setup, metrics, and decision rule
- [`bytetrack_vs_botsort.md`](bytetrack_vs_botsort.md) — recorded benchmark results and interpretation
- [`README_orig.md`](README_orig.md) — preserved pre-consolidation README
