# single source of truth for all tunable values and constants
# change settings here rather than hunting through other files

import cv2  # imported only for named constants like colormaps, no runtime cv2 work here

MODEL_PATH  = "yolov8l.pt"    # yolov8n=fastest, yolov8s/m/l/x=progressively more accurate
CAMERA_INDEX = 0              # 0 = default webcam; 1 = external camera
CAMERA_BACKEND = cv2.CAP_DSHOW  # reliable Windows capture backend; avoids MSMF stream failures

CAPTURE_WIDTH  = 1280         # native resolution before YOLO downsamples to 640x640
CAPTURE_HEIGHT = 720

WINDOW_NAME = "YOLOv8 detection"
WINDOW_W    = 1200
WINDOW_H    = 900

# trackbar names are also used as keys for getTrackbarPos — define once, reference everywhere
TRACKBAR_BRIGHTNESS = "Brightness   0=dark | 100=normal | 200=bright"
TRACKBAR_CONFIDENCE = "Confidence % (detections shown above this)"

BRIGHTNESS_CENTER  = 100     # slider midpoint = no adjustment (beta offset of 0)
BRIGHTNESS_MAX     = 200
CONF_DEFAULT       = 60      # default slider position (%)
CONF_HYSTERESIS_GAP = 0.15   # conf_low = conf_high - this; prevents on/off flicker

BLUR_KERNEL        = (3, 3)  # gaussian kernel for pre-inference denoising; increase for grainier cameras
LABEL_HISTORY_LEN  = 10      # rolling window of past labels used for majority-vote stabilisation

BOX_COLOR = (255, 0, 0)      # BGR green for bounding boxes
BOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_COLOR = (100, 220, 255)  # BGR yellow-white for HUD text

# xai / eigen-cam settings for the live saliency overlay (toggled with the E key)
# eigen-cam reads raw activations from one layer, so which layer we pick decides
# what the heatmap shows. -2 is a late c2f block: deep enough to be semantic
# (responds to whole objects) but still spatial enough to localise them.
# note: an inverted-looking map (hot background, cold objects) is usually the
# arbitrary svd sign, not the wrong layer, and is auto-corrected in explain.py
XAI_TARGET_LAYER_INDEX = -2
TRACK_IMGSZ  = 640           # the tracker's inference width, used to undo letterboxing
CAM_ALPHA    = 0.5           # heatmap blend strength: 0 = invisible, 1 = fully opaque
CAM_COLORMAP = cv2.COLORMAP_JET  # blue = low attention, red = high attention

# fps smoothing: exponential moving average weight for the newest frame
# lower = smoother but laggier readout; higher = jumpier but more responsive
FPS_SMOOTHING = 0.1
