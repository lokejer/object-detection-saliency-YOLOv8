# single source of truth for all tunable values and constants
# change settings here rather than hunting through other files

MODEL_PATH  = "yolov8l.pt"    # yolov8n=fastest, yolov8s/m/l/x=progressively more accurate
CAMERA_INDEX = 0              # 0 = default webcam; 1 = external camera

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
CONF_HYSTERESIS_GAP = 0.25   # conf_low = conf_high - this; prevents on/off flicker

BLUR_KERNEL        = (5, 5)  # gaussian kernel for pre-inference denoising; increase for grainier cameras
LABEL_HISTORY_LEN  = 10      # rolling window of past labels used for majority-vote stabilisation

BOX_COLOR = (0, 255, 0)      # BGR green for bounding boxes
HUD_COLOR = (100, 220, 255)  # BGR yellow-white for HUD text

# fps smoothing: exponential moving average weight for the newest frame
# lower = smoother but laggier readout; higher = jumpier but more responsive
FPS_SMOOTHING = 0.1
