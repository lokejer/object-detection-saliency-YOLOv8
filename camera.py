import cv2
import torch

import config
from detector import Detector
from display import draw_detections, draw_hud

if __name__ == "__main__":
    print("Camera running...\nPress Q to quit.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Model using {device}")

    detector = Detector(config.MODEL_PATH, device)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    # force 720p — higher native resolution gives YOLO more signal before it downsamples to 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

    # guard against the camera being unavailable — VideoCapture(0) succeeds silently even when
    # no camera exists; isOpened() is the actual check
    if not cap.isOpened():
        print("Error: could not open camera.")
        raise SystemExit(1)

    # namedWindow must be called before createTrackbar — trackbars attach to a named window
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_W, config.WINDOW_H)

    # createTrackbar(label, window, default_value, max_value, on_change_callback)
    # lambda _: None = no-op callback; values are read manually each frame with getTrackbarPos
    cv2.createTrackbar(config.TRACKBAR_BRIGHTNESS, config.WINDOW_NAME, config.BRIGHTNESS_CENTER, config.BRIGHTNESS_MAX, lambda _: None)
    cv2.createTrackbar(config.TRACKBAR_CONFIDENCE, config.WINDOW_NAME, config.CONF_DEFAULT, 100, lambda _: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # read live trackbar positions each frame — values update instantly as user drags
        brightness_val = cv2.getTrackbarPos(config.TRACKBAR_BRIGHTNESS, config.WINDOW_NAME)
        conf_high      = cv2.getTrackbarPos(config.TRACKBAR_CONFIDENCE, config.WINDOW_NAME) / 100

        # apply brightness correction before inference so the model sees the adjusted image
        # convertScaleAbs: output = clip(alpha * pixel + beta, 0, 255)
        # beta shifts all pixel values uniformly — positive = brighter, negative = darker
        beta  = brightness_val - config.BRIGHTNESS_CENTER
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

        # gaussian blur smooths out camera sensor noise before inference
        # reduces false detections caused by high-frequency noise being interpreted as texture
        frame = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)

        detections = detector.track(frame, conf_high)

        draw_detections(frame, detections)
        draw_hud(frame, brightness_val, conf_high, device)

        # imshow renders the updated frame; waitKey(1) gives it 1ms to process events
        # 0xFF mask ensures the key code fits in 8 bits across platforms
        cv2.imshow(config.WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # always release camera and destroy windows — not doing so can leave the camera occupied
    cap.release()
    cv2.destroyAllWindows()
