import time

import cv2
import torch

import config
from detector import Detector
from display import draw_detections, draw_heatmap, draw_hud
from explain import Explainer

if __name__ == "__main__":
    print("Camera running...\nPress Q to quit, E to toggle the XAI heatmap.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Model using {device}")

    detector = Detector(config.MODEL_PATH, device)

    # explainer shares the detector's already-loaded model, so no second copy in vram
    # construction is cheap: the cam hook is only attached when the overlay is toggled on
    explainer = Explainer(detector, device)
    xai_on = False  # no overlay by default

    cap = cv2.VideoCapture(config.CAMERA_INDEX, config.CAMERA_BACKEND)
    # force 720p — higher native resolution gives YOLO more signal before it downsamples to 640x640
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)


    # fail loudly if camera does not exist. this guards against the camera being unavailable as VideoCapture(0) succeeds silently
    if not cap.isOpened():
        print("Error: could not open camera.")
        raise SystemExit(1)

    # create window before overlaying trackbars
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_W, config.WINDOW_H)

    # createTrackbar(label, window, default_value, max_value, on_change_callback)
    #   lambda _: None = no-op callback; values are read manually each frame with getTrackbarPos
    cv2.createTrackbar(config.TRACKBAR_BRIGHTNESS, config.WINDOW_NAME, config.BRIGHTNESS_CENTER, config.BRIGHTNESS_MAX, lambda _: None)
    cv2.createTrackbar(config.TRACKBAR_CONFIDENCE, config.WINDOW_NAME, config.CONF_DEFAULT, 100, lambda _: None)

    # fps state: smoothed reading + timestamp of the previous loop iteration
    # perf_counter is a high-resolution monotonic clock — ideal for measuring short durations
    # start at a plausible value so the ema does not slowly ramp up from zero during the first seconds and show a misleading readout
    fps = 30.0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # read live trackbar positions each frame 
        #   values update instantly as user drags
        brightness_val = cv2.getTrackbarPos(config.TRACKBAR_BRIGHTNESS, config.WINDOW_NAME)  # 0-200
        conf_high      = cv2.getTrackbarPos(config.TRACKBAR_CONFIDENCE, config.WINDOW_NAME) / 100

        # ADJUST BRIGHTNESS
        #   applied before inference so the model sees the adjusted image
        #   convertScaleAbs: output = clip(alpha * pixel + beta, 0, 255)
        beta  = brightness_val - config.BRIGHTNESS_CENTER  # 100 - 100 = 0; no change in brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

        # GAUSSIAN BLUR
        #   smooths out camera sensor noise before inference
        #   reduces false detections caused by high-frequency noise being interpreted as texture
        frame = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)

        detections = detector.track(frame, conf_high)

        # PIXEL IMPORTANCE HEATMAP
        #   reuses the activations track() just produced, so it costs only
        #   a small svd. detections guide the sign of the saliency (see explain.py).
        #   drawn first so the boxes and hud stay nicely on top of the blended overlay
        if xai_on:
            cam_map = explainer.heatmap(frame, detections)
            draw_heatmap(frame, cam_map)

        # FRAME SMOOTHING (EXPONENTIAL MOVING AVERAGE )
        #   measure the full loop time (capture + preprocess + inference + draw) for a true
        #   end-to-end throughput number, then smooth it with an exponential moving average
        #   so the readout does not jitter every frame
        now = time.perf_counter()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)  # guard against divide-by-zero
        fps = config.FPS_SMOOTHING * instant_fps + (1 - config.FPS_SMOOTHING) * fps
        prev_time = now

        draw_detections(frame, detections)
        draw_hud(frame, brightness_val, conf_high, device, fps, xai_on)

        # KEY TOGGLES
        #   imshow renders the updated frame; waitKey(1) gives it 1ms to process events
        #   0xFF mask ensures the key code fits in 8 bits across platforms
        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        # QUIT
        if key == ord("q"):
            break

        # PIXEL IMPORTANCE HEATMAP
        if key == ord("e"):

            # flip the overlay state, the hud shows whether it is currently on
            # enable/disable also attaches or removes the cam's forward hook, so
            # activation capture (and its memory cost) only exists while the overlay is on
            xai_on = not xai_on
            if xai_on:
                explainer.enable()
            else:
                explainer.disable()


    cap.release()
    cv2.destroyAllWindows()
