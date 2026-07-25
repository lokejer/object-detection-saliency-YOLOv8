import cv2

import config
from detector import Detection


def draw_detections(frame, detections: list[Detection]) -> None:
    # pure rendering — no model or camera knowledge, just draws onto the frame in place
    for det in detections:
        # draw bounding box rectangle: top-left + bottom-right corners, color, thickness
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), config.BOX_COLOR, 2)

        # draw stable label + live confidence just above the top-left corner of the box
        text = f"{det.label} {det.confidence:.0%}"
        cv2.putText(frame, text, (det.x1, det.y1 - 8), config.BOX_FONT, 0.6, config.BOX_COLOR, 2)


def draw_heatmap(frame, cam_map) -> None:
    # blends a [0, 1] saliency map onto the frame in place
    # called before draw_detections and draw_hud so boxes and text stay crisp on top
    heatmap_color = cv2.applyColorMap((cam_map * 255).astype("uint8"), config.CAM_COLORMAP)
    # addWeighted writes into frame directly: alpha controls how strongly the heatmap shows
    cv2.addWeighted(heatmap_color, config.CAM_ALPHA, frame, 1 - config.CAM_ALPHA, 0, frame)


def draw_hud(frame, brightness_val: int, conf_high: float, device: str, fps: float, xai_on: bool = False) -> None:
    h, w = frame.shape[:2]
    bar_height = 36

    # semi-transparent dark bar at the bottom of the frame
    # only the bar strip is copied and blended, not the whole frame, since
    # blending identical pixels outside the bar would be wasted work every frame
    roi = frame[h - bar_height:h]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

    conf_low = max(0.0, conf_high - config.CONF_HYSTERESIS_GAP)
    offset   = brightness_val - config.BRIGHTNESS_CENTER  # positive = brighter, negative = darker

    brightness_label = f"Brightness: {offset:+d}"
    confidence_label = f"Confidence: {conf_high:.0%} (low: {conf_low:.0%})"
    xai_label        = f"XAI: {'on' if xai_on else 'off'}"
    fps_label        = f"FPS: {fps:4.1f}"
    device_label     = f"Device: {device}"

    # right-side labels are measured and placed from the right edge so a long
    # device string (multi-gpu ids, cpu names) can never overlap the fps readout
    (dev_w, _), _ = cv2.getTextSize(device_label, config.BOX_FONT, 0.55, 1)
    (fps_w, _), _ = cv2.getTextSize(fps_label,    config.BOX_FONT, 0.55, 1)
    device_x = w - dev_w - 12
    fps_x    = device_x - fps_w - 24

    cv2.putText(frame, brightness_label, (12, h - 12),       config.BOX_FONT, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, confidence_label, (220, h - 12),      config.BOX_FONT, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, xai_label,        (520, h - 12),      config.BOX_FONT, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, fps_label,        (fps_x, h - 12),    config.BOX_FONT, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, device_label,     (device_x, h - 12), config.BOX_FONT, 0.55, config.HUD_COLOR, 1)
