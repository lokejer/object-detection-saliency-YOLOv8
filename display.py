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
        cv2.putText(frame, text, (det.x1, det.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.BOX_COLOR, 2)


def draw_hud(frame, brightness_val: int, conf_high: float, device: str, fps: float) -> None:
    h, w = frame.shape[:2]
    bar_height = 36

    # semi-transparent dark bar at the bottom of the frame
    # addWeighted blends two images: dst = alpha*src1 + beta*src2 + gamma
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    conf_low = max(0.0, conf_high - config.CONF_HYSTERESIS_GAP)
    offset   = brightness_val - config.BRIGHTNESS_CENTER  # positive = brighter, negative = darker

    brightness_label = f"Brightness: {offset:+d}"
    confidence_label = f"Confidence: {conf_high:.0%} (low: {conf_low:.0%})"
    fps_label        = f"FPS: {fps:4.1f}"
    device_label     = f"Device: {device}"

    cv2.putText(frame, brightness_label, (12, h - 12),      cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, confidence_label, (220, h - 12),     cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, fps_label,        (w - 290, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.HUD_COLOR, 1)
    cv2.putText(frame, device_label,     (w - 160, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.HUD_COLOR, 1)
