from dataclasses import dataclass
from collections import deque, Counter

from ultralytics import YOLO

import config


@dataclass
class Detection:
    # structured output from Detector.track() — passed to display.py for rendering
    # using a dataclass instead of raw tuples makes it clear what each value means
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float


class Detector:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)

        # lock the underlying network in inference mode so batchnorm and dropout
        # behave deterministically for tracking and for any explainer sharing this model
        self.model.model.eval()

        # per-track state keyed by ByteTrack integer ID (persistent across frames)
        self._label_history: dict[int, deque] = {}
        self._active_tracks: dict[int, bool] = {}

    def track(self, frame, conf_high: float) -> list[Detection]:
        # HYSTERESIS THRESHOLDING
        #   conf_low is derived from conf_high using the fixed hysteresis gap in config
        #   only one slider needed on the HUD. The gap is a design decision, not a user setting
        conf_low = max(0.0, conf_high - config.CONF_HYSTERESIS_GAP)

        # model.track() activates ByteTrack: assigns persistent integer IDs to objects
        # persist=True keeps tracker state alive between calls
        results = self.model.track(frame, device=self.device, verbose=False, persist=True)

        detections = []
        current_ids: set[int] = set()

        for box in results[0].boxes:
            # box.id is None on the very first frame before the tracker assigns ids
            if box.id is None:
                continue

            track_id  = int(box.id[0])
            confidence = float(box.conf[0])
            class_id  = int(box.cls[0])
            raw_label = self.model.names[class_id]  # id→label dict, same role as HuggingFace id2label

            current_ids.add(track_id)

            # hysteresis gate: a box activates at conf_high, deactivates only below conf_low
            # confidence between the two thresholds holds its previous state — no change
            if confidence >= conf_high:
                self._active_tracks[track_id] = True
            elif confidence < conf_low:
                self._active_tracks[track_id] = False

            if not self._active_tracks.get(track_id, False):
                continue

            # label voting: accumulate labels over the last N frames, display the majority
            # prevents class flickering when two labels have similar confidence scores
            if track_id not in self._label_history:
                self._label_history[track_id] = deque(maxlen=config.LABEL_HISTORY_LEN)
            self._label_history[track_id].append(raw_label)
            stable_label = Counter(self._label_history[track_id]).most_common(1)[0][0]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(x1, y1, x2, y2, stable_label, confidence))

        # prune state for track IDs no longer present — prevents unbounded dict growth
        self._prune_stale(current_ids)
        return detections

    def _prune_stale(self, current_ids: set[int]) -> None:
        # retired track IDs are removed each frame to prevent memory leaking
        # over long sessions ByteTrack IDs increase monotonically, so without pruning
        # the dicts grow forever even when few objects are actually on screen
        stale = set(self._label_history) - current_ids
        for tid in stale:
            self._label_history.pop(tid, None)
            self._active_tracks.pop(tid, None)
