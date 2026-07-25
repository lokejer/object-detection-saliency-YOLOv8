import cv2
import numpy as np

import config
from detector import Detection

# power iteration steps for finding the first principal component. an algorithm
# detail rather than a user tunable, so it lives here instead of config.py
_POWER_ITERS = 8


class Explainer:
    # produces an eigen-cam saliency heatmap by reusing the activations the tracker
    # already computed for this frame, instead of running a second forward pass.
    # eigen-cam is just pca over one layer's activations: no gradients, no target
    # class, so the only real cost left is a small svd, which keeps the overlay
    # close to free while it is on
    def __init__(self, detector, device: str):
        self.device = device

        # ultralytics wraps the real network: YOLO -> DetectionModel -> Sequential of blocks
        raw_model = detector.model.model

        # the target layer decides what the heatmap shows: too early = edges and texture,
        # too late = blobs with no location. a late c2f block is the sweet spot
        self.target_layer = raw_model.model[config.XAI_TARGET_LAYER_INDEX]

        # hook state: attached only while the overlay is on, so activation capture
        # (and its memory) does not exist during normal runs
        self._hook = None
        self._activation = None

        # previous frame's principal component, used to warm-start power iteration.
        # activations change slowly between frames, so starting from last frame's
        # answer converges in a couple of steps and keeps the component's sign
        # continuous over time, which stops the heatmap flickering
        self._pc = None

    def enable(self) -> None:
        if self._hook is not None:
            return

        # a forward hook fires whenever the tracker runs the model, handing us the
        # layer output for free. only the latest activation is kept, so memory
        # stays constant no matter how long the session runs
        def _save(module, inputs, output):
            self._activation = output.detach()

        self._hook = self.target_layer.register_forward_hook(_save)

    def disable(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
            self._activation = None

    def heatmap(self, frame, detections: list[Detection]) -> np.ndarray:
        # returns a [0, 1] float32 saliency map the same height and width as the frame
        h, w = frame.shape[:2]

        # no activation yet means the tracker has not run since enable(), show nothing
        if self._activation is None:
            return np.zeros((h, w), np.float32)

        acts = self._activation[0].float().cpu().numpy()  # (channels, hf, wf)
        c, hf, wf = acts.shape

        # eigen-cam core: treat each spatial cell as a c-dimensional vector, centre
        # them, and project onto the first principal component. that component is
        # the dominant activation pattern, which for a detector is normally the objects
        flat = acts.reshape(c, -1).T          # (cells, channels)
        flat = flat - flat.mean(axis=0)

        # power iteration instead of a full svd: repeatedly multiplying by the data
        # pulls any vector toward the largest principal component, and we only need
        # that one. a full svd computes all 200+ components and costs ~45 ms here,
        # this costs well under a millisecond
        v = self._pc if self._pc is not None and self._pc.shape == (c,) else np.ones(c, np.float32)
        for _ in range(_POWER_ITERS):
            v = flat.T @ (flat @ v)
            norm = float(np.linalg.norm(v))
            if norm < 1e-12:
                return np.zeros((h, w), np.float32)  # blank activations, nothing to show
            v = v / norm
        self._pc = v

        proj = (flat @ v).reshape(hf, wf).astype(np.float32)

        # the activation came from the tracker's letterboxed input, so undo that
        # geometry: scale up to input size, crop the vertical padding bars, and work
        # at that content size. this way the heatmap explains the exact image the
        # tracker saw, and stays aligned with the display
        in_w = config.TRACK_IMGSZ
        in_h = round(hf * in_w / wf)
        proj = cv2.resize(proj, (in_w, in_h))
        content_h = round(in_w * h / w)  # frame height after yolo scales width to 640
        if content_h < in_h:
            pad = (in_h - content_h) // 2
            proj = proj[pad:pad + content_h]
        sh, sw = proj.shape

        # the principal component's sign is arbitrary: the same component can come
        # out positive on objects or positive on background. both orientations are
        # relu-clipped (that is what gets displayed), then the detected boxes pick
        # the one that lights up the objects more than the rest of the scene
        pos = np.maximum(proj, 0)
        neg = np.maximum(-proj, 0)
        mask = np.zeros((sh, sw), bool)
        for det in detections:
            x1 = max(int(det.x1 * sw / w), 0)
            y1 = max(int(det.y1 * sh / h), 0)
            x2 = max(int(det.x2 * sw / w), 0)
            y2 = max(int(det.y2 * sh / h), 0)
            mask[y1:y2, x1:x2] = True
        if mask.any() and not mask.all():
            pos_score = pos[mask].mean() - pos[~mask].mean()
            neg_score = neg[mask].mean() - neg[~mask].mean()
            chosen = pos if pos_score >= neg_score else neg
        else:
            # no boxes to guide us, assume the strongest response should be positive
            chosen = pos if pos.max() >= neg.max() else neg

        # upscale to the frame and normalise to [0, 1] for the colormap
        chosen = cv2.resize(chosen, (w, h))
        peak = float(chosen.max())
        return chosen / peak if peak > 0 else chosen
