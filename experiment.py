"""Run a controlled ByteTrack vs BoT-SORT evaluation on one annotated video."""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch

import config
from detector import Detector


# Change these paths after adding the prerecorded video and its annotations.
VIDEO_PATH = Path("evaluation/stationary_webcam_eval.mp4")
GROUND_TRUTH_PATH = Path("evaluation/ground_truth.txt")
OUTPUT_DIR = Path("evaluation/results")

# COCO class 0 is "person". Ground truth should contain only these evaluated objects.
EVALUATED_CLASS_IDS = (0,)
TRACKERS = {
    "ByteTrack": "bytetrack.yaml",
    "BoT-SORT": "botsort.yaml",
}

DETECTION_CONFIDENCE = 0.10
BRIGHTNESS_VALUE = config.BRIGHTNESS_CENTER
MATCH_IOU = 0.50
WARMUP_FRAMES = 5
SAVE_ANNOTATED_VIDEO = True
HOTA_ALPHAS = np.arange(0.05, 0.96, 0.05)

PREDICTION_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "width",
    "height",
    "confidence",
    "class_id",
]


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load MOTChallenge rows: frame,id,x,y,width,height[,confidence,class,visibility]."""
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}\n"
            "Create it from the template described in x.md before running the experiment."
        )

    raw = pd.read_csv(path, header=None, comment="#", sep=r"[,\s]+", engine="python")
    if raw.shape[1] < 6:
        raise ValueError("Ground truth needs at least: frame,id,x,y,width,height")

    names = ["frame", "track_id", "x", "y", "width", "height", "confidence", "class_id", "visibility"]
    raw = raw.iloc[:, : len(names)]
    raw.columns = names[: raw.shape[1]]

    numeric_columns = ["frame", "track_id", "x", "y", "width", "height"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    if "confidence" in raw:
        raw = raw[pd.to_numeric(raw["confidence"], errors="raise") > 0]

    raw["frame"] = raw["frame"].astype(int)
    raw["track_id"] = raw["track_id"].astype(int)
    if raw.empty:
        raise ValueError("Ground truth contains no evaluable rows")
    if (raw["frame"] < 1).any() or (raw[["width", "height"]] <= 0).any().any():
        raise ValueError("Frames must be 1-based and every ground-truth box must have positive size")
    return raw[numeric_columns].sort_values(["frame", "track_id"]).reset_index(drop=True)


def run_tracker(
    tracker_name: str,
    tracker_config: str,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Run one fresh model/tracker over the complete video and collect raw track boxes."""
    cap = cv2.VideoCapture(str(VIDEO_PATH), cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open evaluation video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writer = None
    if SAVE_ANNOTATED_VIDEO:
        output_video = OUTPUT_DIR / f"{tracker_name.lower().replace('-', '_')}.mp4"
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create annotated video: {output_video}")

    detector = Detector(config.MODEL_PATH, device)
    prediction_rows: list[dict] = []
    latencies_ms: list[float] = []
    frame_number = 0
    beta = BRIGHTNESS_VALUE - config.BRIGHTNESS_CENTER

    try:
        while True:
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1

            frame = cv2.add(frame, (beta, beta, beta, 0))
            frame = cv2.GaussianBlur(frame, config.BLUR_KERNEL, 0)
            result = detector.model.track(
                frame,
                device=device,
                verbose=False,
                persist=True,
                tracker=tracker_config,
                conf=DETECTION_CONFIDENCE,
                imgsz=config.TRACK_IMGSZ,
                classes=list(EVALUATED_CLASS_IDS),
            )[0]

            boxes = result.boxes
            if boxes.id is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                track_ids = boxes.id.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)

                for coords, track_id, confidence, class_id in zip(
                    xyxy, track_ids, confidences, class_ids
                ):
                    x1, y1, x2, y2 = coords
                    prediction_rows.append(
                        {
                            "frame": frame_number,
                            "track_id": track_id,
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "confidence": confidence,
                            "class_id": class_id,
                        }
                    )

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000)

            # Rendering and encoding are excluded from tracker latency.
            if writer is not None:
                writer.write(result.plot(
                    font_size=9,
                    line_width=1,
                    # labels=True, conf=True are defaults
                ))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        del detector
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    prediction_path = OUTPUT_DIR / f"{tracker_name.lower().replace('-', '_')}_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    measured_latencies = np.asarray(latencies_ms[WARMUP_FRAMES:], dtype=float)
    if measured_latencies.size == 0:
        raise ValueError(f"Video must contain more than {WARMUP_FRAMES} frames")
    return predictions, measured_latencies, frame_number


def box_iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    """Calculate pairwise IoU for boxes represented as x,y,width,height."""
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt_xy2 = gt_boxes[:, :2] + gt_boxes[:, 2:]
    tracker_xy2 = tracker_boxes[:, :2] + tracker_boxes[:, 2:]
    top_left = np.maximum(gt_boxes[:, None, :2], tracker_boxes[None, :, :2])
    bottom_right = np.minimum(gt_xy2[:, None, :], tracker_xy2[None, :, :])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=2)
    gt_area = np.prod(gt_boxes[:, 2:], axis=1)[:, None]
    tracker_area = np.prod(tracker_boxes[:, 2:], axis=1)[None, :]
    return intersection / np.maximum(gt_area + tracker_area - intersection, np.finfo(float).eps)


def build_metric_data(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    frame_count: int,
) -> dict:
    """Convert DataFrames into contiguous identity arrays used by MOT metrics."""
    max_gt_frame = int(ground_truth["frame"].max())
    if max_gt_frame > frame_count:
        raise ValueError(
            f"Ground truth references frame {max_gt_frame}, but the video has {frame_count} frames"
        )

    gt_ids_unique = sorted(ground_truth["track_id"].unique())
    tracker_ids_unique = sorted(predictions["track_id"].unique()) if not predictions.empty else []
    gt_id_map = {identity: index for index, identity in enumerate(gt_ids_unique)}
    tracker_id_map = {identity: index for index, identity in enumerate(tracker_ids_unique)}

    gt_ids_by_frame = []
    tracker_ids_by_frame = []
    similarity_by_frame = []
    for frame in range(1, frame_count + 1):
        gt_frame = ground_truth[ground_truth["frame"] == frame]
        tracker_frame = predictions[predictions["frame"] == frame]

        gt_ids = np.asarray([gt_id_map[value] for value in gt_frame["track_id"]], dtype=int)
        tracker_ids = np.asarray(
            [tracker_id_map[value] for value in tracker_frame["track_id"]], dtype=int
        )
        gt_boxes = gt_frame[["x", "y", "width", "height"]].to_numpy(dtype=float)
        tracker_boxes = tracker_frame[["x", "y", "width", "height"]].to_numpy(dtype=float)

        gt_ids_by_frame.append(gt_ids)
        tracker_ids_by_frame.append(tracker_ids)
        similarity_by_frame.append(box_iou_matrix(gt_boxes, tracker_boxes))

    return {
        "num_timesteps": frame_count,
        "num_gt_ids": len(gt_ids_unique),
        "num_tracker_ids": len(tracker_ids_unique),
        "num_gt_dets": len(ground_truth),
        "num_tracker_dets": len(predictions),
        "gt_ids": gt_ids_by_frame,
        "tracker_ids": tracker_ids_by_frame,
        "similarity_scores": similarity_by_frame,
    }


def calculate_hota(data: dict) -> dict:
    """Calculate HOTA, DetA, and AssA using the official TrackEval formulation."""
    alpha_count = len(HOTA_ALPHAS)
    if data["num_tracker_dets"] == 0 or data["num_gt_dets"] == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0}

    potential_matches = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))
    gt_id_count = np.zeros((data["num_gt_ids"], 1))
    tracker_id_count = np.zeros((1, data["num_tracker_ids"]))

    for gt_ids, tracker_ids, similarity in zip(
        data["gt_ids"], data["tracker_ids"], data["similarity_scores"]
    ):
        denominator = (
            similarity.sum(axis=0)[None, :]
            + similarity.sum(axis=1)[:, None]
            - similarity
        )
        normalized_similarity = np.divide(
            similarity,
            denominator,
            out=np.zeros_like(similarity),
            where=denominator > np.finfo(float).eps,
        )
        potential_matches[np.ix_(gt_ids, tracker_ids)] += normalized_similarity
        gt_id_count[gt_ids] += 1
        tracker_id_count[0, tracker_ids] += 1

    global_alignment = np.divide(
        potential_matches,
        gt_id_count + tracker_id_count - potential_matches,
        out=np.zeros_like(potential_matches),
        where=(gt_id_count + tracker_id_count - potential_matches) > np.finfo(float).eps,
    )

    true_positives = np.zeros(alpha_count)
    false_negatives = np.zeros(alpha_count)
    false_positives = np.zeros(alpha_count)
    match_counts = [np.zeros_like(potential_matches) for _ in HOTA_ALPHAS]

    for gt_ids, tracker_ids, similarity in zip(
        data["gt_ids"], data["tracker_ids"], data["similarity_scores"]
    ):
        if len(gt_ids) == 0:
            false_positives += len(tracker_ids)
            continue
        if len(tracker_ids) == 0:
            false_negatives += len(gt_ids)
            continue

        score = global_alignment[np.ix_(gt_ids, tracker_ids)] * similarity
        match_rows, match_columns = linear_sum_assignment(-score)
        for index, alpha in enumerate(HOTA_ALPHAS):
            accepted = similarity[match_rows, match_columns] >= alpha - np.finfo(float).eps
            rows = match_rows[accepted]
            columns = match_columns[accepted]
            true_positives[index] += len(rows)
            false_negatives[index] += len(gt_ids) - len(rows)
            false_positives[index] += len(tracker_ids) - len(rows)
            if len(rows):
                match_counts[index][gt_ids[rows], tracker_ids[columns]] += 1

    association_accuracy = np.zeros(alpha_count)
    for index, matches in enumerate(match_counts):
        pair_accuracy = matches / np.maximum(1.0, gt_id_count + tracker_id_count - matches)
        association_accuracy[index] = (
            np.sum(matches * pair_accuracy) / max(1.0, true_positives[index])
        )

    detection_accuracy = true_positives / np.maximum(
        1.0, true_positives + false_negatives + false_positives
    )
    hota = np.sqrt(detection_accuracy * association_accuracy)
    return {
        "HOTA": float(hota.mean()),
        "DetA": float(detection_accuracy.mean()),
        "AssA": float(association_accuracy.mean()),
    }


def calculate_identity(data: dict) -> dict:
    """Calculate IDF1, ID precision, and ID recall with global trajectory assignment."""
    if data["num_tracker_dets"] == 0:
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0}

    gt_count = np.zeros(data["num_gt_ids"])
    tracker_count = np.zeros(data["num_tracker_ids"])
    potential_matches = np.zeros((data["num_gt_ids"], data["num_tracker_ids"]))

    for gt_ids, tracker_ids, similarity in zip(
        data["gt_ids"], data["tracker_ids"], data["similarity_scores"]
    ):
        eligible_gt, eligible_tracker = np.nonzero(similarity >= MATCH_IOU)
        potential_matches[gt_ids[eligible_gt], tracker_ids[eligible_tracker]] += 1
        gt_count[gt_ids] += 1
        tracker_count[tracker_ids] += 1

    gt_total = data["num_gt_ids"]
    tracker_total = data["num_tracker_ids"]
    size = gt_total + tracker_total
    false_positive_cost = np.zeros((size, size))
    false_negative_cost = np.zeros((size, size))
    false_positive_cost[gt_total:, :tracker_total] = 1e10
    false_negative_cost[:gt_total, tracker_total:] = 1e10

    for gt_id in range(gt_total):
        false_negative_cost[gt_id, :tracker_total] = gt_count[gt_id]
        false_negative_cost[gt_id, tracker_total + gt_id] = gt_count[gt_id]
    for tracker_id in range(tracker_total):
        false_positive_cost[:gt_total, tracker_id] = tracker_count[tracker_id]
        false_positive_cost[gt_total + tracker_id, tracker_id] = tracker_count[tracker_id]

    false_negative_cost[:gt_total, :tracker_total] -= potential_matches
    false_positive_cost[:gt_total, :tracker_total] -= potential_matches
    rows, columns = linear_sum_assignment(false_negative_cost + false_positive_cost)

    idfn = int(round(false_negative_cost[rows, columns].sum()))
    idfp = int(round(false_positive_cost[rows, columns].sum()))
    idtp = int(round(gt_count.sum() - idfn))
    id_precision = idtp / max(1, idtp + idfp)
    id_recall = idtp / max(1, idtp + idfn)
    idf1 = 2 * idtp / max(1, 2 * idtp + idfp + idfn)
    return {"IDF1": idf1, "IDP": id_precision, "IDR": id_recall}


def calculate_clear(data: dict) -> dict:
    """Calculate CLEAR MOT diagnostics at the configured IoU matching threshold."""
    if data["num_gt_dets"] == 0:
        raise ValueError("CLEAR metrics require at least one ground-truth detection")

    gt_id_count = np.zeros(data["num_gt_ids"])
    gt_matched_count = np.zeros(data["num_gt_ids"])
    gt_fragment_count = np.zeros(data["num_gt_ids"])
    previous_tracker_id = np.full(data["num_gt_ids"], np.nan)
    previous_timestep_tracker_id = np.full(data["num_gt_ids"], np.nan)
    true_positives = false_negatives = false_positives = identity_switches = 0

    for gt_ids, tracker_ids, similarity in zip(
        data["gt_ids"], data["tracker_ids"], data["similarity_scores"]
    ):
        if len(gt_ids) == 0:
            false_positives += len(tracker_ids)
            continue
        if len(tracker_ids) == 0:
            false_negatives += len(gt_ids)
            gt_id_count[gt_ids] += 1
            continue

        score = 1000 * (
            tracker_ids[None, :] == previous_timestep_tracker_id[gt_ids, None]
        ) + similarity
        score[similarity < MATCH_IOU - np.finfo(float).eps] = 0
        match_rows, match_columns = linear_sum_assignment(-score)
        accepted = score[match_rows, match_columns] > np.finfo(float).eps
        match_rows = match_rows[accepted]
        match_columns = match_columns[accepted]
        matched_gt_ids = gt_ids[match_rows]
        matched_tracker_ids = tracker_ids[match_columns]

        prior_ids = previous_tracker_id[matched_gt_ids]
        identity_switches += int(
            np.sum(~np.isnan(prior_ids) & (matched_tracker_ids != prior_ids))
        )

        gt_id_count[gt_ids] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_tracked_previous_frame = np.isnan(previous_timestep_tracker_id)
        previous_tracker_id[matched_gt_ids] = matched_tracker_ids
        previous_timestep_tracker_id[:] = np.nan
        previous_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        tracked_now = ~np.isnan(previous_timestep_tracker_id)
        gt_fragment_count += not_tracked_previous_frame & tracked_now

        true_positives += len(matched_gt_ids)
        false_negatives += len(gt_ids) - len(matched_gt_ids)
        false_positives += len(tracker_ids) - len(matched_tracker_ids)

    fragmentations = int(np.maximum(gt_fragment_count - 1, 0).sum())
    gt_detections = true_positives + false_negatives
    mota = (true_positives - false_positives - identity_switches) / max(1, gt_detections)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, gt_detections)
    return {
        "MOTA": mota,
        "IDSW": identity_switches,
        "Frag": fragmentations,
        "FP": false_positives,
        "FN": false_negatives,
        "Precision": precision,
        "Recall": recall,
    }


def summarize_tracker(
    tracker_name: str,
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    latencies_ms: np.ndarray,
    frame_count: int,
) -> dict:
    """Calculate accuracy and performance metrics for one tracker run."""
    data = build_metric_data(ground_truth, predictions, frame_count)
    metrics = {
        **calculate_hota(data),
        **calculate_identity(data),
        **calculate_clear(data),
    }
    return {
        "Tracker": tracker_name,
        "Frames": frame_count,
        "HOTA (%)": metrics["HOTA"] * 100,
        "DetA (%)": metrics["DetA"] * 100,
        "AssA (%)": metrics["AssA"] * 100,
        "IDF1 (%)": metrics["IDF1"] * 100,
        "MOTA (%)": metrics["MOTA"] * 100,
        "IDSW": metrics["IDSW"],
        "Fragments": metrics["Frag"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],
        "Precision (%)": metrics["Precision"] * 100,
        "Recall (%)": metrics["Recall"] * 100,
        "Median latency (ms)": float(np.median(latencies_ms)),
        "P95 latency (ms)": float(np.percentile(latencies_ms, 95)),
        "Throughput (FPS)": float(1000 / np.mean(latencies_ms)),
    }


def main() -> None:
    """Execute ByteTrack followed by BoT-SORT and present one comparison DataFrame."""
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation video not found: {VIDEO_PATH}\n"
            "Place the prerecorded video there or change VIDEO_PATH at the top of experiment.py."
        )

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    summaries = []

    for tracker_name, tracker_config in TRACKERS.items():
        print(f"\nRunning {tracker_name} ({tracker_config}) on {device}...")
        predictions, latencies_ms, frame_count = run_tracker(
            tracker_name, tracker_config, device
        )
        summaries.append(
            summarize_tracker(
                tracker_name,
                ground_truth,
                predictions,
                latencies_ms,
                frame_count,
            )
        )

    results = pd.DataFrame(summaries).set_index("Tracker")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "tracker_comparison.csv"
    results.to_csv(results_path)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("\nTracker comparison:\n")
        print(results.round(3).to_string())
    print(f"\nSaved DataFrame to {results_path}")


if __name__ == "__main__":
    main()
