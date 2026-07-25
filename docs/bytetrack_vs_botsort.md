# ByteTrack vs BoT-SORT

This experiment compared ByteTrack and BoT-SORT on the same prerecorded video. Both runs used the same YOLOv8l detector, the same person-only class filter, and the same detection confidence threshold of `0.10`. This makes the tracker the main variable being compared.

The evaluation video contains 525 frames at 1920×1080 and 30 FPS. Its ground truth contains 5,325 pedestrian annotations across 26 person trajectories. Each tracker processed the complete 17.5-second video.

**What has been achieved**

→ A person-only MOT17 video and matching ground-truth annotations were prepared.

→ ByteTrack and BoT-SORT were selected explicitly through `bytetrack.yaml` and `botsort.yaml`.

→ Each tracker was run from a fresh YOLO model instance to prevent tracking state from carrying between runs.

→ Predictions were saved frame by frame with bounding boxes, confidence scores, class IDs, and track IDs.

→ Annotated videos were generated to allow visual inspection of the boxes and assigned identities.

→ Tracking quality, detection quality, identity consistency, errors, latency, and throughput were calculated automatically and written to a comparison DataFrame and CSV file.

## ByteTrack output

![ByteTrack tracking people in the evaluation video](evaluation/bytetrack_preview.jpg)

[the ByteTrack video](evaluation/results/bytetrack.mp4)

## Results

| **Metric** | **ByteTrack** | **BoT-SORT** | **Better result** |
|---|---:|---:|---|
| HOTA | 44.48% | 46.36% | BoT-SORT by 1.87 points |
| DetA | 48.83% | 50.55% | BoT-SORT by 1.72 points |
| AssA | 40.70% | 42.70% | BoT-SORT by 2.00 points |
| IDF1 | 50.69% | 54.05% | BoT-SORT by 3.36 points |
| MOTA | 47.08% | 49.58% | BoT-SORT by 2.50 points |
| ID switches | 43 | 45 | ByteTrack by 2 switches |
| Track fragments | 61 | 57 | BoT-SORT by 4 fragments |
| False positives | 1,261 | 1,203 | BoT-SORT by 58 detections |
| False negatives | 1,514 | 1,437 | BoT-SORT by 77 detections |
| Precision | 75.14% | 76.37% | BoT-SORT by 1.23 points |
| Recall | 71.57% | 73.01% | BoT-SORT by 1.45 points |
| Median latency | 30.66 ms | 49.55 ms | ByteTrack |
| P95 latency | 34.65 ms | 58.53 ms | ByteTrack |
| Throughput | 32.16 FPS | 19.81 FPS | ByteTrack |

## What the metrics mean

→ **HOTA** balances detection accuracy and identity association. A higher value means the tracker is better at both locating people and preserving their identities over time.

→ **DetA** measures detection accuracy within the tracking result. It improves when there are fewer missed people and fewer incorrect detections.

→ **AssA** measures how consistently matched detections are assigned to the correct trajectories. It focuses on the quality of the identity links between frames.

→ **IDF1** is the identity F1 score. It measures the proportion of correctly identified detections while balancing identity precision and identity recall.

→ **MOTA** combines false positives, false negatives, and ID switches into one score:

`MOTA = 1 - (false negatives + false positives + ID switches) / ground-truth detections`

→ An **ID switch** occurs when a ground-truth person changes from one predicted track ID to another.

→ A **fragment** occurs when tracking of a person is interrupted and later resumes. Fewer fragments indicate more continuous tracks.

→ **Precision** describes how many reported detections were correct. **Recall** describes how many ground-truth people were successfully detected.

→ **Median latency** represents the typical processing time for one frame. **P95 latency** is the time that 95% of measured frames met or beat, so it helps reveal occasional slow frames.

→ **Throughput** is the average number of frames processed per second. Higher throughput is better for live video.

## Interpretation

BoT-SORT produced the stronger overall tracking-quality result. It achieved higher HOTA, DetA, AssA, IDF1, and MOTA scores. It also produced fewer false positives, fewer false negatives, and fewer fragmented tracks. Its largest quality advantage was a 3.36-point improvement in IDF1.

ByteTrack still produced two fewer ID switches. An ID-switch count describes individual switching events, while AssA and IDF1 measure identity quality across the full sequence. For that reason, BoT-SORT can have slightly more switches but still achieve better overall identity scores.

ByteTrack was much faster. Its median latency was 30.66 ms rather than 49.55 ms, and it processed 32.16 FPS rather than 19.81 FPS. Its median processing time was approximately 38% lower, or equivalently BoT-SORT took approximately 62% longer per typical frame.

For a live 30 FPS laptop camera, ByteTrack is the more practical choice from this experiment. Its average throughput slightly exceeds 30 FPS, although its P95 latency of 34.65 ms is slightly above the 33.33 ms frame budget for consistent 30 FPS processing.

BoT-SORT is the better choice when the modest improvement in tracking quality matters more than real-time speed. Its throughput of 19.81 FPS is suitable for offline processing, but it may lag or require frame dropping with a 30 FPS live stream on the tested hardware.

## Important limitations

The results are a controlled comparison on one short, stationary-camera sequence. They show how the trackers behaved on this video and hardware, but they do not prove that one tracker will always be better.

The evaluator retains only MOT17 pedestrian ground truth. It does not implement the official MOTChallenge handling of distractors and ignore regions. A prediction overlapping one of those excluded annotations can therefore be counted as a false positive. The absolute scores may be pessimistic and should not be reported as official MOT17 benchmark results.

This limitation affects both trackers under the same evaluation rules, so the relative comparison remains useful. More videos with different crowd densities, occlusion levels, lighting conditions, and camera movement would provide stronger evidence.

The timing measurement includes detection and tracking but excludes drawing labels and encoding the output video. It therefore compares model-processing speed rather than complete display or video-export speed.

## Current conclusion

→ Choose **ByteTrack** when maintaining approximately real-time performance on the laptop camera is the priority.

→ Choose **BoT-SORT** when offline processing is acceptable and the observed improvement in tracking and identity quality is more important.

→ Treat this as an initial experiment. A stronger final decision should use several representative videos and the official MOTChallenge ignore-region rules.

The result table is stored in `evaluation/results/tracker_comparison.csv`. Prediction records are stored in `bytetrack_predictions.csv` and `bot_sort_predictions.csv` in the same directory.

The most recent failed rendering attempt left `evaluation/results/bytetrack.mp4` incomplete because OpenCV rejected a fractional `line_width`. Using an integer such as `line_width=1` or `line_width=2` and rerunning the experiment will regenerate the video and all result files.
