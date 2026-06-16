# improvements timeline

a record of the changes made this session and how each one improved the real-time cv instance.

---

## 1. moved from classification to detection

swapped the cifar/imagenet classifier for yolov8.

a classifier only answers "what is this whole image" with a single label. yolov8 is a detector,
so it finds many objects at once and returns a box around each one. this is the core jump that
turned the project from "name the picture" into "locate every object in the scene".

## 2. gpu acceleration

installed pytorch nightly cu132 for the rtx 5060 (blackwell cc 12.0) and moved the model to cuda.

inference on the cpu was slow and could not keep up with a live feed. running on the gpu cut the
per-frame time by a large margin, which is what makes real-time frame rates possible at all.

## 3. fixed mirrored feed

added cv2.flip on each captured frame.

the raw webcam image comes in mirrored, so motion looked reversed and was confusing to work with.
flipping it makes the on-screen view match real life. this is a usability fix, not an accuracy one.

## 4. 720p capture

forced the camera to capture at 1280x720.

yolo downsamples its input to 640x640 internally. starting from a higher native resolution means
more real detail survives that shrink, so small or distant objects are clearer to the model and
get detected more reliably.

## 5. live trackbars

added brightness and confidence sliders on the window.

different lighting and scenes need different settings. being able to drag a slider live, instead of
editing code and restarting, makes it fast to find the values that give clean detections right now.

## 6. gaussian blur denoising

applied a small gaussian blur to each frame before inference.

cheap camera sensors add grainy high-frequency noise. the model can mistake that noise for texture
and fire off false detections. a light blur smooths the noise away so the model reacts to real
shapes instead of speckle. the trade-off is that too much blur also removes real detail.

## 7. hysteresis thresholding

used two confidence thresholds instead of one. a box turns on at the high threshold and only turns
off once confidence drops below a lower threshold.

with a single cutoff, an object sitting right at the threshold makes its confidence wobble just above
and just below, so the box flickers on and off every frame. the gap between the two thresholds creates
a dead zone, so once a box is shown it stays shown through small wobbles. the result is stable boxes
that do not strobe.

## 8. temporal label voting

kept a rolling history of the last n class predictions per tracked object and showed the most common one.

frame to frame the model can flip a label between two similar classes, for example chair and bench.
voting over recent frames smooths out the odd wrong guess and locks onto the label the model picks most
often, so the displayed name stops flickering.

## 9. bytetrack ids and stale-track pruning

turned on bytetrack so each object keeps the same id across frames, and removed retired ids each frame.

persistent ids are what make label voting possible, since votes must be tied to the same object over
time. bytetrack ids only ever go up, so without cleanup the state dictionaries would grow forever and
slowly leak memory. pruning ids that have left the scene keeps memory flat over long sessions.

## 10. modularised the code

split the single script into config, detector, display, and camera, one responsibility each.

config holds every tunable value, detector owns the model and all the stability logic, display only
draws, and camera just wires it together and runs the loop. this did not change what the model does,
but it made every later change easier to find, test, and adjust without breaking the rest.

## 11. fps counter

added an end-to-end fps reading on the hud, smoothed with an exponential moving average.

fps is measured over the whole loop of capture, preprocess, inference, and draw, so it reflects true
throughput. this makes it easy to benchmark, for example swapping yolov8l for yolov8n to see the real
speed gain on the gpu. the smoothing stops the number jumping around every frame.

---

## additional fixes

small correctness and safety changes made along the way.

- camera open check. videocapture can silently fail, so isopened is checked and the program exits
  cleanly instead of crashing on an empty frame.
- main guard. camera.py runs under an if name == main guard so importing it does not start the camera.
- single source for thresholds. conf_low is derived from conf_high using one gap value in config, so
  the two thresholds can never drift out of sync.
- divide by zero guard. the fps maths clamps the time delta so a zero gap between frames cannot crash.

---

## potential gaps and next steps

honest limitations still left in the current build.

- camera quality is still software side only. setting the mjpg codec and a faster backend, plus manual
  exposure and gain, would cut grain at the source and let the blur be reduced for sharper input.
- the blur is a blunt tool. it removes real detail along with noise, so fixing exposure first would let
  the blur kernel shrink or be dropped.
- fixed model. the model path is set in config and not switchable live, so comparing variants needs a
  restart.
- coco classes only. yolov8 is limited to its 80 trained classes, so anything outside that set cannot
  be detected without fine-tuning.
- no recording or logging. detections are drawn but never saved, so there is no way to review a session
  after it ends.
- single camera and single thread. capture and inference run in one loop, so a slow model directly drops
  the capture frame rate. a capture thread would decouple them.
