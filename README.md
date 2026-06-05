# RealSense Pointing Demo

Dockerized RGB-D pointing demo:

```text
RealSense aligned RGB-D
→ YOLO segmentation detects objects in RGB
→ MediaPipe detects shoulder, wrist, index finger
→ depth lifts object masks and keypoints to 3D
→ 3D pointing ray selects the object closest to the ray
→ optional text filter narrows candidates
```

## Run

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000
```

All detected objects are drawn red. The selected pointed-at object is drawn green.

## Text Filter

Use the input box in the browser. For example:

```text
cup
```

Then only detected objects whose YOLO label contains `cup` are candidates.

## Notes

- This version does not use ROS.
- The container needs access to the RealSense USB device, so `docker-compose.yml` uses `privileged: true`.
- The default model is `yolov8x-seg.pt` for better masks and object labels. It will download inside the container on first run and is much heavier than `yolov8n-seg.pt`.
- For slower machines, set `MODEL_PATH: yolov8n-seg.pt` or `MODEL_PATH: yolov8s-seg.pt` in `docker-compose.yml`.
