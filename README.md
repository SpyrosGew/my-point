# RealSense Pointing Object Detection Demo

To use this application you need a realsense RGB-D camera and a pc with a decent graphics card that can run yolo.

The pipeline is simple.

1. The Intel RealSense RGB-D camera captures synchronized RGB and depth frames of the environment.

2. A real-time object detection model processes the RGB frame and detects objects in 2D space. Any detector can be used, as long as it runs at a minimum of 15 FPS.

3. In parallel, a hand pose estimation model tracks the user’s hand and extracts two key points on the index finger to define the pointing direction.

4. The extracted finger points are mapped to the depth frame and converted into 3D coordinates.

5. A 3D pointing ray is constructed from these two points and extended into the scene to estimate where the user is pointing.

6. Based on the ray direction, depth information, and distance from the camera, a region of interest is selected. Only objects inside this region are kept as candidate targets.

7. The user provides a textual description of the desired object while pointing toward it.

8. A vision-language model receives:
   - cropped images of the candidate objects inside the selected region
   - a full-scene image for global context
   - the user’s textual description of the target object

9. Using both the visual inputs and the text description, the VLM selects the object that best matches the user’s intent.


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
