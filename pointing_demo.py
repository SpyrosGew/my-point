from __future__ import annotations

import math
import os
import threading
import traceback
import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import pyrealsense2 as rs
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_tokens(name: str, default: str) -> set[str]:
    raw = os.getenv(name, default)
    return {token.strip().lower() for token in raw.replace(",", " ").split() if token.strip()}


@dataclass
class ObjectCandidate:
    index: int
    label: str
    confidence: float
    mask: np.ndarray
    points_3d: np.ndarray
    centroid_3d: np.ndarray
    centroid_2d: tuple[int, int]
    ray_distance_m: float = float("inf")
    ray_angle_deg: float = float("inf")
    pointing_score: float = 0.0
    language_score: float = 0.0
    final_score: float = 0.0
    selected: bool = False


@dataclass
class PointingRay:
    origin_3d: np.ndarray
    direction_3d: np.ndarray
    wrist_2d: tuple[int, int]
    index_2d: tuple[int, int]


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.text_filter = ""
        self.status = "starting"
        self.metrics: dict[str, float | int | str] = {}
        self.running = True

    def set_frame(self, image: np.ndarray, status: str, metrics: dict[str, float | int | str] | None = None) -> None:
        encode_start = time.perf_counter()
        jpeg_quality = int(np.clip(env_int("JPEG_QUALITY", 92), 50, 100))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            return
        encode_ms = (time.perf_counter() - encode_start) * 1000.0
        with self.lock:
            self.jpeg = encoded.tobytes()
            self.status = status
            if metrics is not None:
                self.metrics = {**metrics, "jpeg_ms": round(encode_ms, 1)}

    def get_frame(self) -> tuple[bytes | None, str]:
        with self.lock:
            return self.jpeg, self.status

    def get_metrics(self) -> dict[str, float | int | str]:
        with self.lock:
            return dict(self.metrics)

    def get_filter(self) -> str:
        with self.lock:
            return self.text_filter.strip().lower()

    def set_filter(self, value: str) -> None:
        with self.lock:
            self.text_filter = value.strip().lower()


class RealSenseCamera:
    def __init__(self) -> None:
        self.width = env_int("FRAME_WIDTH", 1280)
        self.height = env_int("FRAME_HEIGHT", 720)
        self.fps = env_int("FRAME_FPS", 30)
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.intrinsics: rs.intrinsics | None = None

    def start(self) -> None:
        requested = (self.width, self.height, self.fps)
        candidates = [
            requested,
            (1280, 720, min(self.fps, 30)),
            (848, 480, min(self.fps, 30)),
            (640, 480, min(self.fps, 30)),
        ]
        seen: set[tuple[int, int, int]] = set()
        last_error: Exception | None = None
        for width, height, fps in candidates:
            if (width, height, fps) in seen:
                continue
            seen.add((width, height, fps))
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = self.pipeline.start(config)
            except Exception as exc:
                last_error = exc
                continue

            self.width = width
            self.height = height
            self.fps = fps
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            self.intrinsics = color_profile.get_intrinsics()
            return

        raise RuntimeError(f"RealSense stream unavailable: {last_error}")

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("missing RealSense color/depth frame")
        color = np.asanyarray(color_frame.get_data())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        return color, depth_m

    def stop(self) -> None:
        self.pipeline.stop()


class ObjectSegmenter:
    def __init__(self) -> None:
        self.model_path = os.getenv("MODEL_PATH", "yolov8n-seg.pt")
        self.device = os.getenv("YOLO_DEVICE", "") or None
        self.confidence = env_float("CONFIDENCE", 0.35)
        self.imgsz = env_int("YOLO_IMGSZ", 640)
        self.model = YOLO(self.model_path)

    def detect(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> list[ObjectCandidate]:
        result = self.model.predict(
            image_bgr,
            conf=self.confidence,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        if result.masks is None or result.boxes is None:
            return []

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        candidates: list[ObjectCandidate] = []
        for idx, mask_small in enumerate(masks):
            mask = cv2.resize(mask_small, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask > 0.5
            points = mask_to_points_3d(mask_bool, depth_m, intrinsics)
            if len(points) < 20:
                continue
            centroid = robust_centroid(points)
            centroid_2d = mask_centroid_2d(mask_bool)
            label = str(names.get(int(classes[idx]), classes[idx]))
            candidates.append(
                ObjectCandidate(
                    index=idx,
                    label=label,
                    confidence=float(confidences[idx]),
                    mask=mask_bool,
                    points_3d=points,
                    centroid_3d=centroid,
                    centroid_2d=centroid_2d,
                )
            )
        return candidates


class PointingEstimator:
    def __init__(self) -> None:
        self.backend = os.getenv("HAND_BACKEND", "yolo").strip().lower()
        self.smoothing = env_float("POINTING_SMOOTHING", 0.35)
        self.min_hand_ray_length_m = env_float("MIN_HAND_RAY_LENGTH_M", 0.035)
        self.keypoint_confidence = env_float("HAND_KEYPOINT_CONFIDENCE", 0.25)
        self.smoothed_origin_3d: np.ndarray | None = None
        self.smoothed_direction_3d: np.ndarray | None = None
        self.last_hand_confidence = 0.0
        self.last_hand_candidates = 0
        self.pointing_start_landmark = env_int("POINTING_START_LANDMARK", 5)
        self.pointing_end_landmark = env_int("POINTING_END_LANDMARK", 8)

        if self.backend == "yolo":
            self.init_yolo()
        elif self.backend == "mediapipe":
            self.init_mediapipe()
        else:
            raise RuntimeError(f"unsupported HAND_BACKEND={self.backend!r}; use yolo or mediapipe")

    def init_yolo(self) -> None:
        self.hand_model_path = os.getenv("HAND_MODEL_PATH", "models/yolo26_hand_pose_fp16.onnx")
        self.hand_confidence = env_float("HAND_CONFIDENCE", 0.10)
        self.hand_keypoint_visibility = env_float("HAND_KEYPOINT_VISIBILITY", 0.05)
        self.hand_imgsz = env_int("HAND_IMGSZ", 640)
        self.max_hands = env_int("MAX_HANDS", 1)
        self.hand_device = os.getenv("HAND_DEVICE", os.getenv("YOLO_DEVICE", "0"))
        self.hand_model_kind = os.path.splitext(self.hand_model_path)[1].lower()

        if not os.path.exists(self.hand_model_path):
            raise RuntimeError(f"YOLO hand model not found: {self.hand_model_path}")

        if self.hand_model_kind == ".pt":
            self.hand_model = YOLO(self.hand_model_path)
            return

        try:
            import onnxruntime as ort
            import torch  # Preloads PyTorch's bundled CUDA/cuDNN libraries for ONNXRuntime.
        except ImportError as exc:
            raise RuntimeError("ONNXRuntime GPU is not installed. Rebuild the container after updating requirements.txt.") from exc

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.hand_session = ort.InferenceSession(self.hand_model_path, providers=providers)
        self.hand_input = self.hand_session.get_inputs()[0]
        self.hand_input_name = self.hand_input.name
        self.hand_input_float16 = "float16" in self.hand_input.type

    def init_mediapipe(self) -> None:
        self.tracking_confidence = env_float("HAND_TRACKING_CONFIDENCE", self.keypoint_confidence)
        self.max_hands = env_int("MAX_HANDS", 1)
        self.hand_process_width = env_int("HAND_PROCESS_WIDTH", 640)
        self.hand_model_complexity = env_int("HAND_MODEL_COMPLEXITY", 1)

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed. Rebuild the container after updating requirements.txt.") from exc

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=self.hand_model_complexity,
            min_detection_confidence=self.keypoint_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    def estimate(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        if self.backend == "yolo":
            return self.estimate_yolo(image_bgr, depth_m, intrinsics)
        return self.estimate_mediapipe(image_bgr, depth_m, intrinsics)

    def estimate_yolo(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        if self.hand_model_kind == ".pt":
            return self.estimate_yolo_pt(image_bgr, depth_m, intrinsics)
        return self.estimate_yolo_onnx(image_bgr, depth_m, intrinsics)

    def estimate_yolo_pt(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        result = self.hand_model.predict(
            image_bgr,
            conf=self.hand_confidence,
            imgsz=self.hand_imgsz,
            device=self.hand_device,
            verbose=False,
        )[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            self.reset_smoothing()
            return None

        boxes_conf = result.boxes.conf.detach().cpu().numpy()
        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
        if result.keypoints.conf is None:
            keypoints_conf = np.ones(keypoints_xy.shape[:2], dtype=np.float32)
        else:
            keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

        self.last_hand_confidence = float(np.max(boxes_conf)) if len(boxes_conf) else 0.0
        self.last_hand_candidates = int(np.count_nonzero(boxes_conf >= self.hand_confidence))

        best_ray: PointingRay | None = None
        best_score = -1.0
        selected = 0
        for hand_index in np.argsort(-boxes_conf):
            if selected >= max(1, self.max_hands):
                break
            confidence = float(boxes_conf[hand_index])
            if confidence < self.hand_confidence or keypoints_xy.shape[1] <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue

            start_vis = float(keypoints_conf[hand_index, self.pointing_start_landmark])
            end_vis = float(keypoints_conf[hand_index, self.pointing_end_landmark])
            if start_vis < self.hand_keypoint_visibility or end_vis < self.hand_keypoint_visibility:
                continue

            start_px = keypoint_to_pixel(keypoints_xy[hand_index, self.pointing_start_landmark], image_bgr.shape[1], image_bgr.shape[0])
            end_px = keypoint_to_pixel(keypoints_xy[hand_index, self.pointing_end_landmark], image_bgr.shape[1], image_bgr.shape[0])
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue

            score = confidence + 0.20 * min(start_vis, end_vis)
            selected += 1
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def estimate_yolo_onnx(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        height, width = image_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self.preprocess_yolo(image_bgr)
        output = self.hand_session.run(None, {self.hand_input_name: tensor})[0]
        detections = np.asarray(output).reshape(-1, output.shape[-1])
        if len(detections) == 0:
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            self.reset_smoothing()
            return None

        confidences = detections[:, 4]
        self.last_hand_confidence = float(np.max(confidences))
        self.last_hand_candidates = int(np.count_nonzero(confidences >= self.hand_confidence))

        best_ray: PointingRay | None = None
        best_score = -1.0
        max_hands = max(1, self.max_hands)
        selected = 0

        for detection in detections[np.argsort(-detections[:, 4])]:
            if selected >= max_hands:
                break
            confidence = float(detection[4])
            if confidence < self.hand_confidence:
                continue
            keypoints = detection[6:].reshape(21, 3)
            if len(keypoints) <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue
            start_vis = float(keypoints[self.pointing_start_landmark, 2])
            end_vis = float(keypoints[self.pointing_end_landmark, 2])
            if start_vis < self.hand_keypoint_visibility or end_vis < self.hand_keypoint_visibility:
                continue

            start_px = self.yolo_keypoint_to_pixel(keypoints[self.pointing_start_landmark], scale, pad_x, pad_y, width, height)
            end_px = self.yolo_keypoint_to_pixel(keypoints[self.pointing_end_landmark], scale, pad_x, pad_y, width, height)
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue

            hand_length = float(np.linalg.norm(ray.direction_3d))
            score = confidence + 0.20 * min(start_vis, end_vis) + 0.05 * hand_length
            selected += 1
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def preprocess_yolo(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        height, width = image_bgr.shape[:2]
        size = self.hand_imgsz
        scale = min(size / width, size / height)
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - resized_w) / 2.0
        pad_y = (size - resized_h) / 2.0
        x0 = int(round(pad_x - 0.1))
        y0 = int(round(pad_y - 0.1))
        padded[y0:y0 + resized_h, x0:x0 + resized_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32) / 255.0
        if self.hand_input_float16:
            tensor = tensor.astype(np.float16)
        return tensor, scale, float(x0), float(y0)

    def yolo_keypoint_to_pixel(
        self,
        keypoint: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        x = (float(keypoint[0]) - pad_x) / scale
        y = (float(keypoint[1]) - pad_y) / scale
        return int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))

    def estimate_mediapipe(self, image_bgr: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> PointingRay | None:
        height, width = image_bgr.shape[:2]
        process_image = image_bgr
        if self.hand_process_width > 0 and width > self.hand_process_width:
            process_height = max(1, round(height * self.hand_process_width / width))
            process_image = cv2.resize(image_bgr, (self.hand_process_width, process_height), interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            self.last_hand_confidence = 0.0
            self.last_hand_candidates = 0
            self.reset_smoothing()
            return None

        self.last_hand_confidence = 1.0
        self.last_hand_candidates = len(result.multi_hand_landmarks)

        best_ray: PointingRay | None = None
        best_score = -1.0
        handedness_scores = []
        if result.multi_handedness:
            handedness_scores = [float(item.classification[0].score) for item in result.multi_handedness]

        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark
            if len(landmarks) <= max(self.pointing_start_landmark, self.pointing_end_landmark):
                continue

            start_px = landmark_to_pixel(landmarks[self.pointing_start_landmark], width, height)
            end_px = landmark_to_pixel(landmarks[self.pointing_end_landmark], width, height)
            ray = self.ray_from_pixels(start_px, end_px, depth_m, intrinsics)
            if ray is None:
                continue

            handedness_score = handedness_scores[hand_index] if hand_index < len(handedness_scores) else 1.0
            score = 0.20 * handedness_score
            if score > best_score:
                best_score = score
                best_ray = ray

        if best_ray is None:
            self.reset_smoothing()
            return None
        return self.smooth_ray(best_ray)

    def ray_from_pixels(
        self,
        wrist_px: tuple[int, int],
        index_px: tuple[int, int],
        depth_m: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> PointingRay | None:
        wrist_3d = deproject_pixel_with_patch_depth(wrist_px, depth_m, intrinsics, patch_radius=6)
        index_3d = deproject_pixel_with_patch_depth(index_px, depth_m, intrinsics, patch_radius=4)
        if wrist_3d is None or index_3d is None:
            return None

        vector = index_3d - wrist_3d
        length = float(np.linalg.norm(vector))
        if length < self.min_hand_ray_length_m:
            return None
        return PointingRay(
            origin_3d=wrist_3d,
            direction_3d=vector / length,
            wrist_2d=wrist_px,
            index_2d=index_px,
        )

    def reset_smoothing(self) -> None:
        self.smoothed_origin_3d = None
        self.smoothed_direction_3d = None

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "hand_backend": self.backend,
            "hand_conf": round(self.last_hand_confidence, 3),
            "hand_candidates": self.last_hand_candidates,
            "pointing_start_lm": self.pointing_start_landmark,
            "pointing_end_lm": self.pointing_end_landmark,
        }

    def smooth_ray(self, ray: PointingRay) -> PointingRay:
        alpha = float(np.clip(self.smoothing, 0.0, 1.0))
        if self.smoothed_origin_3d is None or self.smoothed_direction_3d is None or alpha <= 0.0:
            self.smoothed_origin_3d = ray.origin_3d
            self.smoothed_direction_3d = ray.direction_3d
            return ray

        origin = alpha * ray.origin_3d + (1.0 - alpha) * self.smoothed_origin_3d
        direction = normalize(alpha * ray.direction_3d + (1.0 - alpha) * self.smoothed_direction_3d)
        self.smoothed_origin_3d = origin
        self.smoothed_direction_3d = direction
        return PointingRay(
            origin_3d=origin,
            direction_3d=direction,
            wrist_2d=ray.wrist_2d,
            index_2d=ray.index_2d,
        )

def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def keypoint_to_pixel(keypoint: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(round(float(keypoint[0])), 0, width - 1))
    y = int(np.clip(round(float(keypoint[1])), 0, height - 1))
    return x, y


def valid_depth(depth_m: np.ndarray) -> np.ndarray:
    depth_min = env_float("DEPTH_MIN_M", 0.15)
    depth_max = env_float("DEPTH_MAX_M", 4.0)
    return np.isfinite(depth_m) & (depth_m >= depth_min) & (depth_m <= depth_max)


def mask_to_points_3d(mask: np.ndarray, depth_m: np.ndarray, intrinsics: rs.intrinsics) -> np.ndarray:
    valid = mask & valid_depth(depth_m)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32)

    max_points = 2500
    if len(xs) > max_points:
        step = max(1, len(xs) // max_points)
        xs = xs[::step]
        ys = ys[::step]

    z = depth_m[ys, xs]
    x3 = (xs - intrinsics.ppx) / intrinsics.fx * z
    y3 = (ys - intrinsics.ppy) / intrinsics.fy * z
    return np.stack([x3, y3, z], axis=1).astype(np.float32)


def deproject_pixel_with_patch_depth(
    pixel: tuple[int, int],
    depth_m: np.ndarray,
    intrinsics: rs.intrinsics,
    patch_radius: int = 4,
) -> np.ndarray | None:
    x, y = pixel
    y0 = max(0, y - patch_radius)
    y1 = min(depth_m.shape[0], y + patch_radius + 1)
    x0 = max(0, x - patch_radius)
    x1 = min(depth_m.shape[1], x + patch_radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    values = patch[valid_depth(patch)]
    if len(values) == 0:
        return None
    z = float(np.median(values))
    return np.array(
        [
            (x - intrinsics.ppx) / intrinsics.fx * z,
            (y - intrinsics.ppy) / intrinsics.fy * z,
            z,
        ],
        dtype=np.float32,
    )


def robust_centroid(points: np.ndarray) -> np.ndarray:
    median = np.median(points, axis=0)
    distances = np.linalg.norm(points - median, axis=1)
    cutoff = np.percentile(distances, 75)
    trimmed = points[distances <= cutoff]
    if len(trimmed) == 0:
        return median.astype(np.float32)
    return np.mean(trimmed, axis=0).astype(np.float32)


def mask_centroid_2d(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0)
    return int(np.median(xs)), int(np.median(ys))


def squared_pixel_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def score_candidates(
    candidates: list[ObjectCandidate],
    ray: PointingRay | None,
    text_filter: str,
) -> ObjectCandidate | None:
    for candidate in candidates:
        candidate.selected = False
        candidate.ray_distance_m = float("inf")
        candidate.ray_angle_deg = float("inf")
        candidate.pointing_score = 0.0
        candidate.language_score = 0.0
        candidate.final_score = 0.0

    if ray is None:
        return None

    query = parse_query(text_filter)
    filtered = filter_candidates(candidates, query)
    excluded_labels = env_tokens("EXCLUDED_TARGET_LABELS", "person")
    filtered = [candidate for candidate in filtered if candidate.label.lower() not in excluded_labels]
    if not filtered:
        return None

    max_angle_deg = env_float("MAX_RAY_ANGLE_DEG", 35.0)
    for candidate in filtered:
        vectors = candidate.points_3d - ray.origin_3d
        forward = vectors @ ray.direction_3d
        in_front = forward > 0.03
        if not np.any(in_front):
            continue

        forward_vectors = vectors[in_front]
        cross = np.cross(forward_vectors, ray.direction_3d)
        distances = np.linalg.norm(cross, axis=1)
        nearest_distance = float(np.percentile(distances, 5))

        centroid_vector = candidate.centroid_3d - ray.origin_3d
        centroid_distance = float(np.linalg.norm(centroid_vector))
        if centroid_distance <= 1e-6:
            continue

        angle = math.degrees(
            math.acos(float(np.clip(np.dot(normalize(centroid_vector), ray.direction_3d), -1.0, 1.0)))
        )
        candidate.ray_distance_m = nearest_distance
        candidate.ray_angle_deg = angle

        distance_score = math.exp(-nearest_distance / 0.12)
        angle_score = max(0.0, 1.0 - angle / max(max_angle_deg, 1.0))
        candidate.pointing_score = float(np.clip(0.70 * distance_score + 0.30 * angle_score, 0.0, 1.0))

    score_language(filtered, query)
    language_weight = env_float("LANGUAGE_WEIGHT", 0.55 if query.has_language else 0.0)
    language_weight = float(np.clip(language_weight, 0.0, 1.0))

    best: ObjectCandidate | None = None
    best_score = -1.0
    for candidate in filtered:
        if not np.isfinite(candidate.ray_distance_m):
            continue
        candidate.final_score = (
            (1.0 - language_weight) * candidate.pointing_score
            + language_weight * candidate.language_score
            + 0.05 * candidate.confidence
        )
        if candidate.final_score > best_score:
            best_score = candidate.final_score
            best = candidate

    if best is not None:
        best.selected = True
    return best


@dataclass
class SelectionQuery:
    raw: str
    label_tokens: set[str]
    spatial_terms: set[str]

    @property
    def has_language(self) -> bool:
        return bool(self.label_tokens or self.spatial_terms)


SPATIAL_ALIASES = {
    "left": "left",
    "right": "right",
    "middle": "middle",
    "center": "middle",
    "centre": "middle",
    "central": "middle",
    "closest": "closest",
    "nearest": "closest",
    "near": "closest",
    "front": "closest",
    "farthest": "farthest",
    "furthest": "farthest",
    "far": "farthest",
    "back": "farthest",
    "top": "top",
    "upper": "top",
    "bottom": "bottom",
    "lower": "bottom",
}
QUERY_STOPWORDS = {
    "a", "an", "and", "at", "by", "for", "i", "in", "is", "it", "of", "on", "one", "that", "the", "this", "to",
    "want", "with",
}


def parse_query(text_filter: str) -> SelectionQuery:
    raw_tokens = [token for token in text_filter.replace(",", " ").split() if token]
    label_tokens: set[str] = set()
    spatial_terms: set[str] = set()
    for token in raw_tokens:
        normalized = normalize_query_token(token)
        if not normalized or normalized in QUERY_STOPWORDS:
            continue
        spatial = SPATIAL_ALIASES.get(normalized)
        if spatial is not None:
            spatial_terms.add(spatial)
            continue
        label_tokens.add(normalized)
    return SelectionQuery(raw=text_filter, label_tokens=label_tokens, spatial_terms=spatial_terms)


def normalize_query_token(token: str) -> str:
    normalized = "".join(ch for ch in token.lower() if ch.isalnum() or ch in {"_", "-"})
    if len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized


def filter_candidates(candidates: Iterable[ObjectCandidate], query: SelectionQuery) -> list[ObjectCandidate]:
    if not query.label_tokens:
        return list(candidates)
    return [candidate for candidate in candidates if label_matches(candidate.label, query.label_tokens)]


def label_matches(label: str, tokens: set[str]) -> bool:
    label_tokens = {normalize_query_token(token) for token in label.replace("_", " ").replace("-", " ").split()}
    label_text = normalize_query_token(label)
    return any(token in label_tokens or token in label_text for token in tokens)


def score_language(candidates: list[ObjectCandidate], query: SelectionQuery) -> None:
    if not candidates:
        return
    if not query.has_language:
        for candidate in candidates:
            candidate.language_score = 0.0
        return

    spatial_scores = spatial_language_scores(candidates, query.spatial_terms)
    for candidate in candidates:
        label_score = 1.0 if not query.label_tokens or label_matches(candidate.label, query.label_tokens) else 0.0
        spatial_score = spatial_scores.get(candidate.index, 1.0 if not query.spatial_terms else 0.0)
        if query.label_tokens and query.spatial_terms:
            candidate.language_score = 0.35 * label_score + 0.65 * spatial_score
        elif query.label_tokens:
            candidate.language_score = label_score
        else:
            candidate.language_score = spatial_score


def spatial_language_scores(candidates: list[ObjectCandidate], terms: set[str]) -> dict[int, float]:
    if not terms:
        return {candidate.index: 1.0 for candidate in candidates}

    xs = np.array([candidate.centroid_2d[0] for candidate in candidates], dtype=np.float32)
    ys = np.array([candidate.centroid_2d[1] for candidate in candidates], dtype=np.float32)
    zs = np.array([candidate.centroid_3d[2] for candidate in candidates], dtype=np.float32)

    scores: dict[int, list[float]] = {candidate.index: [] for candidate in candidates}
    add_rank_scores(candidates, xs, scores, reverse="right" in terms, enabled_terms=terms & {"left", "right"})
    add_middle_scores(candidates, xs, scores, "middle" in terms)
    add_rank_scores(candidates, ys, scores, reverse="bottom" in terms, enabled_terms=terms & {"top", "bottom"})
    add_rank_scores(candidates, zs, scores, reverse="farthest" in terms, enabled_terms=terms & {"closest", "farthest"})

    return {
        candidate.index: float(np.mean(scores[candidate.index])) if scores[candidate.index] else 0.0
        for candidate in candidates
    }


def add_rank_scores(
    candidates: list[ObjectCandidate],
    values: np.ndarray,
    scores: dict[int, list[float]],
    reverse: bool,
    enabled_terms: set[str],
) -> None:
    if not enabled_terms:
        return
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    span = max(value_max - value_min, 1.0)
    for candidate, value in zip(candidates, values):
        normalized = (float(value) - value_min) / span
        scores[candidate.index].append(normalized if reverse else 1.0 - normalized)


def add_middle_scores(
    candidates: list[ObjectCandidate],
    values: np.ndarray,
    scores: dict[int, list[float]],
    enabled: bool,
) -> None:
    if not enabled:
        return
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    middle = (value_min + value_max) * 0.5
    half_span = max((value_max - value_min) * 0.5, 1.0)
    for candidate, value in zip(candidates, values):
        scores[candidate.index].append(max(0.0, 1.0 - abs(float(value) - middle) / half_span))

def draw_overlay(
    image: np.ndarray,
    candidates: list[ObjectCandidate],
    ray: PointingRay | None,
    selected: ObjectCandidate | None,
    text_filter: str,
    intrinsics: rs.intrinsics | None,
) -> np.ndarray:
    output = image.copy()
    red = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 220, 0], dtype=np.uint8)

    tint = np.zeros_like(output)
    for candidate in candidates:
        color = green if candidate.selected else red
        tint[candidate.mask] = color
    if candidates:
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)

    for candidate in candidates:
        color = green if candidate.selected else red
        ys, xs = np.where(candidate.mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(output, (x0, y0), (x1, y1), color.tolist(), 2)
        label = f"{candidate.label} {candidate.confidence:.2f}"
        if np.isfinite(candidate.ray_distance_m):
            label += f" p={candidate.pointing_score:.2f} l={candidate.language_score:.2f}"
        draw_label(output, label, (x0, max(18, y0 - 6)), color.tolist())

    if ray is not None:
        cv2.circle(output, ray.wrist_2d, 5, (255, 255, 0), -1)
        cv2.circle(output, ray.index_2d, 6, (0, 255, 255), -1)
        end = selected.centroid_2d if selected is not None else project_ray_endpoint_2d(
            ray, intrinsics, output.shape[1], output.shape[0]
        )
        if end is not None:
            cv2.line(output, ray.wrist_2d, end, (0, 255, 255), 2)
            cv2.circle(output, end, 5, (0, 255, 255), -1)

    status = "selected: none"
    if selected is not None:
        status = (
            f"selected: {selected.label} score={selected.final_score:.2f} "
            f"p={selected.pointing_score:.2f} l={selected.language_score:.2f}"
        )
    if text_filter:
        status += f" filter={text_filter}"
    draw_label(output, status, (12, 28), (20, 20, 20), background=(245, 245, 245))
    return output


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: list[int] | tuple[int, int, int],
    background: tuple[int, int, int] = (0, 0, 0),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(image, (x - 4, y - text_h - 6), (x + text_w + 4, y + baseline + 4), background, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def project_ray_endpoint_2d(
    ray: PointingRay,
    intrinsics: rs.intrinsics | None,
    width: int,
    height: int,
    max_distance_m: float = 6.0,
) -> tuple[int, int] | None:
    if intrinsics is None:
        return ray.index_2d

    far_pixel: tuple[int, int] | None = None
    for distance_m in np.linspace(0.15, max_distance_m, 48):
        point = ray.origin_3d + ray.direction_3d * float(distance_m)
        projected = project_point_3d(point, intrinsics)
        if projected is not None:
            far_pixel = projected

    if far_pixel is None:
        return ray.index_2d

    rect = (0, 0, width, height)
    clipped = cv2.clipLine(rect, ray.wrist_2d, far_pixel)
    if not clipped[0]:
        return (
            int(np.clip(far_pixel[0], 0, width - 1)),
            int(np.clip(far_pixel[1], 0, height - 1)),
        )
    return clipped[2]


def project_point_3d(point: np.ndarray, intrinsics: rs.intrinsics) -> tuple[int, int] | None:
    x, y, z = [float(value) for value in point]
    if not np.isfinite(z) or z <= 0.03:
        return None
    pixel_x = int(round((x / z) * intrinsics.fx + intrinsics.ppx))
    pixel_y = int(round((y / z) * intrinsics.fy + intrinsics.ppy))
    return pixel_x, pixel_y


def blank_frame(message: str, width: int = 960, height: int = 540) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    draw_label(image, message, (24, 48), (20, 20, 20), background=(245, 245, 245))
    return image


def worker(state: SharedState) -> None:
    camera = RealSenseCamera()
    segmenter: ObjectSegmenter | None = None
    pointing = PointingEstimator()
    frame_count = 0

    try:
        state.set_frame(blank_frame("Starting RealSense..."), "starting RealSense")
        camera.start()
        if camera.intrinsics is None:
            raise RuntimeError("RealSense color intrinsics unavailable")

        state.set_frame(blank_frame("Loading YOLO segmentation model..."), "loading YOLO")
        segmenter = ObjectSegmenter()

        while state.running:
            frame_start = time.perf_counter()
            start = frame_start
            color, depth = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            candidates = segmenter.detect(color, depth, camera.intrinsics)
            yolo_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            ray = pointing.estimate(color, depth, camera.intrinsics)
            handpose_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            text_filter = state.get_filter()
            selected = score_candidates(candidates, ray, text_filter)
            score_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            overlay = draw_overlay(color, candidates, ray, selected, text_filter, camera.intrinsics)
            draw_ms = (time.perf_counter() - start) * 1000.0

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            metrics: dict[str, float | int | str] = {
                "frame": frame_count,
                "fps": round(fps, 2),
                "total_ms": round(total_ms, 1),
                "read_ms": round(read_ms, 1),
                "yolo_ms": round(yolo_ms, 1),
                "handpose_ms": round(handpose_ms, 1),
                "score_ms": round(score_ms, 1),
                "draw_ms": round(draw_ms, 1),
                "objects": len(candidates),
                "resolution": f"{color.shape[1]}x{color.shape[0]}",
                "model": segmenter.model_path,
                "device": segmenter.device or "auto",
                **pointing.metrics(),
            }
            state.set_frame(
                overlay,
                f"fps={fps:.2f} yolo={yolo_ms:.0f}ms handpose={handpose_ms:.0f}ms objects={len(candidates)} selected={selected.label if selected else 'none'}",
                metrics,
            )
            frame_count += 1
    except Exception as exc:
        traceback.print_exc()
        state.set_frame(blank_frame(f"Error: {exc}"), f"error: {exc}")
        while state.running:
            time.sleep(0.5)
    finally:
        try:
            camera.stop()
        except Exception:
            pass


app = FastAPI()
state = SharedState()


@app.on_event("startup")
def start_worker() -> None:
    thread = threading.Thread(target=worker, args=(state,), daemon=True)
    thread.start()


@app.on_event("shutdown")
def stop_worker() -> None:
    state.running = False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html>
  <head>
    <title>RealSense Pointing Demo</title>
    <style>
      body { margin: 0; font-family: system-ui, sans-serif; background: #101114; color: #f5f5f5; }
      main { min-height: 100vh; display: grid; grid-template-rows: auto 1fr; }
      header { display: flex; gap: 12px; align-items: center; padding: 12px 16px; background: #1b1d22; }
      input { width: 260px; max-width: 45vw; padding: 8px 10px; border-radius: 6px; border: 1px solid #555; background: #111; color: #fff; }
      button { padding: 8px 12px; border-radius: 6px; border: 0; background: #2f80ed; color: white; cursor: pointer; }
      img { width: 100%; height: calc(100vh - 57px); object-fit: contain; background: #050505; }
      .status { margin-left: auto; opacity: .8; font-size: 14px; }
    </style>
  </head>
  <body>
    <main>
      <header>
        <form method="post" action="/filter">
          <input name="text_filter" placeholder="e.g. right chair, closest cup, middle bottle" />
          <button type="submit">Apply</button>
        </form>
        <form method="post" action="/filter">
          <input name="text_filter" type="hidden" value="" />
          <button type="submit">Clear</button>
        </form>
        <div class="status">red = detected, green = selected</div>
      </header>
      <img src="/stream" />
    </main>
  </body>
</html>
"""


@app.post("/filter")
def set_filter(text_filter: str = Form(default="")) -> HTMLResponse:
    state.set_filter(text_filter)
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/" />',
        status_code=303,
        headers={"Location": "/"},
    )


@app.get("/status")
def status() -> dict[str, str | dict[str, float | int | str]]:
    _, current = state.get_frame()
    return {"status": current, "filter": state.get_filter(), "metrics": state.get_metrics()}


def stream_frames():
    while True:
        jpeg, _ = state.get_frame()
        if jpeg is None:
            image = blank_frame("Waiting for first frame...")
            ok, encoded = cv2.imencode(".jpg", image)
            jpeg = encoded.tobytes() if ok else b""
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / 20.0)


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(stream_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
