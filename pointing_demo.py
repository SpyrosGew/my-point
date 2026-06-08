from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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
    ray_forward_m: float = 0.0
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


@dataclass
class SceneSnapshot:
    image: np.ndarray
    depth_m: np.ndarray
    candidates: list[ObjectCandidate]
    ray: PointingRay | None
    intrinsics: rs.intrinsics | None
    text_filter: str
    depth_min_m: float = 0.15
    depth_max_m: float = 4.0


@dataclass
class CommandResult:
    command: str
    target_label: str
    target_index: int | None
    target_3d: list[float] | None
    confidence: float
    reason: str
    mode: str
    annotated_path: str
    target_class: str | None = None
    depth_intent: str | None = None
    position_intent: str | None = None
    command_ms: float = 0.0
    pointing_ms: float = 0.0
    vlm_ms: float = 0.0
    snapshots_tried: int = 0


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.text_filter = ""
        self.status = "starting"
        self.metrics: dict[str, float | int | str] = {}
        self.latest_scene: SceneSnapshot | None = None
        self.recent_scenes: list[SceneSnapshot] = []
        self.last_command_result: CommandResult | None = None
        self.locked_target_label: str | None = None
        self.locked_target_3d: np.ndarray | None = None
        self.command_ray: PointingRay | None = None
        self.command_depth_min_m = 0.15
        self.command_depth_max_m = 4.0
        self.paused_for_command = False
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

    def set_scene(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        candidates: list[ObjectCandidate],
        ray: PointingRay | None,
        intrinsics: rs.intrinsics | None,
        text_filter: str,
    ) -> None:
        with self.lock:
            depth_min_m, depth_max_m = frame_depth_range(depth_m)
            scene = SceneSnapshot(
                image.copy(), depth_m.copy(), list(candidates), ray, intrinsics, text_filter, depth_min_m, depth_max_m
            )
            self.latest_scene = scene
            self.recent_scenes.append(scene)
            max_recent = max(1, env_int("COMMAND_FRAME_BUFFER", 8))
            self.recent_scenes = self.recent_scenes[-max_recent:]

    def get_scene(self) -> SceneSnapshot | None:
        with self.lock:
            return self.latest_scene

    def get_recent_scenes(self) -> list[SceneSnapshot]:
        with self.lock:
            return list(self.recent_scenes)

    def set_command_result(self, result: CommandResult) -> None:
        with self.lock:
            self.last_command_result = result

    def get_command_result(self) -> CommandResult | None:
        with self.lock:
            return self.last_command_result

    def set_command_debug_area(self, scene: SceneSnapshot | None) -> None:
        with self.lock:
            if scene is None:
                self.command_ray = None
                return
            self.command_ray = scene.ray
            self.command_depth_min_m = scene.depth_min_m
            self.command_depth_max_m = scene.depth_max_m

    def get_command_debug_area(self) -> tuple[PointingRay | None, float, float]:
        with self.lock:
            return self.command_ray, self.command_depth_min_m, self.command_depth_max_m

    def set_command_debug_ray(self, ray: PointingRay | None, depth_min_m: float, depth_max_m: float) -> None:
        with self.lock:
            self.command_ray = ray
            self.command_depth_min_m = depth_min_m
            self.command_depth_max_m = depth_max_m

    def set_locked_target(self, candidate: ObjectCandidate | None) -> None:
        with self.lock:
            if candidate is None:
                self.locked_target_label = None
                self.locked_target_3d = None
            else:
                self.locked_target_label = candidate.label
                self.locked_target_3d = candidate.centroid_3d.copy()

    def get_locked_target(self) -> tuple[str | None, np.ndarray | None]:
        with self.lock:
            target_3d = None if self.locked_target_3d is None else self.locked_target_3d.copy()
            return self.locked_target_label, target_3d

    def set_paused(self, value: bool) -> None:
        with self.lock:
            self.paused_for_command = value

    def is_paused(self) -> bool:
        with self.lock:
            return self.paused_for_command


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


def frame_depth_range(depth_m: np.ndarray) -> tuple[float, float]:
    values = depth_m[valid_depth(depth_m)]
    if len(values) == 0:
        return env_float("DEPTH_MIN_M", 0.15), env_float("DEPTH_MAX_M", 4.0)
    return float(np.min(values)), float(np.max(values))


def pointing_radius_for_depth(distance_m: float, depth_min_m: float, depth_max_m: float) -> float:
    min_radius = env_float("POINTING_CANDIDATE_MIN_RADIUS_M", env_float("POINTING_CANDIDATE_NEAR_RADIUS_M", 0.08))
    max_radius = env_float("POINTING_CANDIDATE_MAX_RADIUS_M", env_float("POINTING_CANDIDATE_FAR_RADIUS_M", 0.80))
    span = max(depth_max_m - depth_min_m, 0.01)
    t = float(np.clip((distance_m - depth_min_m) / span, 0.0, 1.0))
    return min_radius + (max_radius - min_radius) * t


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
        candidate.ray_forward_m = 0.0
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
        candidate.ray_forward_m = max(0.0, float(np.dot(centroid_vector, ray.direction_3d)))

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
    "a", "an", "and", "at", "by", "for", "i", "in", "is", "it", "me", "of", "on", "one", "that", "the", "this", "to",
    "bring", "fetch", "get", "give", "grab", "pick", "please", "robot", "take", "want", "with",
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
    depth_min_m: float | None = None,
    depth_max_m: float | None = None,
) -> np.ndarray:
    output = image.copy()
    red = np.array([0, 0, 255], dtype=np.uint8)
    green = np.array([0, 220, 0], dtype=np.uint8)

    selected_only = selected is not None and env_tokens("DRAW_ONLY_SELECTED_WHEN_LOCKED", "0") not in ({"0"}, {"false"}, {"no"})
    visible_candidates = [selected] if selected_only else candidates

    selected_masks = [candidate.mask for candidate in visible_candidates if candidate is not None and candidate.selected]
    if selected_masks:
        tint = np.zeros_like(output)
        for mask in selected_masks:
            tint[mask] = green
        output = cv2.addWeighted(output, 1.0, tint, 0.42, 0.0)

    for candidate in visible_candidates:
        if candidate is None:
            continue
        color = green if candidate.selected else red
        ys, xs = np.where(candidate.mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(output, (x0, y0), (x1, y1), color.tolist(), 2)
        label = f"{candidate.label} {candidate.confidence:.2f}"
        if ray is not None and np.isfinite(candidate.ray_distance_m):
            label += f" p={candidate.pointing_score:.2f} l={candidate.language_score:.2f}"
        draw_label(output, label, (x0, max(18, y0 - 6)), color.tolist())

    if ray is not None:
        draw_pointing_area(
            output,
            ray,
            intrinsics,
            selected,
            depth_min_m if depth_min_m is not None else env_float("DEPTH_MIN_M", 0.15),
            depth_max_m if depth_max_m is not None else env_float("DEPTH_MAX_M", 4.0),
        )
        cv2.circle(output, ray.wrist_2d, 5, (255, 255, 0), -1)
        cv2.circle(output, ray.index_2d, 6, (0, 255, 255), -1)
        end = selected.centroid_2d if selected is not None else project_ray_endpoint_2d(
            ray, intrinsics, output.shape[1], output.shape[0]
        )
        if end is not None:
            cv2.line(output, ray.wrist_2d, end, (0, 255, 255), 2)
            cv2.circle(output, end, 5, (0, 255, 255), -1)

    status = "selected: none"
    if selected is not None and ray is None:
        status = f"locked: {selected.label}"
    elif selected is not None:
        status = (
            f"selected: {selected.label} score={selected.final_score:.2f} "
            f"p={selected.pointing_score:.2f} l={selected.language_score:.2f}"
        )
    if text_filter:
        status += f" filter={text_filter}"
    draw_label(output, status, (12, 28), (20, 20, 20), background=(245, 245, 245))
    return output



def project_ray_area_polygon(
    ray: PointingRay,
    intrinsics: rs.intrinsics | None,
    width: int,
    height: int,
    depth_min_m: float,
    depth_max_m: float,
    max_distance_m: float | None = None,
    radius_m: float | None = None,
) -> np.ndarray | None:
    if intrinsics is None:
        return None
    max_distance = max_distance_m if max_distance_m is not None else env_float("DEBUG_RAY_AREA_DISTANCE_M", 4.0)
    far_radius = radius_m if radius_m is not None else env_float("POINTING_CANDIDATE_MAX_RADIUS_M", env_float("DEBUG_RAY_AREA_RADIUS_M", 0.80))
    near_radius = env_float("POINTING_CANDIDATE_MIN_RADIUS_M", env_float("DEBUG_RAY_AREA_NEAR_RADIUS_M", 0.08))
    direction = normalize(ray.direction_3d.astype(np.float32))

    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_a = np.cross(direction, up_hint)
    if np.linalg.norm(axis_a) < 1e-4:
        axis_a = np.cross(direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    axis_a = normalize(axis_a)
    axis_b = normalize(np.cross(direction, axis_a))

    projected: list[tuple[int, int]] = []
    ring_angles = np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False)
    for distance_m in np.linspace(0.15, max_distance, 22):
        center = ray.origin_3d + direction * float(distance_m)
        grow = pointing_radius_for_depth(float(distance_m), depth_min_m, depth_max_m)
        for angle in ring_angles:
            offset = axis_a * (math.cos(float(angle)) * grow) + axis_b * (math.sin(float(angle)) * grow)
            projected_point = project_point_3d(center + offset, intrinsics)
            if projected_point is not None:
                projected.append((int(np.clip(projected_point[0], 0, width - 1)), int(np.clip(projected_point[1], 0, height - 1))))

    if len(projected) < 3:
        return None
    points = np.array(projected, dtype=np.int32)
    return cv2.convexHull(points)


def project_ray_area_rings(
    ray: PointingRay,
    intrinsics: rs.intrinsics | None,
    width: int,
    height: int,
    depth_min_m: float,
    depth_max_m: float,
    max_distance_m: float | None = None,
    radius_m: float | None = None,
) -> list[np.ndarray]:
    if intrinsics is None:
        return []
    max_distance = max_distance_m if max_distance_m is not None else env_float("DEBUG_RAY_AREA_DISTANCE_M", 4.0)
    far_radius = radius_m if radius_m is not None else env_float("POINTING_CANDIDATE_MAX_RADIUS_M", env_float("DEBUG_RAY_AREA_RADIUS_M", 0.80))
    near_radius = env_float("POINTING_CANDIDATE_MIN_RADIUS_M", env_float("DEBUG_RAY_AREA_NEAR_RADIUS_M", 0.08))
    direction = normalize(ray.direction_3d.astype(np.float32))
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_a = np.cross(direction, up_hint)
    if np.linalg.norm(axis_a) < 1e-4:
        axis_a = np.cross(direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    axis_a = normalize(axis_a)
    axis_b = normalize(np.cross(direction, axis_a))

    rings: list[np.ndarray] = []
    ring_angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    for distance_m in np.linspace(max_distance * 0.25, max_distance, 4):
        center = ray.origin_3d + direction * float(distance_m)
        grow = pointing_radius_for_depth(float(distance_m), depth_min_m, depth_max_m)
        ring: list[tuple[int, int]] = []
        for angle in ring_angles:
            offset = axis_a * (math.cos(float(angle)) * grow) + axis_b * (math.sin(float(angle)) * grow)
            projected_point = project_point_3d(center + offset, intrinsics)
            if projected_point is not None:
                ring.append((int(np.clip(projected_point[0], 0, width - 1)), int(np.clip(projected_point[1], 0, height - 1))))
        if len(ring) >= 3:
            rings.append(np.array(ring, dtype=np.int32))
    return rings

def project_ray_area_disk(
    ray: PointingRay,
    intrinsics: rs.intrinsics | None,
    width: int,
    height: int,
    distance_m: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray | None:
    if intrinsics is None:
        return None
    direction = normalize(ray.direction_3d.astype(np.float32))
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis_a = np.cross(direction, up_hint)
    if np.linalg.norm(axis_a) < 1e-4:
        axis_a = np.cross(direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    axis_a = normalize(axis_a)
    axis_b = normalize(np.cross(direction, axis_a))

    radius = pointing_radius_for_depth(distance_m, depth_min_m, depth_max_m)
    center = ray.origin_3d + direction * float(distance_m)
    points: list[tuple[int, int]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False):
        offset = axis_a * (math.cos(float(angle)) * radius) + axis_b * (math.sin(float(angle)) * radius)
        projected = project_point_3d(center + offset, intrinsics)
        if projected is not None:
            points.append((int(np.clip(projected[0], 0, width - 1)), int(np.clip(projected[1], 0, height - 1))))
    if len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def draw_pointing_area(
    image: np.ndarray,
    ray: PointingRay,
    intrinsics: rs.intrinsics | None,
    selected: ObjectCandidate | None,
    depth_min_m: float,
    depth_max_m: float,
) -> None:
    if selected is not None and np.isfinite(selected.ray_forward_m) and selected.ray_forward_m > 0.0:
        distance_m = selected.ray_forward_m
    else:
        distance_m = (depth_min_m + depth_max_m) * 0.5

    disk = project_ray_area_disk(ray, intrinsics, image.shape[1], image.shape[0], distance_m, depth_min_m, depth_max_m)
    center = project_point_3d(ray.origin_3d + ray.direction_3d * float(distance_m), intrinsics) if intrinsics is not None else None
    if disk is not None:
        area = image.copy()
        cv2.fillPoly(area, [disk], (0, 210, 255))
        cv2.polylines(area, [disk], True, (0, 255, 255), 3)
        cv2.addWeighted(area, 0.34, image, 0.66, 0.0, dst=image)
    if center is not None:
        cv2.circle(image, center, 5, (0, 255, 255), -1)

    if selected is not None and np.isfinite(selected.ray_distance_m):
        cv2.circle(image, selected.centroid_2d, 16, (0, 255, 255), 3)
        draw_label(
            image,
            f"affected area: {selected.label} p={selected.pointing_score:.2f}",
            (selected.centroid_2d[0] + 12, max(28, selected.centroid_2d[1] - 12)),
            (20, 20, 20),
            background=(0, 255, 255),
        )

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


class VLMTargetSelector:
    def __init__(self) -> None:
        self.model_name = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.device = os.getenv("VLM_DEVICE", "cuda")
        self.max_new_tokens = env_int("VLM_MAX_NEW_TOKENS", 96)
        self.enabled = env_tokens("VLM_ENABLED", "1") not in ({"0"}, {"false"}, {"no"})
        self.processor = None
        self.model = None

    def choose(self, image_bgr: np.ndarray, candidate_records: list[dict], command: str) -> dict:
        if not self.enabled:
            raise RuntimeError("VLM is disabled")
        self.load()
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        candidates_text = "\n".join(
            f"{item['id']}: {item['label']} box={item['box']} pointing_score={item['pointing_score']:.2f} "
            f"spatial_score={item['language_score']:.2f}"
            for item in candidate_records
        )
        prompt = (
            "You are selecting the object a robot should fetch. The image has a yellow pointing line "
            "and candidate objects marked with letter IDs. Use the pointing line plus the user's command.\n"
            f"User command: {command!r}\n"
            f"Candidates:\n{candidates_text}\n"
            "Return strict JSON only: {\"target_id\":\"A\",\"confidence\":0.0," 
            "\"reason\":\"short reason\"}. If unsure, choose the closest reasonable candidate and lower confidence."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return parse_vlm_json(output)

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def release(self) -> None:
        if self.model is None and self.processor is None:
            return
        self.model = None
        self.processor = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


vlm_selector = VLMTargetSelector()



def choose_with_vlm_subprocess(
    image_path: str, crop_sheet_path: str | None, group_crop_path: str | None, records: list[dict], command: str, target_class: str | None
) -> dict:
    request_path = os.path.join("outputs", "vlm_request.json")
    request = {
        "image_path": image_path,
        "crop_sheet_path": crop_sheet_path,
        "group_crop_path": group_crop_path,
        "command": command,
        "candidates": records,
        "target_class": target_class,
        "model": os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "device": os.getenv("VLM_DEVICE", "cuda"),
        "max_new_tokens": env_int("VLM_MAX_NEW_TOKENS", 96),
    }
    with open(request_path, "w", encoding="utf-8") as file:
        json.dump(request, file)

    timeout_s = env_int("VLM_TIMEOUT_S", 90)
    completed = subprocess.run(
        [sys.executable, "-m", "vlm_select", request_path],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout or "VLM subprocess failed").strip()
        raise RuntimeError(error[-800:])
    return parse_vlm_json(completed.stdout)

def parse_vlm_json(output: str) -> dict:
    match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"VLM did not return JSON: {output[:300]}")
    data = json.loads(match.group(0))
    return {
        "target_id": str(data.get("target_id", "")).strip().upper(),
        "confidence": float(data.get("confidence", 0.0)),
        "reason": str(data.get("reason", ""))[:240],
    }



def command_target_label(command: str, candidates: list[ObjectCandidate]) -> str | None:
    labels = sorted({candidate.label for candidate in candidates}, key=len, reverse=True)
    command_tokens = command_label_tokens(command)
    best_label: str | None = None
    best_score = 0
    for label in labels:
        label_tokens = command_label_tokens(label)
        if not label_tokens:
            continue
        score = len(label_tokens & command_tokens)
        label_text = normalize_query_token(label)
        if label_text and label_text in command_tokens:
            score += 2
        for token in label_tokens:
            if token in command_tokens:
                score += 1
        if score > best_score:
            best_score = score
            best_label = label
    return best_label if best_score > 0 else None


def command_label_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in text.replace("_", " ").replace("-", " ").replace(",", " ").split():
        token = normalize_query_token(raw)
        if token and token not in QUERY_STOPWORDS and token not in SPATIAL_ALIASES:
            tokens.add(token)
    return tokens


def command_position_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in text.replace("_", " ").replace("-", " ").replace(",", " ").split():
        token = normalize_query_token(raw)
        if token and token not in QUERY_STOPWORDS:
            tokens.add(token)
    return tokens


def filter_to_target_label(candidates: list[ObjectCandidate], target_label: str | None) -> list[ObjectCandidate]:
    if target_label is None:
        return candidates
    return [candidate for candidate in candidates if candidate.label == target_label]



def candidate_bbox(candidate: ObjectCandidate) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(candidate.mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def segments_intersect(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int], d: tuple[int, int]) -> bool:
    def orient(p, q, r) -> int:
        value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(value) < 1e-9:
            return 0
        return 1 if value > 0 else 2

    def on_segment(p, q, r) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def bbox_overlaps_polygon(bbox: tuple[int, int, int, int], polygon: np.ndarray | None) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    x0, y0, x1, y1 = bbox
    poly = polygon.reshape(-1, 2).astype(np.int32)
    if np.any((poly[:, 0] >= x0) & (poly[:, 0] <= x1) & (poly[:, 1] >= y0) & (poly[:, 1] <= y1)):
        return True

    rect_corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for corner in rect_corners:
        if cv2.pointPolygonTest(poly, corner, False) >= 0:
            return True

    rect_edges = list(zip(rect_corners, rect_corners[1:] + rect_corners[:1]))
    poly_points = [tuple(int(v) for v in point) for point in poly]
    poly_edges = list(zip(poly_points, poly_points[1:] + poly_points[:1]))
    return any(segments_intersect(a, b, c, d) for a, b in rect_edges for c, d in poly_edges)


def bbox_overlaps_pointing_area(candidate: ObjectCandidate, scene: SceneSnapshot) -> bool:
    if scene.ray is None or scene.intrinsics is None or not np.isfinite(candidate.ray_forward_m) or candidate.ray_forward_m <= 0.0:
        return False
    bbox = candidate_bbox(candidate)
    if bbox is None:
        return False
    radius_scale = env_float("VLM_CANDIDATE_AREA_SCALE", 1.35)
    original_min = os.environ.get("POINTING_CANDIDATE_MIN_RADIUS_M")
    original_max = os.environ.get("POINTING_CANDIDATE_MAX_RADIUS_M")
    try:
        os.environ["POINTING_CANDIDATE_MIN_RADIUS_M"] = str(env_float("POINTING_CANDIDATE_MIN_RADIUS_M", 0.12) * radius_scale)
        os.environ["POINTING_CANDIDATE_MAX_RADIUS_M"] = str(env_float("POINTING_CANDIDATE_MAX_RADIUS_M", 1.20) * radius_scale)
        polygon = project_ray_area_disk(
            scene.ray,
            scene.intrinsics,
            scene.image.shape[1],
            scene.image.shape[0],
            candidate.ray_forward_m,
            scene.depth_min_m,
            scene.depth_max_m,
        )
    finally:
        if original_min is None:
            os.environ.pop("POINTING_CANDIDATE_MIN_RADIUS_M", None)
        else:
            os.environ["POINTING_CANDIDATE_MIN_RADIUS_M"] = original_min
        if original_max is None:
            os.environ.pop("POINTING_CANDIDATE_MAX_RADIUS_M", None)
        else:
            os.environ["POINTING_CANDIDATE_MAX_RADIUS_M"] = original_max
    return bbox_overlaps_polygon(bbox, polygon)


def command_depth_intent(command: str) -> str | None:
    tokens = command_position_tokens(command)
    back_words = {"back", "behind", "far", "farthest", "furthest", "rear"}
    front_words = {"front", "near", "nearest", "closest"}
    if tokens & back_words:
        return "back"
    if tokens & front_words:
        return "front"
    return None


def choose_by_depth_intent(candidates: list[ObjectCandidate], target_label: str | None, intent: str | None) -> ObjectCandidate | None:
    if intent is None or target_label is None:
        return None
    matching = [candidate for candidate in candidates if candidate.label == target_label]
    if len(matching) < 2:
        return None
    if intent == "back":
        return max(matching, key=lambda candidate: float(candidate.centroid_3d[2]))
    if intent == "front":
        return min(matching, key=lambda candidate: float(candidate.centroid_3d[2]))
    return None


def command_2d_position_intent(command: str) -> str | None:
    tokens = command_position_tokens(command)
    if "middle" in tokens or "center" in tokens or "centre" in tokens or "central" in tokens:
        return "middle"
    if "left" in tokens or "leftmost" in tokens:
        return "left"
    if "right" in tokens or "rightmost" in tokens:
        return "right"
    if "top" in tokens or "upper" in tokens or "above" in tokens:
        return "top"
    if "bottom" in tokens or "lower" in tokens or "below" in tokens or "under" in tokens:
        return "bottom"
    return None


def choose_by_2d_position_intent(candidates: list[ObjectCandidate], target_label: str | None, intent: str | None) -> ObjectCandidate | None:
    if intent is None or target_label is None:
        return None
    matching = [candidate for candidate in candidates if candidate.label == target_label]
    if len(matching) < 2:
        return None
    if intent == "left":
        return min(matching, key=lambda candidate: candidate.centroid_2d[0])
    if intent == "right":
        return max(matching, key=lambda candidate: candidate.centroid_2d[0])
    if intent == "top":
        return min(matching, key=lambda candidate: candidate.centroid_2d[1])
    if intent == "bottom":
        return max(matching, key=lambda candidate: candidate.centroid_2d[1])
    if intent == "middle":
        ordered = sorted(matching, key=lambda candidate: candidate.centroid_2d[0])
        return ordered[len(ordered) // 2]
    return None


def choose_by_positional_intent(
    candidates: list[ObjectCandidate], target_label: str | None, depth_intent: str | None, position_intent: str | None
) -> tuple[ObjectCandidate | None, str | None]:
    depth_choice = choose_by_depth_intent(candidates, target_label, depth_intent)
    if depth_choice is not None:
        return depth_choice, depth_intent
    position_choice = choose_by_2d_position_intent(candidates, target_label, position_intent)
    if position_choice is not None:
        return position_choice, position_intent
    return None, None

def build_command_candidates(scene: SceneSnapshot, command: str, target_label: str | None, max_candidates: int = 8) -> list[ObjectCandidate]:
    working = list(scene.candidates)
    score_candidates(working, scene.ray, "")
    usable = [
        candidate for candidate in working
        if np.isfinite(candidate.ray_distance_m) and bbox_overlaps_pointing_area(candidate, scene)
    ]
    usable.sort(key=lambda item: (item.label == target_label, item.pointing_score), reverse=True)
    return usable[:max_candidates]


def build_candidate_crop_sheet(
    image: np.ndarray,
    candidates: list[ObjectCandidate],
    records: list[dict],
    tile_size: int = 240,
    columns: int = 4,
) -> np.ndarray | None:
    if not records:
        return None
    rows = int(math.ceil(len(records) / max(1, columns)))
    sheet = np.full((rows * tile_size, columns * tile_size, 3), 245, dtype=np.uint8)
    for idx, record in enumerate(records):
        x0, y0, x1, y1 = record["box"]
        pad = 24
        x0p = max(0, int(x0) - pad)
        y0p = max(0, int(y0) - pad)
        x1p = min(image.shape[1] - 1, int(x1) + pad)
        y1p = min(image.shape[0] - 1, int(y1) + pad)
        crop = image[y0p:y1p + 1, x0p:x1p + 1]
        if crop.size == 0:
            continue
        scale = min((tile_size - 18) / max(crop.shape[1], 1), (tile_size - 46) / max(crop.shape[0], 1))
        new_w = max(1, int(round(crop.shape[1] * scale)))
        new_h = max(1, int(round(crop.shape[0] * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        row = idx // columns
        col = idx % columns
        ox = col * tile_size + (tile_size - new_w) // 2
        oy = row * tile_size + 38 + (tile_size - 46 - new_h) // 2
        sheet[oy:oy + new_h, ox:ox + new_w] = resized
        title = f"{record['id']}: {record['label']}"
        if record.get("same_class_position") not in (None, "unknown", "only"):
            title += f" {record['same_class_position']}"
        if record.get("same_class_depth_position") not in (None, "unknown", "only"):
            title += f" {record['same_class_depth_position']}"
        if record.get("depth_m") is not None:
            title += f" z={record['depth_m']:.2f}m"
        draw_label(sheet, title, (col * tile_size + 8, row * tile_size + 26), (255, 255, 255), background=(20, 20, 20))
    return sheet

def build_candidate_group_crop(annotated: np.ndarray, records: list[dict]) -> np.ndarray | None:
    if not records:
        return None
    height, width = annotated.shape[:2]
    boxes = [record["box"] for record in records if record.get("box")]
    if not boxes:
        return None
    x0 = min(int(box[0]) for box in boxes)
    y0 = min(int(box[1]) for box in boxes)
    x1 = max(int(box[2]) for box in boxes)
    y1 = max(int(box[3]) for box in boxes)
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = max(80, int(box_w * env_float("VLM_GROUP_CROP_PAD_SCALE", 0.45)))
    pad_y = max(80, int(box_h * env_float("VLM_GROUP_CROP_PAD_SCALE", 0.45)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width - 1, x1 + pad_x)
    y1 = min(height - 1, y1 + pad_y)
    crop = annotated[y0:y1 + 1, x0:x1 + 1].copy()
    if crop.size == 0:
        return None
    max_side = env_int("VLM_GROUP_CROP_MAX_SIDE", 1280)
    scale = min(1.0, max_side / max(crop.shape[0], crop.shape[1], 1))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    draw_label(crop, "group crop: all candidates in pointing area", (10, 26), (255, 255, 255), background=(20, 20, 20))
    return crop

def draw_vlm_query_image(
    image: np.ndarray,
    candidates: list[ObjectCandidate],
    ray: PointingRay | None,
    intrinsics: rs.intrinsics | None,
    depth_min_m: float | None = None,
    depth_max_m: float | None = None,
) -> tuple[np.ndarray, list[dict], dict[str, ObjectCandidate]]:
    output = image.copy()
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    records: list[dict] = []
    by_id: dict[str, ObjectCandidate] = {}

    if ray is not None:
        draw_pointing_area(
            output,
            ray,
            intrinsics,
            None,
            depth_min_m if depth_min_m is not None else env_float("DEPTH_MIN_M", 0.15),
            depth_max_m if depth_max_m is not None else env_float("DEPTH_MAX_M", 4.0),
        )
        end = project_ray_endpoint_2d(ray, intrinsics, output.shape[1], output.shape[0])
        if end is not None:
            cv2.line(output, ray.wrist_2d, end, (0, 255, 255), 4)
            cv2.circle(output, ray.wrist_2d, 8, (255, 255, 0), -1)
            cv2.circle(output, end, 7, (0, 255, 255), -1)

    same_label_groups: dict[str, list[ObjectCandidate]] = {}
    for candidate in candidates[:len(labels)]:
        same_label_groups.setdefault(candidate.label, []).append(candidate)
    same_label_ranks: dict[int, str] = {}
    same_label_depth_ranks: dict[int, str] = {}
    for group in same_label_groups.values():
        ordered = sorted(group, key=lambda item: item.centroid_2d[0])
        if len(ordered) == 1:
            same_label_ranks[ordered[0].index] = "only"
        else:
            names = ["leftmost"] + ["middle"] * max(0, len(ordered) - 2) + ["rightmost"]
            for item, name in zip(ordered, names):
                same_label_ranks[item.index] = name

        depth_ordered = sorted(group, key=lambda item: float(item.centroid_3d[2]))
        if len(depth_ordered) == 1:
            same_label_depth_ranks[depth_ordered[0].index] = "only"
        else:
            depth_names = ["frontmost"] + ["middle_depth"] * max(0, len(depth_ordered) - 2) + ["backmost"]
            for item, name in zip(depth_ordered, depth_names):
                same_label_depth_ranks[item.index] = name

    for idx, candidate in enumerate(candidates[:len(labels)]):
        object_id = labels[idx]
        ys, xs = np.where(candidate.mask)
        if len(xs) == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        color = (0, 220, 0) if idx == 0 else (0, 0, 255)
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 3)
        draw_label(
            output,
            f"{object_id}: {candidate.label} {same_label_ranks.get(candidate.index, '')} {same_label_depth_ranks.get(candidate.index, '')} z={candidate.centroid_3d[2]:.2f}m",
            (x0, max(24, y0 - 8)),
            (255, 255, 255),
            background=color,
        )
        records.append({
            "id": object_id,
            "index": candidate.index,
            "label": candidate.label,
            "box": [x0, y0, x1, y1],
            "center": [candidate.centroid_2d[0], candidate.centroid_2d[1]],
            "same_class_position": same_label_ranks.get(candidate.index, "unknown"),
            "same_class_depth_position": same_label_depth_ranks.get(candidate.index, "unknown"),
            "depth_m": round(float(candidate.centroid_3d[2]), 3),
            "pointing_score": candidate.pointing_score,
            "language_score": candidate.language_score,
            "final_score": candidate.final_score,
        })
        by_id[object_id] = candidate
    return output, records, by_id


command_pointing: PointingEstimator | None = None


def estimate_command_ray(scene: SceneSnapshot) -> PointingRay | None:
    global command_pointing
    if scene.intrinsics is None:
        return None
    if command_pointing is None:
        command_pointing = PointingEstimator()
    return command_pointing.estimate(scene.image, scene.depth_m, scene.intrinsics)


def choose_target_for_command(
    scenes: list[SceneSnapshot],
    command: str,
    fallback_ray: PointingRay | None = None,
    fallback_depth_min_m: float = 0.15,
    fallback_depth_max_m: float = 4.0,
) -> tuple[CommandResult, ObjectCandidate | None, SceneSnapshot | None]:
    command_start = time.perf_counter()
    pointing_ms = 0.0
    tried = 0
    selected_scene: SceneSnapshot | None = None
    candidates: list[ObjectCandidate] = []
    all_candidates = [candidate for scene in scenes for candidate in scene.candidates]
    target_class = command_target_label(command, all_candidates)
    depth_intent = command_depth_intent(command)
    position_intent = command_2d_position_intent(command)

    for scene in reversed(scenes[-max(1, env_int("COMMAND_FRAME_TRIES", 5)):]):
        tried += 1
        start = time.perf_counter()
        scene.ray = estimate_command_ray(scene)
        pointing_ms += (time.perf_counter() - start) * 1000.0
        if scene.ray is None and fallback_ray is not None:
            scene.ray = fallback_ray
            scene.depth_min_m = fallback_depth_min_m
            scene.depth_max_m = fallback_depth_max_m
        if scene.ray is None:
            continue
        candidates = build_command_candidates(scene, command, target_class)
        if candidates:
            selected_scene = scene
            break

    if selected_scene is None:
        fallback_scene = scenes[-1]
        if all(scene.ray is None for scene in scenes[-tried:]) and fallback_ray is None:
            reason = "No pointing ray was detected."
        elif target_class is None:
            reason = "Could not map the command to a detected object class."
        else:
            reason = f"No detected {target_class} matched near the pointing ray."
        annotated, _, _ = draw_vlm_query_image(fallback_scene.image, [], fallback_scene.ray, fallback_scene.intrinsics, fallback_scene.depth_min_m, fallback_scene.depth_max_m)
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", "latest_vlm_query.jpg")
        cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return CommandResult(
            command=command,
            target_label="none",
            target_index=None,
            target_3d=None,
            confidence=0.0,
            reason=reason,
            mode="no_target",
            annotated_path=path,
            target_class=target_class,
            depth_intent=depth_intent,
            position_intent=position_intent,
            command_ms=round((time.perf_counter() - command_start) * 1000.0, 1),
            pointing_ms=round(pointing_ms, 1),
            vlm_ms=0.0,
            snapshots_tried=tried,
        ), None, fallback_scene

    annotated, records, by_id = draw_vlm_query_image(selected_scene.image, candidates, selected_scene.ray, selected_scene.intrinsics, selected_scene.depth_min_m, selected_scene.depth_max_m)
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", "latest_vlm_query.jpg")
    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    crop_sheet_path = os.path.join("outputs", "latest_vlm_crops.jpg")
    crop_sheet = build_candidate_crop_sheet(selected_scene.image, candidates, records)
    if crop_sheet is not None:
        cv2.imwrite(crop_sheet_path, crop_sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        crop_sheet_path = None
    group_crop_path = os.path.join("outputs", "latest_vlm_group.jpg")
    group_crop = build_candidate_group_crop(annotated, records)
    if group_crop is not None:
        cv2.imwrite(group_crop_path, group_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        group_crop_path = None

    if target_class is not None and not any(candidate.label == target_class for candidate in candidates):
        available = sorted({candidate.label for candidate in candidates})
        available_text = ", ".join(available) if available else "none"
        return CommandResult(
            command=command,
            target_label="none",
            target_index=None,
            target_3d=None,
            confidence=0.0,
            reason=f"No {target_class} is inside the pointing area. Area contains: {available_text}.",
            mode="no_target",
            annotated_path=path,
            target_class=target_class,
            depth_intent=depth_intent,
            position_intent=position_intent,
            command_ms=round((time.perf_counter() - command_start) * 1000.0, 1),
            pointing_ms=round(pointing_ms, 1),
            vlm_ms=0.0,
            snapshots_tried=tried,
        ), None, selected_scene

    mode = "spatial"
    chosen = candidates[0]
    confidence = float(chosen.final_score)
    reason = "Spatial fallback selected the best candidate from pointing plus text."
    positional_rule_choice, positional_rule_name = choose_by_positional_intent(candidates, target_class, depth_intent, position_intent)
    if positional_rule_choice is not None:
        chosen = positional_rule_choice
        confidence = max(confidence, 0.65)
        mode = "position_hint"
        reason = f"Position hint selected the {positional_rule_name} {target_class}; VLM can override using the group crop."
    vlm_ms = 0.0
    if env_tokens("VLM_ENABLED", "1") not in ({"0"}, {"false"}, {"no"}):
        try:
            start = time.perf_counter()
            decision = choose_with_vlm_subprocess(path, crop_sheet_path, group_crop_path, records, command, target_class)
            vlm_ms = (time.perf_counter() - start) * 1000.0
            target_id = decision["target_id"]
            if target_id in by_id:
                chosen = by_id[target_id]
                confidence = float(decision["confidence"])
                reason = decision["reason"]
                mode = "vlm"
        except Exception as exc:
            reason = f"VLM failed, used spatial fallback: {exc}"

    result = CommandResult(
        command=command,
        target_label=chosen.label,
        target_index=chosen.index,
        target_3d=[round(float(value), 3) for value in chosen.centroid_3d],
        confidence=round(confidence, 3),
        reason=reason,
        mode=mode,
        annotated_path=path,
        target_class=target_class,
        depth_intent=depth_intent,
        position_intent=position_intent,
        command_ms=round((time.perf_counter() - command_start) * 1000.0, 1),
        pointing_ms=round(pointing_ms, 1),
        vlm_ms=round(vlm_ms, 1),
        snapshots_tried=tried,
    )
    return result, chosen, selected_scene

def select_locked_target(
    candidates: list[ObjectCandidate],
    target_label: str | None,
    target_3d: np.ndarray | None,
) -> ObjectCandidate | None:
    for candidate in candidates:
        candidate.selected = False
    if target_label is None or target_3d is None:
        return None

    same_label = [candidate for candidate in candidates if candidate.label == target_label]
    if not same_label:
        return None
    selected = min(same_label, key=lambda candidate: float(np.linalg.norm(candidate.centroid_3d - target_3d)))
    distance = float(np.linalg.norm(selected.centroid_3d - target_3d))
    if distance > env_float("LOCK_MAX_DISTANCE_M", 0.25):
        return None
    selected.selected = True
    return selected


def blank_frame(message: str, width: int = 960, height: int = 540) -> np.ndarray:
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    draw_label(image, message, (24, 48), (20, 20, 20), background=(245, 245, 245))
    return image


def worker(state: SharedState) -> None:
    camera = RealSenseCamera()
    segmenter: ObjectSegmenter | None = None
    debug_live_pointing = env_tokens("DEBUG_LIVE_POINTING", "0") not in ({"0"}, {"false"}, {"no"})
    debug_pointing = PointingEstimator() if debug_live_pointing else None
    debug_pointing_every_n = max(1, env_int("DEBUG_POINTING_EVERY_N", 3))
    last_debug_ray: PointingRay | None = None
    frame_count = 0

    try:
        state.set_frame(blank_frame("Starting RealSense..."), "starting RealSense")
        camera.start()
        if camera.intrinsics is None:
            raise RuntimeError("RealSense color intrinsics unavailable")

        state.set_frame(blank_frame("Loading YOLO segmentation model..."), "loading YOLO")
        segmenter = ObjectSegmenter()

        while state.running:
            if state.is_paused():
                time.sleep(0.05)
                continue

            frame_start = time.perf_counter()
            start = frame_start
            color, depth = camera.read()
            read_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            candidates = segmenter.detect(color, depth, camera.intrinsics)
            yolo_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            handpose_ran = debug_pointing is not None and (last_debug_ray is None or frame_count % debug_pointing_every_n == 0)
            if handpose_ran:
                estimated_ray = debug_pointing.estimate(color, depth, camera.intrinsics)
                last_debug_ray = estimated_ray
            debug_ray = last_debug_ray if debug_pointing is not None else None
            handpose_ms = (time.perf_counter() - start) * 1000.0 if handpose_ran else 0.0

            start = time.perf_counter()
            text_filter = state.get_filter()
            target_label, target_3d = state.get_locked_target()
            locked_selected = select_locked_target(candidates, target_label, target_3d)
            if debug_ray is not None:
                debug_selected = score_candidates(candidates, debug_ray, "")
                if locked_selected is not None:
                    for candidate in candidates:
                        candidate.selected = candidate is locked_selected
                    selected = locked_selected
                else:
                    selected = debug_selected
            else:
                selected = locked_selected
            if selected is not None and selected is locked_selected:
                state.set_locked_target(selected)
            score_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            depth_min_m, depth_max_m = frame_depth_range(depth)
            if debug_live_pointing:
                state.set_command_debug_ray(debug_ray, depth_min_m, depth_max_m)
            elif debug_ray is not None:
                state.set_command_debug_ray(debug_ray, depth_min_m, depth_max_m)
            command_ray, command_depth_min_m, command_depth_max_m = state.get_command_debug_area()
            display_ray = debug_ray if debug_live_pointing else command_ray
            display_depth_min = depth_min_m if debug_live_pointing else command_depth_min_m
            display_depth_max = depth_max_m if debug_live_pointing else command_depth_max_m
            overlay = draw_overlay(
                color, candidates, display_ray, selected, text_filter, camera.intrinsics, display_depth_min, display_depth_max
            )
            state.set_scene(color, depth, candidates, debug_ray, camera.intrinsics, text_filter)
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
                "hand_backend": "live_debug" if debug_live_pointing else "on_command",
                "hand_conf": round(debug_pointing.last_hand_confidence, 3) if debug_pointing is not None else 0.0,
                "hand_candidates": debug_pointing.last_hand_candidates if debug_pointing is not None else 0,
                "debug_live_pointing": int(debug_live_pointing),
                "debug_pointing_every_n": debug_pointing_every_n,
                "handpose_ran": int(handpose_ran),
            }
            state.set_frame(
                overlay,
                f"fps={fps:.2f} yolo={yolo_ms:.0f}ms handpose={handpose_ms:.0f}ms objects={len(candidates)} ray={'yes' if display_ray is not None else 'no'} selected={selected.label if selected else 'none'}",
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
      .notice { min-width: 260px; max-width: 42vw; font-size: 14px; color: #f5f5f5; opacity: .95; }
      .notice.warn { color: #ffd166; }
      .notice.good { color: #7ee787; }
    </style>
  </head>
  <body>
    <main>
      <header>
        <form id="command-form" method="post" action="/command">
          <input name="command" placeholder="command, e.g. get the chair on the right" autocomplete="off" />
          <button type="submit">Ask VLM</button>
        </form>
        <div id="notice" class="notice"></div>
        <div class="status">red = detected, green = selected</div>
      </header>
      <img src="/stream" />
      <script>
        const form = document.getElementById('command-form');
        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          const button = form.querySelector('button');
          button.disabled = true;
          button.textContent = 'Thinking';
          const notice = document.getElementById('notice');
          notice.className = 'notice';
          notice.textContent = '';
          try {
            const response = await fetch('/command', { method: 'POST', body: new FormData(form), cache: 'no-store' });
            const result = await response.json();
            const command = result.last_command || {};
            if (command.mode === 'no_target') {
              notice.className = 'notice warn';
              notice.textContent = command.reason || 'No pointing target found';
            } else if (command.target_label) {
              notice.className = 'notice good';
              notice.textContent = `Selected: ${command.target_label}`;
            } else {
              notice.className = 'notice warn';
              notice.textContent = 'No result';
            }
            form.reset();
          } catch (error) {
            notice.className = 'notice warn';
            notice.textContent = 'Command failed';
          } finally {
            button.disabled = false;
            button.textContent = 'Ask VLM';
          }
        });
      </script>
    </main>
  </body>
</html>
"""


@app.post("/filter")
def set_filter(text_filter: str = Form(default="")) -> HTMLResponse:
    state.set_filter("")
    return HTMLResponse(
        '<meta http-equiv="refresh" content="0; url=/" />',
        status_code=303,
        headers={"Location": "/"},
    )


@app.post("/command")
def command(command: str = Form(default="")) -> HTMLResponse:
    command = command.strip()
    if not command:
        return HTMLResponse('<meta http-equiv="refresh" content="0; url=/" />', status_code=303, headers={"Location": "/"})
    state.set_paused(True)
    try:
        scenes = state.get_recent_scenes()
        if not scenes:
            raise RuntimeError("No live scene available yet")
        fallback_ray, fallback_depth_min_m, fallback_depth_max_m = state.get_command_debug_area()
        result, chosen, debug_scene = choose_target_for_command(
            scenes, command, fallback_ray, fallback_depth_min_m, fallback_depth_max_m
        )
        state.set_command_debug_area(debug_scene)
        state.set_locked_target(chosen)
    finally:
        state.set_paused(False)
    state.set_command_result(result)
    state.set_filter("")
    return JSONResponse({
        "last_command": result.__dict__,
        "locked_target": {
            "label": chosen.label if chosen is not None else None,
            "centroid_3d": None if chosen is None else [round(float(value), 3) for value in chosen.centroid_3d],
        },
    })


@app.get("/status")
def status() -> dict[str, object]:
    _, current = state.get_frame()
    result = state.get_command_result()
    target_label, target_3d = state.get_locked_target()
    command_ray, _, _ = state.get_command_debug_area()
    return {
        "status": current,
        "filter": state.get_filter(),
        "metrics": state.get_metrics(),
        "paused_for_command": state.is_paused(),
        "command_pointing_area_visible": command_ray is not None,
        "locked_target": {
            "label": target_label,
            "centroid_3d": None if target_3d is None else [round(float(value), 3) for value in target_3d],
        },
        "last_command": result.__dict__ if result is not None else None,
    }


@app.get("/vlm-query.jpg")
def vlm_query_image() -> FileResponse:
    path = os.path.join("outputs", "latest_vlm_query.jpg")
    if not os.path.exists(path):
        raise RuntimeError("No VLM query image has been created yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/vlm-crops.jpg")
def vlm_crops_image() -> FileResponse:
    path = os.path.join("outputs", "latest_vlm_crops.jpg")
    if not os.path.exists(path):
        raise RuntimeError("No VLM crop sheet has been created yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/vlm-group.jpg")
def vlm_group_image() -> FileResponse:
    path = os.path.join("outputs", "latest_vlm_group.jpg")
    if not os.path.exists(path):
        raise RuntimeError("No VLM group crop has been created yet")
    return FileResponse(path, media_type="image/jpeg")


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
