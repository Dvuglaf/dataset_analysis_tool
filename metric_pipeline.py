import json
import re
from pathlib import Path
from typing import Literal, Any
import sys
import hashlib

import cv2

import clip

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal
from multiprocessing import Process, Queue, Event, cpu_count, shared_memory, Pool
from concurrent.futures import ProcessPoolExecutor

from pydantic import BaseModel
from pyiqa.archs.musiq_arch import MUSIQ

from cache_manager import CacheManager
from computations.metrics.background_dominance import BackgroundDominanceMetric
from computations.metrics.base import Metric
from computations.metrics.base_metrics import get_brightness_metric, get_blur_metric, get_contrast_metric, \
    get_saturation_metric, get_artifacts_metric, get_optical_flow_metric
from computations.metrics.cluster_metrics import ClusterMetric
from computations.metrics.entropy import EntropyMetrics
from computations.metrics.ics_metrics import ICSMetric
from computations.metrics.imaging_quality import ImagingQualityMetrics
from computations.metrics.motion_smoothness import MotionSmoothness, MotionSmoothnessMetrics
from computations.metrics.objects import ObjectDetector
from computations.metrics.psnr import PSNRMetrics
from computations.metrics.ssim import SSIMMetrics
from computations.metrics.temporal_clip_consistency import ClipFeaturesMetric
from custom_types import SceneSegment, VideoWrapper


class MetricsComputer:
    DEVICE = "cpu"
    THIRD_PARTY_DIR = Path("./computations/third_party")

    BATCH_SIZE = 4

    TARGET_HEIGHT = 480

    IMAGING_QUALITY_MODEL_PATH = Path("./computations/musiq_ava_ckpt-e8d3f067.pth")
    MOTION_SMOOTHNESS_CFG = THIRD_PARTY_DIR / "amt/cfgs/AMT-S.yaml"
    MOTION_SMOOTHNESS_MODEL_PATH = THIRD_PARTY_DIR / "amt_model/amt-s.pth"
    OPTICAL_FLOW_MODEL_PATH = THIRD_PARTY_DIR / "raft_model/models/raft-things.pth"

    YOLOE_CLASSES = THIRD_PARTY_DIR / "yoloe" / "data" / "coco_classes.txt"
    YOLOE_MODEL_PATH = THIRD_PARTY_DIR / "yoloe" / "checkpoints" / "yoloe-26n-seg.pt"

    def __init__(self, object_classes: list[str], group_classes: dict[str, list[str]]):
        # initialize models, queues, shared memory, etc.
        self._image_quality_model = MUSIQ(
            pretrained_model_path=str(self.IMAGING_QUALITY_MODEL_PATH), num_class=10
        ).to(self.DEVICE)

        self._motion_smoothness_model = MotionSmoothness(
            config=str(self.MOTION_SMOOTHNESS_CFG),
            ckpt=self.MOTION_SMOOTHNESS_MODEL_PATH,
            device=self.DEVICE,
            target_height=self.TARGET_HEIGHT
        )

        self._object_detector = ObjectDetector(
            model_checkpoint=self.YOLOE_MODEL_PATH,
            classes=object_classes,
            device=self.DEVICE,
            group_classes=group_classes
        )

        self.entropy_metric = EntropyMetrics(self.BATCH_SIZE, device_id=self.DEVICE, resize_height=self.TARGET_HEIGHT)
        self.blur_metric = get_blur_metric
        self.brightness_metric = get_brightness_metric
        self.contrast_metric = get_contrast_metric
        self.saturation_metric = get_saturation_metric
        self.compression_metric = get_artifacts_metric
        self.image_quality_metric = ImagingQualityMetrics(self._image_quality_model, self.BATCH_SIZE, device_id=self.DEVICE)

        self.optic_flow_metric = get_optical_flow_metric
        self.motion_smoothness_metric = MotionSmoothnessMetrics(self._motion_smoothness_model, self.BATCH_SIZE, device_id=self.DEVICE)
        self.psnr_metric = PSNRMetrics(batch_size=self.BATCH_SIZE)
        self.ssim_metric = SSIMMetrics(batch_size=self.BATCH_SIZE)
        self.background_dominance_metric = BackgroundDominanceMetric()

        clip_model, _ = clip.load("ViT-B/32", device=self.DEVICE)

        self.clip_features = ClipFeaturesMetric(clip_model, batch_size=self.BATCH_SIZE, device_id=self.DEVICE)
        self.cluster_metrics = ClusterMetric()
        self.ics_metrics = ICSMetric()


def safe_run(func, *args, **kwargs) -> Metric | Any | None:
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error in {func.__name__}: {e}")
        return None


def override_clamp_dicts(base_dict: dict, override_dict: dict) -> dict:
    for key, value in override_dict.items():
        if key in base_dict.keys() and key is not None:
            base_dict[key] = value
    return base_dict


def run_quality_group(torch_frames,
                      grayscale_frames,
                      hsv_frames,
                      torch_grayscale,
                      processor_instance: MetricsComputer,
                      cached_motion_metrics: list[Metric | None]
                      ):
    cached_result_dict = {
        metric.name: metric for metric in cached_motion_metrics if metric is not None
    }
    return override_clamp_dicts(
        {
            "entropy": safe_run(processor_instance.entropy_metric.compute, torch_grayscale),
            # "imaging_quality": safe_run(processor_instance.image_quality_metric.compute, torch_frames),
            "brightness": safe_run(processor_instance.brightness_metric, grayscale_frames),
            "contrast": safe_run(processor_instance.contrast_metric, grayscale_frames),
            "saturation": safe_run(processor_instance.saturation_metric, hsv_frames),
            "blur": safe_run(processor_instance.blur_metric, grayscale_frames),
            "artifacts": safe_run(processor_instance.compression_metric, grayscale_frames),
        },
        cached_result_dict
    )


def run_motion_group(torch_frames,
                     grayscale_frames,
                     processor_instance: MetricsComputer,
                     cached_motion_metrics: list[Metric | None]
                     ):
    if "flow" in [m.name for m in cached_motion_metrics if m is not None]:
        optic_float = next(m for m in cached_motion_metrics if m is not None and m.name == "flow")
    else:
        optic_float = safe_run(processor_instance.optic_flow_metric, grayscale_frames)

    cached_result_dict = {
        metric.name: metric for metric in cached_motion_metrics if metric is not None
    }

    return override_clamp_dicts(
        {
            "smoothness": safe_run(processor_instance.motion_smoothness_metric.compute, torch_frames),
            "flow": optic_float,
            "psnr": safe_run(processor_instance.psnr_metric.compute, torch_frames),
            "ssim": safe_run(processor_instance.ssim_metric.compute, torch_frames),
            "background_dominance": (
                safe_run(processor_instance.background_dominance_metric.compute, optic_float.raw_data)
                if optic_float is not None else None
            )
        },
        cached_result_dict
    )


def run_objects_group(scene_frames, path_to_video,
                      processor_instance: MetricsComputer,
                      cached_object_metrics: list[Metric | None]
                      ):
    cached_result_dict = {
        metric.name: metric for metric in cached_object_metrics if metric is not None
    }
    if len(cached_result_dict) in (6, 7):
        return cached_result_dict

    object_detections = safe_run(processor_instance._object_detector.detect_objects, scene_frames)
    tracks = safe_run(processor_instance._object_detector.track_objects, object_detections, path_to_video)

    if tracks is None:
        return {
            "duration_sec": Metric(name="duration_sec", value=None, raw_data=None),
            "persistence": Metric(name="persistence", value=None, raw_data=None),
            "norm_avg_velocity": Metric(name="norm_avg_velocity", value=None, raw_data=None),
            "norm_displacement": Metric(name="norm_displacement", value=None, raw_data=None),
            "frames_count": Metric(name="frames_count", value=None, raw_data=None),
            "domain_concentration": Metric(name="domain_concentration", value=None, raw_data=None)
        }
    track_metrics = safe_run(processor_instance._object_detector.calculate_metrics, tracks['result'], path_to_video)

    if track_metrics is None:
        return {
            "duration_sec": Metric(name="duration_sec", value=None, raw_data=None),
            "persistence": Metric(name="persistence", value=None, raw_data=None),
            "norm_avg_velocity": Metric(name="norm_avg_velocity", value=None, raw_data=None),
            "norm_displacement": Metric(name="norm_displacement_v", value=None, raw_data=None),
            "frames_count": Metric(name="frames_count", value=None, raw_data=None),
            "domain_concentration": Metric(name="domain_concentration", value=None, raw_data=None)
        }

    track_metrics: dict

    duration_sec_v = (sum(track_metric["duration_sec"] for track_metric in track_metrics)
                      / len(track_metrics))
    persistence_v = (sum(track_metric["persistence"] for track_metric in track_metrics)
                     / len(track_metrics))
    norm_avg_velocity_v = (sum(track_metric["norm_avg_velocity"] for track_metric in track_metrics)
                           / len(track_metrics))
    norm_displacement_v = (sum(track_metric["norm_displacement"] for track_metric in track_metrics)
                           / len(track_metrics))
    frames_count_v = (sum(track_metric["frames_count"] for track_metric in track_metrics)
                      / len(track_metrics))
    domain_concentration_v = processor_instance._object_detector.get_domain_concentration(object_detections)

    return {
        "duration_sec": Metric(name="duration_sec", value=duration_sec_v, raw_data=None),
        "persistence": Metric(name="persistence", value=persistence_v, raw_data=None),
        "norm_avg_velocity": Metric(name="norm_avg_velocity", value=norm_avg_velocity_v, raw_data=None),
        "norm_displacement": Metric(name="norm_displacement_v", value=norm_displacement_v, raw_data=None),
        "frames_count": Metric(name="frames_count", value=frames_count_v, raw_data=None),
        "domain_concentration": Metric(name="domain_concentration", value=domain_concentration_v, raw_data=None),

    }


def get_scene_features(torch_frames, processor_instance: MetricsComputer, cached_diversity) -> dict:
    feature_metric = next((metric for metric in cached_diversity if metric.name == "features"), None)
    if feature_metric is None:
        features = safe_run(processor_instance.clip_features.compute, torch_frames)
    else:
        features = feature_metric.value

    if isinstance(features, Metric):
        value = features.value
    else:
        value = features
    return {"features": Metric(name="features", value=value, raw_data=None)}


def get_cluster_metrics(features, processor_instance: MetricsComputer):
    cluster_metrics = processor_instance.cluster_metrics.compute(features)
    ics_metrics = processor_instance.ics_metrics.compute(features)

    return {
        "clip_ics": Metric(name="clip_ics", value=ics_metrics["mean_knn_dist"], raw_data=None),
        "num_clusters": Metric(name="num_clusters", value=cluster_metrics["num_clusters"], raw_data=None),
        "r_R_score": Metric(name="r_R_score", value=cluster_metrics["r_R_score"], raw_data=None),
        "noise_ratio": Metric(name="noise_ratio", value=cluster_metrics["noise_ratio"], raw_data=None),
    }


class MetricsProcessor:
    def __init__(self, cache_manager: CacheManager, object_classes: list[str], group_classes: dict[str, list[str]]):
        self.metrics_computer = MetricsComputer(object_classes, group_classes)
        self.cache_manager = cache_manager
        self.pool = ProcessPoolExecutor()

    def compute_scene_parallel(self, path_to_video, scene_segment, callback):
        # 1. Единый препроцессинг (в текущем потоке/процессе)
        data = self._prepare_data_bundle(path_to_video, scene_segment)
        cached_metrics = self.cache_manager.get_scene_cached_metrics(path_to_video, scene_segment)

        if cached_metrics is not None:
            cached_quality_metrics = [metric for metric in cached_metrics if metric.name in (
                "entropy", "brightness", "contrast", "saturation", "blur", "artifacts",
            )]
            cached_motion_metrics = [metric for metric in cached_metrics if metric.name in (
                "smoothness", "flow", "psnr", "ssim", "background_dominance"
            )]
            cached_objects_metrics = [metric for metric in cached_metrics if metric.name in (
                "duration_sec", "persistence", "norm_avg_velocity", "norm_displacement",
                "frames_count", "domain_concentration"
            )]
        else:
            cached_quality_metrics = []
            cached_motion_metrics = []
            cached_objects_metrics = []

        # 2. Формируем задачи
        # Используем partial, чтобы прокинуть данные в функции воркеров
        tasks = [
            ("quality", run_quality_group, (data["gray"], data["hsv"], data["torch_gray"], self.metrics_computer, cached_quality_metrics)),
            ("motion", run_motion_group, (data["torch"], data["gray"], self.metrics_computer, cached_motion_metrics)),
            ("objects", run_objects_group, (data["scene_frames"], data["video_path"], self.metrics_computer, cached_objects_metrics)),
        ]

        # 3. Запуск
        for name, func, args in tasks:
            future = self.pool.submit(func, *args)
            # Добавляем обработчик завершения, чтобы сразу отправить в UI
            future.add_done_callback(lambda f, n=name, p=path_to_video, s=scene_segment:
                                        callback(str(p), s, n, f.result()))

    def _prepare_data_bundle(self, path_to_video: str | Path, scene_segment: SceneSegment) -> dict:
        video_wrapper = VideoWrapper(path_to_video)
        scene_frames = list(video_wrapper.frames_generator(
            start=scene_segment.start_frame,
            end=scene_segment.end_frame,
            step=1,
        ))
        resized_frames = [cv2.resize(frame, (853, 480)) for frame in scene_frames]
        grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in resized_frames]
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) for frame in resized_frames]
        torch_ready_frames = [
            torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            for frame in resized_frames
        ]
        torch_ready_grayscale_frames = [
            torch.tensor(frame, dtype=torch.float32) / 255.0 for frame in grayscale_frames
        ]
        return {
            "video_path": path_to_video,
            "scene_frames": scene_frames,
            "gray": grayscale_frames,
            "hsv": hsv_frames,
            "torch": torch_ready_frames,
            "torch_gray": torch_ready_grayscale_frames
        }

    def compute_scene_sync(self, path_to_video: str | Path, scene_segment: SceneSegment) -> None:
        """
        Синхронно вычисляет все метрики по сцене в текущем потоке и сохраняет в кэш.
        Используется пайплайном в фоновом потоке.
        """
        data = self._prepare_data_bundle(path_to_video, scene_segment)
        print("build_data")
        cached_metrics = None  #TODO
        # cached_metrics = self.cache_manager.get_scene_cached_metrics(path_to_video, scene_segment)
        # scene_features = next(metric for metric in cached_metrics if metric.name == "features").value

        # calculate temporal_consistency (compute cosine similarity)
        # normalized = scene_features / np.linalg.norm(scene_features, axis=1, keepdims=True)
        # result_value_v = np.sum(normalized[:-1] * normalized[1:], axis=1).mean()
        #
        # temporary_consistency_clip_score = Metric(name="temporary_consistency_clip_score", value=result_value_v, raw_data=None)
        # self.cache_manager.save_temporal_cons(path_to_video, scene_segment, temporary_consistency_clip_score)
        # return
        # recalculate clusters
        # cluster_metrics = get_cluster_metrics(scene_features, self.metrics_computer)

        if cached_metrics is not None:  # here norm
            cached_quality = [m for m in cached_metrics if m and m.name in (
                "entropy", "brightness", "contrast", "saturation", "blur", "artifacts",
            )]
            cached_motion = [m for m in cached_metrics if m and m.name in (
                "smoothness", "flow", "psnr", "ssim", "background_dominance"
            )]
            cached_objects = [m for m in cached_metrics if m and m.name in (
                "duration_sec", "persistence", "norm_avg_velocity", "norm_displacement",
                "frames_count", "domain_concentration"
            )]
            cached_diversity = [m for m in cached_metrics if m and m.name in (
                "clip_ics", "num_clusters", "r_R_score", "noise_ratio", "features"
            )]
        else:
            cached_quality = []
            cached_motion = []
            cached_objects = []
            cached_diversity = []

        quality_res = run_quality_group(
            data["torch"], data["gray"], data["hsv"], data["torch_gray"],
            self.metrics_computer, cached_quality
        )
        print("get quality_res")

        del data["hsv"]
        del data["torch_gray"]

        motion_res = run_motion_group(
            data["torch"], data["gray"],
            self.metrics_computer, cached_motion
        )
        print("get motion_res")
        del data["gray"]
        objects_res = run_objects_group(
            data["scene_frames"], data["video_path"],
            self.metrics_computer, cached_objects
        )
        print("get objects_res")
        del data["scene_frames"]
        scene_features = get_scene_features(data["torch"], self.metrics_computer, cached_diversity)

        print("get scene_features")

        # self.cache_manager.resave_scene_features(path_to_video, scene_segment, scene_features["features"].value)
        # return
        del data["torch"]
        all_metrics = []
        for d in (quality_res, motion_res, objects_res, scene_features):
            if isinstance(d, dict):
                for v in d.values():
                    if v is not None and hasattr(v, "name"):
                        all_metrics.append(v)
        if all_metrics:
            self.cache_manager.save_scene_metrics_to_cache(path_to_video, scene_segment, all_metrics)


if __name__ == "__main__":
    video_path = "/Users/dvuglaf/Downloads/2024.04.09_15.20.37.mov"
    cache_manager = CacheManager("./cache")
    print(cache_manager._get_video_hash(video_path))
    # scene = SceneSegment(start_frame=100, end_frame=200, start_time_s=4.0, end_time_s=8.0)
    #
    # processor = MetricsProcessor()
    # processor.load_scene(video_path, scene)
    # # processor.compute_quality_metrics()
    # print(processor.compute_motion_metrics()[-1])