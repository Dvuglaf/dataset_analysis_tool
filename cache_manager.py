import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from computations.metrics.base import Metric
from custom_types import SceneSegment


class SceneMetricData(BaseModel):
    metric_name: str
    metric_value: float | str | None


class SceneMetrics(BaseModel):
    scene_id: str
    metrics: list[SceneMetricData]


class VideoMetricData(BaseModel):
    metric_name: str
    metric_value: float | str | None


class VideoMetrics(BaseModel):
    metrics: list[VideoMetricData]


class CacheManager:
    def __init__(self, path_to_cache_dir: str | Path):
        self.path_to_cache_dir = Path(path_to_cache_dir)
        self.path_to_cache_dir.mkdir(exist_ok=True, parents=True)

    def get_features(self, video: str | Path, scene: SceneSegment):
        metrics = self.get_scene_cached_metrics(video, scene)
        if metrics is None:
            return None
        return next((self._restore_metric_value(metric.value) for metric in metrics if metric is not None and metric.name == "features"), None)

    def get_dataset_level_metrics(self, video_paths: list[str | Path]):
        def metric_map(name: str) -> str:
            if name == "norm_displacement":
                return "norm_displacement_v"
            return name

        all_metrics = {}
        for path_to_video in video_paths:
            video_metrics = self.get_video_cached_metrics(path_to_video)
            if len(all_metrics.keys()) == 0:
                all_metrics.update({metric_map(metric.name): [] for metric in video_metrics if metric.name != "features"})
            for metric in video_metrics:
                if metric.name == "features":
                    continue
                all_metrics[metric_map(metric.name)].append(metric.value)

        for key in all_metrics.keys():
            valid_values = [v for v in all_metrics[key] if v is not None]
            all_metrics[key] = float(np.mean(valid_values))
        all_metrics.update({"ics_score": 0.19781219959259033})
        return all_metrics

    def get_video_cached_metrics(self,
                                 path_to_video: str | Path,
                                 scenes: list[SceneSegment] | None = None,
                                 cluster_metrics=None
                                 ) -> list[Metric | None] | None:
        video_hash = self._get_video_hash(path_to_video)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"
        if not cache_file.exists():
            if scenes is None:
                raise ValueError
            if not all(self._get_scene_cache_file(path_to_video, scene).exists() for scene in scenes):
                return None

            self._unite_scenes_metrics(path_to_video, scenes, cluster_metrics)

        with open(cache_file, 'r') as f:
            cache_data = VideoMetrics.model_validate(json.load(f))

        return [
            Metric(name=metric.metric_name,
                   value=self._restore_metric_value(metric.metric_value),
                   raw_data=None
                   )
            for metric in cache_data.metrics
        ]

    def _unite_scenes_metrics(self, path_to_video: str | Path, scenes: list[SceneSegment], cluster_metrics: dict[str, Metric] | None = None):
        video_hash = self._get_video_hash(path_to_video)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"

        metrics_map = {}

        for scene in scenes:
            scene_metrics = self.get_scene_cached_metrics(path_to_video, scene)
            for metric in scene_metrics:
                name = metric.name
                value = metric.value
                if name not in metrics_map.keys():
                    metrics_map.update({name: []})
                metrics_map[name].append(value)

        aggregated_metrics: list[VideoMetricData] = []

        for metric_name, values in metrics_map.items():
            # Убираем null
            valid_values = [v for v in values if v is not None]

            # Если всё null
            if not valid_values:
                metric_data = VideoMetricData(metric_name=metric_name, metric_value=None)
                aggregated_metrics.append(metric_data)
                continue

            # Если это npy
            if isinstance(valid_values[0], np.ndarray):
                # arrays = []
                # for filename in valid_values:
                #     path = self.path_to_cache_dir / filename
                #     arr = np.load(path)
                #     arrays.append(arr)

                # if metric_name == "features":
                #     valid_values = [v for v in valid_values if not isinstance(v, float)]

                if metric_name == "features":
                    mean_array = np.concatenate(valid_values, axis=0)
                else:
                    mean_array = np.mean(valid_values, axis=0)
                out_name = f"{video_hash}_{metric_name}_mean.npy"
                out_path = self.path_to_cache_dir / out_name
                np.save(out_path, mean_array)

                metric_data = VideoMetricData(metric_name=metric_name, metric_value=out_name)
                aggregated_metrics.append(metric_data)
            else:  # число
                print(valid_values)
                nums = [float(v) for v in valid_values]
                mean_value = float(np.mean(nums))
                metric_data = VideoMetricData(metric_name=metric_name, metric_value=mean_value)
                aggregated_metrics.append(metric_data)

        if cluster_metrics is not None:
            for cluster_metric_name, cluster_metric in cluster_metrics.items():
                aggregated_metrics.append(VideoMetricData(
                    metric_name=cluster_metric_name,
                    metric_value=cluster_metric.value
                ))

        video_metrics = VideoMetrics(metrics=aggregated_metrics)
        cache_file.write_text(video_metrics.model_dump_json(indent=2))

    def get_scene_cached_metrics(self,
                                 path_to_video: str | Path,
                                 scene_segment: SceneSegment,
                                 ) -> list[Metric | None] | None:
        cache_file = self._get_scene_cache_file(path_to_video, scene_segment)
        if not cache_file.exists():
            return None

        with open(cache_file, 'r') as f:
            cache_data = SceneMetrics.model_validate(json.load(f))

        return [
            Metric(name=metric.metric_name,
                   value=self._restore_metric_value(metric.metric_value),
                   raw_data=None
                   ) for metric in cache_data.metrics
        ]

    def save_video_metrics_to_cache(self, path_to_video: str | Path, metrics: list[Metric]) -> None:
        video_hash = self._get_video_hash(path_to_video)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"
        for metric in metrics:
            self._save_video_metric_to_cache(cache_file, metric)

    def resave_video_features(self,
                              path_to_video: str | Path,
                              scenes: list[SceneSegment]
                              ):
        all_features = []
        for scene in scenes:
            cache_file = self._get_scene_cache_file(path_to_video, scene)
            with open(cache_file, 'r') as f:
                cache_data = SceneMetrics.model_validate(json.load(f))
            target_metric = next(
                (metric for metric in cache_data.metrics if metric.metric_name == "features"), None)
            path_to_features = self.path_to_cache_dir / target_metric.metric_value
            features = np.load(path_to_features)
            all_features.append(features)

        all_features = np.concatenate(all_features, axis=0)
        video_hash = self._get_video_hash(path_to_video)
        np.save(self.path_to_cache_dir / f"{video_hash}_features.npy", all_features)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"
        with open(cache_file, 'r') as f:
            cache_data = VideoMetrics.model_validate(json.load(f))
        target_metric = next(
            (metric for metric in cache_data.metrics if metric.metric_name == "features"), None)
        target_metric.metric_value = f"{cache_file.stem}_features.npy"
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    def save_video_ics(self, path_to_video: str | Path, value):
        video_hash = self._get_video_hash(path_to_video)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"
        with open(cache_file, 'r') as f:
            cache_data = VideoMetrics.model_validate(json.load(f))
        cache_data.metrics.append(VideoMetricData(
            metric_name="clip_ics",
            metric_value=value
        ))
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    def save_clip_video(self, path_to_video: str | Path,
                              scenes: list[SceneSegment]):
        all_values = []
        for scene in scenes:
            cache_file = self._get_scene_cache_file(path_to_video, scene)
            with open(cache_file, 'r') as f:
                cache_data = SceneMetrics.model_validate(json.load(f))
            target_metric = next(
                (metric for metric in cache_data.metrics if metric.metric_name == "temporary_consistency_clip_score"), None)
            all_values.append(target_metric.metric_value)

        mean_value = np.mean(all_values)
        video_hash = self._get_video_hash(path_to_video)
        cache_file = self.path_to_cache_dir / f"{video_hash}.json"
        with open(cache_file, 'r') as f:
            cache_data = VideoMetrics.model_validate(json.load(f))
        cache_data.metrics.append(VideoMetricData(
            metric_name="temporary_consistency_clip_score",
            metric_value=float(mean_value)
        ))
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    # TODO: delete
    def save_temporal_cons(self,
                           path_to_video: str | Path,
                           scene_segment: SceneSegment,
                           metric: Metric):
        cache_file = self._get_scene_cache_file(path_to_video, scene_segment)
        with open(cache_file, 'r') as f:
            cache_data = SceneMetrics.model_validate(json.load(f))
        cache_data.metrics.append(SceneMetricData(metric_name="temporary_consistency_clip_score",
                                                  metric_value=metric.value))
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)


    def resave_scene_features(self,
                              path_to_video: str | Path,
                              scene_segment: SceneSegment,
                              features: np.ndarray
                              ):
        cache_file = self._get_scene_cache_file(path_to_video, scene_segment)
        with open(cache_file, 'r') as f:
            cache_data = SceneMetrics.model_validate(json.load(f))
        path_to_npy = self._cache_numpy_array(features, cache_file, "features", "value")
        target_metric = next((metric for metric in cache_data.metrics if metric.metric_name == "features"), None)
        target_metric.metric_value = path_to_npy.name
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    def save_scene_metrics_to_cache(self,
                                    path_to_video: str | Path,
                                    scene_segment: SceneSegment,
                                    metrics: list[Metric]
                                    ) -> None:
        cache_file = self._get_scene_cache_file(path_to_video, scene_segment)
        for metric in metrics:
            self._save_scene_metric_to_cache(cache_file, metric, scene_segment.label)

    def _get_scene_cache_file(self, path_to_video: str | Path, scene: SceneSegment) -> Path:
        video_hash = self._get_video_hash(path_to_video)
        return self.path_to_cache_dir / f"{video_hash}_{scene.start_frame}_{scene.end_frame}.json"

    def _restore_metric_value(self, cached_metric: str | float):
        if isinstance(cached_metric, str) and cached_metric.endswith(".npy"):
            try:
                value = np.load(self.path_to_cache_dir / cached_metric)
            except Exception as e:
                print(f"Caught exception in restore_metric_value: {e}")
                return None
            else:
                return value
        else:
            return cached_metric

    @staticmethod
    def _cache_numpy_array(array: np.ndarray,
                           cache_file: Path,
                           metric_name: str,
                           metric_type: Literal["value", "raw"]
                           ) -> Path:
        value_data_path = cache_file.parent / f"{cache_file.stem}_{metric_name}_{metric_type}.npy"
        np.save(value_data_path, array)
        return value_data_path

    def _save_scene_metric_to_cache(self, cache_file: Path, metric: Metric, scene_id: str) -> None:
        metric_name = metric.name
        metric_value = metric.value
        if isinstance(metric.value, np.ndarray):
            metric_value = self._cache_numpy_array(metric.value, cache_file, metric_name, "value").name

        new_metric = SceneMetricData(metric_name=metric_name,
                                     metric_value=metric_value,
                                     )
        # Читаем существующий кеш
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = SceneMetrics.model_validate(json.load(f))
        else:
            cache_data = SceneMetrics(scene_id=scene_id, metrics=[])

        cache_data.metrics.append(new_metric)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    def _save_video_metric_to_cache(self, cache_file: Path, metric: Metric) -> None:
        metric_name = metric.name
        metric_value = metric.value
        if isinstance(metric.value, np.ndarray):
            metric_value = self._cache_numpy_array(metric.value, cache_file, metric_name, "value").name

        new_metric = VideoMetricData(metric_name=metric_name,
                                     metric_value=metric_value,
                                     )
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = VideoMetrics.model_validate(json.load(f))
        else:
            cache_data = VideoMetrics(metrics=[])

        cache_data.metrics.append(new_metric)
        with open(cache_file, 'w') as f:
            json.dump(cache_data.model_dump(), f, indent=2)

    @staticmethod
    def _get_video_hash(path_to_video: str | Path) -> str:
        BUF_SIZE = 65536

        sha256 = hashlib.sha256()

        with open(path_to_video, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)

        return sha256.hexdigest()
