import json
from pathlib import Path

import cv2
import numpy as np
from pyexpat import features

from PIL.features import features

from cache_manager import CacheManager
from custom_types import VideoWrapper, SceneSegment
from metric_pipeline import MetricsProcessor, get_cluster_metrics
from project import Project


def temporal_consistency(features: np.ndarray) -> float:
    """
    features: (N, 512)
    """
    if len(features) < 2:
        return 0.0

    # normalize
    features = features / np.linalg.norm(
        features, axis=1, keepdims=True
    )

    # cosine similarity
    sims = np.sum(
        features[:-1] * features[1:],
        axis=1
    )

    return float(sims.mean())


def scene_checker(path: str,
                  start_frame: int,
                  end_frame: int,
                  low_thresh: float=45,
                  high_thresh: float=220):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    brightnesses = []

    while True:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 == end_frame:
            break
        _, frame = cap.read()
        if frame is None: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses.append(np.mean(frame))

    cap.release()
    return np.mean(brightnesses) < low_thresh or np.mean(brightnesses) > high_thresh


def detect_fade_segments(path,
                         slope_thresh=2.0,
                         min_length=16,
                         smooth_window=6):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не открыл видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(fps)

    means = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(np.mean(gray))
    cap.release()
    if len(means) < 2:
        return []

    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        means = np.convolve(means, kernel, mode='same')

    diffs = np.diff(means)

    def extract(mask):
        segs, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            if not v and start is not None:
                if i - start >= min_length:
                    segs.append((start, i))
                start = None
        if start is not None and len(mask) - start >= min_length:
            segs.append((start, len(mask)))
        return segs

    inc = diffs >= slope_thresh
    dec = diffs <= -slope_thresh
    seg_in  = extract(inc)
    seg_out = extract(dec)

    res = []
    for s,e in seg_in:
        res.append({'type':'in',  'start': s/fps, 'end': e/fps})
    for s,e in seg_out:
        res.append({'type':'out', 'start': s/fps, 'end': e/fps})
    return sorted(res, key=lambda x: x['start'])

# print(detect_fade_segments("/Users/dvuglaf/Desktop/industrial videos/tyres.mp4"))
# exit()

video_paths = list(Path("/Users/dvuglaf/Desktop/industrial videos").glob("**/*.mp4"))

cache_manager = CacheManager("./.analysis_cache_industrial_v2/")

with open("./computations/third_party/yoloe/classes.json") as rf:
    group_classes = json.load(rf)

object_classes = []
for group, values in group_classes.items():
    object_classes.extend(values)

total = len(video_paths)
processor = MetricsProcessor(cache_manager, object_classes, group_classes)
scene_config = {
    "algorithm": "pyscenedetect (Adaptive)",
    "parameters": {}
}
project = Project("/Users/dvuglaf/Desktop/industrial_videos_project.vdproj")

videos_features = []

# for idx, video_path in enumerate(video_paths):
video_path = "/Users/dvuglaf/Desktop/industrial videos/tyres.mp4"
print("fps", VideoWrapper(video_path).fps, VideoWrapper(video_path).num_frames)
print(f"process video: {video_path}")

# Scene detection
from workers import detect_scenes_worker

result = detect_scenes_worker(
    str(video_path),
    scene_config["algorithm"],
    scene_config["parameters"],
)

scenes = result["scenes"]
print(f"detected {len(scenes)} scenes: {scenes}")

print(detect_fade_segments(str(video_path)))
exit()

#     scene_features = [
#         cache_manager.get_features(video_path, scene)
#         for scene in scenes
#     ]
#     saved = np.concatenate(scene_features, axis=0)
#     print(saved.shape)
#     np.save("./.analysis_cache_industrial_v2/c00edeba6d8b6fc870f09df3d3aecbd3ca90d0ee2252dbb53322bccc7ad206c5_features.npy", saved)
#
#     exit()
#     # cache_manager.get_video_cached_metrics(video_path, scenes)
#     exit()
#     # scene_features = [
#     #     cache_manager.get_features(video_path, scene)
#     #     for scene in scenes
#     # ]
#     # videos_features.append(scene_features)
#     # video_ics_metric = processor.metrics_computer.ics_metrics.compute(scene_features)["mean_knn_dist"]
#     # cache_manager.save_video_ics(video_path, video_ics_metric)
#     # print("get features")
#     # continue
#
# # for scene in scenes:
# #     project.add_scene(video_path, scene, scene_config["algorithm"],
# #                       {"adaptive_threshold": 3.0,
# #                                "min_scene_len": 2,
# #                                "window_width": 30,
# #                                "min_content_val": 15.0})
# #
# # project.save()
#
# # video_metrics = cache_manager.get_video_cached_metrics(video_path, scenes)
# # if video_metrics is not None:
# #     print("found video features for video")
# #     continue
#
#     if not scenes:
#         try:
#             vw = VideoWrapper(video_path)
#             if vw.num_frames > 0:
#                 scenes = [
#                     SceneSegment(
#                         start_frame=0,
#                         end_frame=vw.num_frames,
#                         start_time_s=0.0,
#                         end_time_s=vw.num_frames / vw.fps,
#                         label="full",
#                     )
#                 ]
#         except Exception:
#             scenes = [
#                 SceneSegment(
#                     start_frame=0,
#                     end_frame=1,
#                     start_time_s=0.0,
#                     end_time_s=0.04,
#                     label="full",
#                 )
#             ]
#         print(f"corrected scenes: {len(scenes)} scenes: {scenes}")
#
#     exception_caught = False
#     # Metrics per scene (quality, motion, objects)
#     for s_idx, scene in enumerate(scenes):
#         print(f"computing metrics (scene {s_idx + 1}/{len(scenes)})")
#         try:
#             processor.compute_scene_sync(str(video_path), scene)
#         except Exception as e:
#             exception_caught = True
#             print(f"Caught exception in processor.compute_scene_sync: {e}")
#             raise
#     break
#
# # if not exception_caught:
# #     features = [
# #         cache_manager.get_features(video_path, scene)
# #         for scene in scenes
# #     ]
# #     features = [feature for feature in features if not isinstance(feature, float)]
# #     features = np.array(features)
# #
# #     cluster_metrics = get_cluster_metrics(features, processor.metrics_computer)
# #
# #     try:
# #         cache_manager._unite_scenes_metrics(video_path, scenes, cluster_metrics)
# #     except Exception as e:
# #         print(f"caught exception in cache_manager._unite_scenes_metrics: {e}")
