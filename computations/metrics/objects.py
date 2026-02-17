from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from argparse import Namespace
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics import YOLOE

from custom_types import VideoWrapper


@dataclass
class ObjectDetection:
    frame_index: int
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetector:
    """Детектор объектов на основе модели YOLOE"""

    def __init__(self,
                 model_checkpoint: str | Path,
                 group_classes: dict[str, list[str]],
                 classes: list[str],
                 device: str = "cpu",
                 track_high_thresh=0.45,
                 track_low_thresh=0.1,
                 new_track_thresh=0.4,
                 track_buffer=90,
                 match_thresh=0.8,
                 fuse_score=True,
                 min_iou_match=0.4,
                 min_track_length=5,
                 num_processes=None,
                 velocity_round_digits=5,
                 jitter_round_digits=5,
                 persistence_round_digits=3,
                 displacement_round_digits=5,
                 ):

        self._classes = classes
        self._group_classes = group_classes

        self.min_track_length = min_track_length
        self.velocity_round = velocity_round_digits
        self.jitter_round = jitter_round_digits
        self.persistence_round = persistence_round_digits
        self.displacement_round = displacement_round_digits

        self.device: str = device
        self.classes = classes
        self.model = YOLOE(model_checkpoint).to(device)
        self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))

        self.tracker_args = Namespace(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        self.min_iou_match = min_iou_match

    def detect_objects(self,
                       frames: list[np.ndarray],
                       frame_stride: int = 1,
                       conf_threshold: float = 0.65,
                       batch_size: int = 16
                       ) -> list[ObjectDetection]:
        video_detections = []

        try:
            results_gen = self.model.predict(
                source=frames,
                device=self.device,
                stream=True,
                vid_stride=frame_stride,
                conf=conf_threshold,
                batch=batch_size,
                verbose=True,
            )

            for i, res in enumerate(results_gen):
                real_frame_idx = i * frame_stride
                if res.boxes:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    confs = res.boxes.conf.cpu().numpy()

                    for j in range(len(boxes)):
                        row = ObjectDetection(
                            real_frame_idx,
                            cls_ids[j],
                            confs[j],
                            boxes[j][0],
                            boxes[j][1],
                            boxes[j][2],
                            boxes[j][3],
                        )
                        video_detections.append(row)

            del results_gen
            return video_detections

        except Exception as e:
            print("Caught exception in detect_objects:", e)
            return []

    def track_objects(self,
                      detections: list[ObjectDetection],
                      path_to_video: str | Path
                      ) -> dict | None:
        fps = VideoWrapper(path_to_video).fps
        tracker = BYTETracker(args=self.tracker_args, frame_rate=int(fps))

        data = np.array(
            [[d.frame_index, d.class_id, d.confidence, d.x1, d.y1, d.x2, d.y2] for d in detections])
        data = data[data[:, 0].argsort()]
        final_results = []

        start_f = int(data[:, 0].min())
        end_f = int(data[:, 0].max())

        for fid in range(start_f, end_f + 1):
            frame_data = data[data[:, 0] == fid]

            if len(frame_data) > 0:
                xyxy = frame_data[:, 3:7]
                conf = frame_data[:, 2].copy()
                if conf.max() > 1.0:
                    conf /= 100.0
                online_targets = tracker.update(TrackBox(xyxy, conf, frame_data[:, 1]))
            else:
                online_targets = tracker.update(
                    TrackBox(np.empty((0, 4)), np.empty(0), np.empty(0)))

            if len(online_targets) > 0 and len(frame_data) > 0:
                for target in online_targets:
                    t_xyxy, t_id = target[0:4], int(target[4])

                    best_iou, best_idx = 0, -1
                    for i in range(len(frame_data)):
                        iou = get_iou(t_xyxy, frame_data[i, 3:7])
                        if iou > best_iou:
                            best_iou, best_idx = iou, i

                    if best_idx != -1 and best_iou > 0.4:
                        new_row = np.append(frame_data[best_idx], t_id)
                        final_results.append(new_row)

        if final_results:
            res = np.vstack(final_results)
            return {
                'result': res,
                'unique_tracks': len(np.unique(res[:, -1])),
                'total_frames': len(np.unique(res[:, 0])),
            }

        return None

    def calculate_metrics(self, data: np.ndarray, path_to_video: str | Path):
        video_wrapper = VideoWrapper(path_to_video)

        if data.size == 0:
            return None

        w, h = video_wrapper.width, video_wrapper.height
        fps = video_wrapper.fps

        unique_tracks = np.unique(data[:, -1])
        results = []

        for tid in unique_tracks:
            track_data = data[data[:, -1] == tid]
            track_data = track_data[track_data[:, 0].argsort()]

            if len(track_data) < self.min_track_length:
                continue

            # Нормализованные координаты центра
            cx = ((track_data[:, 3] + track_data[:, 5]) / 2) / w
            cy = ((track_data[:, 4] + track_data[:, 6]) / 2) / h
            coords = np.column_stack([cx, cy])

            # Векторы перемещений между последовательными кадрами
            diffs = np.diff(coords, axis=0)
            step_distances = np.sqrt(np.sum(diffs**2, axis=1))

            # Метрики
            avg_vel = np.mean(step_distances)
            accel = np.diff(step_distances)
            jitter = np.mean(np.abs(accel)) if len(accel) > 0 else 0.0

            # Persistence = заполненность трека кадрами
            total_frames_range = int(track_data[-1, 0] - track_data[0, 0] + 1)
            persistence = len(track_data) / total_frames_range if total_frames_range > 0 else 0.0

            # Общее перемещение (start → end)
            displacement = np.sqrt(np.sum((coords[-1] - coords[0])**2))

            # Время удержания трека
            duration_sec = total_frames_range / fps

            results.append({
                'track_id'           : int(tid),
                'class_id'           : int(track_data[0, 1]),
                'frames_count'       : len(track_data),
                'duration_sec'       : round(duration_sec, 2),
                'norm_avg_velocity'  : round(avg_vel, self.velocity_round),
                'norm_jitter'        : round(jitter, self.jitter_round),
                'persistence'        : round(persistence, self.persistence_round),
                'norm_displacement'  : round(displacement, self.displacement_round),
                'res_w'              : w,
                'res_h'              : h,
                'fps'                : round(fps, 2)
            })

            return results if results else None

    def get_domain_concentration(self, detections: list[ObjectDetection]) -> float | None:
        def build_class_to_domain(domain_map: dict[str, list[str]]):
            class2domain = {}

            for domain, classes in domain_map.items():
                for cls in classes:
                    class2domain[cls] = domain

            return class2domain

        if not detections:
            return None

        id2name = {idx:value for idx, value in enumerate(self._classes)}
        class2domain = build_class_to_domain(self._group_classes)

        domain_counter = Counter()
        total = 0

        for det in detections:
            class_name = id2name.get(det.class_id)

            if class_name is None:
                continue

            domain = class2domain.get(class_name, "other")

            domain_counter[domain] += 1
            total += 1

        if total == 0:
            return None

        top_domain_count = max(domain_counter.values())
        concentration = (top_domain_count / total) * 100.0

        return round(concentration, 2)


class TrackBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.ascontiguousarray(xyxy, dtype=np.float32)
        self.conf = np.ascontiguousarray(conf, dtype=np.float32)
        self.cls = np.ascontiguousarray(cls, dtype=np.float32)
        w = self.xyxy[:, 2] - self.xyxy[:, 0]
        h = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.column_stack([self.xyxy[:, 0] + w / 2, self.xyxy[:, 1] + h / 2, w, h])

    def __len__(self): return len(self.conf)

    def __getitem__(self, idx): return TrackBox(self.xyxy[idx], self.conf[idx], self.cls[idx])


def get_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


if __name__ == "__main__":
    model_checkpoint = "./yoloe/checkpoints/yoloe-26n-seg.pt"
    classes = ["person", "car", "bicycle"]
    device = "cpu"

    path_to_video = "/Users/dvuglaf/2024.04.09_15.20.37.mov"
    video_wrapper = VideoWrapper(path_to_video)

    detector = ObjectDetector(model_checkpoint, classes, device)
    detections = detector.detect_objects(path_to_video, frame_stride=4, conf_threshold=0.5, batch_size=4)
    print(detections)

    tracks = detector.track_objects(detections, path_to_video)
    print(tracks)

    metrics = detector.calculate_metrics(tracks['result'], path_to_video)
    print(metrics)

