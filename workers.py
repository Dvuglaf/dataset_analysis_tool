import subprocess
from pathlib import Path

import cv2
import numpy as np
from scenedetect import VideoStreamCv2, SceneManager

from config import SCENES
from custom_types import SceneSegment, VideoWrapper


def detect_fade_segments(path,
                         slope_thresh=2.0,
                         min_length=16,
                         smooth_window=6):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Не открыл видео: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

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


class SceneDetector:
    def __init__(self, algo: str, params: dict):
        self._algo = algo
        self._params = params

        if self._is_pyscenedetect_pipeline(algo):
            cls_object = SCENES.get(algo).get("object")
            self._scene_manager = SceneManager()
            self._scene_manager.add_detector(cls_object(**params))

    def detect_scenes(self, video_wrapper: VideoWrapper) -> list[SceneSegment]:
        if self._is_pyscenedetect_pipeline(self._algo):
            return self._detect_pyscenedetect(video_wrapper)
        else:
            return self._detect_custom_algo(video_wrapper)

    def _detect_pyscenedetect(self, video_wrapper: VideoWrapper) -> list[SceneSegment]:
        video = VideoStreamCv2(str(video_wrapper.path_to_file))
        self._scene_manager.detect_scenes(video)


        scene_timecodes = self._scene_manager.get_scene_list()
        scenes: list[SceneSegment] = [
            SceneSegment(
                start_frame=start.get_frames(),
                end_frame=end.get_frames(),
                start_time_s=start.get_seconds(),
                end_time_s=end.get_seconds(),
                label=f"{self._algo} #{idx}",
            )
            for idx, (start, end) in enumerate(scene_timecodes)
        ]

        if len(scenes) == 0:
            scenes = [SceneSegment(start_frame=0, start_time_s=0,
                                   end_frame=video_wrapper.num_frames,
                                   end_time_s=video_wrapper.num_frames / video_wrapper.fps,
                                   label="full")]

        fade_segments = detect_fade_segments(str(video_wrapper.path_to_file))
        for fade_segment in fade_segments:
            scene_corr_start = next(scene for scene in scenes if scene.start_time_s < fade_segment["start"] < scene.end_time_s)
            scene_corr_start.end_time_s = fade_segment["start"]
            scene_corr_start.end_frame = int(fade_segment["start"] * video_wrapper.fps)

            scene_corr_end = next((scene for scene in scenes if scene.start_time_s < fade_segment["end"] < scene.end_time_s), None)
            if scene_corr_end is not None:  # not end of video
                scene_corr_end.start_time_s = fade_segment["end"]
                scene_corr_end.start_frame = int(fade_segment["end"] * video_wrapper.fps)

        corrected_scenes: list[SceneSegment] = []

        adjusted_first_scene = None

        for idx, scene in enumerate(scenes):
            if adjusted_first_scene is not None:
                scene.start_frame = adjusted_first_scene.start_frame
                scene.start_time_s = adjusted_first_scene.start_time_s
                adjusted_first_scene = None

            if scene.end_frame - scene.start_frame < 16:
                if len(corrected_scenes) != 0:
                    corrected_scenes[-1].end_frame = scene.end_frame
                    corrected_scenes[-1].end_time_s = scene.end_time_s
                else:
                    adjusted_first_scene = scene
            else:
                corrected_scenes.append(scene)

        return corrected_scenes

    def _detect_custom_algo(self, video_wrapper: VideoWrapper) -> list[SceneSegment]:
        frame_step = int(self._params.get("scene_length", 5.0) * video_wrapper.fps)
        scenes = [
            SceneSegment(start_frame=i,
                         end_frame=min(i + frame_step, video_wrapper.num_frames),
                         start_time_s=i / video_wrapper.fps,
                         end_time_s=min((i + frame_step) / video_wrapper.fps, video_wrapper.num_frames / video_wrapper.fps)
                         )
            for i in range(0, video_wrapper.num_frames, frame_step)
        ]
        if self._params.get("adjust_last_scene", False):
            if scenes:
                last_scene = scenes[-1]
                if last_scene.end_frame < video_wrapper.num_frames:
                    scenes[-1] = SceneSegment(
                        start_frame=last_scene.start_frame,
                        end_frame=video_wrapper.num_frames,
                        start_time_s=last_scene.start_time_s,
                        end_time_s=video_wrapper.num_frames / video_wrapper.fps
                    )
        return scenes

    @staticmethod
    def _is_pyscenedetect_pipeline(algo: str) -> bool:
        return "pyscenedetect" in algo.lower()


def detect_scenes_worker(video_path: str | Path, algo: str, params: dict) -> dict:
    scene_detector = SceneDetector(algo, params)

    return {
        "video_path": str(video_path),
        "algo": algo,
        "params": params,
        "scenes": scene_detector.detect_scenes(VideoWrapper(video_path)),
    }


def export_scenes_worker(video_path: str | Path, scenes: list[SceneSegment], output_directory: str | Path,
                         file_prefix: str, file_ext: str, overwrite_files: bool,
                         progress_callback=None,
                         ) -> dict:
    output_directory = Path(output_directory)
    total = max(1, len(scenes))
    num_errors = 0
    for i, scene in enumerate(scenes, start=1):
        file_name = f"{file_prefix}_{i:03d}{file_ext}"
        path_to_output_file = output_directory / file_name
        if path_to_output_file.exists() and not overwrite_files:
            pass
        else:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path), "-ss", str(scene.start_time_s), "-to", str(scene.end_time_s),
                "-c", "copy", str(path_to_output_file)
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL)
            except Exception as e:
                print(f"Error exporting scene {i}: {e}")
                num_errors += 1

        if progress_callback is not None:
            progress_callback(i / total)

    return {"output_directory": output_directory, "num_scenes": len(scenes), "num_errors": num_errors}
