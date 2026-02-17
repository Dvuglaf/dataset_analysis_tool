from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from pydantic import BaseModel


@dataclass
class SceneSegment:
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    label: str = ""


class VideoModel(BaseModel):
    id: int
    path: Path


class SceneAlgoModel(BaseModel):
    id: int
    name: str
    params: dict


class SceneModel(BaseModel, SceneSegment):
    scene_algo_id: int
    video_id: int


class ProjectModel(BaseModel):
    version: str
    dataset_path: Path | None
    created_at: str
    videos: list[VideoModel] | None
    scenes: list[SceneModel] | None
    scene_algos: list[SceneAlgoModel] | None
    settings: dict


class VideoWrapper:
    def __init__(self, path_to_file: str | Path):
        self._path_to_file = Path(path_to_file)
        self.cap = cv2.VideoCapture(str(self._path_to_file))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frames(self) -> Generator[np.ndarray, None, None]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            yield frame

    def frames_generator(self, start: int = 0, end: int = -1, step: int = 1, target_height: int | None = None) -> Generator[np.ndarray, None, None]:
        if target_height is not None:
            if target_height == 480:
                target_width = 640
            if target_height == 720:
                target_width = 1280
            if target_height == 240:
                target_width = 320

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        current_frame = start
        while True:
            if end != -1 and current_frame >= end:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            if target_height is not None:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            yield frame

            current_frame += step
            if step != 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    @property
    def path_to_file(self) -> Path:
        return self._path_to_file

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def file_size(self) -> int:
        return self._path_to_file.stat().st_size

    def __getitem__(self, item: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, item)
        ret, frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
