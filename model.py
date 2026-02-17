import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, cast

from PySide6 import QtCore
from PySide6.QtCore import Signal

from custom_types import VideoWrapper, SceneSegment, SceneModel


@dataclass
class ScenesStats:
    total_scenes: int
    average_duration: float
    shortest_scene: float
    longest_scene: float


class Model(QtCore.QObject):
    media_changed = Signal(VideoWrapper)
    frame_position_changed = Signal(int)
    scenes_changed = Signal(list)
    current_scene_changed = Signal(SceneSegment)
    current_classes_changed = Signal(list[str])
    current_tab_changed = Signal(str)

    PATH_TO_CLASSES_FILE = "./computations/third_party/yoloe/classes.json"

    def __init__(self):
        super().__init__()
        self._current_media = None
        self._current_frame = 0
        self._root_directory = None
        self._scenes: List[SceneSegment] = []
        self._current_scene = None
        self._current_tab: Literal["scene", "analysis"] = "scene"

        self._current_classes: dict[str, list[str]] = self._extract_classes_from_file(self.PATH_TO_CLASSES_FILE)

    @property
    def current_tab(self) -> Literal["scene", "analysis"]:
        return self._current_tab

    @current_tab.setter
    def current_tab(self, tab: Literal["scene", "analysis"]):
        self._current_tab = tab
        self.current_tab_changed.emit(tab)

    @property
    def current_media(self) -> VideoWrapper:
        return self._current_media

    @current_media.setter
    def current_media(self, media_path: Path):
        self._current_media = VideoWrapper(media_path)
        self.media_changed.emit(self._current_media)

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, position: int):
        self._current_frame = position
        self.frame_position_changed.emit(position)

    @property
    def root_directory(self):
        return self._root_directory

    @root_directory.setter
    def root_directory(self, directory: Path):
        self._root_directory = directory

    @property
    def scenes(self) -> List[SceneSegment]:
        return self._scenes

    @scenes.setter
    def scenes(self, scenes: List[SceneSegment] | None):
        self._scenes = scenes or []
        self.scenes_changed.emit(self._scenes)

    @property
    def current_scene(self) -> SceneSegment | None:
        return self._current_scene

    @current_scene.setter
    def current_scene(self, scene: SceneSegment | SceneModel):
        scene_segment = SceneSegment(
            start_frame=scene.start_frame,
            end_frame=scene.end_frame,
            start_time_s=scene.start_time_s,
            end_time_s=scene.end_time_s,
            label=scene.label
        )
        # if self._current_scene != scene_segment:
        self._current_scene = scene_segment
        self.current_scene_changed.emit(scene)

    def get_scenes_stats(self) -> ScenesStats | None:
        if not self._scenes:
            return None

        scene_durations_s = [scene.end_time_s - scene.start_time_s for scene in self._scenes]

        return ScenesStats(
            total_scenes=len(self._scenes),
            average_duration=sum(scene_durations_s) / len(self._scenes),
            shortest_scene=min(scene_durations_s),
            longest_scene=max(scene_durations_s)
        )

    @property
    def current_classes(self) -> dict[str, list[str]]:
        return self._current_classes

    @current_classes.setter
    def current_classes(self, info: str | Path | dict[str, list[str]]):
        if isinstance(info, (str, Path)):
            classes = self._extract_classes_from_file(info)
        else:
            classes = info

        self._current_classes = classes
        self.current_classes_changed.emit(classes)

    @staticmethod
    def _extract_classes_from_file(path_to_file: str | Path) -> dict[str, list[str]]:
        path_to_file = Path(path_to_file)
        if path_to_file.suffix == ".json":
            with open(path_to_file, mode='r', encoding="utf-8") as rf:
                json_data = json.load(rf)

            if (isinstance(json_data, dict)
                    and all(isinstance(key, str) for key in json_data.keys())
                    and all(isinstance(value, list) for value in json_data.values())
            ):
                return json_data
            else:
                matched = re.findall(r"\".*\"", path_to_file.read_text(encoding="utf-8"))
                return {"default": matched}
        else:
            raise NotImplementedError(f"Extract classes not implemented "
                                      f"for suffix {path_to_file.suffix}")
