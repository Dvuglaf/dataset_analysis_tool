import json
from pathlib import Path
from datetime import datetime

from pydantic import ValidationError

from custom_types import ProjectModel, SceneModel, VideoModel, SceneSegment, SceneAlgoModel


def get_relative_video_path(video_path: Path, root_path: Path) -> Path:
    try:
        return video_path.relative_to(root_path)
    except ValueError:
        return video_path


class Project:
    def __init__(self, project_path: str | Path | None):
        self._project_path = None
        self._project_data = None

        if project_path:
            self._project_path = Path(project_path)
            try:
                self._project_data = self._load_project_data()
            except ValidationError as e:
                print(f"Error loading project data: {e}")
                self._project_data = None
                raise
        else:
            self._project_data = ProjectModel(
                version='1',
                dataset_path=None,
                created_at=datetime.now().isoformat(),
                videos=None,
                scenes=None,
                scene_algos=None,
                settings={}
            )
    
    @property
    def project_path(self) -> Path | None:
        return self._project_path
    
    @property
    def dataset_path(self) -> Path | None:
        return self._project_data.dataset_path

    @dataset_path.setter
    def dataset_path(self, dataset_path: str | Path):
        self._project_data.dataset_path = Path(dataset_path)
    
    @property
    def has_project_file(self) -> bool:
        return self._project_path is not None

    def _load_project_data(self) -> ProjectModel:
        data = ProjectModel.model_validate(
            json.loads(self._project_path.read_text(encoding="utf-8"))
        )
        return data

    def get_video_scenes(self, path_to_video: str | Path) -> tuple[list[SceneModel], SceneAlgoModel] | None:
        if self._project_data.scenes is None or self._project_data.videos is None:
            return None
        relative_video_path = get_relative_video_path(Path(path_to_video), self._project_data.dataset_path)
        video_id = next((v.id for v in self._project_data.videos if v.path == relative_video_path), None)
        if video_id is None:
            return None
        scenes = sorted([scene for scene in self._project_data.scenes if scene.video_id == video_id],
                        key=lambda s: s.start_frame)
        if not all(scene.scene_algo_id == scenes[0].scene_algo_id for scene in scenes):
            raise ValueError("All scenes for the same video should have the same scene algo")
        scene_algo = next((a for a in self._project_data.scene_algos if a.id == scenes[0].scene_algo_id), None)
        if scene_algo is None:
            raise ValueError("Scene algo not found for the video scenes")
        return scenes, scene_algo

    def add_video(self, video_path: str | Path) -> VideoModel:
        if self._project_data.videos is None:
            self._project_data.videos = []
        relative_video_path = get_relative_video_path(Path(video_path), self._project_data.dataset_path)
        existed_video = next((v for v in self._project_data.videos if v.path == relative_video_path), None)
        if existed_video is not None:
            return existed_video
        video_id = max((v.id for v in self._project_data.videos), default=0) + 1
        new_video = VideoModel(id=video_id, path=relative_video_path)
        self._project_data.videos.append(new_video)
        return new_video

    def add_scene_algo(self, name: str, params: dict) -> SceneAlgoModel:
        if self._project_data.scene_algos is None:
            self._project_data.scene_algos = []
        existed_scene_algo = next(
            (a for a in self._project_data.scene_algos if a.name == name and a.params == params),
            None
        )
        if existed_scene_algo is not None:
            return existed_scene_algo
        algo_id = max((a.id for a in self._project_data.scene_algos), default=0) + 1
        new_algo = SceneAlgoModel(id=algo_id, name=name, params=params)
        self._project_data.scene_algos.append(new_algo)
        return new_algo

    def add_scene(self, video_path: str | Path, scene_info: SceneSegment, algo_name: str, params: dict):
        video = self.add_video(video_path)
        scene_algo = self.add_scene_algo(name=algo_name, params=params)
        # remove all scenes for the video with different algorithm
        if self._project_data.scenes is not None:
            self._project_data.scenes = [
                scene for scene in self._project_data.scenes
                if not (scene.video_id == video.id and scene.scene_algo_id != scene_algo.id)
            ]
        new_scene = SceneModel(
            scene_algo_id=scene_algo.id,
            video_id=video.id,
            start_frame=scene_info.start_frame,
            end_frame=scene_info.end_frame,
            start_time_s=scene_info.start_time_s,
            end_time_s=scene_info.end_time_s,
            label=scene_info.label
        )
        if self._project_data.scenes is None:
            self._project_data.scenes = []

        self._project_data.scenes.append(new_scene)

    def save(self, path: str | Path | None = None):
        save_path = path or self._project_path
        if save_path is None:
            raise ValueError("No path specified for saving the project.")

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(self._project_data.model_dump_json(indent=2))
            
            self._project_path = Path(save_path)
        except Exception as e:
            print(f"Error saving project: {e}")
            return False
