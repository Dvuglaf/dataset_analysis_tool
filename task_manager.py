import uuid
from pathlib import Path
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from cache_manager import CacheManager
from custom_types import SceneSegment
from metric_pipeline import MetricsProcessor


class BaseTask(QObject):
    # Сигналы для связи с UI
    started = Signal(str)  # task_id
    progress = Signal(str, int)  # task_id, percentage
    partial_result = Signal(str, SceneSegment, str, dict)  # task_id, data_chunk
    finished = Signal(str, dict)  # task_id, final_data
    error = Signal(str, str)   # task_id, error_msg

    def __init__(self, video_path: Path, scene: SceneSegment | list[SceneSegment],
                 cache_manager: CacheManager, object_classes: dict[str, list[str]]):
        super().__init__()
        self.video_path = video_path
        self.scene = scene
        self.task_id = self._generate_id()
        self.cache_manager = cache_manager
        self._object_classes = object_classes

    @staticmethod
    def _generate_id():
        return str(uuid.uuid4())

    def run(self):
        """Переопределяется в подклассах"""
        raise NotImplementedError


class ComputeSceneMetricsTask(BaseTask):
    def run(self):
        self.started.emit(self.task_id)

        results = {}

        object_classes = []
        for value in self._object_classes.values():
            object_classes.extend(value)

        try:
            # Вызываем диспетчер
            metrics_processor = MetricsProcessor(self.cache_manager, object_classes, self._object_classes)
            metrics_processor.compute_scene_parallel(
                self.video_path,
                self.scene,
                callback=lambda video, scene, name, res:
                    self.partial_result.emit(
                        video, scene, self.task_id, {"group": name, "data": res}
                    )
            )

            self.finished.emit(self.task_id, results)

        except Exception as e:
            print("EXCEPTION in ComputeSceneMetricsTask:", e)
            self.error.emit(self.task_id, str(e))


class ComputeVideoMetricsTask(BaseTask):
    def run(self):
        self.started.emit(self.task_id)

        results = {}

        object_classes = []
        for _, value in self._object_classes.values():
            object_classes.extend(value)

        try:
            # Вызываем диспетчер
            metrics_processor = MetricsProcessor(self.cache_manager, object_classes, self._object_classes)
            for scene in self.scene:
                metrics_processor.compute_scene_parallel(
                    self.video_path,
                    scene,
                    callback=lambda video, sc, name, res:
                        self.partial_result.emit(
                            video, sc, self.task_id, {"group": name, "data": res}
                        )
                )
            self.finished.emit(self.task_id, results)

        except Exception as e:
            print("EXCEPTION in ComputeVideoMetricsTask:", e)
            self.error.emit(self.task_id, str(e))


class AnalysisManager(QObject):
    def __init__(self):
        super().__init__()
        self.active_tasks = {}
        self.thread_pool = QThreadPool.globalInstance()
        # Можно ограничить количество одновременных задач для CPU
        self.thread_pool.setMaxThreadCount(4)

    def submit_task(self, task: BaseTask):
        if task.task_id in self.active_tasks:
            print("Task already running")
            return

        # Пробрасываем сигналы задачи в менеджер или сразу в UI
        task.finished.connect(lambda tid: self.active_tasks.pop(tid, None))

        # Обертка для запуска в QThreadPool
        class RunnableWrapper(QRunnable):
            def run(self): task.run()

        self.active_tasks[task.task_id] = task
        self.thread_pool.start(RunnableWrapper())
