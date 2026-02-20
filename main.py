import sys
import os
from pathlib import Path

from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from analysis_tab import AnalysisTab, MetricRow
from cache_manager import CacheManager
from config import SCENES, FILTERS
from custom_types import SceneSegment, VideoWrapper
from dataset_browser import VideoBrowserWindow, build_tree
from filter_block import FilterPanel
from filter_timeline_grid import FilterTimelineWidget
from metric_pipeline import MetricsProcessor
from metric_table import TableRow
from model import Model
from media_panel import VideoArea
from project import Project
from report_window import ReportWindow
from scene_detector_block import SceneDetectorPanel
from scene_edit_timeline import SceneEditTimeline
from job_manager import JobManager, JobInfo, DoneJobInfo
from start_dialog import StartDialog
from task_manager import AnalysisManager
from tasks_dock import TasksDock
from export_scenes_dialog import ExportScenesDialog
from workers import detect_scenes_worker, export_scenes_worker


# -----------------------------
# Dataset status bar (left side of status bar)
# -----------------------------
def _sep() -> QFrame:
    s = QFrame()
    s.setFrameShape(QFrame.Shape.VLine)
    s.setFrameShadow(QFrame.Shadow.Sunken)
    s.setStyleSheet("color: #9ca3af;")
    return s


class DatasetStatusWidget(QWidget):
    """Shows: status icon + text | X video(s) | X scenes | Duration(s) | X GB."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)

        self._status_icon = QLabel()
        self._status_icon.setFixedSize(10, 10)
        self._status_icon.setStyleSheet("border-radius: 5px; background: #9ca3af;")
        self._status_label = QLabel("no video processed")
        self._status_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self._status_icon)
        layout.addWidget(self._status_label)
        layout.addWidget(_sep())

        self._video_icon = QLabel()
        self._video_icon.setPixmap(QIcon("./icons/video.png").pixmap(14, 14))
        self._video_label = QLabel("0 videos")
        self._video_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self._video_icon)
        layout.addWidget(self._video_label)
        layout.addWidget(_sep())

        self._scene_icon = QLabel()
        self._scene_icon.setPixmap(QIcon("./icons/scene.png").pixmap(14, 14))
        self._scene_label = QLabel("0 scenes")
        self._scene_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self._scene_icon)
        layout.addWidget(self._scene_label)
        layout.addWidget(_sep())

        self._duration_icon = QLabel()
        self._duration_icon.setPixmap(QIcon("./icons/time.png").pixmap(14, 14))
        self._duration_label = QLabel("— Duration(s)")
        self._duration_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self._duration_icon)
        layout.addWidget(self._duration_label)
        layout.addWidget(_sep())

        self._storage_icon = QLabel()
        self._storage_icon.setPixmap(QIcon("./icons/database.png").pixmap(14, 14))
        self._storage_label = QLabel("— GB")
        self._storage_label.setStyleSheet("font-size: 12px; color: #374151;")
        layout.addWidget(self._storage_icon)
        layout.addWidget(self._storage_label)
        layout.addWidget(_sep())

        layout.addStretch(1)

    def upd(self):
        self._status_label.setText("All processed")

    def update_stats(
        self,
        video_paths: list,
        project: Project | None,
        total_duration_sec: float | None = None,
    ):
        n_videos = len(video_paths) if video_paths else 0
        self._video_label.setText(f"{n_videos} video" + ("s" if n_videos != 1 else ""))

        if not project or not video_paths:
            self._status_icon.setStyleSheet("border-radius: 5px; background: #9ca3af;")
            self._status_label.setText("no video processed")
            self._scene_label.setText("0 scenes")
            self._duration_label.setText("— Duration(s)")
            self._storage_label.setText("— GB")
            return

        num_processed = 0
        for p in video_paths:
            try:
                if project.get_video_scenes(p) is not None:
                    num_processed += 1
            except Exception:
                pass

        if num_processed == 0:
            self._status_icon.setStyleSheet("border-radius: 5px; background: #dc2626;")
            self._status_label.setText("No processed")
        elif num_processed >= n_videos:
            self._status_icon.setStyleSheet("border-radius: 5px; background: #16a34a;")
            self._status_label.setText("Part processed")  # TODO
        else:
            self._status_icon.setStyleSheet("border-radius: 5px; background: #ca8a04;")
            self._status_label.setText("Part processed")

        scenes = project._project_data.scenes
        if scenes:
            n_scenes = len(scenes)
        else:
            n_scenes = 0
        self._scene_label.setText(f"{n_scenes} scene" + ("s" if n_scenes != 1 else ""))

        if total_duration_sec is not None:
            self._duration_label.setText(f"{total_duration_sec:.1f} Duration(s)")
        else:
            total_duration_s = sum(scene.end_time_s - scene.start_time_s for scene in scenes)
            self._duration_label.setText(f"{total_duration_s:.1f} Duration(s)")

        total_bytes = 0
        for p in video_paths:
            try:
                total_bytes += Path(p).stat().st_size
            except OSError:
                pass
        total_gb = total_bytes / (1024 ** 3)
        self._storage_label.setText(f"{total_gb:.2f} GB")


# -----------------------------
# Control Panel (Right)
# -----------------------------
class ControlPanel(QTabWidget):
    def __init__(self, model: Model, analysis_manager: AnalysisManager,
                 cache_manager: CacheManager):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.scene_tab = SceneDetectorPanel(model, SCENES)
        self.analysis_tab = AnalysisTab(model, analysis_manager, cache_manager)

        self.addTab(self.scene_tab, "Scene")
        self.addTab(self.analysis_tab, "Analysis")

        self.setMinimumWidth(280)


class MainWindow(QMainWindow):
    def __init__(self, model: Model, project: Project):
        super().__init__()

        self._analysis_manager = AnalysisManager()
        self._cache_manager = CacheManager("./demo/analysis_cache/")

        self._model = model
        self._project = project

        self._job_manager = JobManager(self)
        self._job_manager.job_updated.connect(self._on_job_updated)

        self.setWindowTitle("Video Dataset Analysis Tool")
        self.resize(1400, 850)

        central = QWidget()
        central.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 8, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.dataset = VideoBrowserWindow(self._model, self._cache_manager)
        self.media = VideoArea(self._model)
        self.controls = ControlPanel(self._model, self._analysis_manager, self._cache_manager)

        self.scene_edit_timeline = SceneEditTimeline(self._model)
        self.scene_edit_timeline.setMinimumHeight(125)
        self.scene_edit_timeline.setVisible(False)

        self.splitter.addWidget(self.dataset)
        self.splitter.addWidget(self.media)
        self.splitter.setSizes([200, 800])
        self.splitter.widget(0).setMinimumWidth(150)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        top_layout.addWidget(self.splitter, 1)
        top_layout.addSpacing(20)
        top_layout.addWidget(self.controls)

        self._timeline_container = QWidget()
        timeline_container_layout = QVBoxLayout(self._timeline_container)
        timeline_container_layout.setContentsMargins(0, 0, 0, 0)
        timeline_container_layout.addWidget(self.scene_edit_timeline)
        main_layout.addLayout(top_layout, 1)
        main_layout.addWidget(self._timeline_container, 0)

        # меню
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        save_action = file_menu.addAction("Save Project")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_project)
        
        save_as_action = file_menu.addAction("Save Project As...")
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_project_as)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._dataset_status = DatasetStatusWidget()
        self._status_bar.addWidget(self._dataset_status, 1)

        self._tasks_panel = TasksDock(self._job_manager, self)
        self._tasks_panel.hide()

        self._tasks_button = QPushButton("Tasks: 0", self)
        self._tasks_button.setFlat(True)
        self._tasks_button.clicked.connect(self._toggle_tasks_panel)
        self._status_bar.addPermanentWidget(self._tasks_button)

        # обновление статуса по количеству задач
        self._job_manager.jobs_changed.connect(self._update_tasks_button)
        self._update_tasks_button(self._job_manager.jobs())

        # связи UI
        self.controls.analysis_tab.tyres_compute_clicked.connect(self.on_tyres_compute_clicked)
        self.media.empty_scene.connect(self.controls.analysis_tab.on_empty_scene)
        self.controls.analysis_tab.report_clicked.connect(self.on_report_clicked)
        self.controls.scene_tab.on_detect_clicked(self._on_detect_scenes)
        self.controls.scene_tab.on_export_clicked(self._on_export_scenes)
        self.controls.currentChanged.connect(self._on_tab_changed)

        self._model.media_changed.connect(self._on_video_selected)
        self._model.scenes_changed.connect(self._on_scene_changed)
        self.dataset.directory_loaded.connect(self._refresh_dataset_status)

        if self._project and self._project.has_project_file:
            # self._status_bar.showMessage(f"Project: {self._project.project_path.name}", 5000)
            pass
        else:
            self._status_bar.showMessage("Working without project (changes not saved)", 5000)
        
        if self._project and self._project.dataset_path:
            self._model.root_directory = self._project.dataset_path

        if self._project.dataset_path:
            self.dataset.load_directory(self._project.dataset_path)
        self._refresh_dataset_status()

    def on_tyres_compute_clicked(self):
        self._dataset_status.upd()

    def _on_scene_changed(self, scenes: list[SceneSegment]):
        if scenes:
            self.scene_edit_timeline.setVisible(True)
        self._refresh_dataset_status()

    def _refresh_dataset_status(self):
        video_paths = getattr(self.dataset, "_video_paths", None) or []
        self._dataset_status.update_stats(video_paths, self._project, total_duration_sec=None)

    def resizeEvent(self, event):
        """При изменении размера главного окна обновляем позицию панели задач."""
        super().resizeEvent(event)
        if hasattr(self, "_tasks_panel") and self._tasks_panel.isVisible():
            self._tasks_panel._update_position()

    def moveEvent(self, event):
        """При перемещении главного окна обновляем позицию панели задач."""
        super().moveEvent(event)
        if hasattr(self, "_tasks_panel") and self._tasks_panel.isVisible():
            self._tasks_panel._update_position()

    # ------------------------
    # JobManager helpers
    # ------------------------
    @property
    def job_manager(self) -> JobManager:
        return self._job_manager

    def _toggle_tasks_panel(self):
        """Показывает/скрывает панель задач."""
        if self._tasks_panel.isVisible():
            self._tasks_panel.hide()
        else:
            self._tasks_panel.show()
            self._tasks_panel.raise_()
            self._tasks_panel.activateWindow()

    def _update_tasks_button(self, jobs):
        """Обновляет кнопку задач с индикатором непросмотренных задач."""
        total = len(jobs)
        from job_manager import JobStatus
        
        running = sum(1 for j in jobs if getattr(j, "status", None) == JobStatus.RUNNING)
        unviewed = self._tasks_panel.get_unviewed_count() if hasattr(self._tasks_panel, "get_unviewed_count") else 0
        
        if total == 0:
            text = "Tasks: 0"
            self._tasks_button.setStyleSheet("")
        else:
            text = f"Tasks: {running}/{total}"
            # если есть непросмотренные задачи (особенно с ошибками), подсвечиваем кнопку
            if unviewed > 0:
                self._tasks_button.setStyleSheet("""
                    QPushButton {
                        background: #ffcccc;
                        border: 1px solid #ff9999;
                        border-radius: 3px;
                        padding: 2px 6px;
                    }
                    QPushButton:hover {
                        background: #ffaaaa;
                    }
                """)
                text = f"Tasks: {running}/{total} ({unviewed} new)"
            else:
                self._tasks_button.setStyleSheet("""
                    QPushButton {
                        border: 1px solid #ddd;
                        border-radius: 3px;
                        padding: 2px 6px;
                    }
                    QPushButton:hover {
                        background: #f0f0f0;
                    }
                """)
        
        self._tasks_button.setText(text)

    # ------------------------
    # Scene detection / export
    # ------------------------
    def _on_detect_scenes(self):
        video = self._model.current_media
        if video is None:
            QMessageBox.warning(self, "No video", "Please select a video before detecting scenes.")
            self._status_bar.showMessage("Detect scenes: no video selected", 5000)
            return

        cfg = self.controls.scene_tab.get_config()
        algo_name = cfg["algorithm"]
        params = cfg["parameters"]

        self._job_manager.run_job(
            job_type="detect_scenes",
            description=f"Detect scenes with '{algo_name}' on '{os.path.basename(str(video.path_to_file))}'",
            func=detect_scenes_worker,
            video_path=str(video.path_to_file),
            algo=algo_name,
            params=params,
        )

    def _on_export_scenes(self):
        video = self._model.current_media
        if video is None:
            QMessageBox.warning(self, "No video", "Please select a video before exporting scenes.")
            self._status_bar.showMessage("Export scenes: no video selected", 5000)
            return

        if not self._model.scenes:
            QMessageBox.information(self, "No scenes", "There are no scenes to export. Detect scenes first.")
            self._status_bar.showMessage("Export scenes: no scenes to export", 5000)
            return

        default_dir = str(Path(video.path_to_file).parent)
        dlg = ExportScenesDialog(self, default_output=default_dir)
        if dlg.exec() != QDialog.Accepted:
            self._status_bar.showMessage("Export scenes: cancelled", 3000)
            return

        output_dir = dlg.output_dir
        prefix = dlg.prefix
        ext = dlg.extension
        overwrite = dlg.overwrite

        output_dir.mkdir(parents=True, exist_ok=True)

        desc = f"Export {len(self._model.scenes)} scenes from '{os.path.basename(str(video.path_to_file))}'"
        self._status_bar.showMessage(f"{desc} started…", 3000)

        self._job_manager.run_job(
            job_type="export_scenes",
            description=desc,
            func=export_scenes_worker,
            video_path=video.path_to_file,
            scenes=list(self._model.scenes),
            output_directory=output_dir,
            file_prefix=prefix,
            file_ext=ext,
            overwrite_files=overwrite,
        )

    def on_report_clicked(self):
        video_paths = getattr(self.dataset, "_video_paths", None) or []

        dataset_metrics = self._cache_manager.get_dataset_level_metrics(video_paths)

        dataset_row = TableRow(
            dataset=self._project.dataset_path.name,
            entropy=dataset_metrics["entropy"],
            optical_flow=dataset_metrics["optical_flow"],
            imaging_quality=None,
            motion_smoothness=dataset_metrics["motion_smoothness"],
            psnr=dataset_metrics["psnr"],
            ssim=dataset_metrics["ssim"],
            temporal_coherence_clip_score=dataset_metrics["temporary_consistency_clip_score"],
            background_dominance=dataset_metrics["background_dominance"],
            clip_ics=dataset_metrics["ics_score"],
            num_clusters=dataset_metrics["num_clusters"],
            r_R_score=dataset_metrics["r_R_score"],
            noise_ratio=dataset_metrics["noise_ratio"],
            duration_sec=dataset_metrics["duration_sec"],
            persistence=dataset_metrics["persistence"],
            norm_avg_velocity=dataset_metrics["norm_avg_velocity"],
            frames=dataset_metrics["frames_count"]
        )
        dlg = ReportWindow(dataset_row=dataset_row, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

    # ------------------------
    # Tabs / timelines
    # ------------------------
    def _on_tab_changed(self, index: int):
        # 0: Filter, 1: Analysis
        if index == 0:
            self._status_bar.showMessage("Filter tab selected", 2000)
            if self._model.scenes:
                self.scene_edit_timeline.setVisible(True)
        else:
            self._status_bar.showMessage("Analysis tab selected", 2000)
            self.scene_edit_timeline.setVisible(False)
        self._model.current_tab = "scene" if index == 0 else "analysis"

    def _on_job_updated(self, job: JobInfo):
        if isinstance(job, DoneJobInfo) and job.type == "detect_scenes":
            scenes = job.result.get("scenes", [])
            algo_name = job.result.get("algo", "")
            params = job.result.get("params", {})
            
            self._model.scenes = scenes
            
            if self._project and self._model.current_media:
                for scene in scenes:
                    self._project.add_scene(
                        video_path=self._model.current_media.path_to_file,
                        scene_info=scene,
                        algo_name=algo_name,
                        params=params
                    )
                self._project.save()
                self._refresh_dataset_status()

            msg = f"Detected {len(scenes)} scenes with '{algo_name}'"
            self._status_bar.showMessage(msg)
    
    def _on_video_selected(self, video: VideoWrapper):
        if self._project and video:
            scene_description = self._project.get_video_scenes(video.path_to_file)
            if scene_description is None:
                # Clear scenes for this video if none are stored in the project
                self._model.scenes = []
                return
            scenes, algorithm = scene_description
            # Always update model.scenes, even if list is empty
            self._model.scenes = scenes or []
            if scenes and self.controls.currentIndex() == 0:
                self.controls.scene_tab.set_detector(algorithm.name, algorithm.params)
                self._status_bar.showMessage(f"Loaded {len(scenes)} scenes for selected video", 2000)
            if self._model.current_tab == "analysis":
                self.scene_edit_timeline.setVisible(False)
    
    def _save_project(self):
        if not self._project.has_project_file:
            self._save_project_as()
            return
        
        if self._project.save():
            self._status_bar.showMessage("Project saved", 3000)
        else:
            QMessageBox.warning(self, "Error", "Failed to save project")
    
    def _save_project_as(self):
        if not self._project.dataset_path:
            QMessageBox.warning(self, "Error", "No dataset selected")
            return
        
        default_path = str(self._project.project_path) if self._project.project_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            default_path,
            "Video Project (*.vdproj)"
        )
        
        if not path:
            return
        
        if not path.endswith(".vdproj"):
            path += ".vdproj"
        
        if self._project.save_as(Path(path)):
            self._status_bar.showMessage(f"Project saved: {Path(path).name}", 3000)
        else:
            QMessageBox.warning(self, "Error", "Failed to save project")


def main():
    from qt_material import apply_stylesheet
    app = QApplication(sys.argv)
    # apply_stylesheet(app, theme='dark_teal.xml')
    dlg = StartDialog()

    if dlg.exec() != QDialog.Accepted:
        sys.exit(0)

    model = Model()
    
    project_path = dlg.selected_path()
    project = Project(project_path)

    dataset_path = dlg.selected_dataset()
    if dataset_path:
        project.dataset_path = dataset_path

    window = MainWindow(model, project)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
