# Pipeline configuration and run dialog (PySide6)

import json
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QMimeData, QPoint, QEvent
from PySide6.QtGui import QIcon, QDrag, QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QFrame,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QFormLayout,
    QCheckBox,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QProgressBar,
    QScrollArea,
    QWidget,
    QApplication,
)

from config import SCENES
from cache_manager import CacheManager
from custom_types import SceneSegment, VideoWrapper
from model import Model
from widgets import GroupWidget, ValueSlider
from workers import detect_scenes_worker
from metric_pipeline import MetricsProcessor, get_cluster_metrics
from scene_detector_block import fix_label_width


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv")
FOLDER_ICON = QIcon.fromTheme("folder")
VIDEO_ICON = QIcon.fromTheme("video")


# MIME type for video path drag (so list can accept only our drags)
VIDEO_PATH_MIME = "application/x-video-path"


def _is_video_path(s: str) -> bool:
    if not s or not s.strip():
        return False
    return Path(s).suffix.lower() in (".mp4", ".avi", ".mkv")


class PipelineVideoListWidget(QListWidget):
    """List of videos to process; accepts drag-and-drop of video paths (MimeData text or VIDEO_PATH_MIME)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._add_path_callback = lambda p: None

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasText() and _is_video_path(mime.text()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        mime = event.mimeData()
        if mime.hasText() and _is_video_path(mime.text()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasText():
            path_str = mime.text().strip()
            if path_str and _is_video_path(path_str):
                self._add_path_callback(Path(path_str))
        event.acceptProposedAction()

    def set_add_path_callback(self, callback):
        self._add_path_callback = callback


class PipelineTreeWidget(QTreeWidget):
    """Directory tree; dragging a video item passes its path in MimeData."""

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if item:
            path = item.data(0, Qt.UserRole)
            if path and _is_video_path(path):
                mime = QMimeData()
                mime.setText(path)
                mime.setData(VIDEO_PATH_MIME, path.encode("utf-8"))
                drag = QDrag(self)
                drag.setMimeData(mime)
                drag.exec(Qt.CopyAction)
                return
        super().startDrag(supportedActions)

def _deepest_child_at(widget: QWidget, pos: QPoint) -> QWidget | None:
    """Return the deepest child at the given position (in widget coordinates)."""
    if not widget:
        return None
    child = widget.childAt(pos)
    if not child:
        return widget
    return _deepest_child_at(child, child.mapFrom(widget, pos)) or child


class PipelineScrollArea(QScrollArea):
    """Scroll area that does not scroll when the mouse is over a child that has its own scroll (tree, list, table)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.viewport() and event.type() == QEvent.Type.Wheel:
            w = self.widget()
            if w:
                pos_in_content = w.mapFrom(self.viewport(), event.position().toPoint())
                target = _deepest_child_at(w, pos_in_content)
                while target and target != w:
                    if isinstance(target, (QTreeWidget, QListWidget, QTableWidget)):
                        QApplication.sendEvent(target, event)
                        return True  # consume so viewport does not scroll
                    target = target.parentWidget()
        return super().eventFilter(obj, event)


BTN_STYLE = """
    QPushButton {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #f3f4f6;
        padding: 4px 10px;
    }
    QPushButton:hover {
        background: #e5e7eb;
    }
    QPushButton:disabled {
        background: #f9fafb;
        color: #9ca3af;
    }
"""


def _build_tree_from_dir(
    parent_item: QTreeWidgetItem,
    folder_path: Path,
    collected_videos: list[Path] | None = None,
) -> None:
    folder_path = Path(folder_path)
    try:
        entries = list(folder_path.iterdir())
    except PermissionError:
        return
    folders = sorted(
        [e for e in entries if (folder_path / e).is_dir()],
        key=lambda x: str(x).lower(),
    )
    files = sorted(
        [
            e
            for e in entries
            if (folder_path / e).is_file()
            and str(e).lower().endswith(VIDEO_EXTENSIONS)
        ],
        key=lambda x: str(x).lower(),
    )
    for f in folders:
        item = QTreeWidgetItem(parent_item, [f.name])
        item.setIcon(0, FOLDER_ICON)
        item.setData(0, Qt.UserRole, None)
        _build_tree_from_dir(item, folder_path / f, collected_videos)
    for f in files:
        item = QTreeWidgetItem(parent_item, [f.name])
        item.setIcon(0, VIDEO_ICON)
        full_path = folder_path / f
        item.setData(0, Qt.UserRole, str(full_path))
        if collected_videos is not None:
            collected_videos.append(full_path)


class PipelineRunnerThread(QThread):
    """Thread that runs the pipeline: scene detection and metrics per video."""

    progress = Signal(int, int, str, str)  # current_index, total, video_name, step
    finished_ok = Signal()
    failed = Signal(str)

    def __init__(
        self,
        video_paths: list[Path],
        scene_config: dict,
        save_scenes_path: Path | None,
        root_for_structure: Path | None,
        object_classes: list[str],
        group_classes: dict[str, list[str]],
        cache_manager: CacheManager,
        parent=None,
    ):
        super().__init__(parent)
        self._video_paths = list(video_paths)
        self._scene_config = scene_config
        self._save_scenes_path = save_scenes_path
        self._root_for_structure = root_for_structure
        self._object_classes = object_classes
        self._group_classes = group_classes
        self._cache_manager = cache_manager
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            total = len(self._video_paths)
            processor = MetricsProcessor(self._cache_manager, self._object_classes, self._group_classes)

            for idx, video_path in enumerate(self._video_paths):
                if self._cancelled:
                    break
                video_name = video_path.name
                self.progress.emit(idx, total, video_name, "scene detection")

                # Scene detection
                from workers import detect_scenes_worker
                result = detect_scenes_worker(
                    str(video_path),
                    self._scene_config["algorithm"],
                    self._scene_config["parameters"],
                )
                scenes = result["scenes"]

                if not scenes:
                    try:
                        vw = VideoWrapper(video_path)
                        if vw.num_frames > 0:
                            scenes = [
                                SceneSegment(
                                    start_frame=0,
                                    end_frame=vw.num_frames,
                                    start_time_s=0.0,
                                    end_time_s=vw.num_frames / vw.fps,
                                    label="full",
                                )
                            ]
                    except Exception:
                        scenes = [
                            SceneSegment(
                                start_frame=0,
                                end_frame=1,
                                start_time_s=0.0,
                                end_time_s=0.04,
                                label="full",
                            )
                        ]

                # Save scenes to disk (optional)
                if self._save_scenes_path and self._root_for_structure and scenes:
                    self.progress.emit(idx, total, video_name, "saving scenes to disk")
                    try:
                        rel = video_path.relative_to(self._root_for_structure)
                        out_dir = self._save_scenes_path / rel.parent / (video_path.stem)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        from workers import export_scenes_worker
                        export_scenes_worker(
                            video_path,
                            scenes,
                            out_dir,
                            "scene",
                            ".mp4",
                            overwrite_files=True,
                        )
                    except Exception:
                        pass

                # Metrics per scene (quality, motion, objects)
                for s_idx, scene in enumerate(scenes):
                    if self._cancelled:
                        break
                    self.progress.emit(
                        idx, total, video_name,
                        f"computing metrics (scene {s_idx + 1}/{len(scenes)})",
                    )

                    processor.compute_scene_sync(str(video_path), scene)

                features = np.array([
                    self._cache_manager.get_features(video_path, scene)
                    for scene in scenes
                ])

                cluster_metrics = get_cluster_metrics(features, processor.metrics_computer)

                try:
                    self._cache_manager._unite_scenes_metrics(video_path, scenes, cluster_metrics)
                except Exception as e:
                    print(e)
                    raise

                self.progress.emit(idx + 1, total, video_name, "done")

            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


# --------------- Scene detection block ---------------
class SceneDetectorBlock(QFrame):
    def __init__(self, detectors_config: dict, parent=None):
        super().__init__(parent)
        self.detectors_config = detectors_config
        self.param_widgets = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("Algorithm:")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        self.combo = QComboBox()
        self.combo.addItems(list(detectors_config.keys()))
        self.combo.currentTextChanged.connect(self._on_detector_changed)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.combo)

        self.params_layout = QFormLayout()
        self.params_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        layout.addLayout(header)
        layout.addLayout(self.params_layout)
        self._on_detector_changed(self.combo.currentText())

    def _on_detector_changed(self, name: str):
        while self.params_layout.rowCount():
            self.params_layout.removeRow(0)
        self.param_widgets.clear()

        cfg = self.detectors_config.get(name)
        if not cfg:
            return
        for pname, pdata in cfg["parameters"].items():
            widget_t = pdata["widget"]
            if widget_t == ValueSlider:
                w = widget_t(
                    pname,
                    pdata["min"],
                    pdata["max"],
                    pdata["default"],
                )
            else:
                w = widget_t()
            w.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            if hasattr(w, "setMinimum"):
                w.setMinimum(pdata.get("min", 0))
            if hasattr(w, "setMaximum"):
                w.setMaximum(pdata.get("max", 100))
            if isinstance(w, QCheckBox):
                w.setChecked(pdata["default"])
            elif widget_t != ValueSlider:
                w.setValue(pdata["default"])
            w.setToolTip(pdata.get("help", ""))
            if widget_t == ValueSlider:
                self.params_layout.addRow(w)
            else:
                self.params_layout.addRow(QLabel(pname), w)
            self.param_widgets[pname] = w
        fix_label_width(self.params_layout)

    def get_config(self) -> dict:
        name = self.combo.currentText()
        params = {}
        for pname, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                params[pname] = widget.isChecked()
            else:
                params[pname] = getattr(widget, "get_value", lambda: None)()
        return {"algorithm": name, "parameters": params}


# --------------- Classes and groups table ---------------
class ClassesTableWidget(QWidget):
    def __init__(self, initial_classes: dict[str, list[str]] | None = None, parent=None):
        super().__init__(parent)
        self._initial_classes = initial_classes
        self._groups = ["default"]
        if isinstance(initial_classes, dict):
            self._groups = list(initial_classes.keys())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Class", "Group"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add row")
        add_btn.setStyleSheet(BTN_STYLE)
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self._add_row)
        remove_btn = QPushButton("Remove row")
        remove_btn.setStyleSheet(BTN_STYLE)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self._remove_row)
        import_btn = QPushButton("Import from file")
        import_btn.setStyleSheet(BTN_STYLE)
        import_btn.setCursor(Qt.PointingHandCursor)
        import_btn.clicked.connect(self._import_from_file)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(import_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        for group, classes in self._initial_classes.items():
            for cls_name in classes:
                self._append_row(cls_name, group)

    def _append_row(self, class_name: str, group: str):
        if group not in self._groups:
            self._groups.append(group)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(class_name))
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self._groups)
        combo.setCurrentText(group)
        combo.currentTextChanged.connect(self._on_group_edited)
        self.table.setCellWidget(row, 1, combo)

    def _on_group_edited(self, text: str):
        if text and text not in self._groups:
            self._groups.append(text)

    def _add_row(self):
        self._append_row("", "default")

    def _remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _ensure_group_exists(self, group: str):
        if group and group not in self._groups:
            self._groups.append(group)

    def _import_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import classes",
            "",
            "JSON (*.json);;All (*)",
        )
        if not path:
            return
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        self._append_row(item, "default")
                    elif isinstance(item, dict):
                        self._append_row(
                            item.get("class", item.get("name", "")),
                            item.get("group", "default"),
                        )
            elif isinstance(data, dict) and "classes" in data:
                for c in data["classes"]:
                    if isinstance(c, str):
                        self._append_row(c, "default")
                    else:
                        self._append_row(
                            c.get("class", c.get("name", "")),
                            c.get("group", "default"),
                        )
            elif isinstance(data, dict):
                for group, classes in data.items():
                    for cls_name in classes:
                        self._append_row(cls_name, group)
            else:
                # plain list of strings from JSON array like ["a","b"]
                matched = re.findall(r'"([^"]+)"', raw)
                for s in matched:
                    self._append_row(s, "default")
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to load file: {e}",
            )

    def get_classes(self) -> list[str]:
        classes = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            name = item.text().strip() if item else ""
            if name:
                classes.append(name)
        return classes

    def get_classes_with_groups(self) -> list[tuple[str, str]]:
        result = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            name = (item.text().strip() if item else "") or ""
            w = self.table.cellWidget(row, 1)
            group = w.currentText().strip() if isinstance(w, QComboBox) else "default"
            if name:
                result.append((name, group))
        return result


# --------------- Pipeline dialog ---------------
class PipelineDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        model: Model,
        cache_manager: CacheManager,
        root_directory: Path | None,
        initial_video_paths: list[Path] | None,
    ):
        super().__init__(parent)
        self._model = model
        self._cache_manager = cache_manager
        self._root_directory = Path(root_directory) if root_directory else None
        self._initial_video_paths = initial_video_paths or []
        self._runner: PipelineRunnerThread | None = None

        self.setWindowTitle("Dataset processing pipeline")
        self.setMinimumSize(820, 620)
        self.resize(900, 700)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Dataset processing pipeline")
        title.setStyleSheet("font-weight: 700; font-size: 18px;")
        main_layout.addWidget(title)

        scroll = PipelineScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        # ---- 1) Video selection ----
        content_layout.addWidget(self._build_videos_group())

        # ---- 2) Scene detection + save to disk ----
        scene_layout = QVBoxLayout()
        self._scene_block = SceneDetectorBlock(SCENES)
        scene_layout.addWidget(self._scene_block)

        self._save_scenes_cb = QCheckBox("Save scenes to disk")
        self._save_scenes_cb.toggled.connect(self._on_save_scenes_toggled)
        scene_layout.addWidget(self._save_scenes_cb)
        self._save_scenes_path_edit = QLineEdit()
        self._save_scenes_path_edit.setPlaceholderText("Path to save scenes (directory structure is mirrored)")
        self._save_scenes_path_edit.setEnabled(False)
        path_btn = QPushButton("Browse...")
        path_btn.setStyleSheet(BTN_STYLE)
        path_btn.setCursor(Qt.PointingHandCursor)
        path_btn.clicked.connect(self._browse_save_scenes_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self._save_scenes_path_edit, 1)
        path_row.addWidget(path_btn)
        scene_layout.addLayout(path_row)
        scene_group = GroupWidget("Scene detection", scene_layout, collapsible=True, default_expanded=False)
        content_layout.addWidget(scene_group)

        # ---- 3) Metrics / classes ----
        self._classes_widget = ClassesTableWidget(initial_classes=model.current_classes)
        classes_outer = QVBoxLayout()
        classes_outer.addWidget(self._classes_widget)
        metrics_group = GroupWidget("Metrics (classes and groups)", classes_outer, collapsible=True, default_expanded=False)
        content_layout.addWidget(metrics_group)

        # ---- Progress (hidden until run) ----
        self._progress_widget = QFrame()
        progress_layout = QVBoxLayout(self._progress_widget)
        self._progress_label = QLabel("")
        self._progress_label.setWordWrap(True)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_label)
        progress_layout.addWidget(self._progress_bar)
        self._progress_widget.hide()
        content_layout.addWidget(self._progress_widget)

        content_layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)

        # ---- Bottom buttons ----
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._run_btn = QPushButton("Run pipeline")
        self._run_btn.setStyleSheet(BTN_STYLE)
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.clicked.connect(self._start_pipeline)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet(BTN_STYLE)
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self._run_btn)
        btn_layout.addWidget(self._cancel_btn)
        main_layout.addLayout(btn_layout)

        self._fill_tree()
        for p in self._initial_video_paths:
            self._add_video_path(Path(p))

    def _build_videos_group(self) -> QWidget:
        self._tree_videos = PipelineTreeWidget()
        self._tree_videos.setHeaderLabel("Directory")
        self._tree_videos.setColumnCount(1)
        self._tree_videos.setIndentation(20)
        self._tree_videos.setDragEnabled(True)
        self._tree_videos.setStyleSheet("QTreeWidget { border: none; background: transparent; }")

        self._list_to_process = PipelineVideoListWidget()
        self._list_to_process.set_add_path_callback(self._add_video_path)
        self._list_to_process.setStyleSheet("QListWidget { border: none; background: transparent; }")

        tree_frame = QFrame()
        tree_frame.setStyleSheet(
            "QFrame { border-radius: 8px; border: 1px solid #dcdde1; background: #ffffff; }"
        )
        tree_inner = QVBoxLayout(tree_frame)
        tree_inner.setContentsMargins(6, 6, 6, 6)
        tree_lbl = QLabel("Directory (drag videos to the list on the right)")
        tree_lbl.setStyleSheet("border: none; background: transparent;")
        tree_inner.addWidget(tree_lbl)
        tree_inner.addWidget(self._tree_videos)
        list_frame = QFrame()
        list_frame.setStyleSheet(
            "QFrame { border-radius: 8px; border: 1px solid #dcdde1; background: #ffffff; }"
        )
        list_inner = QVBoxLayout(list_frame)
        list_inner.setContentsMargins(6, 6, 6, 6)
        list_lbl = QLabel("Videos to process")
        list_lbl.setStyleSheet("border: none; background: transparent;")
        list_inner.addWidget(list_lbl)
        list_inner.addWidget(self._list_to_process)

        videos_btns = QVBoxLayout()
        add_all_btn = QPushButton("Add all videos")
        add_all_btn.setStyleSheet(BTN_STYLE)
        add_all_btn.setCursor(Qt.PointingHandCursor)
        add_all_btn.clicked.connect(self._add_all_videos)
        clear_btn = QPushButton("Clear list")
        clear_btn.setStyleSheet(BTN_STYLE)
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self._list_to_process.clear)
        remove_btn = QPushButton("Remove item")
        remove_btn.setStyleSheet(BTN_STYLE)
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self._remove_selected_from_list)
        videos_btns.addWidget(add_all_btn)
        videos_btns.addWidget(clear_btn)
        videos_btns.addWidget(remove_btn)
        videos_btns.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(tree_frame)
        splitter.addWidget(list_frame)
        splitter.setSizes([400, 400])
        row = QHBoxLayout()
        row.addWidget(splitter, 1)
        row.addLayout(videos_btns, 0)
        return GroupWidget("Video selection", row, collapsible=True, default_expanded=True)

    def _fill_tree(self):
        self._tree_videos.clear()
        self._all_video_paths = []
        if not self._root_directory or not self._root_directory.exists():
            return
        root_item = QTreeWidgetItem(self._tree_videos, [self._root_directory.name])
        root_item.setIcon(0, FOLDER_ICON)
        root_item.setData(0, Qt.UserRole, None)
        _build_tree_from_dir(
            root_item,
            self._root_directory,
            self._all_video_paths,
        )
        root_item.setExpanded(True)
        self._tree_videos.expandToDepth(1)

    def _add_video_path(self, path: Path):
        path = Path(path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        for i in range(self._list_to_process.count()):
            item = self._list_to_process.item(i)
            if item and item.data(Qt.UserRole) == str(path):
                return
        item = QListWidgetItem(path.name)
        item.setData(Qt.UserRole, str(path))
        self._list_to_process.addItem(item)

    def _add_all_videos(self):
        for p in self._all_video_paths:
            self._add_video_path(p)

    def _remove_selected_from_list(self):
        row = self._list_to_process.currentRow()
        if row >= 0:
            self._list_to_process.takeItem(row)

    def _on_save_scenes_toggled(self, checked: bool):
        self._save_scenes_path_edit.setEnabled(checked)
        if not checked:
            self._save_scenes_path_edit.clear()

    def _browse_save_scenes_path(self):
        d = QFileDialog.getExistingDirectory(self, "Folder to save scenes")
        if d:
            self._save_scenes_path_edit.setText(d)

    def _get_video_paths_to_process(self) -> list[Path]:
        paths = []
        for i in range(self._list_to_process.count()):
            item = self._list_to_process.item(i)
            if item and item.data(Qt.UserRole):
                paths.append(Path(item.data(Qt.UserRole)))
        return paths

    def _start_pipeline(self):
        paths = self._get_video_paths_to_process()
        if not paths:
            QMessageBox.warning(
                self,
                "No videos",
                "Add at least one video to process.",
            )
            return
        classes = self._classes_widget.get_classes()
        if not classes:
            QMessageBox.warning(
                self,
                "No classes",
                "Add at least one class in the metrics table.",
            )
            return

        save_path = None
        if self._save_scenes_cb.isChecked():
            t = self._save_scenes_path_edit.text().strip()
            if t:
                save_path = Path(t)
        root_for_structure = self._root_directory if save_path else None

        self._progress_widget.show()
        self._progress_bar.setValue(0)
        self._progress_label.setText("")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setText("Close")

        self._runner = PipelineRunnerThread(
            video_paths=paths,
            scene_config=self._scene_block.get_config(),
            save_scenes_path=save_path,
            root_for_structure=root_for_structure,
            object_classes=classes,
            cache_manager=self._cache_manager,
            parent=self,
            group_classes=self._model.current_classes
        )
        self._runner.progress.connect(self._on_progress)
        self._runner.finished_ok.connect(self._on_pipeline_finished)
        self._runner.failed.connect(self._on_pipeline_failed)
        self._runner.start()

    def _on_progress(self, current: int, total: int, video_name: str, step: str):
        if total <= 0:
            pct = 0
        else:
            pct = int(100 * current / total)
        self._progress_bar.setValue(pct)
        self._progress_label.setText(f"{video_name}: {step}")

    def _on_pipeline_finished(self):
        self._progress_bar.setValue(100)
        self._progress_label.setText("Pipeline finished.")
        self._run_btn.setEnabled(True)
        self._cancel_btn.setText("Close")
        QMessageBox.information(self, "Done", "Pipeline completed successfully.")

    def _on_pipeline_failed(self, msg: str):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setText("Close")
        QMessageBox.critical(self, "Error", f"Pipeline failed:\n{msg}")

    def _on_cancel_clicked(self):
        if self._runner is not None and self._runner.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit pipeline?",
                "Are you sure you want to exit? Progress will not be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._runner.cancel()
                self._runner.terminate()
                if not self._runner.wait(3000):
                    self._runner.terminate()
                self._runner = None
                self.reject()
        else:
            self.reject()

    def closeEvent(self, event):
        if self._runner is not None and self._runner.isRunning():
            reply = QMessageBox.question(
                self,
                "Exit pipeline?",
                "Are you sure you want to exit? Progress will not be saved.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._runner.cancel()
                self._runner.terminate()
                if not self._runner.wait(3000):
                    self._runner.terminate()
                self._runner = None
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
