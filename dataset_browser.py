import sys
import os
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QTreeWidget,
    QTreeWidgetItem,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QFrame,
    QPushButton,
    QHBoxLayout,
    QLabel,
)
from PySide6.QtGui import QIcon, QPainter, QColor, QFont, QAction, QFontMetrics
from PySide6.QtCore import Qt, Signal, Slot, QSize

from model import Model

try:
    from pipeline_window import PipelineDialog
except ImportError:
    PipelineDialog = None
from cache_manager import CacheManager


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv')  # только видео
FOLDER_ICON = QIcon.fromTheme("folder")       # можно заменить на кастомный
VIDEO_ICON = QIcon.fromTheme("video")  # тоже можно кастомизировать


class ClickableOverlay(QFrame):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Рисуем полупрозрачный фон (черный с 150/255 прозрачностью)
        painter.fillRect(self.rect(), QColor(30, 30, 30, 200))

        # 2. Рисуем текст
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 16, QFont.Bold))

        # Отрисовка текста по центру
        text = "Click to open dataset"
        painter.drawText(self.rect(), Qt.AlignCenter, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


def build_tree(
    parent_item: QTreeWidgetItem,
    folder_path: str | Path,
    collected_videos: list[Path] | None = None,
):
    folder_path = Path(folder_path)
    try:
        entries = list(folder_path.iterdir())
    except PermissionError:
        return

    # сортируем: сначала папки, потом файлы
    folders = sorted([e for e in entries if (folder_path / e).is_dir()],
                     key=lambda x: str(x).lower())
    files = sorted([e for e in entries if (folder_path / e).is_file()
                    and str(e).lower().endswith(VIDEO_EXTENSIONS)],
                   key=lambda x: str(x).lower())

    # добавляем папки
    for f in folders:
        item = QTreeWidgetItem(parent_item, [f.name])
        item.setIcon(0, FOLDER_ICON)
        build_tree(item, folder_path / f, collected_videos)

    # добавляем видео файлы
    for f in files:
        item = QTreeWidgetItem(parent_item, [f.name])
        item.setIcon(0, VIDEO_ICON)
        if collected_videos is not None:
            collected_videos.append(folder_path / f)


class DatasetListItem(QWidget):
    """
    Виджет-элемент для list-представления видео:
    превью слева (заглушка), справа название и мета‑информация.
    """

    def __init__(self, path: Path, parent: QWidget | None = None):
        super().__init__(parent)

        self._path = path
        self._selected = False
        self._hover = False
        self._full_name = path.name

        self.thumb = QFrame()
        self.thumb.setFixedSize(72, 48)
        self.thumb.setStyleSheet(
            """
            QFrame {
                background: #111827;
                border-radius: 6px;
            }
            """
        )

        self.name_lbl = QLabel(path.name)
        self.name_lbl.setStyleSheet("font-weight: 600;")
        self.name_lbl.setWordWrap(False)

        # простая демо-информация; реальную длительность/сцены можно подставить позже
        self.meta_lbl = QLabel("⏱ 0:00   🎬 0 scenes")
        self.meta_lbl.setStyleSheet("color: #6b7280; font-size: 11px;")

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        text_layout.addWidget(self.name_lbl)
        text_layout.addWidget(self.meta_lbl)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)
        row.addWidget(self.thumb)
        row.addLayout(text_layout, 1)

        self._apply_style()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        base = super().sizeHint()
        if base.height() < 68:
            base.setHeight(68)
        return base

    def _apply_style(self):
        if self._selected:
            style = """
            QWidget {
                background: #eef2ff;
                border-radius: 10px;
                border: 2px solid #4f46e5;
            }
            """
        elif self._hover:
            style = """
            QWidget {
                background: #f9fafb;
                border-radius: 10px;
                border: 1px solid #cbd5f5;
            }
            """
        else:
            style = """
            QWidget {
                background: #ffffff;
                border-radius: 10px;
                border: 1px solid #e5e7eb;
            }
            """
        self.setStyleSheet(style)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def enterEvent(self, event):
        """Подсветка целого элемента при hover."""
        self._hover = True
        self._apply_style()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self._apply_style()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        """Эллипсис (...) для длинных имён, которые не помещаются."""
        super().resizeEvent(event)
        fm: QFontMetrics = self.name_lbl.fontMetrics()
        max_width = self.name_lbl.width()
        elided = fm.elidedText(self._full_name, Qt.ElideRight, max_width)
        self.name_lbl.setText(elided)


# ------------------------
# Main Window
# ------------------------
class VideoBrowserWindow(QWidget):
    MIN_WIDTH: int = 400
    directory_loaded = Signal()

    def __init__(self, model: Model, cache_manager: CacheManager | None = None):
        super().__init__()
        self._model = model
        self._cache_manager = cache_manager
        self._video_paths: list[Path] = []
        self._current_dir: Path | None = None

        self.setWindowTitle("Video Browser")
        self.resize(600, 800)

        layout = QVBoxLayout()
        layout.setSpacing(8)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(8, 8, 8, 0)
        header_layout.setSpacing(6)

        self.title_label = QLabel("Dataset")
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-weight: bold; 
                font-size: 18px;
                color: #2c3e50;
                margin-left: 2px;
            }
            """
        )

        self.count_label = QLabel("0 videos")
        self.count_label.setStyleSheet(
            """
            QLabel {
                background: #e5e7eb;
                border-radius: 10px;
                padding: 2px 8px;
                color: #374151;
                font-size: 11px;
            }
            """
        )

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.count_label)

        # строка поиска + Reload
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(8, 0, 8, 0)
        search_layout.setSpacing(6)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search videos...")
        self.search_box.textChanged.connect(self.on_search)
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.search_box.setFixedHeight(32)
        self.search_box.setStyleSheet(
            """
            QLineEdit {
                border-radius: 10px;
                padding: 4px 8px;
            }
            """
        )

        # иконка лупы внутри поля
        search_action = QAction(self)
        search_action.setIcon(QIcon.fromTheme("edit-find"))
        self.search_box.addAction(search_action, QLineEdit.LeadingPosition)

        self.reload_btn = QPushButton()
        self.reload_btn.setToolTip("Reload from disk")
        self.reload_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.reload_btn.setEnabled(False)
        self.reload_btn.clicked.connect(self._reload_directory)
        self.reload_btn.setFixedHeight(32)
        self.reload_btn.setCursor(Qt.PointingHandCursor)
        self.reload_btn.setStyleSheet(
            """
            QPushButton {
                border-radius: 10px;
                padding: 4px 10px;
                border: 1px solid #e5e7eb;
                background: #f9fafb;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
            """
        )
        self.pipeline_btn = QPushButton()
        self.pipeline_btn.setToolTip("Run pipeline")
        self.pipeline_btn.setIcon(QIcon("./icons/play.png"))
        self.pipeline_btn.clicked.connect(self._on_pipeline_btn_clicked)
        self.pipeline_btn.setFixedHeight(32)
        self.pipeline_btn.setCursor(Qt.PointingHandCursor)
        self.pipeline_btn.setStyleSheet(
            """
            QPushButton {
                border-radius: 10px;
                padding: 4px 10px;
                border: 1px solid #e5e7eb;
                background: #f9fafb;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
            """
        )

        search_layout.addWidget(self.search_box, 1)
        search_layout.addWidget(self.reload_btn)
        search_layout.addWidget(self.pipeline_btn)

        layout.addLayout(header_layout)
        layout.addLayout(search_layout)

        self.tree = QTreeWidget()
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.setHeaderLabel("Video Datasets")
        self.tree.setColumnCount(1)
        self.tree.setSortingEnabled(True)
        self.tree.setIndentation(24)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setStyleSheet(
            """
            QTreeWidget {
                border: none;
                background: transparent;
            }
            QTreeView::item {
                padding: 2px 4px;
            }
            """
        )

        content_frame = QFrame()
        content_frame.setStyleSheet(
            """
            QFrame {
                border-radius: 12px;
                border: 1px solid #e5e7eb;
                background: #ffffff;
            }
            """
        )
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.addWidget(self.tree)

        layout.addWidget(content_frame, 1)

        self.overlay = ClickableOverlay(self.tree)
        self.overlay.clicked.connect(self._on_select_directory)

        self.update_overlay_visibility()

        self.setMinimumWidth(self.MIN_WIDTH)
        self.setLayout(layout)

    def _on_pipeline_btn_clicked(self):
        if PipelineDialog is None:
            return
        if not self._model.root_directory or not self._model.root_directory.exists():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No directory",
                "Please select a dataset directory first.",
            )
            return
        cache = self._cache_manager
        if cache is None:
            from pathlib import Path
            cache = CacheManager(Path(".analysis_cache"))
        dlg = PipelineDialog(
            parent=self,
            model=self._model,
            cache_manager=cache,
            root_directory=self._model.root_directory,
            initial_video_paths=self._video_paths,
        )
        dlg.exec()

    def on_item_clicked(self, item: QTreeWidgetItem):
        if not item.text(0).endswith(".mp4"):
            return
        # Получаем полный путь к файлу
        path_parts = []
        current = item
        while current is not None:
            path_parts.insert(0, current.text(0))
            current = current.parent()
        full_path = self._model.root_directory.parent / os.path.join(*path_parts)
        self._model.current_media = full_path

    def update_overlay_visibility(self):
        """Показывает оверлей, если в дереве нет элементов"""
        is_empty = self.tree.topLevelItemCount() == 0
        self.overlay.setVisible(is_empty)
        # Если дерево не пустое, оверлей скрывается и не мешает кликать по файлам

    def resizeEvent(self, event):
        """Следим, чтобы оверлей всегда был по размеру дерева"""
        super().resizeEvent(event)
        self.overlay.resize(self.tree.size())

    # ------------------------
    # Search
    # ------------------------
    def on_search(self, text):
        text = text.lower()

        for i in range(self.tree.topLevelItemCount()):
            self._filter_item(self.tree.topLevelItem(i), text)

    def _filter_item(self, item, text):
        """
        Рекурсивно скрываем/показываем элементы
        """
        # Проверяем потомков
        child_match = False
        for i in range(item.childCount()):
            child = item.child(i)
            if self._filter_item(child, text):
                child_match = True

        # Проверяем текущий элемент
        self_match = text in item.text(0).lower()

        # Показываем, если совпадает сам или любой потомок
        item.setHidden(not (self_match or child_match))

        return self_match or child_match

    def select_directory(self) -> Path:
        # folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        folder = "/Users/dvuglaf/Desktop"
        return Path(folder)

    def load_directory(self, path_to_directory: str | Path):
        path_to_directory = Path(path_to_directory)
        self._current_dir = path_to_directory
        self.tree.clear()
        root_item = QTreeWidgetItem(self.tree, [path_to_directory.name])
        root_item.setIcon(0, FOLDER_ICON)
        self._video_paths = []
        build_tree(root_item, path_to_directory, self._video_paths)
        root_item.setExpanded(True)
        self.tree.expandToDepth(1)

        # Скрываем оверлей после загрузки
        self.update_overlay_visibility()
        self._model.root_directory = path_to_directory.resolve()

        self._update_count_label()
        self.reload_btn.setEnabled(True)
        self.directory_loaded.emit()

    @Slot()
    def _on_select_directory(self):
        directory = self.select_directory()
        self.load_directory(directory)

    def _update_count_label(self):
        n = len(self._video_paths)
        self.count_label.setText(f"{n} video" + ("s" if n != 1 else ""))

    def _reload_directory(self):
        if self._current_dir is None:
            return
        self.load_directory(self._current_dir)


# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = VideoBrowserWindow()
    win.show()
    sys.exit(app.exec())