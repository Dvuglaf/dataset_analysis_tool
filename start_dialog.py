from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import QSettings, Qt


class StartDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Start Project")
        self.setFixedSize(420, 400)

        self._selected_path = None
        self._selected_dataset = None  # для режима без проекта
        self._mode = None  # "project" или "no_project"

        main = QVBoxLayout(self)
        main.setSpacing(12)

        # Заголовок
        title = QLabel("Video Dataset Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        main.addWidget(title)

        # Recent
        recent_label = QLabel("Recent projects:")
        main.addWidget(recent_label)

        self.recent_list = QListWidget()
        main.addWidget(self.recent_list, 1)

        # Кнопки
        btns = QHBoxLayout()

        self.new_btn = QPushButton("New Project")
        self.open_btn = QPushButton("Open Project")
        self.work_without_btn = QPushButton("Work Without Project")
        self.quit_btn = QPushButton("Quit")

        btns.addWidget(self.new_btn)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.work_without_btn)
        btns.addStretch()
        btns.addWidget(self.quit_btn)

        main.addLayout(btns)

        # Connections
        self.new_btn.clicked.connect(self._new_project)
        self.open_btn.clicked.connect(self._open_project)
        self.work_without_btn.clicked.connect(self._work_without_project)
        self.quit_btn.clicked.connect(self.reject)

        self.recent_list.itemDoubleClicked.connect(
            self._open_recent
        )

        self._load_recent()

    def _load_recent(self):
        settings = QSettings("MyCompany", "VideoDatasetTool")
        paths = settings.value("recent_projects", [], list)

        self.recent_list.clear()

        for p in paths:
            self.recent_list.addItem(p)

    @staticmethod
    def add_recent(path: str):
        settings = QSettings("MyCompany", "VideoDatasetTool")
        paths = settings.value("recent_projects", [], list)

        if path in paths:
            paths.remove(path)

        paths.insert(0, path)

        paths = paths[:5]  # максимум 5
        settings.setValue("recent_projects", paths)

    def _new_project(self):
        # 1. Выбор датасета
        dataset = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder"
        )

        if not dataset:
            return

        # 2. Куда сохранить проект
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Create Project",
            "",
            "Video Project (*.vdproj)"
        )

        if not path:
            return

        if not path.endswith(".vdproj"):
            path += ".vdproj"

        # Создаём пустой проект
        import json
        from datetime import datetime

        data = {
            "version": 1,
            "dataset_path": dataset,
            "created": datetime.now().isoformat(),
            "scenes": [],
            "settings": {}
        }

        try:
            with open(path, "w", encoding="utf8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )
            return

        self._selected_path = path
        self._mode = "project"

        self.add_recent(path)

        self.accept()

    def _open_project(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "Video Project (*.vdproj)"
        )

        if not path:
            return

        self._selected_path = path
        self._mode = "project"

        self.add_recent(path)

        self.accept()

    def _open_recent(self, item):

        path = item.text()

        import os

        if not os.path.exists(path):

            QMessageBox.warning(
                self,
                "Missing",
                "Project file not found"
            )

            return

        self._selected_path = path
        self._mode = "project"

        self.add_recent(path)

        self.accept()

    def _work_without_project(self):
        """Режим работы без проекта - только выбор датасета."""
        dataset = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Folder"
        )
        
        if not dataset:
            return
        
        self._selected_dataset = dataset
        self._mode = "no_project"
        self.accept()
    
    def selected_path(self):
        """Возвращает путь к проекту (если режим с проектом)."""
        return self._selected_path if self._mode == "project" else None
    
    def selected_dataset(self):
        """Возвращает путь к датасету (для режима без проекта)."""
        return self._selected_dataset
    
    def mode(self):
        """Возвращает режим: 'project' или 'no_project'."""
        return self._mode