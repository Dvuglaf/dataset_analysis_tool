from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent, QImage, QPixmap
from PySide6.QtWidgets import QWidget

from custom_types import SceneSegment, VideoWrapper
from model import Model


@dataclass
class SceneView(SceneSegment):
    @classmethod
    def from_segment(cls, segment: SceneSegment) -> "SceneView":
        return cls(
            start_frame=segment.start_frame,
            start_time_s=segment.start_time_s,
            end_frame=segment.end_frame,
            end_time_s=segment.end_time_s,
            label=segment.label,
        )


class SimpleSceneTimeline(QWidget):
    """
    Таймлайн сцен:
    - горизонтальная полоса, вся длина видео
    - сцены отображаются как цветные прямоугольники
    - кликом по сцене можно перейти к ней (в начало/центр)
    - вертикальная красная линия показывает текущий кадр
    """

    scene_selected = Signal(int)  # индекс выбранной сцены

    def __init__(self, model: Model, parent=None):
        super().__init__(parent)

        self._model = model
        self._scenes: List[SceneView] = []
        self._scene_rects: List[QRect] = []
        self._hover_index: int = -1
        self._selected_index: int = -1

        self._background: QPixmap | None = None

        self.setMinimumHeight(50)
        self.setMouseTracking(True)

        self._palette = [
            QColor("#66c2a5"),
            QColor("#fc8d62"),
            QColor("#8da0cb"),
            QColor("#e78ac3"),
            QColor("#a6d854"),
            QColor("#ffd92f"),
            QColor("#e5c494"),
            QColor("#b3b3b3"),
        ]

        self._model.media_changed.connect(self.on_media_changed)
        self._model.scenes_changed.connect(self.on_scenes_changed)  # type: ignore[attr-defined]
        self._model.frame_position_changed.connect(self.on_frame_position_changed)

    # ------------------------
    # Модель / сигналы
    # ------------------------
    @Slot(VideoWrapper)
    def on_media_changed(self, video: VideoWrapper):
        self._scenes = []
        self._hover_index = -1
        self._selected_index = -1
        self._background = None
        self.update()

    @Slot(list)
    def on_scenes_changed(self, scenes: list[SceneSegment]):
        # ожидаем список сцен из model (dataclass) или совместимых dict/объектов
        converted: List[SceneView] = []
        for s in scenes:
            if isinstance(s, SceneSegment):
                converted.append(SceneView.from_segment(s))

        self._scenes = converted

        self._selected_index = -1 if not self._scenes else 0
        self._background = None
        self.update()

    @Slot(int)
    def on_frame_position_changed(self, frame_idx: int):
        if self._model.current_media is None or not self._scenes:
            return

        idx = self._scene_index_for_frame(frame_idx)
        if idx is not None and idx != self._selected_index:
            self._selected_index = idx

        self.update()

    # ------------------------
    # Отрисовка
    # ------------------------
    def paintEvent(self, event):
        video = self._model.current_media
        if video is None or video.num_frames <= 0 or not self._scenes:
            return

        total_frames = video.num_frames
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()

        top_margin = 8
        bottom_margin = 8
        scene_height = max(10, h - top_margin - bottom_margin)

        self._scene_rects.clear()

        # сцены
        for idx, scene in enumerate(self._scenes):
            start = max(0, min(scene.start_frame, total_frames - 1))
            end = max(start + 1, min(scene.end_frame, total_frames))

            x1 = int(start / total_frames * w)
            x2 = int(end / total_frames * w)

            width = max(2, x2 - x1)

            rect = QRect(x1, top_margin, width, scene_height)

            self._scene_rects.append(rect)

            overlay = QColor(self._palette[idx % len(self._palette)])
            if idx == self._hover_index:
                alpha = 120
            elif scene.start_frame <= self._model.current_frame < scene.end_frame:
                alpha = 150
            else:
                alpha = 80
            overlay.setAlpha(alpha)

            painter.fillRect(rect, overlay)

            # рамка
            border = QPen(QColor(120, 120, 120))
            if idx == self._selected_index:
                border.setWidth(2)
            else:
                border.setWidth(1)
            painter.setPen(border)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 4, 4)

        # вертикальная красная линия текущего кадра
        self._draw_current_frame_cursor(painter, total_frames)

    def _draw_current_frame_cursor(self, painter: QPainter, total_frames: int):
        if self._model.current_media is None:
            return

        current = self._model.current_frame
        if current < 0 or current >= total_frames:
            return

        x = int(current / total_frames * self.width())
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x, 0, x, self.height())

    # ------------------------
    # Мышь / взаимодействие
    # ------------------------
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return

        video = self._model.current_media
        if video is None or video.num_frames <= 0 or not self._scenes:
            return

        clicked_scene = self._scene_index_at(event.position().x())
        if clicked_scene is None:
            return

        self._selected_index = clicked_scene
        scene = self._scenes[clicked_scene]

        local_x = event.position().x() - self._scene_rects[clicked_scene].x()

        if local_x < self._scene_rects[clicked_scene].width() / 2:
            # кликаем в левую половину сцены - переходим к началу
            self._model.current_frame = scene.start_frame
        else:
            # переходим к центру сцены
            center_frame = int((scene.start_frame + scene.end_frame) / 2)
            center_frame = max(0, min(center_frame, video.num_frames - 1))
            self._model.current_frame = center_frame

        self.scene_selected.emit(clicked_scene)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._scenes or self._model.current_media is None:
            return

        idx = self._scene_index_at(event.position().x())

        if idx != self._hover_index:
            self._hover_index = idx if idx is not None else -1
            self.update()

        if idx is not None:
            self.setToolTip(self._scenes[idx].label or f"Scene {idx + 1}")
        else:
            self.setToolTip('')

    def leaveEvent(self, _event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.update()

    # ------------------------
    # Утилиты
    # ------------------------
    def _scene_index_for_frame(self, frame: int) -> int | None:
        for idx, scene in enumerate(self._scenes):
            if scene.start_frame <= frame < scene.end_frame:
                return idx
        return None

    def _scene_index_at(self, x_pos: float) -> int | None:
        """
        Возвращает индекс сцены по координате X (в пикселях), если есть.
        """
        video = self._model.current_media
        if video is None or video.num_frames <= 0 or not self._scenes:
            return None

        total_frames = video.num_frames
        w = self.width()
        if w <= 0:
            return None

        H_MARGIN = 1
        usable_w = max(1, w - 2 * H_MARGIN)

        rel_x = x_pos - H_MARGIN
        if rel_x < 0:
            rel_x = 0
        if rel_x > usable_w:
            rel_x = usable_w

        frame = int(rel_x / usable_w * total_frames)
        return self._scene_index_for_frame(frame)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    model = Model()
    model.current_media = Path("/Users/dvuglaf/Downloads/IMG_3559_drone_cropped (1).mp4")
    print(model.current_media.num_frames)
    timeline = SimpleSceneTimeline(model)
    model.scenes = [
        SceneSegment(0, 7000, 0, 5.0, label="Intro"),
        SceneSegment(7000, 15000, 0, 5.0, label="Scene 2"),
        SceneSegment(19000, 21000, 0, 5.0, label="Scene 2"),
    ]
    timeline.show()

    sys.exit(app.exec())
