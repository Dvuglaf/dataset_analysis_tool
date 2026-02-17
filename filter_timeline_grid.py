from typing import List

import numpy as np
from PySide6.QtCore import Qt, Slot, QRect, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout

from custom_types import SceneSegment, VideoWrapper
from frame_sequence import FrameGridView
from model import Model


class SceneListPanel(QWidget):
    """
    Вертикальный список сцен (слева от сетки кадров).

    Показывает сцены сверху вниз, позволяет выбирать сцену кликом.
    """

    scene_selected = Signal(int)

    def __init__(self, model: Model, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = model
        self._scenes: List[SceneSegment] = []
        self._scene_rects: List[QRect] = []
        self._hover_index: int = -1
        self._selected_index: int = -1

        self.setMinimumWidth(160)
        self.setMaximumWidth(260)
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

        self._model.media_changed.connect(self._on_media_changed)
        self._model.scenes_changed.connect(self._on_scenes_changed)
        self._model.frame_position_changed.connect(self._on_frame_changed)

    # ------------------------
    # Модель / сигналы
    # ------------------------
    @Slot(VideoWrapper)
    def _on_media_changed(self, video: VideoWrapper):
        self._scenes = []
        self._scene_rects = []
        self._hover_index = -1
        self._selected_index = -1
        self.update()

    @Slot(list)
    def _on_scenes_changed(self, scenes: list[SceneSegment]):
        self._scenes = list(scenes or [])
        self._scene_rects = []
        self._hover_index = -1
        self._selected_index = 0 if self._scenes else -1
        self.update()

    @Slot(int)
    def _on_frame_changed(self, frame_idx: int):
        if not self._scenes:
            return
        idx = self._scene_index_for_frame(frame_idx)
        if idx is not None and idx != self._selected_index:
            self._selected_index = idx
            self.update()

    # ------------------------
    # Отрисовка
    # ------------------------
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.fillRect(self.rect(), QColor(248, 248, 248))
        self._scene_rects.clear()

        if not self._scenes:
            # Пустое состояние
            painter.setPen(QColor(140, 140, 140))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No scenes.\nDetect scenes on the Scene tab.",
            )
            return

        margin_x = 6
        margin_y = 6
        row_h = 40

        for idx, scene in enumerate(self._scenes):
            y = margin_y + idx * (row_h + margin_y)
            rect = QRect(
                margin_x,
                y,
                max(10, self.width() - 2 * margin_x),
                row_h,
            )
            self._scene_rects.append(rect)

            base_color = QColor(self._palette[idx % len(self._palette)])
            bg = QColor(base_color)
            bg.setAlpha(80)

            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 6, 6)

            border = QPen(QColor(120, 120, 120))
            if idx == self._selected_index:
                border.setWidth(2)
            else:
                border.setWidth(1)
            painter.setPen(border)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 6, 6)

            # текст: label и диапазон кадров
            label = scene.label or f"Scene {idx + 1}"
            text = f"{label}  [{scene.start_frame} – {scene.end_frame}]"

            painter.setPen(QColor(40, 40, 40))
            inner = rect.adjusted(8, 0, -8, 0)
            painter.drawText(inner, Qt.AlignVCenter | Qt.AlignLeft, text)

    # ------------------------
    # Мышь
    # ------------------------
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        idx = self._index_at(event.position().toPoint())
        if idx is None:
            return
        self._selected_index = idx
        self.scene_selected.emit(idx)
        self.update()

    def mouseMoveEvent(self, event):
        idx = self._index_at(event.position().toPoint())
        if idx != self._hover_index:
            self._hover_index = idx if idx is not None else -1
            self.update()

    def leaveEvent(self, _event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.update()

    # ------------------------
    # Утилиты
    # ------------------------
    def _index_at(self, pos) -> int | None:
        for i, rect in enumerate(self._scene_rects):
            if rect.contains(pos):
                return i
        return None

    def _scene_index_for_frame(self, frame: int) -> int | None:
        for idx, scene in enumerate(self._scenes):
            if scene.start_frame <= frame < scene.end_frame:
                return idx
        return None

    @Slot(int)
    def select_scene(self, index: int):
        """
        Внешний слот для синхронизации выбора (например, из SceneTimeline).
        """
        if index < 0 or index >= len(self._scenes):
            return
        if index != self._selected_index:
            self._selected_index = index
            self.update()


class FilterTimelineWidget(QWidget):
    """
    Таймлайн для вкладки Filter на основе FrameGridView.

    Показывает выборку кадров внутри выбранной сцены (или всего видео) и
    визуализирует для каждого кадра метрики:
      - MAE (межкадровое отличие)
      - Brightness (яркость кадра)
      - Fade (модуль производной яркости)

    Кадры подвыбираются до максимального количества, чтобы не блокировать UI.
    """

    def __init__(self, model: Model, parent: QWidget | None = None):
        super().__init__(parent)

        self._model = model
        self._scenes: List[SceneSegment] = []
        self._current_scene_index: int | None = None

        self._scene_panel = SceneListPanel(model)
        self._grid = FrameGridView()
        self._empty_label = QLabel(
            "No scene selected.\nClick a scene in the Scene tab\nor on the left scene list to inspect frames here."
        )
        self._empty_label.setAlignment(Qt.AlignCenter)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        main_layout.addWidget(self._scene_panel, 0)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._grid, 1)
        right_layout.addWidget(self._empty_label, 1)

        main_layout.addLayout(right_layout, 1)

        self._update_empty_state()

        self._model.media_changed.connect(self._on_media_changed)
        self._model.scenes_changed.connect(self._on_scenes_changed)

        # выбор сцены слева обновляет сетку кадров
        self._scene_panel.scene_selected.connect(self.on_scene_selected)

    # ------------------------
    # Модель / сигналы
    # ------------------------
    @Slot(VideoWrapper)
    def _on_media_changed(self, video: VideoWrapper):
        self._scenes = []
        self._current_scene_index = None
        self._grid.set_frames([], [], [])
        self._update_empty_state()

    @Slot(list)
    def _on_scenes_changed(self, scenes: list[SceneSegment]):
        # Просто сохраняем список сцен; реальное отображение по клику из SceneTimeline
        self._scenes = list(scenes or [])
        self._current_scene_index = None
        self._grid.set_frames([], [], [])
        self._update_empty_state()

    @Slot(int)
    def on_scene_selected(self, index: int):
        """
        Публичный слот, который можно связать с SceneTimeline.scene_selected.
        Вызывает перерасчёт кадров и метрик для выбранной сцены.
        """
        if index < 0 or index >= len(self._scenes):
            self._current_scene_index = None
            self._grid.set_frames([], [], [])
            self._update_empty_state()
            return

        self._current_scene_index = index
        # синхронизируем выбор со списком сцен слева
        self._scene_panel.select_scene(index)
        scene = self._scenes[index]
        self._show_scene(scene)

    # ------------------------
    # Основная логика
    # ------------------------
    def _update_empty_state(self):
        has_frames = self._grid._pixmaps  # type: ignore[attr-defined]
        show_empty = not bool(has_frames)
        self._empty_label.setVisible(show_empty)
        self._grid.setVisible(not show_empty)

    def _show_scene(self, scene: SceneSegment):
        video = self._model.current_media
        if video is None or video.num_frames <= 0:
            self._grid.set_frames([], [], [])
            self._update_empty_state()
            return

        # Ограничиваем количество кадров, чтобы не блокировать UI
        max_frames = 300
        start = max(0, int(scene.start_frame))
        end = max(start + 1, min(int(scene.end_frame), video.num_frames))
        total = end - start
        if total <= 0:
            self._grid.set_frames([], [], [])
            self._update_empty_state()
            return

        step = max(1, total // max_frames)
        frame_indices = list(range(start, end, step))[:max_frames]

        pixmaps: list[QPixmap] = []
        metrics_list: list[dict] = []
        removed_flags: list[bool] = []

        prev_frame_arr: np.ndarray | None = None
        prev_brightness: float | None = None

        # Простейшие пороги для визуализации; в будущем можно сделать настраиваемыми
        mae_threshold = 0.05  # в долях от максимального MAE
        brightness_threshold = 0.5  # 0..1
        fade_threshold = 0.05  # 0..1

        # Для нормализации MAE нам нужен максимальный масштаб; предварительно оценим его по нескольким кадрам
        mae_values: list[float] = []
        brightness_values: list[float] = []
        fade_values: list[float] = []

        # Первый проход — собираем значения
        frames_cache: dict[int, np.ndarray] = {}
        for idx in frame_indices:
            try:
                frame_arr = video[idx]
            except Exception:
                continue

            frames_cache[idx] = frame_arr

            # яркость
            gray = (
                0.299 * frame_arr[:, :, 0]
                + 0.587 * frame_arr[:, :, 1]
                + 0.114 * frame_arr[:, :, 2]
            )
            brightness = float(gray.mean() / 255.0)
            brightness_values.append(brightness)

            # MAE
            if prev_frame_arr is not None:
                diff = np.abs(frame_arr.astype(np.float32) - prev_frame_arr.astype(np.float32))
                mae = float(diff.mean() / 255.0)
            else:
                mae = 0.0
            mae_values.append(mae)

            # Fade (производная яркости)
            if prev_brightness is not None:
                fade_val = abs(brightness - prev_brightness)
            else:
                fade_val = 0.0
            fade_values.append(fade_val)

            prev_frame_arr = frame_arr
            prev_brightness = brightness

        if not frames_cache:
            self._grid.set_frames([], [], [])
            self._update_empty_state()
            return

        # Нормализация порогов относительно реальных значений
        max_mae = max(mae_values) if mae_values else 1.0
        if max_mae <= 0:
            max_mae = 1.0
        mae_thr_abs = mae_threshold * max_mae

        max_fade = max(fade_values) if fade_values else 1.0
        if max_fade <= 0:
            max_fade = 1.0
        fade_thr_abs = fade_threshold * max_fade

        # Второй проход — строим pixmap и словари для FrameGridView
        prev_frame_arr = None
        prev_brightness = None

        for idx in frame_indices:
            frame_arr = frames_cache.get(idx)
            if frame_arr is None:
                continue

            h, w, ch = frame_arr.shape
            img = QImage(
                frame_arr.data,
                w,
                h,
                ch * w,
                QImage.Format_RGB888,
            )
            pm = QPixmap.fromImage(img)
            pixmaps.append(pm)

            gray = (
                0.299 * frame_arr[:, :, 0]
                + 0.587 * frame_arr[:, :, 1]
                + 0.114 * frame_arr[:, :, 2]
            )
            brightness = float(gray.mean() / 255.0)

            # MAE
            if prev_frame_arr is not None:
                diff = np.abs(frame_arr.astype(np.float32) - prev_frame_arr.astype(np.float32))
                mae = float(diff.mean() / 255.0)
            else:
                mae = 0.0

            # Fade strength
            if prev_brightness is not None:
                fade_val = abs(brightness - prev_brightness)
            else:
                fade_val = 0.0

            metrics = {
                "mae": {
                    "value": mae,
                    "threshold": mae_thr_abs,
                    "better": "low",
                },
                "brightness": {
                    "value": brightness,
                    "threshold": brightness_threshold,
                    "better": "high",
                },
                "fade": {
                    "value": fade_val,
                    "threshold": fade_thr_abs,
                    "better": "low",
                },
            }

            metrics_list.append(metrics)
            removed_flags.append(False)

            prev_frame_arr = frame_arr
            prev_brightness = brightness

        self._grid.set_frames(pixmaps, metrics_list, removed_flags)
        self._update_empty_state()

