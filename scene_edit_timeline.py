from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, Signal, Slot, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent, QImage, QPixmap, QIcon
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from custom_types import SceneSegment, VideoWrapper
from model import Model


@dataclass
class SceneView(SceneSegment):
    preview: QPixmap | None = None

    @classmethod
    def from_segment(cls, segment: SceneSegment, preview: QPixmap | None = None) -> "SceneView":
        return cls(
            start_frame=segment.start_frame,
            start_time_s=segment.start_time_s,
            end_frame=segment.end_frame,
            end_time_s=segment.end_time_s,
            label=segment.label,
            preview=preview,
        )


class SceneEditTimelineView(QWidget):
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

        # zoom / scroll state
        self._zoom: float = 1.0  # 1.0 = весь таймлайн целиком
        self._offset_frames: float = 0.0  # левый видимый кадр

        # drag state
        self._dragging_cursor: bool = False
        self._dragging_edge: str | None = None  # "left" | "right"
        self._drag_scene_index: int = -1
        self._edge_drag_start_frame: int = 0

        self._hover_edge: str | None = None  # для смены курсора

        self._background: QPixmap | None = None

        self.setMinimumHeight(150)
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
        self._zoom = 1.0
        self._offset_frames = 0.0
        self.update()

    @Slot(list)
    def on_scenes_changed(self, scenes: list[SceneSegment]):
        # ожидаем список сцен из model (dataclass) или совместимых dict/объектов
        converted: List[SceneView] = []
        for s in scenes:
            if isinstance(s, SceneSegment):
                converted.append(SceneView.from_segment(s))

        self._scenes = converted

        if self._model.current_media is not None:
            for scene in self._scenes:
                try:
                    frame = self._model.current_media[scene.start_frame]
                except Exception:
                    continue
                qimg = QImage(
                    frame.data,
                    frame.shape[1],
                    frame.shape[0],
                    QImage.Format_RGB888,
                )
                scene.preview = QPixmap.fromImage(qimg)

        self._selected_index = -1 if not self._scenes else 0
        self._background = None
        # при загрузке сцен показываем весь таймлайн
        self._zoom = 1.0
        self._offset_frames = 0.0
        self.update()

    @Slot(int)
    def on_frame_position_changed(self, frame_idx: int):
        if self._model.current_media is None or not self._scenes:
            return

        idx = self._scene_index_for_frame(frame_idx)
        # если курсор внутри сцены — выделяем её,
        # если вне всех сцен — снимаем выделение
        new_selected = idx if idx is not None else -1
        if new_selected != self._selected_index:
            self._selected_index = new_selected

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

        # фон
        painter.fillRect(self.rect(), QColor(248, 248, 248))

        top_margin = 8
        bottom_margin = 8
        scene_height = max(10, h - top_margin - bottom_margin)

        self._scene_rects.clear()

        # зум: вычисляем, сколько кадров помещается в текущую ширину
        visible_frames = max(1.0, total_frames / max(1.0, self._zoom))
        left_frame = max(0.0, min(self._offset_frames, total_frames - visible_frames))
        self._offset_frames = left_frame
        px_per_frame = w / visible_frames

        # сцены
        for idx, scene in enumerate(self._scenes):
            start = max(0, min(scene.start_frame, total_frames - 1))
            end = max(start + 1, min(scene.end_frame, total_frames))

            # пропускаем сцены которые не попали в видимый диапазон
            if end < left_frame or start > left_frame + visible_frames:
                self._scene_rects.append(QRect())  # placeholder, чтобы индексы совпадали
                continue

            x1 = int((start - left_frame) * px_per_frame)
            x2 = int((end - left_frame) * px_per_frame)

            width = max(2, x2 - x1)

            rect = QRect(x1, top_margin, width, scene_height)

            self._scene_rects.append(rect)

            if hasattr(scene, "preview"):
                painter.save()

                if idx == self._hover_index or scene.start_frame <= self._model.current_frame < scene.end_frame:
                    alpha = 0.4
                else:
                    alpha = 0.25

                painter.setOpacity(alpha)
                painter.drawPixmap(
                    rect,
                    scene.preview,
                    scene.preview.rect()
                )

                painter.restore()

            overlay = QColor(self._palette[idx % len(self._palette)])
            overlay.setAlpha(80)

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

            if rect.width() > 40:
                text = scene.label or f"Scene {idx + 1}"

                # прямоугольник под текст
                text_rect = QRect(rect)

                text_rect.setHeight(22)
                text_rect.moveCenter(rect.center())

                # фон текста
                bg = QColor(0, 0, 0, 120)
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg)
                painter.drawRoundedRect(text_rect, 6, 6)

                # сам текст
                painter.setPen(Qt.white)
                painter.drawText(
                    text_rect,
                    Qt.AlignCenter,
                    text
                )

        # вертикальная красная линия текущего кадра
        self._draw_current_frame_cursor(painter, total_frames, left_frame, px_per_frame)

    def _draw_current_frame_cursor(self, painter: QPainter, total_frames: int, left_frame: float, px_per_frame: float):
        if self._model.current_media is None:
            return

        current = self._model.current_frame
        if current < 0 or current >= total_frames:
            return

        x = int((current - left_frame) * px_per_frame)
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
        if video is None or video.num_frames <= 0:
            return

        x = event.position().x()
        clicked_scene = self._scene_index_at(x)

        # проверяем, не попали ли мы в область границы сцены для resize
        if clicked_scene is not None and 0 <= clicked_scene < len(self._scene_rects):
            rect = self._scene_rects[clicked_scene]
            if rect.isValid():
                local_x = x - rect.x()
                EDGE_MARGIN = 6
                if 0 <= local_x <= EDGE_MARGIN:
                    # drag левой границы
                    self._start_edge_drag(clicked_scene, "left")
                    return
                if rect.width() - EDGE_MARGIN <= local_x <= rect.width():
                    # drag правой границы
                    self._start_edge_drag(clicked_scene, "right")
                    return

        # иначе начинаем drag курсора по таймлайну
        self._dragging_cursor = True
        self._update_current_frame_from_x(x)

        if clicked_scene is not None:
            self._selected_index = clicked_scene
            self.scene_selected.emit(clicked_scene)

        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._scenes or self._model.current_media is None:
            return

        x = event.position().x()

        # если двигаем курсор - перемещаем текущий кадр
        if self._dragging_cursor and event.buttons() & Qt.LeftButton:
            self._update_current_frame_from_x(x)
            return

        # если двигаем границу сцены
        if self._dragging_edge is not None and self._drag_scene_index != -1 and (event.buttons() & Qt.LeftButton):
            self._update_edge_drag(x)
            return

        # hover логика (подсветка, tooltip, курсор)
        idx = self._scene_index_at(x)

        if idx != self._hover_index:
            self._hover_index = idx if idx is not None else -1
            self.update()

        self._hover_edge = None
        if idx is not None and 0 <= idx < len(self._scene_rects):
            rect = self._scene_rects[idx]
            if rect.isValid():
                local_x = x - rect.x()
                EDGE_MARGIN = 6
                if 0 <= local_x <= EDGE_MARGIN:
                    self._hover_edge = "left"
                elif rect.width() - EDGE_MARGIN <= local_x <= rect.width():
                    self._hover_edge = "right"

        if self._hover_edge:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if idx is not None:
            self.setToolTip(self._scenes[idx].label or f"Scene {idx + 1}")
        else:
            self.setToolTip('')

    def leaveEvent(self, _event):
        if self._hover_index != -1:
            self._hover_index = -1
            self.update()
        self._hover_edge = None
        self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self._dragging_cursor = False
            self._dragging_edge = None
            self._drag_scene_index = -1

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

        frame = self._frame_at_x(x_pos)
        return self._scene_index_for_frame(frame)

    def _frame_at_x(self, x_pos: float) -> int:
        """
        Конвертирует координату X в индекс кадра с учётом зума и смещения.
        """
        video = self._model.current_media
        if video is None or video.num_frames <= 0:
            return 0

        total_frames = video.num_frames
        w = max(1, self.width())

        visible_frames = max(1.0, total_frames / max(1.0, self._zoom))
        px_per_frame = w / visible_frames

        left_frame = self._offset_frames
        rel_x = max(0.0, min(float(x_pos), float(w)))

        frame_f = left_frame + rel_x / px_per_frame
        frame = int(max(0, min(frame_f, total_frames - 1)))
        return frame

    def _update_current_frame_from_x(self, x_pos: float):
        """
        Обновляет текущий кадр модели по X-координате (используется для перемещения курсора).
        """
        frame = self._frame_at_x(x_pos)
        self._model.current_frame = frame
        self.update()

    # ------------------------
    # Зум / скролл публичный API
    # ------------------------
    def zoom_in(self):
        if self._model.current_media is None:
            return
        # увеличиваем зум в 2 раза, максимум 16x
        new_zoom = min(16.0, self._zoom * 2.0)
        self._apply_zoom(new_zoom)

    def zoom_out(self):
        if self._model.current_media is None:
            return
        # минимальный зум = 1.0 (весь таймлайн)
        new_zoom = max(1.0, self._zoom / 2.0)
        self._apply_zoom(new_zoom)

    def _apply_zoom(self, new_zoom: float):
        if self._model.current_media is None:
            return
        if new_zoom == self._zoom:
            return

        total_frames = self._model.current_media.num_frames
        if total_frames <= 0:
            return

        # сохраняем текущий кадр по центру вьюпорта при смене зума
        current = self._model.current_frame
        current = max(0, min(current, total_frames - 1))

        self._zoom = new_zoom

        visible_frames = max(1.0, total_frames / max(1.0, self._zoom))
        desired_offset = current - visible_frames / 2.0
        max_offset = max(0.0, total_frames - visible_frames)
        self._offset_frames = max(0.0, min(desired_offset, max_offset))
        self.update()

    def set_offset_ratio(self, ratio: float):
        """
        Позволяет внешнему скроллбару задавать offset (0..1) по всему таймлайну.
        """
        if self._model.current_media is None:
            return
        total_frames = self._model.current_media.num_frames
        if total_frames <= 0:
            return
        visible_frames = max(1.0, total_frames / max(1.0, self._zoom))
        max_offset = max(0.0, total_frames - visible_frames)
        self._offset_frames = max(0.0, min(ratio * max_offset, max_offset))
        self.update()

    def offset_ratio(self) -> float:
        """
        Возвращает нормализованный offset 0..1 для привязки к скроллбару.
        """
        if self._model.current_media is None:
            return 0.0
        total_frames = self._model.current_media.num_frames
        if total_frames <= 0:
            return 0.0
        visible_frames = max(1.0, total_frames / max(1.0, self._zoom))
        max_offset = max(0.0, total_frames - visible_frames)
        if max_offset <= 0:
            return 0.0
        return float(self._offset_frames) / max_offset

    # ------------------------
    # Редактирование сцен (Cut / Split / Resize)
    # ------------------------
    def cut_current_scene(self):
        """
        Полностью вырезает сцену, в которой находится курсор (или выбранную),
        без изменения остальных сцен.
        """
        if not self._scenes:
            return

        idx = self._scene_index_for_frame(self._model.current_frame)
        if idx is None:
            idx = self._selected_index if 0 <= self._selected_index < len(self._scenes) else None
        if idx is None:
            return

        del self._scenes[idx]
        # корректируем выбранный индекс
        if not self._scenes:
            self._selected_index = -1
        else:
            self._selected_index = min(idx, len(self._scenes) - 1)
        self.update()

    def split_current_scene(self):
        """
        Делит сцену на две по текущей позиции курсора.
        """
        if not self._scenes:
            return

        frame = self._model.current_frame
        idx = self._scene_index_for_frame(frame)
        if idx is None:
            return

        scene = self._scenes[idx]
        if frame <= scene.start_frame or frame >= scene.end_frame:
            # курсор на границе - нет смысла делить
            return

        left = SceneView(
            start_frame=scene.start_frame,
            start_time_s=scene.start_time_s,
            end_frame=frame,
            end_time_s=scene.start_time_s,  # время можно пересчитать позже при сохранении
            label=scene.label,
            preview=scene.preview,
        )
        right = SceneView(
            start_frame=frame,
            start_time_s=scene.start_time_s,
            end_frame=scene.end_frame,
            end_time_s=scene.end_time_s,
            label=scene.label,
            preview=scene.preview,
        )

        self._scenes[idx:idx + 1] = [left, right]
        # текущий кадр лежит в новой правой сцене, выберем именно её
        new_idx = self._scene_index_for_frame(self._model.current_frame)
        if new_idx is None:
            new_idx = idx + 1  # по идее должен быть именно этот индекс
        self._selected_index = new_idx
        self.scene_selected.emit(new_idx)
        self.update()

    # ------------------------
    # Внутренняя логика drag границ сцен
    # ------------------------
    def _start_edge_drag(self, scene_index: int, edge: str):
        self._dragging_edge = edge
        self._drag_scene_index = scene_index
        scene = self._scenes[scene_index]
        self._edge_drag_start_frame = scene.start_frame if edge == "left" else scene.end_frame

    def _update_edge_drag(self, x_pos: float):
        video = self._model.current_media
        if video is None or self._dragging_edge is None or self._drag_scene_index == -1:
            return

        total_frames = video.num_frames
        frame = self._frame_at_x(x_pos)

        idx = self._drag_scene_index
        scene = self._scenes[idx]

        # "зона прилипания" к соседним границам, в кадрах.
        # Усиливаем ещё сильнее, чтобы границы заметно "прилипали" друг к другу.
        SNAP_FRAMES = 30

        if self._dragging_edge == "left":
            # левая граница не может быть правее конца самой сцены - 1
            max_left = scene.end_frame - 1
            new_start = max(0, min(frame, max_left))

            # прилипание к началу/концу соседних сцен
            if idx > 0:
                prev_scene = self._scenes[idx - 1]
                neighbor_points = [prev_scene.start_frame, prev_scene.end_frame]
                for p in neighbor_points:
                    if abs(new_start - p) <= SNAP_FRAMES:
                        new_start = p
                        break

                # если залезли левой границей в предыдущую сцену — уменьшаем её
                if new_start < prev_scene.end_frame:
                    # правая граница предыдущей сцены не может быть левее её начала + 1
                    prev_scene.end_frame = max(prev_scene.start_frame + 1, new_start)
                    # и левая граница текущей сцены не может уйти правее новой правой границы предыдущей
                    new_start = max(new_start, prev_scene.end_frame)

            scene.start_frame = new_start

        elif self._dragging_edge == "right":
            # правая граница не может быть левее начала самой сцены + 1
            min_right = scene.start_frame + 1
            new_end = max(min_right, min(frame, total_frames))

            # прилипание к началу/концу соседних сцен
            if idx + 1 < len(self._scenes):
                next_scene = self._scenes[idx + 1]
                neighbor_points = [next_scene.start_frame, next_scene.end_frame]
                for p in neighbor_points:
                    if abs(new_end - p) <= SNAP_FRAMES:
                        new_end = p
                        break

                # если залезли правой границей в следующую сцену — уменьшаем её
                if new_end > next_scene.start_frame:
                    # левая граница следующей сцены не может быть правее её конца - 1
                    next_scene.start_frame = min(next_scene.end_frame - 1, new_end)
                    # и правая граница текущей сцены не может уйти левее новой левой границы следующей
                    new_end = min(new_end, next_scene.start_frame)

            scene.end_frame = new_end

        # во время движения границы обновляем текущий кадр, чтобы
        # media_panel мог показывать соответствующие кадры
        self._model.current_frame = frame
        self.update()


class SceneEditTimeline(QWidget):
    """
    Обёртка над SceneEditTimelineView с кнопками Cut / Split / Save и управлением зумом.
    """

    # сигнал наружу при явном сохранении
    scenes_saved = Signal(list)

    def __init__(self, model: Model):
        super().__init__()
        self._model = model
        self._timeline_view = SceneEditTimelineView(model)

        header_label = QLabel("Scene Timeline")
        header_label.setStyleSheet(
            """
            QLabel {
                font-weight: bold; 
                font-size: 18px;
                color: #2c3e50;
                margin-left: 2px;
            }
            """
        )

        self.split_btn = QPushButton("")
        self.split_btn.setIcon(QIcon("./icons/split.png"))

        self.cut_btn = QPushButton("")
        self.cut_btn.setIcon(QIcon("./icons/cut.png"))

        self.save_btn = QPushButton("")
        self.save_btn.setIcon(QIcon("./icons/icons8-save-96.png"))

        self.zoom_in_btn = QPushButton("")
        self.zoom_in_btn.setIcon(QIcon("./icons/icons8-zoom-in-50.png"))

        self.zoom_out_btn = QPushButton("")
        self.zoom_out_btn.setIcon(QIcon("./icons/icons8-zoom-out-50.png"))

        # for btn in (self.split_btn, self.cut_btn, self.save_btn, self.zoom_in_btn, self.zoom_out_btn):
        #     btn.setFixedSize(24, 24)

        self.zoom_out_btn.setEnabled(False)  # изначально зум-аута быть не может (и так весь таймлайн)

        header_layout = QHBoxLayout()
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.zoom_out_btn)
        header_layout.addWidget(self.zoom_in_btn)
        header_layout.addWidget(self.split_btn)
        header_layout.addWidget(self.cut_btn)
        header_layout.addWidget(self.save_btn)

        # горизонтальный скролл для зума
        self._scroll = QHBoxLayout()
        self._scroll_bar = None

        from PySide6.QtWidgets import QScrollBar  # локальный импорт чтобы не тянуть в начало файла
        self._scroll_bar = QScrollBar(Qt.Horizontal)
        self._scroll_bar.setMinimum(0)
        self._scroll_bar.setMaximum(1000)
        self._scroll_bar.setPageStep(100)
        self._scroll_bar.valueChanged.connect(self._on_scroll_changed)
        # изначально таймлайн показывает всё видео целиком, поэтому скролл не нужен
        self._scroll_bar.setVisible(False)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addLayout(header_layout)
        layout.addWidget(self._timeline_view)
        layout.addWidget(self._scroll_bar)
        self.setLayout(layout)

        # связи кнопок
        self.cut_btn.clicked.connect(self._on_cut_clicked)
        self.split_btn.clicked.connect(self._on_split_clicked)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.zoom_in_btn.clicked.connect(self._on_zoom_in)
        self.zoom_out_btn.clicked.connect(self._on_zoom_out)

        # изначально редактирующие кнопки неактивны, пока курсор не попал в сцену
        self.cut_btn.setEnabled(False)
        self.split_btn.setEnabled(False)

        # следим за перемещением курсора, чтобы включать/выключать кнопки
        self._model.frame_position_changed.connect(self._on_frame_position_changed)
        # инициализируем состояние по текущему кадру
        self._on_frame_position_changed(self._model.current_frame)

    # ------------------------
    # Кнопки редактирования
    # ------------------------
    @Slot()
    def _on_cut_clicked(self):
        self._timeline_view.cut_current_scene()

    @Slot()
    def _on_split_clicked(self):
        self._timeline_view.split_current_scene()

    @Slot()
    def _on_save_clicked(self):
        """
        Явно применяет изменения: обновляет model.scenes и шлёт сигнал наружу.
        """
        scenes: list[SceneSegment] = []
        for sv in self._timeline_view._scenes:
            scenes.append(
                SceneSegment(
                    start_frame=sv.start_frame,
                    end_frame=sv.end_frame,
                    start_time_s=sv.start_time_s,
                    end_time_s=sv.end_time_s,
                    label=sv.label,
                )
            )

        # обновляем модель (это также вызовет scenes_changed)
        self._model.scenes = scenes
        # и шлём собственный сигнал наружу
        self.scenes_saved.emit(scenes)

    # ------------------------
    # Зум / скролл
    # ------------------------
    @Slot()
    def _on_zoom_in(self):
        self._timeline_view.zoom_in()
        # при первом зум-ин включаем zoom_out
        if self._timeline_view._zoom > 1.0:
            self.zoom_out_btn.setEnabled(True)
            if self._scroll_bar is not None:
                self._scroll_bar.setVisible(True)
        self._sync_scroll_with_view()

    @Slot()
    def _on_zoom_out(self):
        self._timeline_view.zoom_out()
        # если снова показан весь таймлайн - делаем zoom_out недоступной
        if self._timeline_view._zoom <= 1.0:
            self.zoom_out_btn.setEnabled(False)
            if self._scroll_bar is not None:
                self._scroll_bar.setVisible(False)
        self._sync_scroll_with_view()

    def _sync_scroll_with_view(self):
        if self._scroll_bar is None:
            return
        ratio = self._timeline_view.offset_ratio()
        self._scroll_bar.blockSignals(True)
        self._scroll_bar.setValue(int(ratio * self._scroll_bar.maximum()))
        self._scroll_bar.blockSignals(False)

    def _on_scroll_changed(self, value: int):
        if self._scroll_bar is None:
            return
        max_v = self._scroll_bar.maximum()
        ratio = 0.0 if max_v == 0 else float(value) / max_v
        self._timeline_view.set_offset_ratio(ratio)

    # ------------------------
    # Состояние кнопок в зависимости от позиции курсора
    # ------------------------
    @Slot(int)
    def _on_frame_position_changed(self, frame_idx: int):
        """
        Когда курсор выходит за границы любой сцены:
        - кнопки Cut/Split становятся неактивными
        - в самом виджете сцены выделение уже снимается (смотри on_frame_position_changed у SceneEditTimelineView)
        """
        # ищем сцену, в которой находится кадр
        in_scene = False
        for s in self._timeline_view._scenes:
            if s.start_frame <= frame_idx < s.end_frame:
                in_scene = True
                break

        self.cut_btn.setEnabled(in_scene)
        self.split_btn.setEnabled(in_scene)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    model = Model()
    model.current_media = Path("/Users/dvuglaf/Downloads/IMG_3559_drone_cropped (1).mp4")
    timeline = SceneEditTimeline(model)
    model.scenes = [
        SceneSegment(0, 7000, 0, 5.0, label="Intro"),
        SceneSegment(7000, 15000, 0, 5.0, label="Scene 2"),
        SceneSegment(19000, 21000, 0, 5.0, label="Scene 2"),
    ]
    timeline.resize(800, 150)
    timeline.show()

    sys.exit(app.exec())
