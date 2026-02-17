from PySide6.QtWidgets import *
from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor, QPen, QMouseEvent

from custom_types import VideoWrapper
from model import Model

FRAME_HEIGHT = 80
BASE_FRAME_WIDTH = 120  # "базовая" ширина кадра при frame_view = 1
MAX_DRAWN_FRAMES = 100  # максимум кадров, которые одновременно рисуем/декодируем


class FrameStripWidget(QWidget):
    """
    Лента кадров с зумом, сеткой по кадрам и вертикальной красной линией
    текущего кадра. Кадры подгружаются лениво: декодируются только видимые
    (до ~MAX_DRAWN_FRAMES вокруг вьюпорта).
    """

    def __init__(self, model: Model):
        super().__init__()

        self._model = model

        # данные
        self.total_frames: int = 0

        # кэш миниатюр: frame_idx -> QPixmap
        self._frame_cache: dict[int, QPixmap] = {}
        self._max_cache_size = 500

        # Zoom / scroll
        # offset: смещение в пикселях от начала видео по оси X
        self.offset: float = 0.0
        # frame_view: логический "шаг" по кадрам, влияет на зум
        # px_per_frame = BASE_FRAME_WIDTH / frame_view
        self.frame_view: int = 1

        # настройки сетки
        self._min_grid_px = 80  # минимальное расстояние между линиями сетки

        self.setMinimumHeight(FRAME_HEIGHT + 24)
        self.setMouseTracking(True)

        self._model.media_changed.connect(self.on_media_changed)
        self._model.frame_position_changed.connect(self.on_frame_position_changed)

    # ------------------------
    # Модель / сигналы
    # ------------------------
    @Slot(VideoWrapper)
    def on_media_changed(self, video: VideoWrapper):
        self.total_frames = video.num_frames
        self._frame_cache.clear()
        self.offset = 0
        self.frame_view = 1
        self.update()

    @Slot(int)
    def on_frame_position_changed(self, frame_idx: int):
        # только обновляем позицию красного курсора, без автоскролла
        self.update()

    # ------------------------
    # Вспомогательные свойства
    # ------------------------
    def _px_per_frame(self) -> float:
        # ширина одного кадра в пикселях на текущем зуме
        return BASE_FRAME_WIDTH / max(1, float(self.frame_view))

    def _clamp_offset(self, offset: float) -> float:
        if self.total_frames <= 0:
            return 0.0
        px_per_frame = self._px_per_frame()
        total_length_px = self.total_frames * px_per_frame
        max_offset = max(0.0, total_length_px - self.width())
        return max(0.0, min(offset, max_offset))

    # ------------------------
    # Публичный API
    # ------------------------
    def set_offset(self, offset: float):
        """
        Устанавливаем offset напрямую (в пикселях).
        """
        self.offset = self._clamp_offset(offset)
        self.update()

    def update_frame_view(self, zoom_in: bool):
        """
        Обновляет frame_view (зум). Чем меньше frame_view, тем сильнее зум.
        """
        if zoom_in:
            new_view = max(1, self.frame_view // 2)
        else:
            new_view = min(1024, self.frame_view * 2)
        if new_view == self.frame_view:
            return
        # держим текущий кадр в центре при смене зума
        current = self._model.current_frame
        self.frame_view = new_view
        self.offset = self._get_offset_by_current_frame(current)
        self.update()

    # ------------------------
    # Расчёт offset по кадру
    # ------------------------
    def _get_offset_by_current_frame(self, current_frame: int) -> float:
        """
        Возвращает offset, чтобы current_frame был видим и по возможности
        в центре вьюпорта.
        """
        if self._model.current_media is None or self.total_frames <= 0:
            return 0.0

        px_per_frame = self._px_per_frame()
        frame_width = px_per_frame
        x_pos = current_frame * px_per_frame

        desired_offset = x_pos - self.width() / 2 + frame_width / 2
        return self._clamp_offset(desired_offset)

    # ------------------------
    # Работа с кэшем кадров
    # ------------------------
    def _get_frame_pixmap(self, frame_idx: int) -> QPixmap | None:
        """
        Возвращает QPixmap для кадра frame_idx, используя ленивый кэш.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None

        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]

        # лениво читаем кадр из VideoWrapper
        video = self._model.current_media
        if video is None:
            return None

        try:
            frame = video[frame_idx]  # np.ndarray, RGB
        except Exception:
            return None

        # numpy -> QImage -> QPixmap
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            BASE_FRAME_WIDTH,
            FRAME_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # простой контроль размера кэша
        if len(self._frame_cache) >= self._max_cache_size:
            # удаляем какой‑то произвольный элемент (без LRU для простоты)
            self._frame_cache.pop(next(iter(self._frame_cache)))

        self._frame_cache[frame_idx] = pix
        return pix

    # ------------------------
    # Обработчики событий
    # ------------------------
    def paintEvent(self, event):
        if self._model.current_media is None or self.total_frames <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        px_per_frame = self._px_per_frame()
        if px_per_frame <= 0:
            return

        # видимый диапазон кадров
        view_left_px = self.offset
        view_right_px = self.offset + self.width()

        start_frame = max(0, int(view_left_px / px_per_frame))
        end_frame = min(self.total_frames, int(view_right_px / px_per_frame) + 1)

        # ограничиваем кол-во реально отрисованных кадров
        frames_in_view = max(1, end_frame - start_frame)
        # stride, чтобы не рисовать > MAX_DRAWN_FRAMES
        stride_by_density = max(1, int(frames_in_view / MAX_DRAWN_FRAMES))
        # также принимаем во внимание логический зум (frame_view)
        stride = max(self.frame_view, stride_by_density)

        # нормализуем старт под stride
        start_frame = (start_frame // stride) * stride

        # --- фон под кадрами ---
        painter.fillRect(self.rect(), QColor(245, 245, 245))

        # --- отрисовка кадров ---
        for f in range(start_frame, end_frame, stride):
            pix = self._get_frame_pixmap(f)
            if pix is None:
                continue

            x = int(f * px_per_frame - self.offset)
            if x > self.width():
                break

            target_w = max(1, int(px_per_frame))
            target_h = FRAME_HEIGHT
            target_rect = QRect(x, 0, target_w, target_h)
            painter.drawPixmap(target_rect, pix)

        # --- сетка по кадрам ---
        self._draw_grid(painter, px_per_frame)

        # --- вертикальная линия текущего кадра ---
        self._draw_current_frame_cursor(painter, px_per_frame)

    def mousePressEvent(self, event: QMouseEvent):
        """
        Клик по ленте переводит текущий кадр в соответствующую позицию.
        """
        if self._model.current_media is None or self.total_frames <= 0:
            return

        if event.button() != Qt.LeftButton:
            return

        px_per_frame = self._px_per_frame()
        frame = int((event.position().x() + self.offset) / px_per_frame)
        frame = max(0, min(frame, self.total_frames - 1))
        self._model.current_frame = frame

    # ------------------------
    # Отрисовка сетки и курсора
    # ------------------------
    def _draw_grid(self, painter: QPainter, px_per_frame: float):
        if self.total_frames <= 0:
            return

        # шаг по кадрам для линий сетки так, чтобы между ними было >= _min_grid_px
        frames_step = max(1, int(self._min_grid_px / px_per_frame))

        view_left_px = self.offset
        view_right_px = self.offset + self.width()

        start_frame = max(0, int(view_left_px / px_per_frame))
        start_frame = (start_frame // frames_step) * frames_step

        # стиль линий
        grid_pen = QPen(QColor(210, 210, 210))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        text_color = QColor(80, 80, 80)

        baseline_y = FRAME_HEIGHT

        for f in range(start_frame, self.total_frames, frames_step):
            x = int(f * px_per_frame - self.offset)
            if x > self.width():
                break

            # вертикальная линия
            painter.drawLine(x, 0, x, baseline_y + 10)

            # подпись кадра
            painter.setPen(text_color)
            painter.drawText(x + 2, baseline_y + 10, str(f))
            painter.setPen(grid_pen)

    def _draw_current_frame_cursor(self, painter: QPainter, px_per_frame: float):
        if self._model.current_media is None:
            return

        current_frame = self._model.current_frame
        if current_frame < 0 or current_frame >= self.total_frames:
            return

        x = int(current_frame * px_per_frame - self.offset)
        if x < 0 or x > self.width():
            return

        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(x, 0, x, self.height())


class FrameTimeline(QWidget):
    def __init__(self, model: Model):
        super().__init__()

        self._model = model

        layout = QVBoxLayout()

        # Widget
        self.timeline = FrameStripWidget(model)
        layout.addWidget(self.timeline)

        # Scroll bar
        self.scroll = QScrollBar(Qt.Horizontal)
        layout.addWidget(self.scroll)

        # Zoom buttons
        zoom_layout = QHBoxLayout()
        btn_plus = QPushButton("+")
        btn_minus = QPushButton("-")
        btn_plus.clicked.connect(self.zoom_in)
        btn_minus.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(btn_minus)
        zoom_layout.addWidget(btn_plus)
        layout.addLayout(zoom_layout)

        # Инициализация scroll
        self.scroll.setMinimum(0)
        self.scroll.valueChanged.connect(self.on_scroll_offset)

        self.setLayout(layout)

        self._model.media_changed.connect(self.on_media_changed)

    @Slot(VideoWrapper)
    def on_media_changed(self, video: VideoWrapper):
        self.update_scrollbar()

    @Slot(int)
    def on_frame_position_changed(self, frame_idx: int):
        offset = self.timeline._get_offset_by_current_frame(frame_idx)
        self.scroll.setValue(offset)

    def update_scrollbar(self):
        if self._model.current_media is None:
            self.scroll.setMinimum(0)
            self.scroll.setMaximum(0)
            return

        total_frames = self._model.current_media.num_frames
        px_per_frame = self.timeline._px_per_frame()
        total_length = int(total_frames * px_per_frame)
        viewport_width = max(1, self.timeline.width())

        self.scroll.setMinimum(0)
        self.scroll.setMaximum(max(0, total_length - viewport_width))
        self.scroll.setPageStep(viewport_width)

    def on_scroll_offset(self, val):
        self.timeline.set_offset(val)

    def zoom_in(self):
        self.timeline.update_frame_view(zoom_in=True)
        self.update_scrollbar()
        self.timeline.update()

    def zoom_out(self):
        self.timeline.update_frame_view(zoom_in=False)
        self.update_scrollbar()
        self.timeline.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # при изменении размера корректируем диапазон скролла
        self.update_scrollbar()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    model = Model()
    timeline = FrameTimeline(model)
    timeline.setMinimumWidth(1000)
    model.current_media = Path("/Users/dvuglaf/Downloads/IMG_3559_drone_cropped (1).mp4")
    timeline.show()

    sys.exit(app.exec())
