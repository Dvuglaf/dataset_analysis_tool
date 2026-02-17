import math
from typing import Callable, Optional

from custom_types import VideoWrapper, SceneSegment
from model import Model
from filter_timeline_grid import SceneListPanel

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QPushButton,
    QLineEdit,
    QStyle,
    QFormLayout,
    QDialog,
    QSizePolicy, QComboBox,
)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QTimer, QUrl, QSignalBlocker
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PySide6.QtGui import (
    QPixmap, QPainter, QFont, QImage, QShortcut, QKeySequence,
    QMouseEvent, QIntValidator, QFocusEvent, QKeyEvent, QColor
)

from simple_scene_timeline import SimpleSceneTimeline


def get_num_digits(value: int) -> int:
    return int(math.log10(value)) + 1 if value > 0 else 1


def pretty_file_size(size_in_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} TB"


def format_timestamp(timestamp: int) -> str:
    mins = timestamp // 60
    seconds = timestamp % 60
    return "{0:02}:{1:02}".format(mins, seconds)


class BasePushButton(QPushButton):
    def __init__(self,
                 on_button_click: Callable[[], None],
                 btn_size: Optional[QSize] = None,
                 tooltip: Optional[str] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._button_click_handler = on_button_click
        self.clicked.connect(on_button_click)

        if btn_size:
            self.setFixedSize(btn_size)

        if tooltip:
            self.setToolTip(tooltip)

        self.setCursor(Qt.PointingHandCursor)

    def attach_shortcut(self, shortcut_key: str, parent: Optional[QWidget] = None) -> QShortcut:
        shortcut = QShortcut(QKeySequence(shortcut_key), parent)
        shortcut.activated.connect(self._button_click_handler)
        return shortcut


class TextButton(BasePushButton):
    def __init__(self,
                 text: str,
                 on_button_click: Callable[[], None],
                 btn_size: Optional[QSize] = None,
                 tooltip: Optional[str] = None,
                 font: Optional[QFont] = None,
                 *args, **kwargs
                 ):
        BasePushButton.__init__(self, on_button_click, btn_size, tooltip, *args, **kwargs)

        self.setText(text)
        if font:
            self.setFont(font)


class PositionLineEdit(QLineEdit):
    updated_value = Signal(int)

    def __init__(self, maximum_value: int):
        super().__init__()
        self._previous: int = 0
        self._maximum_value = maximum_value
        self.setValidator(QIntValidator(0, self._maximum_value))
        self.textEdited.connect(self.on_text_edit)
        self.editingFinished.connect(self.on_changed_text)
        self.setAlignment(Qt.AlignRight)

    @property
    def maximum_value(self) -> int:
        return self._maximum_value

    @maximum_value.setter
    def maximum_value(self, value: int):
        self._maximum_value = value
        self.setValidator(QIntValidator(0, self._maximum_value))

    @Slot()
    def on_text_edit(self):
        if self.text() and int(self.text()) > self._maximum_value:
            self.setText(self.text()[:-1])
            return

    @Slot()
    def on_changed_text(self):
        if not self.text():
            self.setText(str(self._previous))
        else:
            current_value = int(self.text())
            self._previous = current_value
            self.updated_value.emit(current_value)

    def mouseDoubleClickEvent(self, event: Optional[QMouseEvent]) -> None:
        self.selectAll()
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event: Optional[QFocusEvent]) -> None:
        self.on_changed_text()
        super().focusOutEvent(event)

    def keyPressEvent(self, event: Optional[QKeyEvent]) -> None:
        if (event.key() == Qt.Key_Enter) or (event.key() == Qt.Key_Return):
            self.on_changed_text()
            self.clearFocus()
        super().keyPressEvent(event)


class NavigationWidget(QWidget):
    NAVIGATATION_BUTTON_SIZE = QSize(40, 32)
    FONT = QFont("Arial", 14)

    play_clicked = Signal()
    pause_clicked = Signal()
    backward_move_position = Signal()
    forward_move_position = Signal()
    slider_set_position = Signal(int)
    prompt_set_position = Signal(int)

    def __init__(self, model: Model):
        super().__init__()

        self._model = model

        self._is_playing = False

        self.backward = TextButton(
            text="<",
            on_button_click=lambda: self.backward_move_position.emit(),
            btn_size=self.NAVIGATATION_BUTTON_SIZE,
            font=self.FONT,
        )
        self.forward = TextButton(
            text=">",
            on_button_click=lambda: self.forward_move_position.emit(),
            btn_size=self.NAVIGATATION_BUTTON_SIZE,
            font=self.FONT,
        )
        self.play_pause = TextButton(
            text="",
            on_button_click=self.on_play_pause_clicked,
            btn_size=self.NAVIGATATION_BUTTON_SIZE,
            font=self.FONT,
        )
        self.play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.slider_set_position.emit)
        self.slider.setCursor(Qt.ArrowCursor)

        self.current_frame_prompt = PositionLineEdit(maximum_value=0)
        self.current_frame_prompt.setFont(self.FONT)
        self.current_frame_prompt.setText('0')
        self.current_frame_prompt.setFixedWidth(1 * 10 + 4)
        self.current_frame_prompt.setAlignment(Qt.AlignRight)
        self.current_frame_prompt.updated_value.connect(self.prompt_set_position.emit)

        self.total_frames = QLabel("/ 0")
        self.total_frames.setFont(self.FONT)

        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.backward)
        navigation_layout.addWidget(self.play_pause)
        navigation_layout.addWidget(self.forward)
        navigation_layout.addWidget(self.slider)
        navigation_layout.addWidget(self.current_frame_prompt)
        navigation_layout.addWidget(self.total_frames)
        navigation_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(navigation_layout)

        self._model.frame_position_changed.connect(self.on_changed_frame_position)

    @Slot()
    def on_play_pause_clicked(self):
        if self._is_playing:
            self._is_playing = False
            self.play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
            self.pause_clicked.emit()
        else:
            self._is_playing = True
            self.play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
            self.play_clicked.emit()

    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)

    @Slot(int)
    def on_changed_frame_position(self, frame_position: int):
        self.slider.blockSignals(True)
        self.slider.setValue(frame_position)
        self.slider.blockSignals(False)
        self.current_frame_prompt.setText(str(frame_position))
        self.total_frames.setText(f"/ {self._model.current_media.num_frames}")

    def update_style(self, num_frames: int):
        self.on_changed_frame_position(0)
        self.slider.setCursor(Qt.PointingHandCursor)
        self.slider.setMaximum(num_frames)
        self.current_frame_prompt.setFixedWidth(get_num_digits(num_frames) * 10 + 4)
        self.current_frame_prompt.maximum_value = num_frames


class VideoDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        self._checker = 18

    def paintEvent(self, e):
        painter = QPainter(self)

        s = self._checker

        for y in range(0, self.height(), s):
            for x in range(0, self.width(), s):
                if (x//s + y//s) % 2:
                    painter.fillRect(x, y, s, s, QColor(210, 210, 210))
                else:
                    painter.fillRect(x, y, s, s, QColor(235, 235, 235))

        super().paintEvent(e)


class VideoInfoBar(QWidget):
    info_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.path_label = QLabel("—")
        self.path_label.setStyleSheet("font-weight: 600;")
        # self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.info_btn = QPushButton("i")
        self.info_btn.setStyleSheet("font-style: italic;")
        self.info_btn.setFixedHeight(32)
        self.info_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.info_btn.clicked.connect(self.info_clicked)

        self.info_btn.setStyleSheet(
            """
            QPushButton {
                border-radius: 8px;
                border: 1px solid #e5e7eb;
                background: #f3f4f6;
                padding: 0 6px;
            }
            QPushButton:hover {
                background: #e5e7eb;
            }
            """
        )

        layout = QHBoxLayout(self)
        # layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self.path_label, 1)
        layout.addSpacing(10)
        layout.addWidget(self.info_btn)
        #
        # self.setStyleSheet("""
        #     background:#f4f4f4;
        #     border-bottom:1px solid #ccc;
        # """)

    @Slot(VideoWrapper)
    def on_media_changed(self, media: VideoWrapper):
        self.path_label.setText(media.path_to_file.name)


class MetadataDialog(QDialog):
    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Metadata")
        self.resize(420, 520)

        form = QFormLayout(self)

        for k, v in meta.items():
            key_label = QLabel(k)
            key_label.setFont(QFont("Arial", 12, QFont.Bold))
            form.addRow(key_label, QLabel(str(v)))


class CursorOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(130, 40)
        self.setStyleSheet("""
            background:rgba(0,0,0,160);
            color:white;
            border-radius:8px;
            padding:4px;
            font-size:11px;
        """)
        self.setAlignment(Qt.AlignCenter)
        self.hide()


class SceneOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(130, 40)
        self.setStyleSheet("""
            background:rgba(0,0,0,160);
            color:white;
            border-radius:8px;
            padding:4px;
            font-size:11px;
        """)
        self.setAlignment(Qt.AlignCenter)
        self.hide()


class VideoArea(QWidget):
    empty_scene = Signal()
    def __init__(self, model):
        super().__init__()

        self._model = model
        self._video: VideoWrapper | None = None
        self._scenes: list[SceneSegment] = []

        self._pixmap = QPixmap()

        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)
        self._video_sink = QVideoSink(self)
        self._player.setVideoOutput(self._video_sink)

        self._ignore_position = False

        # UI
        self._info_bar = VideoInfoBar()
        self._info_bar.info_clicked.connect(self._show_metadata)

        self._display = VideoDisplay()
        self._display.setScaledContents(False)
        self._display.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Ignored
        )
        self._display.setMinimumSize(0, 0)
        self._overlay = CursorOverlay(self._display)
        self._scene_overlay = SceneOverlay(self._display)

        # список сцен под настройками (замена отдельным таймлайнам)
        self._nav = NavigationWidget(model)

        self._scene_timeline = SimpleSceneTimeline(model)

        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        main.addWidget(self._info_bar, 0)
        main.addWidget(self._display, 1)
        scene_layout = QHBoxLayout()
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.addWidget(self._scene_timeline)
        main.addWidget(self._nav, 0)
        main.addLayout(scene_layout, 0)

        self.setLayout(main)

        model.media_changed.connect(self._on_media_changed)
        model.frame_position_changed.connect(self._on_frame_position_changed)
        model.scenes_changed.connect(self._on_scenes_changed)

        self._nav.play_clicked.connect(self._play)
        self._nav.pause_clicked.connect(self._pause)
        self._nav.forward_move_position.connect(self._next_frame)
        self._nav.backward_move_position.connect(self._prev_frame)
        self._nav.slider_set_position.connect(self._set_frame)
        self._nav.prompt_set_position.connect(self._set_frame)

        # клик по сцене в списке — переход к началу сцены
        self._scene_timeline.scene_selected.connect(self._on_scene_clicked)

        self._player.positionChanged.connect(self._on_player_position_changed)
        self._video_sink.videoFrameChanged.connect(self._on_video_frame)

        self._display.mouseMoveEvent = self._on_mouse_move

        self._nav.setDisabled(True)

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def _on_media_changed(self, video: VideoWrapper):
        self._pause()

        self._video = video
        # Reset scenes list; new scenes will be provided via scenes_changed signal
        self._scenes = []

        if not video:
            self._reset()
            return

        self._nav.update_style(video.num_frames)
        self._nav.setEnabled(True)

        self._info_bar.path_label.setText(str(video.path_to_file.name))

        url = QUrl.fromLocalFile(str(video.path_to_file))
        self._player.setSource(url)
        self._player.setPosition(0)
        self._player.play()
        self._player.pause()
        self._model.current_frame = 0

    @Slot(int)
    def _on_frame_position_changed(self, frame_idx: int):
        if self._player.isPlaying() or self._ignore_position:
            return
        if not self._video:
            return

        if self._scenes:
            scene_idx = self._scene_timeline._scene_index_for_frame(frame_idx)
            if scene_idx is not None:
                self._scene_overlay.setText(f"Scene {scene_idx}")
                self._scene_overlay.move(12, 8)
                self._scene_overlay.show()
                self._model.current_scene = self._scenes[scene_idx]
            else:
                self._scene_overlay.setText(f"Fade-out")
                self._scene_overlay.move(12, 8)
                self._scene_overlay.show()
                self.empty_scene.emit()

        fps = self._video.fps or 25.0
        # Прыгаем в середину кадра (+0.5), чтобы избежать "граничных" ошибок
        pos_ms = int((frame_idx + 0.5) * 1000.0 / fps)

        self._ignore_position = True
        self._player.setPosition(pos_ms)
        self._ignore_position = False

    def _reset(self):
        self._pixmap = QPixmap()
        self._display.clear()

        self._nav.update_style(0)
        self._nav.setDisabled(True)

        self._info_bar.path_label.setText("—")

        self._scenes = []

    def _play(self):
        if not self._video:
            return
        self._player.play()

    def _pause(self):
        self._player.pause()
        if self._video:
            fps = self._video.fps or 25.0
            # Прыгаем ровно в начало кадра, который сейчас на слайдере
            exact_ms = round((self._model.current_frame + 1) * 1000.0 / fps + (1 / fps))
            self._ignore_position = True
            self._player.setPosition(exact_ms)
            self._ignore_position = False

    def _next_frame(self):
        if not self._video:
            return

        pos = self._model.current_frame + 1

        if pos >= self._video.num_frames:
            self._pause()
            return

        self._model.current_frame = pos

    def _prev_frame(self):
        if not self._video:
            return

        pos = max(0, self._model.current_frame - 1)
        self._model.current_frame = pos

    def _set_frame(self, idx):
        if not self._video:
            return

        idx = max(0, min(idx, self._video.num_frames-1))
        self._model.current_frame = idx

    def _on_player_position_changed(self, pos_ms: int):
        if not self._video or self._ignore_position:
            return

        fps = self._video.fps or 25.0
        # Используем строгое округление вниз (floor)
        # Кадр 0 длится от 0 до 39.999 мс
        frame = int(pos_ms * fps / 1000.0)

        frame = max(0, min(frame, self._video.num_frames - 1))

        if frame != self._model.current_frame:
            self._ignore_position = True
            self._model.current_frame = frame
            self._ignore_position = False

        if self._model.scenes:
            scene_idx = self._scene_timeline._scene_index_for_frame(frame)
            if scene_idx is not None:
                self._scene_overlay.setText(f"Scene {scene_idx}")
                self._scene_overlay.move(12, 8)
                self._scene_overlay.show()
                self._model.current_scene = self._scenes[scene_idx]
            else:
                self._scene_overlay.setText(f"Fade-out")
                self._scene_overlay.move(12, 8)
                self._scene_overlay.show()
                self.empty_scene.emit()

    def _on_video_frame(self, frame):
        if frame.isValid():
            img = frame.toImage().convertToFormat(QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(img)
            self._update_display()

    def resizeEvent(self, e):
        self._update_display()
        super().resizeEvent(e)

    def _update_display(self):
        if self._pixmap.isNull():
            return

        scaled = self._pixmap.scaled(
            self._display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self._display.setPixmap(scaled)

    def _on_mouse_move(self, e):
        if self._pixmap.isNull():
            return

        pm = self._display.pixmap()
        if not pm:
            return

        lw = self._display.width()
        lh = self._display.height()

        pw = pm.width()
        ph = pm.height()

        x0 = (lw - pw)//2
        y0 = (lh - ph)//2

        x = e.pos().x() - x0
        y = e.pos().y() - y0

        if 0 <= x <= pw and 0 <= y <= ph:

            nx = x / pw
            ny = y / ph

            self._overlay.setText(
                f"x:{nx:.3f}\ny:{ny:.3f}"
            )
            self._overlay.move(lw-145, 8)
            self._overlay.show()
        else:
            self._overlay.hide()

    def _show_metadata(self):
        if not self._video:
            return

        meta = {
            "Path:": self._video.path_to_file,
            "FPS:": self._video.fps,
            "Frames:": self._video.num_frames,
            "Resolution:": f"{self._video.width}x{self._video.height}",
            "Size:": pretty_file_size(self._video.file_size),
            "Duration (s)": round(
                self._video.num_frames / self._video.fps, 2
            )
        }

        dlg = MetadataDialog(meta, self)
        dlg.exec()

    # ------------------------
    # Scenes integration
    # ------------------------
    @Slot(list)
    def _on_scenes_changed(self, scenes: list[SceneSegment]):
        """
        Update internal scenes list and combo box when model.scenes changes.
        """
        self._scenes = scenes or []
        if scenes:
            scene_idx = self._scene_timeline._scene_index_for_frame(0)
            self._scene_overlay.setText(f"Scene {scene_idx}")
            self._scene_overlay.move(12, 8)
            self._scene_overlay.show()
            self._model.current_scene = scenes[scene_idx]

    @Slot(int)
    def _on_scene_selected(self, index: int):
        """
        Jump playback to the beginning of the selected scene.
        """
        if index < 0 or index >= len(self._scenes):
            return
        if not self._video:
            return

        scene = self._scenes[index]
        start_frame = int(scene.start_frame)
        # Clamp to valid range
        start_frame = max(0, min(start_frame, self._video.num_frames - 1))
        self._model.current_frame = start_frame
        self._model.current_scene = scene

    @Slot(int)
    def _on_scene_clicked(self, index: int):
        """
        Обработчик клика по сцене в вертикальном списке под плеером.
        Логика такая же, как у выбора в комбобоксе.
        """
        if index < 0 or index >= len(self._scenes):
            return
        if not self._video:
            return

        scene = self._scenes[index]
        start_frame = int(scene.start_frame)
        start_frame = max(0, min(start_frame, self._video.num_frames - 1))
        self._model.current_frame = start_frame
        self._model.current_scene = scene
