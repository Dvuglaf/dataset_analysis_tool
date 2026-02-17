from __future__ import annotations

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QTimer, QPoint
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QScrollArea,
    QSizePolicy,
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIcon, QMouseEvent

from job_manager import JobManager, JobInfo, JobStatus, RunningJobInfo, DoneJobInfo, ErrorJobInfo


class StatusIndicator(QWidget):
    """
    Индикатор статуса задачи: цветной кружок с иконкой или анимацией.
    """
    
    def __init__(self, status: JobStatus, parent=None):
        super().__init__(parent)
        self._status = status
        self._animation_value = 0.0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation)
        self.setFixedSize(20, 20)
        
        if status == JobStatus.RUNNING:
            self._animation_timer.start(50)  # обновление каждые 50ms для плавной анимации
    
    def set_status(self, status: JobStatus):
        if self._status != status:
            self._status = status
            if status == JobStatus.RUNNING:
                self._animation_timer.start(50)
            else:
                self._animation_timer.stop()
            self.update()
    
    def _update_animation(self):
        self._animation_value = (self._animation_value + 0.1) % 1.0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center = self.rect().center()
        radius = 8
        
        # цвет кружка в зависимости от статуса
        if self._status == JobStatus.RUNNING:
            # анимированный кружок загрузки (вращающийся градиент)
            angle = int(self._animation_value * 360)
            painter.setPen(QPen(QColor(70, 130, 180), 2))
            painter.setBrush(QBrush(QColor(70, 130, 180)))
            painter.drawEllipse(center, radius, radius)
            # маленький белый кружок для эффекта загрузки
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            import math
            x = center.x() + int(radius * 0.7 * math.cos(math.radians(angle)))
            y = center.y() + int(radius * 0.7 * math.sin(math.radians(angle)))
            painter.drawEllipse(x - 2, y - 2, 4, 4)
        elif self._status == JobStatus.DONE:
            # зелёный кружок с галочкой
            painter.setPen(QPen(QColor(50, 150, 50), 2))
            painter.setBrush(QBrush(QColor(50, 150, 50)))
            painter.drawEllipse(center, radius, radius)
            # галочка
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawLine(center.x() - 4, center.y(), center.x() - 1, center.y() + 3)
            painter.drawLine(center.x() - 1, center.y() + 3, center.x() + 4, center.y() - 2)
        elif self._status == JobStatus.ERROR:
            # красный кружок с крестиком
            painter.setPen(QPen(QColor(200, 50, 50), 2))
            painter.setBrush(QBrush(QColor(200, 50, 50)))
            painter.drawEllipse(center, radius, radius)
            # крестик
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            cross_size = 6
            painter.drawLine(center.x() - cross_size, center.y() - cross_size,
                           center.x() + cross_size, center.y() + cross_size)
            painter.drawLine(center.x() - cross_size, center.y() + cross_size,
                           center.x() + cross_size, center.y() - cross_size)
        elif self._status == JobStatus.CANCELLED:
            # серый кружок
            painter.setPen(QPen(QColor(150, 150, 150), 2))
            painter.setBrush(QBrush(QColor(150, 150, 150)))
            painter.drawEllipse(center, radius, radius)
        else:
            # серый кружок для pending
            painter.setPen(QPen(QColor(180, 180, 180), 2))
            painter.setBrush(QBrush(QColor(180, 180, 180)))
            painter.drawEllipse(center, radius, radius)


class SmoothProgressBar(QWidget):
    """
    Минималистичный progress bar с плавной анимацией.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._target_value = 0.0
        self._current_value = 0.0
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation)
        self.setFixedHeight(4)
        self.setMinimumWidth(100)
    
    def set_value(self, value: float):
        """Устанавливает целевое значение (0..1), анимация к нему будет плавной."""
        self._target_value = max(0.0, min(1.0, float(value)))
        if not self._animation_timer.isActive():
            self._animation_timer.start(16)  # ~60 FPS
    
    def _update_animation(self):
        # плавное приближение к целевому значению
        diff = self._target_value - self._current_value
        if abs(diff) < 0.01:
            self._current_value = self._target_value
            self._animation_timer.stop()
        else:
            # экспоненциальное сглаживание
            self._current_value += diff * 0.15
        self.update()
    
    def paintEvent(self, event):
        if self._current_value <= 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # фон (стандартный Qt цвет)
        from PySide6.QtWidgets import QApplication
        bg_color = QApplication.palette().color(QApplication.palette().ColorRole.Base)
        painter.fillRect(self.rect(), bg_color)
        
        # прогресс (стандартный Qt highlight цвет)
        progress_width = int(self.width() * self._current_value)
        if progress_width > 0:
            progress_rect = QRect(0, 0, progress_width, self.height())
            highlight_color = QApplication.palette().color(QApplication.palette().ColorRole.Highlight)
            painter.fillRect(progress_rect, highlight_color)


class TaskCard(QFrame):
    """
    Карточка одной задачи с красивым дизайном.
    """
    
    def __init__(self, job: JobInfo, parent=None):
        super().__init__(parent)
        self._job_id = job.id
        self._job = job
        
        self.setObjectName("TaskCard")
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(1)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # верхняя строка: индикатор статуса + название задачи
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        
        self._status_indicator = StatusIndicator(job.status, self)
        top_row.addWidget(self._status_indicator)
        
        # название задачи (жирным)
        self._title_label = QLabel(job.type)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(10)
        self._title_label.setFont(title_font)
        top_row.addWidget(self._title_label, 1)
        
        layout.addLayout(top_row)
        
        # описание (мелким шрифтом)
        self._description_label = QLabel(job.description)
        desc_font = QFont()
        desc_font.setPointSize(9)
        self._description_label.setFont(desc_font)
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("color: palette(text);")
        layout.addWidget(self._description_label)
        
        # progress bar (только для RunningJobInfo)
        self._progress_bar = SmoothProgressBar(self)
        self._progress_bar.setVisible(isinstance(job, RunningJobInfo))
        layout.addWidget(self._progress_bar)
        
        # простой стиль карточки
        self.setStyleSheet("""
            QFrame#TaskCard {
                background: palette(base);
                border: 1px solid palette(mid);
            }
            QFrame#TaskCard:hover {
                background: palette(alternate-base);
            }
        """)
    
    def update_job(self, job: JobInfo):
        """Обновляет карточку с новыми данными задачи."""
        self._job = job
        self._status_indicator.set_status(job.status)
        self._description_label.setText(job.description)
        
        # обновляем progress bar
        if isinstance(job, RunningJobInfo):
            self._progress_bar.setVisible(True)
            self._progress_bar.set_value(job.progress)
        else:
            self._progress_bar.setVisible(False)
    
    @property
    def job_id(self) -> str:
        return self._job_id


class TasksPanel(QWidget):
    """
    Панель задач, которая позиционируется относительно главного окна.
    """
    
    def __init__(self, manager: JobManager, parent: QWidget | None = None):
        super().__init__(parent, Qt.Tool | Qt.FramelessWindowHint)
        
        self._manager = manager
        self._parent_window = parent
        self._task_cards: dict[str, TaskCard] = {}
        self._viewed_job_ids: set[str] = set()  # ID задач, которые были просмотрены
        self._drag_position: QPoint | None = None
        
        self.setWindowTitle("Tasks")
        self.setMinimumWidth(400)
        self.setMaximumWidth(500)
        self.setMinimumHeight(200)
        self.setMaximumHeight(600)
        
        # простой стиль без теней
        self.setStyleSheet("""
            TasksPanel {
                background: palette(window);
                border: 1px solid palette(mid);
            }
        """)
        
        # основной layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # заголовок
        header = QHBoxLayout()
        title = QLabel("Tasks")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch()
        
        # кнопка закрытия
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setFlat(True)
        close_btn.clicked.connect(self.hide)
        header.addWidget(close_btn)
        
        main_layout.addLayout(header)
        
        # область прокрутки для карточек задач
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()
        
        scroll.setWidget(self._cards_container)
        main_layout.addWidget(scroll)
        
        # нижняя панель с кнопкой очистки
        footer = QHBoxLayout()
        footer.addStretch()
        self._clear_btn = QPushButton("Clear finished")
        self._clear_btn.clicked.connect(self._clear_finished)
        footer.addWidget(self._clear_btn)
        main_layout.addLayout(footer)
        
        # связи с JobManager
        manager.job_added.connect(self._on_job_added)
        manager.job_updated.connect(self._on_job_updated)
        manager.jobs_changed.connect(self._on_jobs_changed)
        
        # начальное заполнение
        self._rebuild_cards(manager.jobs())
    
    def showEvent(self, event):
        """При показе панели отмечаем все задачи как просмотренные."""
        super().showEvent(event)
        self._viewed_job_ids = {job.id for job in self._manager.jobs()}
        self._update_parent_button()
        self._update_position()
    
    def _update_position(self):
        """Позиционирует панель в правом нижнем углу родительского окна."""
        if self._parent_window is None:
            return
        
        parent_rect = self._parent_window.geometry()
        margin = 10
        x = parent_rect.right() - self.width() - margin
        y = parent_rect.bottom() - self.height() - margin
        
        # убеждаемся, что окно не выходит за границы экрана
        screen_geometry = self.screen().availableGeometry()
        x = max(screen_geometry.left(), min(x, screen_geometry.right() - self.width()))
        y = max(screen_geometry.top(), min(y, screen_geometry.bottom() - self.height()))
        
        self.move(x, y)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_position()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Начало перетаскивания окна."""
        if event.button() == Qt.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Перетаскивание окна."""
        if event.buttons() == Qt.LeftButton and self._drag_position is not None:
            self.move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Завершение перетаскивания."""
        self._drag_position = None
        event.accept()
    
    def _rebuild_cards(self, jobs: list[JobInfo]):
        """Пересоздаёт все карточки задач."""
        # удаляем старые карточки
        for card in list(self._task_cards.values()):
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._task_cards.clear()
        
        # создаём новые карточки
        for job in jobs:
            card = TaskCard(job, self._cards_container)
            self._task_cards[job.id] = card
            self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
    
    def _on_job_added(self, job: JobInfo):
        """Добавляет новую карточку задачи."""
        if job.id in self._task_cards:
            return
        
        card = TaskCard(job, self._cards_container)
        self._task_cards[job.id] = card
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)
    
    def _on_job_updated(self, job: JobInfo):
        """Обновляет существующую карточку задачи."""
        card = self._task_cards.get(job.id)
        if card is None:
            return
        
        card.update_job(job)
        
        # если задача завершилась с ошибкой, она не считается просмотренной
        if isinstance(job, ErrorJobInfo):
            self._viewed_job_ids.discard(job.id)
    
    def _on_jobs_changed(self, jobs: list[JobInfo]):
        """Полное обновление списка задач."""
        self._rebuild_cards(jobs)
    
    def _clear_finished(self):
        """Удаляет карточки завершённых задач."""
        # находим завершённые задачи
        finished_ids = []
        for job in self._manager.jobs():
            if job.status in (JobStatus.DONE, JobStatus.CANCELLED):
                finished_ids.append(job.id)
        
        # удаляем карточки
        for job_id in finished_ids:
            card = self._task_cards.pop(job_id, None)
            if card is not None:
                self._cards_layout.removeWidget(card)
                card.deleteLater()
            self._viewed_job_ids.discard(job_id)
        
        self._update_parent_button()
    
    def _update_parent_button(self):
        """Обновляет кнопку в родительском окне (если есть метод)."""
        if hasattr(self._parent_window, "_update_tasks_button"):
            self._parent_window._update_tasks_button(self._manager.jobs())
    
    def get_unviewed_count(self) -> int:
        """Возвращает количество непросмотренных задач (с ошибками или новые)."""
        all_job_ids = {job.id for job in self._manager.jobs()}
        return len(all_job_ids - self._viewed_job_ids)


# Для обратной совместимости
TasksDock = TasksPanel
