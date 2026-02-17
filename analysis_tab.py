# main.py
# PySide6 Analysis Tab Demo (Light Theme)
# With Compute / Export buttons, Loading, Animations
# Requirements: pip install PySide6 PySide6-Addons

import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QStackedWidget,
    QFrame, QProgressBar, QFileDialog, QMessageBox,
    QGraphicsDropShadowEffect, QFormLayout, QLineEdit
)
from PySide6.QtGui import QPainter, QFont, QColor, QIcon
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, Slot, QSize, Signal

from PySide6.QtCharts import (
    QChartView, QPolarChart,
    QLineSeries, QCategoryAxis, QValueAxis
)

from cache_manager import CacheManager
from computations.metrics.base import Metric
from custom_types import SceneSegment, VideoWrapper
from metric_interpreter import THRESHOLDS, get_description, GroupVerdict, get_group_score
from metric_table import TableRow
from model import Model
from radar_widget import ModernRadarWidget, StatisticWidget
from task_manager import AnalysisManager, ComputeSceneMetricsTask
from widgets import GroupWidget, add_stretch_row, ModeSwitch, GroupWidgetPlain


# ================== STYLES ==================

# PANEL_STYLE = """
# QFrame {
#     background-color: #ffffff;
#     border: 1px solid #d0d0d0;
#     border-radius: 14px;
# }
# QLabel {
#     color: #222222;
#     border: none;
# }
# """
#
# APP_STYLE = """
# QWidget {
#     background-color: #f4f5f7;
#     font-family: Arial;
# }
# """


# ================== HELPERS ==================


def add_shadow(widget):

    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(20)
    shadow.setOffset(0, 4)
    shadow.setColor(QColor(0, 0, 0, 40))

    widget.setGraphicsEffect(shadow)


# ================== TAB BUTTON ==================

class TabButton(QPushButton):

    def __init__(self, text):
        super().__init__(text)

        self.setCheckable(True)
        self.setMinimumHeight(48)

        self.setStyleSheet("""
        QPushButton {
            background: #ffffff;
            border-radius: 12px;
            color: #222;
            font-size: 16px;
            font-weight: 600;
            border: 1px solid #d0d0d0;
        }
        QPushButton:checked {
            background: #e9eeff;
            border: 1px solid #3d6cff;
        }
        QPushButton:hover {
            background: #f3f3f3;
        }
        QPushButton:disabled {
            background: #fafafa;
            color: #aaa;
        }
        """)

        add_shadow(self)


# ================== LOADING OVERLAY ==================

class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: rgba(255,255,255,180);")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Computing metrics…")
        self.label.setStyleSheet("font-size:18px;font-weight:600;color:#333;")

        layout.addWidget(self.label)

        self.hide()


# ================== RADAR CHART ==================

class RadarWidget(QChartView):
    def __init__(self):
        chart = QPolarChart()
        super().__init__(chart)

        self.setRenderHint(QPainter.Antialiasing)

        self.series = QLineSeries()
        self.series.setPointsVisible(True)

        chart.addSeries(self.series)
        chart.setBackgroundVisible(False)
        chart.legend().hide()

        self.angular = QCategoryAxis()
        self.angular.append("Quality", 0)
        self.angular.append("Motion", 120)
        self.angular.append("Diversity", 240)
        self.angular.append("", 360)
        self.angular.setRange(0, 360)

        self.radial = QValueAxis()
        self.radial.setRange(0, 1)
        self.radial.setTickCount(6)

        chart.addAxis(self.angular, QPolarChart.PolarOrientationAngular)
        chart.addAxis(self.radial, QPolarChart.PolarOrientationRadial)

        self.series.attachAxis(self.angular)
        self.series.attachAxis(self.radial)

        self.setMinimumHeight(280)


    def set_scores(self, q, m, d):

        self.series.clear()

        self.series.append(0, q)
        self.series.append(120, m)
        self.series.append(240, d)
        self.series.append(360, q)


# ================== METRIC ROW ==================

class MetricRow(QWidget):
    def __init__(self, metric_name: str, displayed_name: str, subtitle=None):
        super().__init__()
        self.metric_name = metric_name
        self.displayed_name = displayed_name
        # Берем спецификацию из нашего обновленного словаря THRESHOLDS
        self.spec = THRESHOLDS.get(metric_name.replace(' ', '_').lower(), None)

        self.value = None
        main = QVBoxLayout(self)
        main.setSpacing(4)
        main.setContentsMargins(0, 0, 0, 0)

        # Header
        top = QHBoxLayout()
        self.name_lbl = QLabel(self.displayed_name)
        self.value_lbl = QLabel("—")
        self.value_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
        top.addWidget(self.name_lbl)
        top.addStretch()
        top.addWidget(self.value_lbl)

        # Progress bar
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)  # Увеличим разрешение для плавности анимации
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)

        main.addLayout(top)
        main.addWidget(self.bar)

        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet("color:#777;font-size:11px;")
            main.addWidget(sub)

        self.anim = QPropertyAnimation(self.bar, b"value")
        self.anim.setDuration(700)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

        self._apply_base_style()

    def _apply_base_style(self):
        self.bar.setStyleSheet("""
            QProgressBar {
                background: #e5e5e5;
                border-radius: 6px;
                border: none;
            }
            QProgressBar::chunk {
                border-radius: 6px;
            }
        """)

    def _get_gradient_css(self, current_norm):
        if not self.spec:
            return "#3d6cff"

        red = "#e74c3c"
        orange = "#f39c12"
        green = "#2ecc71"

        # Расстояние, на котором градиент "живет" (растяжение)
        scale = 1.0 / max(current_norm, 0.01)
        r0, r1 = self.spec.range

        if self.spec.mode == "direct":
            # Порог, выше которого всё "зеленое"
            threshold_norm = (self.spec.high_lower - r0) / (r1 - r0)
            threshold_norm = max(0.1, min(0.9, threshold_norm))

            return f"""qlineargradient(x1:0, y1:0, x2:{scale}, y2:0, 
                        stop:0 {red}, 
                        stop:{threshold_norm * 0.7} {orange}, 
                        stop:{threshold_norm} {green}, 
                        stop:1 {green})"""

        elif self.spec.mode == "inverse":
            # Порог, ниже которого всё "зеленое" (low_upper)
            threshold_norm = (self.spec.low_upper - r0) / (r1 - r0)
            threshold_norm = max(0.1, min(0.9, threshold_norm))

            return f"""qlineargradient(x1:0, y1:0, x2:{scale}, y2:0, 
                        stop:0 {green}, 
                        stop:{threshold_norm} {green}, 
                        stop:{threshold_norm + (1 - threshold_norm) * 0.3} {orange}, 
                        stop:1 {red})"""

        elif self.spec.mode == "target":
            target_norm = (self.spec.target - r0) / (r1 - r0)
            # Для target пороги обычно симметричны вокруг цели
            return f"""qlineargradient(x1:0, y1:0, x2:{scale}, y2:0, 
                        stop:0 {red}, 
                        stop:{target_norm * 0.6} {orange}, 
                        stop:{target_norm} {green}, 
                        stop:{target_norm + (1 - target_norm) * 0.4} {orange}, 
                        stop:1 {red})"""

    def reset(self):
        self.set_value(Metric(name="dummy", value=0.0, raw_data=None))
        self.value_lbl.setText("—")

    def set_value(self, metric: Metric | None | str):
        if metric is None or not self.spec:
            return
        if isinstance(metric, str):
            self.reset()
            self.value_lbl.setText(metric)
            return
        self.value = metric.value
        low, high = self.spec.range

        # Нормализация
        norm = (self.value - low) / (high - low)
        norm = max(0.0, min(1.0, norm))

        # Анимация (0..1000 для точности)
        self.anim.stop()
        self.anim.setEndValue(int(norm * 1000))
        self.anim.start()

        self.value_lbl.setText(f"{self.value:.2f}")

        # Применяем динамический градиент
        grad = self._get_gradient_css(norm)
        self.bar.setStyleSheet(f"""
            QProgressBar {{ background: #e5e5e5; border-radius: 6px; }}
            QProgressBar::chunk {{ background: {grad}; border-radius: 6px; }}
        """)


# ================== Verdict description card ==================
class VerdictDescriptionWidget(QFrame):
    """Framed card with icon (high/mid/low), bold title line, and description lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VerdictCard")
        self.setStyleSheet("""
            QFrame#VerdictCard {
                border-radius: 10px;
                border: 1px solid #e5e7eb;
                background-color: #fafafa;
                padding: 10px 12px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        self._top_row = QHBoxLayout()
        self._icon_label = QLabel()
        self._icon_label.setFixedSize(24, 24)
        self._icon_label.setStyleSheet("border: none; background: transparent; font-size: 16px;")
        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #1f2937; border: none; background: transparent;")
        self._title_label.setWordWrap(True)
        self._top_row.addWidget(self._icon_label, 0)
        self._top_row.addWidget(self._title_label, 1)
        layout.addLayout(self._top_row)
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignJustify)
        self._desc_label.setStyleSheet("color: #6b7280; font-size: 12px; border: none; background: transparent;")
        self._desc_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._desc_label, 1)
        self._warning_label = QLabel()
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #b45309; font-size: 11px; border: none; background: transparent;")
        layout.addWidget(self._warning_label, 0)
        self._warning_label.hide()
        self.set_verdict(None)

    def set_verdict(self, verdict: GroupVerdict | None):
        if verdict is None or verdict.empty:
            self._title_label.setText("")
            self._desc_label.setText("")
            self._warning_label.setText("")
            self._warning_label.hide()
            self._icon_label.setText("")
            self._icon_label.setStyleSheet("border: none; background: transparent; font-size: 16px;")
            self.show()
            return
        self._title_label.setText(verdict.title_line)
        self._desc_label.setText(". ".join(verdict.description_lines) if verdict.description_lines else "")
        if verdict.warning_line:
            self._warning_label.setText(verdict.warning_line)
            self._warning_label.show()
        else:
            self._warning_label.hide()

        # Icon: circle or symbol by level
        icon_map = {"high": ("●", "#16a34a"), "low": ("●", "#dc2626"), "mid": ("●", "#ca8a04")}
        sym, color = icon_map.get(verdict.icon_level, ("●", "#6b7280"))
        self._icon_label.setText(sym)
        self._icon_label.setStyleSheet(f"border: none; background: transparent; font-size: 18px; color: {color};")
        self.show()


# ================== PAGES ==================
class OverallPage(QWidget):
    def __init__(self, model: Model, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._cache_manager = cache_manager
        self._current_mode = "scene"

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metrics = {}

        self._radar_widget = StatisticWidget()
        self._radar_widget.setMaximumWidth(350)

        stats_layout = QVBoxLayout()
        stats_layout.addWidget(self._radar_widget)
        self.panel = GroupWidget("Statistics", stats_layout)

        layout.addWidget(self.panel)
        layout.addStretch()

        self._model.media_changed.connect(self.on_current_media_changed)

    def on_mode_changed(self, mode_idx: int):
        self._current_mode = "scene" if mode_idx == 0 else "video"

    def restore_metrics_from_cache(self,
                                   cached_metrics: list[Metric | None],
                                   quality_metric_names: list[str],
                                   motion_metric_names: list[str],
                                   diversity_metric_names: list[str]
                                   ):
        quality_metrics = {
            metric.name: metric.value
            for metric in cached_metrics if metric is not None and metric.name in quality_metric_names
        }
        motion_metrics = {
            metric.name: metric.value
            for metric in cached_metrics if
            metric is not None and metric.name in motion_metric_names
        }
        diversity_metrics = {
            metric.name: metric.value
            for metric in cached_metrics if
            metric is not None and metric.name in diversity_metric_names
        }

        quality_score = get_group_score(quality_metrics)
        motion_score = get_group_score(motion_metrics)
        diversity_score = get_group_score(diversity_metrics)

        self._radar_widget.set_scores(quality_score, motion_score, diversity_score)

    def set_empty_metrics(self):
        self._radar_widget.set_scores(None, None, None)

    def on_current_media_changed(self, video: VideoWrapper):
        pass

    def _reset_current_metrics(self):
        pass


class QualityPage(QWidget):
    def __init__(self, model: Model, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._cache_manager = cache_manager
        self._current_mode = "scene"

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metrics = {}

        self.names = ["entropy", "blur", "contrast", "brightness", "saturation", "artifacts"]

        panel_layout = QVBoxLayout()

        for metric_name in self.names:
            row = MetricRow(
                metric_name,
                THRESHOLDS.get(metric_name).display_name,
                THRESHOLDS.get(metric_name).subtitle
            )
            self.metrics[metric_name] = row
            panel_layout.addWidget(row)

        self.description = VerdictDescriptionWidget()
        self.panel = GroupWidget("Visual Quality", panel_layout)

        layout.addWidget(self.panel)
        layout.addWidget(self.description)
        layout.addStretch()

        self._model.media_changed.connect(self.on_current_media_changed)

    def on_mode_changed(self, mode_idx: int):
        self._current_mode = "scene" if mode_idx == 0 else "video"

    def _quality_metric_names(self):
        return {"entropy", "blur", "contrast", "brightness", "saturation", "artifacts"}

    def _update_verdict(self, metrics=None):
        if metrics is None:
            metrics = {name: row.value for name, row in self.metrics.items() if getattr(row, "value", None) is not None}
        score = get_group_score(metrics)
        verdict = get_description(metrics, "quality", score, self._current_mode)
        self.description.set_verdict(verdict)

    def restore_metrics_from_cache(self, cached_metrics: list[Metric | None]):
        def find_metric(metric_name: str) -> Metric | None:
            return next((metric for metric in cached_metrics if metric and metric.name == metric_name), None)

        for metric_name in self._quality_metric_names():
            c_metric = find_metric(metric_name)
            if c_metric is not None:
                if metric_name in self.metrics:
                    self.metrics[metric_name].set_value(c_metric)
        group_metrics = [m for m in (cached_metrics or []) if m and getattr(m, "name", None) in self._quality_metric_names()]
        self._update_verdict(group_metrics)

    def set_empty_metrics(self):
        for metric_name in self._quality_metric_names():
            if metric_name in self.metrics:
                self.metrics[metric_name].reset()
        self._update_verdict([])

    def on_current_media_changed(self, video: VideoWrapper):
        pass

    def _reset_current_metrics(self):
        for metric in self.metrics.values():
            metric.set_value(None)

    def update_scene(self, scene):
        q = scene.get("quality_metrics") or {}
        for k, v in q.items():
            if k in self.metrics:
                self.metrics[k].set_value(v)
        self._update_verdict(q)


class MotionPage(QWidget):
    def __init__(self, model: Model, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._cache_manager = cache_manager
        self._current_mode = "scene"

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metrics = {}

        self.names = ["optical_flow", "motion_smoothness", "psnr", "ssim", "background_dominance", "temporary_consistency_clip_score"]
        panel_layout = QVBoxLayout()

        for metric_name in self.names:
            row = MetricRow(metric_name,
                            THRESHOLDS.get(metric_name).display_name,
                            THRESHOLDS.get(metric_name).subtitle)
            self.metrics[metric_name] = row
            panel_layout.addWidget(row)

        self.description = VerdictDescriptionWidget()
        self.panel = GroupWidget("Motion & Temporal Coherence", panel_layout)

        layout.addWidget(self.panel)
        layout.addWidget(self.description)
        layout.addStretch()

        self._model.media_changed.connect(self.on_current_media_changed)

    def on_mode_changed(self, mode_idx: int):
        self._current_mode = "scene" if mode_idx == 0 else "video"

    def _motion_metric_names(self):
        return {"optical_flow", "motion_smoothness", "psnr", "ssim", "background_dominance", "flow", "smoothness", "temporary_consistency_clip_score"}

    def _update_verdict(self, metrics=None):
        if metrics is None:
            metrics = {name: row.value for name, row in self.metrics.items() if getattr(row, "value", None) is not None}
        score = get_group_score(metrics)
        verdict = get_description(metrics,"motion", score, self._current_mode)
        self.description.set_verdict(verdict)

    def restore_metrics_from_cache(self, cached_metrics: list[Metric | None]):
        def find_metric(metric_name: str) -> Metric | None:
            return next((metric for metric in cached_metrics if metric and metric.name == metric_name), None)
        for key in ("optical_flow", "flow", "motion_smoothness", "smoothness", "psnr", "ssim", "background_dominance", "temporary_consistency_clip_score"):
            c_metric = find_metric(key)
            if c_metric is not None:
                target = "optical_flow" if key == "flow" else "motion_smoothness" if key == "smoothness" else key
                if target in self.metrics:
                    self.metrics[target].set_value(c_metric)
        group_metrics = [m for m in (cached_metrics or []) if m and getattr(m, "name", None) in self._motion_metric_names()]
        self._update_verdict(group_metrics)

    def set_empty_metrics(self):
        for metric_name in ("optical_flow", "motion_smoothness", "psnr", "ssim", "background_dominance"):
            if metric_name in self.metrics:
                self.metrics[metric_name].reset()
        self._update_verdict([])

    def on_current_media_changed(self, video: VideoWrapper):
        pass

    def _reset_current_metrics(self):
        for metric in self.metrics.values():
            metric.set_value(None)

    def update_scene(self, scene):
        motion = scene.get("motion_metrics") or {}
        name_map = {"flow": "optical_flow", "smoothness": "motion_smoothness"}
        for k, v in motion.items():
            key = name_map.get(k, k)
            if key in self.metrics:
                self.metrics[key].set_value(v)
        self._update_verdict(motion)


class ObjectsPage(QWidget):
    def __init__(self, model: Model, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._cache_manager = cache_manager
        self._current_mode = "scene"

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metrics = {}

        names = ["duration_sec", "persistence", "norm_avg_velocity", "frames_count", ]

        panel_layout = QVBoxLayout()

        for metric_name in names:
            row = MetricRow(metric_name,
                            THRESHOLDS.get(metric_name).display_name,
                            THRESHOLDS.get(metric_name).subtitle)
            self.metrics[metric_name] = row
            panel_layout.addWidget(row)

        self.description = VerdictDescriptionWidget()
        self.panel = GroupWidget("Object Detection & Tracking", panel_layout)

        class_edit_layout = QHBoxLayout()
        self.class_display = QLineEdit()
        self.class_display.setReadOnly(True)
        self.class_display.setStyleSheet(
            """
            QLineEdit {
                border-radius: 8px;
                border: 1px solid #e5e7eb;
                padding: 0 6px;
                color: gray;
            }
            """
        )
        self.class_display.setFixedHeight(32)

        self.class_edit_btn = QPushButton("Edit")
        self.class_edit_btn.setStyleSheet(
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
        self.class_edit_btn.setFixedHeight(32)
        self.class_edit_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        class_edit_layout.addWidget(self.class_display)
        class_edit_layout.addWidget(self.class_edit_btn)
        class_edit_layout.setContentsMargins(0, 0, 0, 0)

        class_edit_widget = GroupWidgetPlain("Object classes", class_edit_layout)

        layout.addWidget(class_edit_widget)
        layout.addSpacing(12)
        layout.addWidget(self.panel)
        layout.addWidget(self.description)
        layout.addStretch(1)

        self.class_edit_btn.clicked.connect(self.on_edit_classes)
        self._model.media_changed.connect(self.on_current_media_changed)
        self._model.current_classes_changed.connect(self.on_changed_current_classes)

        self.on_changed_current_classes(self._model.current_classes)

        self.setLayout(layout)

    def on_edit_classes(self):
        pass

    def on_changed_current_classes(self, classes: list[str]):
        self.class_display.setText(", ".join(classes))

    def on_mode_changed(self, mode_idx: int):
        self._current_mode = "scene" if mode_idx == 0 else "video"

    def _objects_metric_names(self):
        return {"duration_sec", "persistence", "norm_avg_velocity", "frames_count", "domain_concentration", "norm_displacement_v"}

    def _update_verdict(self, metrics=None):
        if metrics is None:
            metrics = {name: row.value for name, row in self.metrics.items() if getattr(row, "value", None) is not None}
        verdict = get_description(metrics, "objects", self._current_mode)
        self.description.set_verdict(verdict)

    def restore_metrics_from_cache(self, cached_metrics: list[Metric | None]):
        def find_metric(metric_name: str) -> Metric | None:
            return next((metric for metric in cached_metrics if metric and metric.name == metric_name), None)

        for metric_name in self._objects_metric_names():
            if metric_name in self.metrics:
                c_metric = find_metric(metric_name)
                if c_metric is not None:
                    self.metrics[metric_name].set_value(c_metric)
        group_metrics = [m for m in (cached_metrics or []) if m and getattr(m, "name", None) in self._objects_metric_names()]
        self._update_verdict(group_metrics)

    def set_empty_metrics(self):
        for metric_name in self._objects_metric_names():
            if metric_name in self.metrics:
                self.metrics[metric_name].reset()
        self._update_verdict([])

    def on_current_media_changed(self, video: VideoWrapper):
        pass

    def update_scene(self, scene):
        ob = scene.get("objects_metrics") or {}
        for k, v in ob.items():
            target = "norm_displacement" if k == "norm_displacement_v" else k
            if target in self.metrics:
                self.metrics[target].set_value(v)
        self._update_verdict(ob)


class DiversityPage(QWidget):
    def __init__(self, model: Model, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._cache_manager = cache_manager
        self._current_mode = "scene"

        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        self.metrics = {}

        self.names = ["duration_sec", "persistence", "norm_avg_velocity", "frames_count", "clip_ics"]
        panel_layout = QVBoxLayout()

        for metric_name in self.names:
            row = MetricRow(metric_name,
                            THRESHOLDS.get(metric_name).display_name,
                            THRESHOLDS.get(metric_name).subtitle)
            self.metrics[metric_name] = row
            panel_layout.addWidget(row)

        self.description = VerdictDescriptionWidget()
        self.panel = GroupWidget("Diversity & Distinguishability", panel_layout)

        layout.addWidget(self.panel)
        layout.addWidget(self.description)
        layout.addStretch()

        self._model.media_changed.connect(self.on_current_media_changed)

    def on_mode_changed(self, mode_idx: int):
        self._current_mode = "scene" if mode_idx == 0 else "video"
        for metric_name in ("clip_ics",):
            if self._current_mode == "scene":
                self.metrics[metric_name].set_value("only for video")
                self.metrics[metric_name].setEnabled(False)
            else:
                self.metrics[metric_name].reset()
                self.metrics[metric_name].setEnabled(True)

    def _diversity_metric_names(self):
        return {"duration_sec", "persistence", "norm_avg_velocity", "frames_count", "clip_ics"}

    def _update_verdict(self, metrics=None):
        if metrics is None:
            metrics = {name: row.value for name, row in self.metrics.items() if getattr(row, "value", None) is not None}
        score = get_group_score(metrics)
        verdict = get_description(metrics, "diversity", score, self._current_mode)
        if verdict is not None and verdict.warning_line is not None and "detected" in verdict.warning_line:
            for metric_name in self._diversity_metric_names():
                if metric_name in self.metrics:
                    self.metrics[metric_name].reset()
        self.description.set_verdict(verdict)

    def restore_metrics_from_cache(self, cached_metrics: list[Metric | None]):
        def find_metric(metric_name: str) -> Metric | None:
            return next((metric for metric in cached_metrics if metric and metric.name == metric_name), None)

        for metric_name in self._diversity_metric_names():
            if metric_name in ("clip_ics", ):
                if self._current_mode == "scene":
                    self.metrics[metric_name].set_value("only for video")
                    self.metrics[metric_name].setEnabled(False)
                    continue
                else:
                    self.metrics[metric_name].set_value(None)
                    self.metrics[metric_name].setEnabled(True)
            if metric_name in self.metrics:
                c_metric = find_metric(metric_name)
                if c_metric is not None and c_metric.value is not None:
                    self.metrics[metric_name].set_value(c_metric)
        group_metrics = [m for m in (cached_metrics or []) if m and getattr(m, "name", None) in self._diversity_metric_names()]
        self._update_verdict(group_metrics)

    def set_empty_metrics(self):
        for metric_name in self._diversity_metric_names():
            if metric_name in self.metrics:
                self.metrics[metric_name].reset()
        self._update_verdict([])

    def on_current_media_changed(self, video: VideoWrapper):
        pass

    def _reset_current_metrics(self):
        for metric in self.metrics.values():
            metric.set_value(None)


# ================== ANALYSIS TAB ==================
class AnalysisTab(QWidget):
    report_clicked = Signal()
    tyres_compute_clicked = Signal()

    def __init__(self, model: Model, analysis_manager: AnalysisManager, cache_manager: CacheManager):
        super().__init__()

        self._model = model
        self._analysis_manager = analysis_manager
        self._cache_manager = cache_manager

        self.current_data = None

        main = QVBoxLayout(self)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)
        self._mode = "scene"
        self.mode_switch = ModeSwitch(("Scene-level", "Video-level"))
        self.mode_switch.idx_changed.connect(self.on_mode_changed)

        main.addWidget(self.mode_switch)
        main.addSpacing(6)

        # ===== Tab buttons =====

        self.buttons = [
            TabButton("Overall"),
            TabButton("Quality"),
            TabButton("Motion"),
            TabButton("Diversity"),
        ]

        grid = QGridLayout()
        grid.setSpacing(8)

        for i, b in enumerate(self.buttons):
            grid.addWidget(b, i // 2, i % 2)

        main.addLayout(grid)
        main.addSpacing(12)

        # ===== Pages =====

        self.stack = QStackedWidget()

        self.overall = OverallPage(self._model, self._cache_manager)
        self.quality = QualityPage(self._model, self._cache_manager)
        self.motion = MotionPage(self._model, self._cache_manager)
        self.diversity = DiversityPage(self._model, self._cache_manager)

        self.stack.addWidget(self.overall)
        self.stack.addWidget(self.quality)
        self.stack.addWidget(self.motion)
        self.stack.addWidget(self.diversity)

        self.compute_btn = QPushButton("Compute metrics")
        self.compute_btn.setStyleSheet(
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
        self.compute_btn.setFixedHeight(32)
        self.compute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.compute_btn.clicked.connect(self.on_compute_clicked)

        self.export_btn = QPushButton("Report")
        self.export_btn.setStyleSheet(
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
        self.export_btn.setFixedHeight(32)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # self.export_btn.setEnabled(False)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.compute_btn)
        btn_layout.addWidget(self.export_btn)
        main.addWidget(self.stack)

        main.addLayout(btn_layout)

        self._no_load = True

        self.loader = LoadingOverlay(self)
        self.loader.raise_()

        # ===== Connect =====
        self.export_btn.clicked.connect(lambda: self.report_clicked.emit())
        self.compute_btn.clicked.connect(self.on_compute_clicked)
        self.mode_switch.idx_changed.connect(self.quality.on_mode_changed)
        self.mode_switch.idx_changed.connect(self.motion.on_mode_changed)
        self.mode_switch.idx_changed.connect(self.diversity.on_mode_changed)

        for i, b in enumerate(self.buttons):
            b.clicked.connect(lambda _, x=i: self.switch(x))

        self.export_btn.clicked.connect(self.export)
        self.switch(0)

        self._model.media_changed.connect(self.on_video_changed)
        self._model.current_scene_changed.connect(self.on_scene_changed)

    @Slot(SceneSegment)
    def on_scene_changed(self, scene_segment: SceneSegment):
        if self._no_load and "tyres" in self._model.current_media.path_to_file.name:
            self.quality.set_empty_metrics()
            self.motion.set_empty_metrics()
            self.diversity.set_empty_metrics()
            self.overall.set_empty_metrics()
            return
        if self._model.current_tab != "analysis" or self._mode != "scene":
            return
        cached_metrics = self._cache_manager.get_scene_cached_metrics(
            self._model.current_media.path_to_file,
            scene_segment
        )
        print(cached_metrics)
        if cached_metrics is None:
            self.quality.set_empty_metrics()
            self.motion.set_empty_metrics()
            self.diversity.set_empty_metrics()
            self.overall.set_empty_metrics()
        else:
            self.quality.restore_metrics_from_cache(cached_metrics)
            self.motion.restore_metrics_from_cache(cached_metrics)
            self.diversity.restore_metrics_from_cache(cached_metrics)
            self.overall.restore_metrics_from_cache(
                cached_metrics,
                self.quality.names,
                self.motion.names,
                self.diversity.names
            )

    def on_empty_scene(self):
        self.quality.set_empty_metrics()
        self.motion.set_empty_metrics()
        self.diversity.set_empty_metrics()
        self.overall.set_empty_metrics()

    def attach_video_metrics(self):
        if self._no_load and "tyres" in self._model.current_media.path_to_file.name:
            self.quality.set_empty_metrics()
            self.motion.set_empty_metrics()
            self.diversity.set_empty_metrics()
            self.overall.set_empty_metrics()
            return
        video_scenes = self._model.scenes
        if video_scenes is None or not video_scenes:
            return
        video_level_metrics = self._cache_manager.get_video_cached_metrics(
            self._model.current_media.path_to_file,
            video_scenes
        )
        if video_level_metrics is None:
            self.quality.set_empty_metrics()
            self.motion.set_empty_metrics()
            self.diversity.set_empty_metrics()
            self.overall.set_empty_metrics()
        else:
            self.quality.restore_metrics_from_cache(video_level_metrics)
            self.motion.restore_metrics_from_cache(video_level_metrics)
            self.diversity.restore_metrics_from_cache(video_level_metrics)
            self.overall.restore_metrics_from_cache(
                video_level_metrics,
                self.quality.names,
                self.motion.names,
                self.diversity.names
            )

    @Slot(VideoWrapper)
    def on_video_changed(self, video_wrapper: VideoWrapper):
        if self._no_load and "tyres" in self._model.current_media.path_to_file.name:
            self.quality.set_empty_metrics()
            self.motion.set_empty_metrics()
            self.diversity.set_empty_metrics()
            self.overall.set_empty_metrics()
            return
        if self._mode == "video":
            self.attach_video_metrics()

    @Slot(int)
    def on_mode_changed(self, mode_idx: int):
        self.overall.set_empty_metrics()
        self.quality.set_empty_metrics()
        self.motion.set_empty_metrics()
        self.diversity.set_empty_metrics()
        if mode_idx == 0:  # Scene-level
            self._mode = "scene"
            self.compute_btn.setEnabled(True)
            self.on_scene_changed(self._model.current_scene)
        else:
            self._mode = "video"
            self.compute_btn.setEnabled(False)
            self.attach_video_metrics()

    def resizeEvent(self, e):
        self.loader.resize(self.size())
        super().resizeEvent(e)

    def on_compute_clicked(self):
        if "tyres" in self._model.current_media.path_to_file.name:
            self.tyres_compute_clicked.emit()
            self._no_load = False
            self.attach_video_metrics()
        # if self._mode == "scene":
        #     task = ComputeSceneMetricsTask(
        #         video_path=self._model.current_media.path_to_file,
        #         scene=self._model.current_scene,
        #         cache_manager=self._cache_manager,
        #         object_classes=self._model.current_classes
        #     )
        #     task.partial_result.connect(self.on_partial_result)
        #     self._analysis_manager.submit_task(task)
        # else:
        #     pass

    def on_partial_result(self, path_to_video: str | Path, scene_segment: SceneSegment,
                          task_name: str, data: dict):
        metric_group = data.get("group", "")

        metrics = []
        for metric_name, metric in data.get("data", {}).items():
            metrics.append(metric)

        self._cache_manager.save_scene_metrics_to_cache(path_to_video, scene_segment, metrics)

        if metric_group == "quality":
            for metric in metrics:
                if metric.name in self.quality.metrics:
                    self.quality.metrics[metric.name].set_value(metric)
            self.quality._update_verdict(metrics)

        if metric_group == "motion":
            name_map = {"flow": "optical_flow", "smoothness": "motion_smoothness"}
            for metric in metrics:
                key = name_map.get(metric.name, metric.name)
                if key in self.motion.metrics:
                    self.motion.metrics[key].set_value(metric)
            self.motion._update_verdict(metrics)

        if metric_group == "diversity":
            for metric in metrics:
                if metric.name in self.diversity.metrics:
                    self.diversity.metrics[metric.name].set_value(metric)
            self.diversity._update_verdict(metrics)

    # ===== Export =====
    def export(self):
        if not self.current_data:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            "analysis.txt",
            "Text Files (*.txt)"
        )

        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:

            d = self.current_data

            f.write(f"Scene ID: {d['id']}\n")
            f.write(f"Type: {d['type']}\n\n")

            f.write("=== Quality ===\n")
            for k, v in d['quality_metrics'].items():
                f.write(f"{k}: {v:.3f}\n")

            f.write("\n=== Motion ===\n")
            for k, v in d['motion_metrics'].items():
                f.write(f"{k}: {v:.3f}\n")

        QMessageBox.information(self, "Export", "Report exported successfully.")

    # ===== Tab switch =====

    def switch(self, index):
        for b in self.buttons:
            b.setChecked(False)

        self.buttons[index].setChecked(True)
        self.stack.setCurrentIndex(index)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Dummy model and cache manager for testing
    model = Model()
    cache_manager = CacheManager(Path("./cache"))
    analysis_manager = AnalysisManager()

    widget = MetricRow("Test Metric", "This is a subtitle")
    widget.show()

    sys.exit(app.exec())


# Scene: Overall, Quality, Motion, Objects
# Video: Overall, Quality, Motion, Diversity, Objects
