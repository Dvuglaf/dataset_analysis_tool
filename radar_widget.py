import sys
import math
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                               QPushButton, QHBoxLayout, QLabel, QFrame)
from PySide6.QtCore import Qt, QVariantAnimation, QPointF, QEasingCurve, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPolygonF, QPen, QFont, QBrush


class ScoreLabel(QWidget):
    """Маленький виджет строки статистики с цветовым индикатором"""

    def __init__(self, label_text):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self.name_lbl = QLabel(label_text)
        self.name_lbl.setStyleSheet(
            "font-weight: bold; color: #555; background: transparent; border: none;")  # Явно убираем рамку

        self.val_lbl = QLabel("—")
        self.val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.val_lbl.setFixedWidth(60)
        self.val_lbl.setStyleSheet(
            "font-family: 'Consolas'; background: transparent; font-weight: bold; font-size: 14px; border: none;")

        layout.addWidget(self.name_lbl)
        layout.addStretch()
        layout.addWidget(self.val_lbl)

    def update_score(self, value, color):
        if value is None:
            self.val_lbl.setText("—")
        else:
            self.val_lbl.setText(f"{value:.2f}")
        self.val_lbl.setStyleSheet(
            f"color: {color.name()}; background: transparent; font-weight: bold; font-family: 'Consolas'; font-size: 14px; border: none;")


class ModernRadarWidget(QWidget):
    valuesChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.labels = ["Quality", "Motion", "Diversity"]
        self.current_values = [0.0, 0.0, 0.0]
        self.target_values = [0.0, 0.0, 0.0]
        self.start_values = [0.0, 0.0, 0.0]

        self.setMinimumSize(350, 350)  # Немного увеличим для подписей
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(1000)
        self.anim.setEasingCurve(QEasingCurve.OutQuart)
        self.anim.valueChanged.connect(self._update_anim)

    def get_gradient_color(self, val):
        if val is None:
            return QColor(0, 0, 0)
        val = max(0.0, min(1.0, val))
        if val < 0.5:
            ratio = val / 0.5
            r, g, b = 255, int(76 + 131 * ratio), int(76 - 12 * ratio)
        else:
            ratio = (val - 0.5) / 0.5
            r, g, b = int(255 - 209 * ratio), int(207 - 3 * ratio), int(64 + 49 * ratio)
        return QColor(r, g, b)

    def _update_anim(self, progress):
        for i in range(3):
            s, t = self.start_values[i], self.target_values[i]
            self.current_values[i] = s + (t - s) * progress
        self.valuesChanged.emit(self.current_values)
        self.update()

    def set_scores(self, q, m, d):
        if all(v is None for v in (q, m, d)):
            self.target_values = [0., 0., 0.]
        else:
            self.target_values = [q, m, d]
        self.start_values = list(self.current_values)
        self.anim.stop()
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        center = rect.center()
        radius = min(rect.width(), rect.height()) / 2 * 0.65  # Уменьшили радиус, чтобы текст влез

        # 1. Рисуем сетку
        painter.setPen(QPen(QColor(220, 220, 220), 1))
        for level in [0.2, 0.4, 0.6, 0.8, 1.0]:
            r = radius * level
            poly = QPolygonF()
            for i in range(3):
                angle = math.radians(i * 120 - 90)
                poly.append(
                    QPointF(center.x() + r * math.cos(angle), center.y() + r * math.sin(angle)))
            painter.drawPolygon(poly)

        # 2. Рисуем подписи углов (Quality, Motion, Diversity)
        painter.setPen(QPen(QColor(100, 100, 100)))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        for i, label in enumerate(self.labels):
            angle = math.radians(i * 120 - 90)
            # Выносим текст чуть дальше радиуса (1.2 от радиуса)
            text_radius = radius * 1.2
            x = center.x() + text_radius * math.cos(angle)
            y = center.y() + text_radius * math.sin(angle)

            # Центрирование текста относительно точки
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(label)
            th = metrics.height()
            painter.drawText(x - tw / 2, y + th / 4, label)

        # 3. Рисуем заполненную фигуру данных
        data_poly = QPolygonF()
        for i, val in enumerate(self.current_values):
            angle = math.radians(i * 120 - 90)
            r = radius * val
            data_poly.append(
                QPointF(center.x() + r * math.cos(angle), center.y() + r * math.sin(angle)))

        avg_val = sum(self.current_values) / 3
        base_color = self.get_gradient_color(avg_val)
        fill_color = QColor(base_color)
        fill_color.setAlpha(120)

        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(base_color, 2))
        painter.drawPolygon(data_poly)

        # 4. Точки на вершинах
        for i, pt in enumerate(data_poly):
            pt_color = self.get_gradient_color(self.current_values[i])
            painter.setBrush(pt_color)
            painter.setPen(QPen(Qt.white, 2))
            painter.drawEllipse(pt, 5, 5)


class StatisticWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #f9f9f9;")

        main_layout = QVBoxLayout(self)

        self.radar = ModernRadarWidget()
        main_layout.addWidget(self.radar)

        # ПАНЕЛЬ СТАТИСТИКИ
        self.info_panel = QFrame()
        self.info_panel.setObjectName("infoPanel")  # Устанавливаем ID для стилей
        self.info_panel.setStyleSheet("""
            QFrame#infoPanel { 
                background-color: white; 
                border-radius: 12px; 
                border: 1px solid #ddd;
            }
            QLabel { border: none; } 
        """)

        panel_layout = QVBoxLayout(self.info_panel)
        panel_layout.setSpacing(0)

        self.stat_rows = {
            "Quality": ScoreLabel("Quality"),
            "Motion": ScoreLabel("Motion"),
            "Diversity": ScoreLabel("Diversity")
        }

        for row in self.stat_rows.values():
            panel_layout.addWidget(row)

        main_layout.addWidget(self.info_panel)
        self.radar.valuesChanged.connect(self.update_panel)

    def update_panel(self, values):
        for i, key in enumerate(["Quality", "Motion", "Diversity"]):
            val = values[i]
            color = self.radar.get_gradient_color(val)
            self.stat_rows[key].update_score(val, color)

    def set_scores(self, q, m, d):
        print("set_scores", q, m, d)
        self.update_panel((q, m, d))
        self.radar.set_scores(q, m, d)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatsApp()
    window.show()
    window.radar.set_scores(0.85, 0.42, 0.68)
    sys.exit(app.exec())