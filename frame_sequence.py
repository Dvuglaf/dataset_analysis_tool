from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class FrameGridView(QAbstractScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._area = QAbstractScrollArea()
        self._pixmaps = []
        self._metrics = []
        self._removed = []

        self._item_rects = []
        self._metric_rects = {}

        self._hover_index = -1
        self._hover_on_metrics = False

        self._selected = set()

        self._cell_w = 160
        self._cell_h = 120
        self._margin = 12
        self._spacing = 12

        self._cols = 1
        self._rows = 0

        self._palette = {
            "psnr": QColor(0, 200, 0),
            "blur": QColor(0, 160, 255),
            "motion": QColor(255, 180, 0),
            "noise": QColor(200, 0, 200)
        }

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.verticalScrollBar().valueChanged.connect(self.viewport().update)

    # ---------------- DATA ----------------

    def set_frames(self, pixmaps, metrics=None, removed=None):
        self._pixmaps = list(pixmaps)
        n = len(self._pixmaps)

        self._metrics = metrics or [{} for _ in range(n)]
        self._removed = removed or [False] * n

        self._selected.clear()

        self._recalc_layout()
        self.viewport().update()

    # ---------------- LAYOUT ----------------

    def resizeEvent(self, e):
        self._recalc_layout()
        super().resizeEvent(e)

    def _recalc_layout(self):
        w = self.viewport().width()

        self._cols = max(1, w // (self._cell_w + self._spacing))
        self._rows = (len(self._pixmaps) + self._cols - 1) // self._cols

        total_h = (
            self._rows * (self._cell_h + self._spacing)
            + self._margin * 2
        )

        self.verticalScrollBar().setRange(0, max(0, total_h - self.viewport().height()))
        self._item_rects.clear()

        y = self._margin

        for r in range(self._rows):
            x = self._margin

            for c in range(self._cols):
                i = r * self._cols + c

                if i >= len(self._pixmaps):
                    break

                rect = QRect(x, y, self._cell_w, self._cell_h)
                self._item_rects.append(rect)

                x += self._cell_w + self._spacing

            y += self._cell_h + self._spacing

    def paintEvent(self, e):
        p = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)

        p.fillRect(self.viewport().rect(), QColor(245, 245, 245))

        scroll = self.verticalScrollBar().value()

        self._metric_rects.clear()

        for i, rect in enumerate(self._item_rects):
            r = rect.translated(0, -scroll)

            if not self.viewport().rect().intersects(r):
                continue

            self._draw_item(p, r, i)

    def _draw_item(self, p, rect, index):
        radius = 8

        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black)
        p.drawRoundedRect(rect, radius, radius)

        pm = self._pixmaps[index]

        if pm:
            p.drawPixmap(rect, pm)

        if self._removed[index]:
            p.setBrush(QColor(255, 0, 0, 80))
            p.drawRoundedRect(rect, radius, radius)

        self._draw_top_bar(p, rect, index)

        bars = self._draw_metric_bars(p, rect, self._metrics[index])
        self._metric_rects[index] = bars

        if self._hover_index == index and self._hover_on_metrics:
            self._draw_metric_overlay(p, rect, self._metrics[index])

        if index in self._selected:
            pen = QPen(QColor(0, 160, 255), 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), radius, radius)

        if self._hover_index == index:
            pen = QPen(QColor(200, 200, 200), 1)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rect.adjusted(2, 2, -2, -2), radius, radius)

    def _draw_top_bar(self, p, rect, index):
        h = 22
        bar = QRect(rect.left(), rect.top(), rect.width(), h)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 140))
        p.drawRoundedRect(bar, 8, 8)

        p.setPen(Qt.white)
        p.drawText(bar, Qt.AlignCenter, f"#{index}")

    def _draw_metric_bars(self, p, rect, metrics):
        if not metrics:
            return QRect()

        bar_w = 12
        bar_h = 6
        gap = 4

        items = list(metrics.items())
        count = len(items)

        total = count * bar_w + (count - 1) * gap

        y = rect.bottom() - bar_h - 8
        x = rect.center().x() - total // 2

        bars_rect = QRect(x, y, total, bar_h)

        for name, data in items:
            base = self._palette.get(name, QColor(180, 180, 180))
            color = self._metric_color(
                data["value"],
                data["threshold"],
                data["better"],
                base
            )

            p.setPen(Qt.NoPen)
            p.setBrush(color)

            r = QRect(x, y, bar_w, bar_h)
            p.drawRoundedRect(r, 2, 2)

            x += bar_w + gap

        return bars_rect

    def _metric_color(self, value, threshold, better, base):
        if better == "high":
            ok = value >= threshold
            near = value >= threshold * 0.9
        else:
            ok = value <= threshold
            near = value <= threshold * 1.1

        if ok:
            a = 230
        elif near:
            a = 150
        else:
            a = 70

        c = QColor(base)
        c.setAlpha(a)
        return c

    def _draw_metric_overlay(self, p, rect, metrics):
        overlay = rect.adjusted(6, 26, -6, -26)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 210))
        p.drawRoundedRect(overlay, 10, 10)

        p.setPen(Qt.white)
        fm = p.fontMetrics()

        y = overlay.top() + 20
        x = overlay.left() + 12

        for name, data in metrics.items():
            v = round(data["value"], 2)
            t = round(data["threshold"], 2)

            txt = f"{name.upper()}: {v} / {t}"

            p.drawText(x, y, txt)
            y += fm.height() + 6

    def mouseMoveEvent(self, e):
        pos = e.pos()
        scroll = self.verticalScrollBar().value()

        self._hover_index = -1
        self._hover_on_metrics = False

        for i, rect in enumerate(self._item_rects):
            r = rect.translated(0, -scroll)
            if not r.contains(pos):
                continue

            self._hover_index = i

            m = self._metric_rects.get(i)
            if m and m.translated(0, -scroll).contains(pos):
                self._hover_on_metrics = True
            break

        self.viewport().update()

    def leaveEvent(self, e):
        self._hover_index = -1
        self.viewport().update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return

        pos = e.pos()
        scroll = self.verticalScrollBar().value()

        for i, rect in enumerate(self._item_rects):
            r = rect.translated(0, -scroll)
            if not r.contains(pos):
                continue

            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                if i in self._selected:
                    self._selected.remove(i)
                else:
                    self._selected.add(i)
            else:
                self._selected = {i}

            self.viewport().update()
            break

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            for i in self._selected:
                self._removed[i] = True

            self.viewport().update()

        super().keyPressEvent(e)


class FrameTimelineWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._area = FrameGridView()

        layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        header_name = QLabel("Frame View")
        self._header_total_frames_info = QLabel("Total: 0 frames")
        self._header_removed_frames_info = QLabel("Removed: 0")
        self._header_kept_frames_info = QLabel("Kept: 0")
        header_layout.addWidget(header_name, 1)
        header_layout.addWidget(self._header_total_frames_info)
        header_layout.addWidget(self._header_removed_frames_info)
        header_layout.addWidget(self._header_kept_frames_info)

        layout.addLayout(header_layout)
        layout.addWidget(self._area)

        self.setLayout(layout)

    def set_frames(self, pixmaps, metrics=None, removed=None):
        self._area.set_frames(pixmaps, metrics, removed)

        total = len(pixmaps)
        removed_count = sum(removed) if removed else 0
        kept_count = total - removed_count

        self._header_total_frames_info.setText(f"Total: {total} frames")
        self._header_removed_frames_info.setText(f"Removed: {removed_count}")
        self._header_kept_frames_info.setText(f"Kept: {kept_count}")


if __name__ == '__main__':
    import sys
    import random

    app = QApplication(sys.argv)

    w = FrameTimelineWidget()

    pix = []
    mets = []

    for i in range(300):
        pm = QPixmap(240, 160)
        pm.fill(QColor(70 + i % 100, 90, 130))
        pix.append(pm)
        m = {
            "psnr": {
                "value": random.uniform(25, 40),
                "threshold": 30,
                "better": "high"
            },
            "blur": {
                "value": random.uniform(0.05, 0.4),
                "threshold": 0.2,
                "better": "low"
            },
            "motion": {
                "value": random.uniform(0.0, 0.6),
                "threshold": 0.3,
                "better": "low"
            }
        }

        mets.append(m)

    removed = [(i % 11 == 0) for i in range(300)]

    # Fix the argument order: metrics should be the second argument, removed the third
    w.set_frames(pix, mets, removed)

    w.resize(1100, 600)
    w.show()

    sys.exit(app.exec_())
