from PySide6.QtCore import Qt, Signal, QMargins
from PySide6.QtWidgets import QWidget, QApplication, QSlider, QVBoxLayout, QHBoxLayout, QLabel, \
    QLineEdit, QSpinBox, \
    QDoubleSpinBox, QLayout, QFrame, QFormLayout, QPushButton, QButtonGroup


def to_value_cast(value: str | int | float, to: type):
    if isinstance(value, str):
        value = value.replace(',', '.')
    if to == int:
        return int(float(value))
    elif to == float:
        return float(value)
    else:
        raise ValueError(f"Unsupported type: {to}")


class FloatSlider(QSlider):

    floatValueChanged = Signal(float)

    def __init__(self, decimals=2, *args, **kwargs):
        super(FloatSlider, self).__init__(*args, **kwargs)
        self._multi = 10 ** decimals

        self.valueChanged.connect(self.on_value_changed)

    def on_value_changed(self):
        value = float(super(FloatSlider, self).value()) / self._multi
        self.floatValueChanged.emit(value)

    def value(self) -> float:
        return float(super(FloatSlider, self).value()) / self._multi

    def setMinimum(self, value: float):
        return super(FloatSlider, self).setMinimum(int(value * self._multi))

    def setMaximum(self, value: float):
        return super(FloatSlider, self).setMaximum(int(value * self._multi))

    def setSingleStep(self, value: float):
        return super(FloatSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(FloatSlider, self).singleStep()) / self._multi

    def setValue(self, value: float):
        super(FloatSlider, self).setValue(int(value * self._multi))


class ValueSlider(QWidget):
    def __init__(self, name: str, min_value: int | float = 0, max_value: int | float = 100, default: int | float = 50, parent=None):
        super().__init__(parent)
        self.general_type = int
        if any(isinstance(v, float) for v in (min_value, max_value, default)):
            self.general_type = float

        if self.general_type == int:
            self.slider = QSlider(Qt.Orientation.Horizontal, self)
            self.slider.setRange(min_value, max_value)
            self.slider.setValue(default)
            self.slider.valueChanged.connect(self.on_slider_value_changed)

            self.value_edit = QSpinBox()
            self.value_edit.setValue(default)
            self.value_edit.setRange(min_value, max_value)
        else:
            self.slider = FloatSlider(decimals=2, orientation=Qt.Orientation.Horizontal, parent=self)
            self.slider.setMinimum(min_value)
            self.slider.setMaximum(max_value)
            self.slider.setSingleStep(0.05)
            self.slider.setValue(default)
            self.slider.floatValueChanged.connect(self.on_slider_value_changed)

            self.value_edit = QDoubleSpinBox(decimals=2)
            self.value_edit.setValue(default)
            self.value_edit.setRange(min_value, max_value)
            self.value_edit.setSingleStep(0.05)

        # self.slider.setTickPosition(QSlider.TicksBelow)
        # self.slider.setTickInterval(50)
        self.value_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.value_edit.textChanged.connect(self.on_edit_value_changed)

        header_layout = QHBoxLayout()
        name_label = QLabel(name)

        header_layout.addWidget(name_label, 1)
        header_layout.addWidget(self.value_edit, 0)

        range_layout = QHBoxLayout()
        min_value_label = QLabel(str(min_value))
        max_value_label = QLabel(str(max_value))
        range_layout.addWidget(min_value_label, 1)
        range_layout.addWidget(max_value_label, 0)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.addLayout(header_layout)
        layout.addWidget(self.slider)
        layout.addLayout(range_layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setLayout(layout)

    def on_slider_value_changed(self, value):
        self.value_edit.setValue(to_value_cast(value, self.general_type))

    def on_edit_value_changed(self, value):
        self.slider.setValue(to_value_cast(value, self.general_type))

    def get_value(self):
        return self.slider.value()


# Общий стиль для заголовка (чтобы менять в одном месте)
TITLE_STYLE = """
    QLabel {
        font-weight: bold; 
        font-size: 15px;
        color: #2c3e50;
        margin-left: 2px;
    }
"""


# Triangle symbols for collapsible header (expanded / collapsed)
COLLAPSE_ICON_EXPANDED = "\u25BC"   # ▼
COLLAPSE_ICON_COLLAPSED = "\u25B6"  # ▶


class GroupWidget(QWidget):
    """Group with border and light background. Optional collapsible header."""

    def __init__(self, name: str, content_layout: QLayout, *, collapsible: bool = False, default_expanded: bool = True):
        super().__init__()
        self._expanded = default_expanded
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # 1. Header row (title + optional collapse button)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._collapse_btn = None
        if collapsible:
            self._collapse_btn = QPushButton(COLLAPSE_ICON_EXPANDED if default_expanded else COLLAPSE_ICON_COLLAPSED)
            self._collapse_btn.setFixedSize(22, 22)
            self._collapse_btn.setFlat(True)
            self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._collapse_btn.setStyleSheet("""
                QPushButton {
                    border: none;
                    background: transparent;
                    font-size: 10px;
                    color: #2c3e50;
                }
                QPushButton:hover { color: #1a73e8; }
            """)
            self._collapse_btn.clicked.connect(self._toggle_collapsed)
            header_row.addWidget(self._collapse_btn, 0)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(TITLE_STYLE)
        header_row.addWidget(self.name_label, 1)
        main_layout.addLayout(header_row, 0)

        # 2. Container frame
        self.container_frame = QFrame()
        self.container_frame.setObjectName("GroupContainer")
        self.container_frame.setStyleSheet("""
            QFrame#GroupContainer {
                border: 1px solid #dcdde1;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)
        content_layout.setContentsMargins(10, 10, 10, 10)
        self.container_frame.setLayout(content_layout)
        main_layout.addWidget(self.container_frame, 1)
        if collapsible and not default_expanded:
            self.container_frame.hide()

    def _toggle_collapsed(self):
        self._expanded = not self._expanded
        self.container_frame.setVisible(self._expanded)
        if self._collapse_btn:
            self._collapse_btn.setText(COLLAPSE_ICON_EXPANDED if self._expanded else COLLAPSE_ICON_COLLAPSED)


class GroupWidgetPlain(QWidget):
    """Group without frame, with small content indent. Optional collapsible header."""

    def __init__(self, name: str, content_layout: QLayout, *, collapsible: bool = False, default_expanded: bool = True):
        super().__init__()
        self._expanded = default_expanded
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        self._collapse_btn = None
        if collapsible:
            self._collapse_btn = QPushButton(COLLAPSE_ICON_EXPANDED if default_expanded else COLLAPSE_ICON_COLLAPSED)
            self._collapse_btn.setFixedSize(22, 22)
            self._collapse_btn.setFlat(True)
            self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._collapse_btn.setStyleSheet("""
                QPushButton { border: none; background: transparent; font-size: 10px; color: #2c3e50; }
                QPushButton:hover { color: #1a73e8; }
            """)
            self._collapse_btn.clicked.connect(self._toggle_collapsed_plain)
            header_row.addWidget(self._collapse_btn, 0)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(TITLE_STYLE)
        header_row.addWidget(self.name_label, 1)
        main_layout.addLayout(header_row, 0)

        self.content_container = QWidget()
        if content_layout.contentsMargins() != QMargins(0, 0, 0, 0):
            content_layout.setContentsMargins(12, 0, 0, 0)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_container.setLayout(content_layout)
        main_layout.addWidget(self.content_container, 1)
        if collapsible and not default_expanded:
            self.content_container.hide()

    def _toggle_collapsed_plain(self):
        self._expanded = not self._expanded
        self.content_container.setVisible(self._expanded)
        if self._collapse_btn:
            self._collapse_btn.setText(COLLAPSE_ICON_EXPANDED if self._expanded else COLLAPSE_ICON_COLLAPSED)


class ModeSwitch(QWidget):
    idx_changed = Signal(int)

    def __init__(self, names: tuple[str, str], parent=None):
        super().__init__(parent)
        self._current_idx = 0

        self.setFixedHeight(34)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.left_btn = QPushButton(names[0])
        self.right_btn = QPushButton(names[1])

        for b in (self.left_btn, self.right_btn):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(28)

        self.left_btn.setChecked(True)

        layout.addWidget(self.left_btn)
        layout.addWidget(self.right_btn)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        self.group.addButton(self.left_btn, 0)
        self.group.addButton(self.right_btn, 1)

        self._apply_style()

        self.group.idClicked.connect(self._on_clicked)

    def _apply_style(self):
        self.setStyleSheet("""
        QPushButton {
            border: none;
            background: transparent;
            border-radius: 8px;
            padding: 4px 12px;
            font-size: 13px;
            color: #374151;
        }

        QPushButton:checked {
            background: #2563eb;
            color: white;
            font-weight: 600;
        }

        QWidget {
            background: #e5e7eb;
            border-radius: 10px;
        }
        """)

    def _on_clicked(self, idx: int):
        if idx != self._current_idx:
            self._current_idx = idx
            self.idx_changed.emit(idx)

    def current_idx(self) -> int:
        return 0 if self.left_btn.isChecked() else 1


def add_stretch_row(layout: QFormLayout, w1: QWidget, w2: QWidget):
    row = QHBoxLayout()
    row.addWidget(w1, 1)
    row.addWidget(w2)
    layout.addRow(row)


if __name__ == "__main__":
    import sys
    app = QApplication([])
    # widget = ValueSlider("Test Float", 0.0, 1.0, 0.05)
    # widget.resize(300, 100)
    # layout = QVBoxLayout()
    #
    # for i in range(4):
    #     row_layout = QHBoxLayout()
    #     row_layout.addWidget(QLabel(f"Parameter{i}:"), 1)
    #     row_layout.addWidget(QLabel(f"Value{i}"), 0)
    #     layout.addLayout(row_layout)
    # widget = GroupWidget("Test Group", layout)
    widget = ModeSwitch(("Mode A", "Mode B"))
    widget.idx_changed.connect(lambda value: print(value))
    widget.show()
    sys.exit(app.exec())