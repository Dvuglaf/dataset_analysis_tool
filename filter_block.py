from PySide6.QtWidgets import *
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config import FILTERS


class FilterBlock(QFrame):
    def __init__(self, name, config, panel, parent=None):
        super().__init__(parent)

        self.panel = panel
        self.name = name
        self.config = config
        self.param_widgets = {}

        self.setObjectName("FilterBlock")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(6)
        self.main_layout.setContentsMargins(8, 6, 8, 6)

        header = QHBoxLayout()
        header.setSpacing(12)

        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(self.on_toggle)

        title = QLabel(name)
        title.setFont(QFont("Arial", 12, QFont.Bold))

        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.clicked.connect(self.show_help)

        help_btn.setObjectName("HelpBtn")

        header.addWidget(self.checkbox)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(help_btn)

        self.main_layout.addLayout(header)

        # ========================
        # Parameters
        # ========================

        self.params_widget = QWidget()
        self.params_layout = QFormLayout(self.params_widget)

        self.params_layout.setContentsMargins(24, 4, 4, 4)
        self.params_layout.setHorizontalSpacing(8)
        self.params_layout.setVerticalSpacing(4)

        self.build_parameters()

        self.params_widget.setVisible(False)

        self.main_layout.addWidget(self.params_widget)

        # ========================
        # Style
        # ========================

        self.setStyleSheet("""
            QFrame#FilterBlock {
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #fafafa;
            }

            QPushButton#HelpBtn {
                border: 1px solid #bbb;
                border-radius: 10px;
                background: #f0f0f0;
            }

            QPushButton#HelpBtn:hover {
                background: #e0e0e0;
            }
        """)

    def build_parameters(self):

        for pname, pdata in self.config["parameters"].items():

            widget = pdata["widget"]()
            widget.setFocusPolicy(Qt.NoFocus)

            widget.setMinimum(pdata["min"])
            widget.setMaximum(pdata["max"])
            widget.setValue(pdata["default"])
            widget.setToolTip(pdata["help"])

            if isinstance(widget, QDoubleSpinBox):
                widget.setDecimals(3)
                widget.setSingleStep(0.1)

            label = QLabel(pname)

            self.params_layout.addRow(label, widget)

            self.param_widgets[pname] = widget

    def on_toggle(self):
        enabled = self.checkbox.isChecked()
        self.panel.reorder_blocks()
        self.params_widget.setVisible(enabled)

    def show_help(self):

        QMessageBox.information(
            self,
            self.name,
            self.config["help"]
        )

    def is_enabled(self):
        return self.checkbox.isChecked()

    def set_enabled(self, value: bool):
        self.checkbox.setChecked(value)

    def get_params(self):
        return {
            name: widget.value()
            for name, widget in self.param_widgets.items()
        }


class FilterPanel(QWidget):

    def __init__(self, filters_config, parent=None):
        super().__init__(parent)

        self.blocks = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(6)
        self.main_layout.setContentsMargins(6, 6, 6, 6)

        btn_layout = QHBoxLayout()

        enable_all = QPushButton("Enable All")
        disable_all = QPushButton("Disable All")

        enable_all.clicked.connect(self.enable_all)
        disable_all.clicked.connect(self.disable_all)

        btn_layout.addWidget(enable_all)
        btn_layout.addWidget(disable_all)
        btn_layout.addStretch()

        self.main_layout.addLayout(btn_layout)

        self.filters_layout = QVBoxLayout()
        self.filters_layout.setSpacing(6)

        self.main_layout.addLayout(self.filters_layout)
        self.main_layout.addStretch()

        # Build filters
        for name, cfg in filters_config.items():
            block = FilterBlock(name, cfg, self)
            self.blocks.append(block)
            self.filters_layout.addWidget(block)

    def reorder_blocks(self):

        enabled = []
        disabled = []

        for b in self.blocks:
            if b.is_enabled():
                enabled.append(b)
            else:
                disabled.append(b)

        ordered = enabled + disabled

        # Clear layout
        while self.filters_layout.count():
            self.filters_layout.takeAt(0)

        # Reinsert
        for b in ordered:
            self.filters_layout.addWidget(b)

    def enable_all(self):
        for b in self.blocks:
            b.set_enabled(True)

    def disable_all(self):
        for b in self.blocks:
            b.set_enabled(False)

    def get_enabled_filters(self):

        result = {}

        for b in self.blocks:
            if b.is_enabled():
                result[b.name] = b.get_params()

        return result


if __name__ == "__main__":

    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    win = QWidget()
    win.setWindowTitle("Filter Panel")
    win.resize(420, 520)

    layout = QVBoxLayout(win)

    panel = FilterPanel(FILTERS)
    layout.addWidget(panel)

    btn = QPushButton("Print Active Filters")

    def dump():
        print(panel.get_enabled_filters())

    btn.clicked.connect(dump)

    layout.addWidget(btn)

    win.show()

    sys.exit(app.exec())