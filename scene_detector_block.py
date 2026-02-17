import sys

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import *
from PySide6.QtGui import QFont, QIcon

from config import SCENES
from custom_types import SceneAlgoModel
from model import Model
from widgets import ValueSlider, GroupWidget, GroupWidgetPlain, add_stretch_row


def fix_label_width(form: QFormLayout):
    max_w = 0

    for i in range(form.rowCount()):
        item = form.itemAt(i, QFormLayout.LabelRole)
        if item and item.widget():
            max_w = max(max_w, item.widget().sizeHint().width())

    for i in range(form.rowCount()):
        item = form.itemAt(i, QFormLayout.LabelRole)
        if item and item.widget():
            item.widget().setFixedWidth(max_w)


# ------------------------
# Scene Detector Panel
# ------------------------
class SceneDetectorPanel(QFrame):
    def __init__(self, model: Model, detectors_config, parent=None):
        super().__init__(parent)

        self._model = model

        self.detectors_config = detectors_config
        self.param_widgets = {}

        self.setObjectName("SceneDetectorPanel")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(12)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        # ========================
        # Header
        # ========================

        header = QHBoxLayout()
        header.setSpacing(6)

        title = QLabel("Algorithm:")
        title.setFont(QFont("Arial", 12, QFont.Bold))

        self.combo = QComboBox()
        self.combo.addItems(detectors_config.keys())
        self.combo.currentTextChanged.connect(self.on_detector_changed)

        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.clicked.connect(self.show_help)
        help_btn.setObjectName("HelpBtn")

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.combo)
        header.addWidget(help_btn)

        # self.main_layout.addLayout(header)

        self.settings_layout = QVBoxLayout()
        self.settings_layout.setSpacing(0)
        self.params_layout = QFormLayout()
        self.params_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.params_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                            | Qt.AlignmentFlag.AlignTop)

        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_layout.setHorizontalSpacing(10)
        self.params_layout.setVerticalSpacing(6)

        self.settings_layout.addLayout(header)
        self.settings_layout.addSpacing(12)
        self.settings_layout.addLayout(self.params_layout)
        self.settings_layout.setContentsMargins(12, 0, 0, 0)
        self.params_widget = GroupWidgetPlain("Scene detection settings", self.settings_layout)
        self.main_layout.addWidget(self.params_widget, 1)

        detected_lbl = QLabel("Scenes detected:")
        self._detected_res_lbl = QLabel("–")
        avg_duration_lbl = QLabel("Avg scene duration:")
        self._avg_duration_res_lbl = QLabel("–")
        shortest_lbl = QLabel("Shortest scene:")
        self._shortest_res_lbl = QLabel("–")
        longest_lbl = QLabel("Longest scene:")
        self._longest_res_lbl = QLabel("–")

        detection_result_layout = QFormLayout()
        detection_result_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        detection_result_layout.setContentsMargins(0, 0, 0, 0)
        detection_result_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft
                                                 | Qt.AlignmentFlag.AlignTop)
        add_stretch_row(detection_result_layout, detected_lbl, self._detected_res_lbl)
        add_stretch_row(detection_result_layout, avg_duration_lbl, self._avg_duration_res_lbl)
        add_stretch_row(detection_result_layout, shortest_lbl, self._shortest_res_lbl)
        add_stretch_row(detection_result_layout, longest_lbl, self._longest_res_lbl)

        self._detection_results = GroupWidget("Detection Results", detection_result_layout)
        self._detection_results.setMaximumHeight(200)

        self.main_layout.addWidget(self._detection_results)

        dummy = QLabel('')
        self.main_layout.addWidget(dummy, 1)
        # ========================
        # Detect / Export buttons
        # ========================

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.detect_btn = QPushButton("Detect Scenes")
        self.detect_btn.setEnabled(False)
        self.detect_btn.setFixedHeight(32)
        self.detect_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.export_btn = QPushButton("Export Scenes")
        self.export_btn.setEnabled(False)
        self.export_btn.setFixedHeight(32)
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self.detect_btn.setStyleSheet(
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

        btn_row.addWidget(self.detect_btn)
        btn_row.addWidget(self.export_btn)

        self.main_layout.addLayout(btn_row)

        # ========================
        # Style
        # ========================

        self.setStyleSheet("""
            QFrame#SceneDetectorPanel {
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
            
            QFrame#ParamsBlock {
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 6px;
            }
        """)

        # Init
        self.on_detector_changed(self.combo.currentText())
        self._model.media_changed.connect(self.on_media_changed)
        self._model.scenes_changed.connect(self.on_scenes_changed)

    def on_media_changed(self, video):
        self.detect_btn.setEnabled(video is not None)
        self.export_btn.setEnabled(False)

    def on_scenes_changed(self, scenes):
        self.export_btn.setEnabled(bool(scenes))
        scenes_stats = self._model.get_scenes_stats()
        if scenes_stats:
            self._detected_res_lbl.setText(str(scenes_stats.total_scenes))
            self._avg_duration_res_lbl.setText(f"{scenes_stats.average_duration:.2f} sec")
            self._shortest_res_lbl.setText(f"{scenes_stats.shortest_scene:.2f} sec")
            self._longest_res_lbl.setText(f"{scenes_stats.longest_scene:.2f} sec")
        else:
            self._detected_res_lbl.setText("–")
            self._avg_duration_res_lbl.setText("–")
            self._shortest_res_lbl.setText("–")
            self._longest_res_lbl.setText("–")


    def set_detector(self, name, params: dict | None = None):
        """
        Позволяет программно установить детектор и его параметры
        """
        index = self.combo.findText(name)
        if index != -1:
            self.clear_params()
            with QSignalBlocker(self.combo):
                self.combo.setCurrentIndex(index)
            self.build_params(name, params)

    # ------------------------
    # Detector changed
    # ------------------------
    def on_detector_changed(self, name):
        self.clear_params()
        self.build_params(name)

    # ------------------------
    # Build parameters
    # ------------------------
    def build_params(self, name, params: dict | None = None):
        cfg = self.detectors_config[name]
        self.param_widgets.clear()

        for pname, pdata in cfg["parameters"].items():
            widget_t = pdata["widget"]
            if widget_t == ValueSlider:
                widget = widget_t(pname, pdata["min"], pdata["max"], pdata["default"])
            else:
                widget = widget_t()
            widget.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

            if params and pname in params:
                pvalue = params.get(pname, pdata)
            else:
                pvalue = pdata["default"]

            # Range
            if hasattr(widget, "setMinimum"):
                widget.setMinimum(pdata.get("min", 0))

            if hasattr(widget, "setMaximum"):
                widget.setMaximum(pdata.get("max", 100))

            # Default
            if isinstance(widget, QCheckBox):
                widget.setChecked(pdata["default"])
            elif widget_t != ValueSlider:
                widget.setValue(pvalue)

            widget.setToolTip(pdata["help"])

            if isinstance(widget, QDoubleSpinBox):
                widget.setDecimals(2)
                widget.setSingleStep(0.05)

            widget.setMinimumWidth(120)
            widget.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed
            )
            widget.setContentsMargins(0, 0, 0, 0)
            if widget_t == ValueSlider:
                self.params_layout.addRow(widget)
            else:
                label = QLabel(pname)
                self.params_layout.addRow(label, widget)

            self.param_widgets[pname] = widget

        fix_label_width(self.params_layout)
        self.params_layout.setSpacing(12)

    # ------------------------
    # Clear params
    # ------------------------
    def clear_params(self):
        while self.params_layout.rowCount():
            self.params_layout.removeRow(0)

        self.param_widgets.clear()

    # ------------------------
    # Help
    # ------------------------
    def show_help(self):
        name = self.combo.currentText()
        text = self.detectors_config[name]["help"]

        QMessageBox.information(self, name, text)

    # ------------------------
    # Public API
    # ------------------------
    def get_config(self):
        """
        Возвращает текущий детектор + параметры
        """

        name = self.combo.currentText()

        params = {}

        for pname, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                params[pname] = widget.isChecked()
            else:
                params[pname] = widget.get_value()
        return {
            "algorithm": name,
            "parameters": params
        }

    def on_detect_clicked(self, callback):
        """
        Позволяет подключить обработчик кнопки Detect
        """
        self.detect_btn.clicked.connect(callback)

    def on_export_clicked(self, callback):
        """
        Позволяет подключить обработчик кнопки Export Scenes
        """
        self.export_btn.clicked.connect(callback)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SceneDetectorPanel(SCENES)
    win.show()
    sys.exit(app.exec())