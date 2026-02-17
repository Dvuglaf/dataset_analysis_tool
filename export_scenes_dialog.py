from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QWidget,
)


class ExportScenesDialog(QDialog):
    """
    Простое диалоговое окно с параметрами экспорта сцен.
    """

    def __init__(self, parent: QWidget | None = None, default_output: str | None = None):
        super().__init__(parent)

        self.setWindowTitle("Export Scenes")
        self.setModal(True)

        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Output directory
        self.output_edit = QLineEdit(self)
        if default_output:
            self.output_edit.setText(default_output)

        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self._browse_dir)

        out_container = QWidget(self)
        out_layout = QFormLayout(out_container)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        out_layout.addRow(self.output_edit, browse_btn)

        form.addRow("Output directory:", out_container)

        # Filename prefix
        self.prefix_edit = QLineEdit("scene", self)
        form.addRow("File prefix:", self.prefix_edit)

        # Format
        self.format_combo = QComboBox(self)
        self.format_combo.addItems([".mp4", ".mkv", ".avi"])
        form.addRow("Format:", self.format_combo)

        # Overwrite flag
        self.overwrite_check = QCheckBox("Overwrite existing files", self)
        form.addRow("", self.overwrite_check)

        # Buttons OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    # ------------------------
    # Helpers / properties
    # ------------------------
    def _browse_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select output directory")
        if directory:
            self.output_edit.setText(directory)

    def _accept_if_valid(self):
        path = self.output_edit.text().strip()
        if not path:
            self.output_edit.setFocus()
            return
        self.accept()

    @property
    def output_dir(self) -> Path:
        return Path(self.output_edit.text().strip()).expanduser()

    @property
    def prefix(self) -> str:
        return self.prefix_edit.text().strip() or "scene"

    @property
    def extension(self) -> str:
        return self.format_combo.currentText()

    @property
    def overwrite(self) -> bool:
        return self.overwrite_check.isChecked()


