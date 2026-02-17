import json
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QLabel,
    QWidget,
    QMessageBox,
)


class ClassEditorDialog(QDialog):
    """
    Модальное окно для редактирования классов с одноуровневой группировкой.

    Формат данных:
    [
        {"name": "person", "group": "people"},
        {"name": "car", "group": "vehicles"}
    ]
    """

    def __init__(self, classes=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Class Editor")
        self.setModal(True)
        self.resize(600, 400)

        self.classes = classes or []

        self._build_ui()
        self._load_classes()

    # ---------------- UI ----------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # List
        self.list_widget = QListWidget()
        main_layout.addWidget(self.list_widget)

        # Editor panel
        editor_layout = QHBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Class name")

        self.group_input = QLineEdit()
        self.group_input.setPlaceholderText("Group (optional)")

        add_btn = QPushButton("Add / Update")
        add_btn.clicked.connect(self.add_or_update_class)

        editor_layout.addWidget(QLabel("Name:"))
        editor_layout.addWidget(self.name_input)
        editor_layout.addWidget(QLabel("Group:"))
        editor_layout.addWidget(self.group_input)
        editor_layout.addWidget(add_btn)

        main_layout.addLayout(editor_layout)

        # Buttons
        btn_layout = QHBoxLayout()

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_from_file)

        self.save_btn = QPushButton("Save to File")
        self.save_btn.clicked.connect(self.save_to_file)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(self.delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)

        main_layout.addLayout(btn_layout)

        # Signals
        self.list_widget.itemSelectionChanged.connect(self.on_item_selected)

    # ---------------- Logic ----------------

    def _load_classes(self):
        self.list_widget.clear()

        for item in self.classes:
            text = self._format_item(item)
            lw_item = QListWidgetItem(text)
            lw_item.setData(Qt.UserRole, item)
            self.list_widget.addItem(lw_item)

    def _format_item(self, item: dict) -> str:
        name = item.get("name", "")
        group = item.get("group", "")

        if group:
            return f"{name}  [{group}]"
        return name

    def add_or_update_class(self):
        name = self.name_input.text().strip()
        group = self.group_input.text().strip()

        if not name:
            QMessageBox.warning(self, "Error", "Class name is empty")
            return

        selected = self.list_widget.currentItem()

        # Update
        if selected:
            data = selected.data(Qt.UserRole)
            data["name"] = name
            data["group"] = group

            selected.setText(self._format_item(data))
            selected.setData(Qt.UserRole, data)

        # Add new
        else:
            data = {
                "name": name,
                "group": group,
            }

            self.classes.append(data)

            item = QListWidgetItem(self._format_item(data))
            item.setData(Qt.UserRole, data)
            self.list_widget.addItem(item)

        self._clear_inputs()

    def delete_selected(self):
        row = self.list_widget.currentRow()

        if row < 0:
            return

        reply = QMessageBox.question(
            self,
            "Delete",
            "Delete selected class?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.list_widget.takeItem(row)
            self.classes.pop(row)
            self._clear_inputs()

    def on_item_selected(self):
        item = self.list_widget.currentItem()

        if not item:
            return

        data = item.data(Qt.UserRole)

        self.name_input.setText(data.get("name", ""))
        self.group_input.setText(data.get("group", ""))

    def _clear_inputs(self):
        self.name_input.clear()
        self.group_input.clear()
        self.list_widget.clearSelection()

    # ---------------- Files ----------------

    def save_to_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save classes",
            "classes.json",
            "JSON (*.json)",
        )

        if not path:
            return

        try:
            data = self.get_classes()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load classes",
            "",
            "JSON (*.json)",
        )

        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid format")

            self.classes = data
            self._load_classes()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Load failed: {e}")

    # ---------------- Public API ----------------

    def get_classes(self):
        """
        Вернуть текущий список классов
        """
        result = []

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            result.append(item.data(Qt.UserRole))

        return result


# ---------------- Example ----------------
def main():
    app = QApplication(sys.argv)

    initial_classes = [
        {"name": "person", "group": "people"},
        {"name": "car", "group": "vehicles"},
        {"name": "bicycle", "group": "vehicles"},
    ]

    dialog = ClassEditorDialog(initial_classes)

    if dialog.exec() == QDialog.Accepted:
        result = dialog.get_classes()
        print("Saved classes:")
        print(result)

    sys.exit(0)


if __name__ == "__main__":
    main()
