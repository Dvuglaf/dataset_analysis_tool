import sys
from PySide6.QtWidgets import (QApplication, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QHeaderView, QLabel,
                               QHBoxLayout, QAbstractItemView, QPushButton, QButtonGroup)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor


# --- Сохраняем логику сортировки из предыдущих ответов ---
class NumericTableWidgetItem(QTableWidgetItem):
    def __init__(self, text, sort_value=None):
        super().__init__(text)
        self.sort_value = sort_value if sort_value is not None else text

    def __lt__(self, other):
        try:
            return float(self.sort_value) < float(other.sort_value)
        except:
            return str(self.sort_value) < str(other.sort_value)


class MetricsTable(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Metrics Paginated")
        self.resize(1100, 500)

        # 1. ОПРЕДЕЛЯЕМ СТРУКТУРУ КОЛОНОК
        # Фиксированная часть (всегда видна)
        self.fixed_cols = ["Dataset"]

        # Первая страница (Quality & Motion) ~10 метрик
        self.page_1_cols = ["Clips", "Classes", "Duration", "MUSIQ", "Entropy",
                            "Optical Flow", "DINO", "CLIP", "Temp-Consist", "Motion-Bias"]

        # Вторая страница (Diversity) ~10 метрик
        self.page_2_cols = ["Diversity", "Distinct.", "Coverage", "Density",
                            "Inception Score", "FID", "Precision", "Recall", "Density-2",
                            "Coverage-2"]

        self.headers = self.fixed_cols + self.page_1_cols + self.page_2_cols

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        toggle_container = QWidget()
        toggle_container.setObjectName("toggleContainer")
        toggle_layout = QHBoxLayout(toggle_container)
        toggle_layout.setContentsMargins(4, 4, 4, 4)
        toggle_layout.setSpacing(2)

        self.btn_group = QButtonGroup(self)

        # Кнопки страниц
        self.btn_page1 = QPushButton("Quality & Motion")
        self.btn_page2 = QPushButton("Diversity Metrics")

        for i, btn in enumerate([self.btn_page1, self.btn_page2]):
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            self.btn_group.addButton(btn, i)
            toggle_layout.addWidget(btn)

        self.btn_page1.setChecked(True)
        self.btn_group.idClicked.connect(self.switch_page)

        # Центрируем переключатель
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        header_layout.addWidget(toggle_container)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # --- ТАБЛИЦА ---
        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)

        self.fill_mock_data()
        self.apply_styles()

        # Инициализируем видимость первой страницы
        self.switch_page(0)

        main_layout.addWidget(self.table)

    def switch_page(self, page_id):
        """Логика скрытия/показа групп колонок"""
        # Индексы колонок для страниц
        # Page 0 (Quality/Motion): индексы с 1 по len(page_1_cols)
        # Page 1 (Diversity): индексы с len(page_1_cols)+1 до конца

        p1_end = len(self.fixed_cols) + len(self.page_1_cols)

        for col_idx in range(len(self.fixed_cols), len(self.headers)):
            if page_id == 0:  # Показываем Quality/Motion
                should_hide = col_idx >= p1_end
            else:  # Показываем Diversity
                should_hide = col_idx < p1_end

            self.table.setColumnHidden(col_idx, should_hide)

    def fill_mock_data(self):
        datasets = ["Kinetics-400", "IDS (Your Dataset)", "SSv2", "Assembly101"]
        self.table.setRowCount(len(datasets))
        for r, name in enumerate(datasets):
            is_target = "Your Dataset" in name
            # Колонка 0
            self.table.setItem(r, 0, NumericTableWidgetItem("", sort_value=name))
            self.table.setCellWidget(r, 0, self.create_dataset_cell(name, is_target))
            # Остальные 20 колонок
            for c in range(1, len(self.headers)):
                item = NumericTableWidgetItem(str(round(0.123 * (c + r), 3)))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)
                if is_target:
                    item.setBackground(QColor("#f8faff"))

    def create_dataset_cell(self, name, is_target):
        w = QWidget() ;
        l = QHBoxLayout(w) ;
        l.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel(name.replace(" (Your Dataset)", ""))
        l.addWidget(lbl)
        if is_target:
            badge = QLabel("Your Dataset")
            badge.setStyleSheet(
                "background:#000; color:#fff; border-radius:8px; padding:2px 6px; font-size:9px; font-weight:bold;")
            l.addWidget(badge)
        l.addStretch() ;
        return w

    def apply_styles(self):
        self.setStyleSheet("""
            /* Контейнер для переключателя */
            #toggleContainer {
                background: #f3f4f6;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
            }

            /* Ваш базовый стиль для кнопок */
            QPushButton {
                border-radius: 8px;
                border: none;
                background: transparent;
                padding: 0 15px;
                color: #6b7280;
                font-weight: 500;
            }

            QPushButton:hover {
                background: #e5e7eb;
                color: #374151;
            }

            /* Состояние выбранной страницы */
            QPushButton:checked {
                background: white;
                color: #111827;
                border: 1px solid #d1d5db; /* Тень или легкая рамка для объема */
            }

            QTableWidget {
                background-color: white;
                gridline-color: #f0f0f0;
                border: 1px solid #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            QHeaderView::section {
                background-color: #fafafa;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #eee;
                font-weight: bold;
                color: #4b5563;
            }
            QHeaderView::section:hover {
                background-color: #f3f4f6;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MetricsTable()
    w.show()
    sys.exit(app.exec())