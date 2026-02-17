import dataclasses
import sys
from dataclasses import dataclass

from PySide6.QtWidgets import (QApplication, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QHeaderView, QLabel,
                               QHBoxLayout, QAbstractItemView, QButtonGroup, QPushButton)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from metric_interpreter import THRESHOLDS


class NumericTableWidgetItem(QTableWidgetItem):
    """Кастомный айтем для правильной сортировки чисел и скрытия текста в первой колонке"""

    def __init__(self, text, sort_value=None):
        super().__init__(text)
        self.sort_value = sort_value if sort_value is not None else text

    def __lt__(self, other):
        try:
            # Сравниваем как числа, если это возможно
            return float(self.sort_value) < float(other.sort_value)
        except (ValueError, TypeError):
            return str(self.sort_value) < str(other.sort_value)


@dataclass
class TableRow:
    dataset: str

    imaging_quality: float | None
    entropy: float | None

    optical_flow: float | None
    motion_smoothness: float | None
    psnr: float | None
    ssim: float | None
    temporal_coherence_clip_score: float | None
    background_dominance: float | None

    clip_ics: float | None
    num_clusters: float | None
    r_R_score: float | None
    noise_ratio: float | None
    duration_sec: float | None
    persistence: float | None
    norm_avg_velocity: float | None
    frames: float | None

    def __iter__(self):
        for field in dataclasses.fields(self):
            print(field.name)
            yield getattr(self, field.name)


class MetricsTable(QWidget):
    def __init__(self, dataset_metrics: TableRow):
        super().__init__()
        self.setWindowTitle("Dataset Metrics Comparison")
        self.resize(1200, 450)
        self.init_ui(dataset_metrics)

    def init_ui(self, dataset_metrics):
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
        self.btn_page1 = QPushButton("Quality / Motion")
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
        self.table = QTableWidget()

        # 1. ЗАПРЕТ РЕДАКТИРОВАНИЯ (Read Only)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)  # Убирает пунктирную рамку при клике

        self.setup_table_structure()
        self.fill_data(dataset_metrics)
        self.apply_styles()
        self.highlight_extremes()

        main_layout.addWidget(self.table)

    def setup_table_structure(self):
        # Фиксированная часть (всегда видна)
        self.fixed_cols = ["Dataset"]

        # Первая страница (Quality & Motion) ~10 метрик
        self.page_1_cols = ["MUSIQ", "Entropy", "Opt. Flow", "Smoothness",
                            "PSNR", "SSIM", "CLIP Consist.", "Back. Dominance"]

        # Вторая страница (Diversity) ~10 метрик
        self.page_2_cols = ["Similarity (CLIP)", "Num Clusters", "r/R score", "Noise Ratio",
                            "Avg Duration", "Persistence", "Avg Velocity", "Frames"]

        self.headers = self.fixed_cols + self.page_1_cols + self.page_2_cols
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSortIndicatorShown(True)
        header.setSectionsClickable(True)

        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(True)

        self.switch_page(0)

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

    def create_dataset_cell(self, name, is_target=False):
        """Виджет для первой колонки: Название + Бейдж"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: 500; color: #333;")
        layout.addWidget(name_label)

        layout.addStretch()
        return container

    def create_metric_widget(self, value, trend=None):
        """Виджет для ячеек с метриками (число + стрелочка тренда)"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(5, 0, 5, 0)
        layout.setAlignment(Qt.AlignCenter)

        value = round(value, 3) if isinstance(value, float) else "—"
        val_label = QLabel(str(value))
        layout.addWidget(val_label)

        if trend:
            trend_label = QLabel(trend)
            # Цвет стрелочки (зеленая вверх, красная вниз, серая тире)
            color = "#2ecc71" if "↗" in trend else "#e74c3c" if "↘" in trend else "#bdc3c7"
            trend_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")
            layout.addWidget(trend_label)

        return container, val_label

    def fill_data(self, dataset_metrics: TableRow):
        dataset_metrics.dataset += " (Your Dataset)"

        data = [
            TableRow(
                dataset="SSv2",
                imaging_quality=56.767,
                entropy=4.828,
                optical_flow=23.398,
                motion_smoothness=0.980,
                psnr=23.172,
                ssim=0.802,
                temporal_coherence_clip_score=0.975,
                background_dominance=0.461,
                clip_ics=0.259,
                num_clusters=159.0,
                r_R_score=0.091,
                noise_ratio=0.053,
                duration_sec=1.90,
                persistence=0.79,
                norm_avg_velocity=0.013264,
                frames=17.22
            ),
            TableRow(
                dataset="Kinetics-400",
                imaging_quality=52.901,
                entropy=4.748,
                optical_flow=14.405,
                motion_smoothness=0.984,
                psnr=25.725,
                ssim=0.849,
                temporal_coherence_clip_score=0.984,
                background_dominance=0.509,
                clip_ics=0.377,
                num_clusters=None,  # Данных по кластерам для K400 в тексте не было
                r_R_score=None,
                noise_ratio=None,
                duration_sec=1.62,
                persistence=0.75,
                norm_avg_velocity=0.011078,
                frames=29.49
            ),
            TableRow(
                dataset="HMDB51",
                imaging_quality=47.109,
                entropy=4.664,
                optical_flow=12.263,
                motion_smoothness=0.985,
                psnr=27.081,
                ssim=0.869,
                temporal_coherence_clip_score=0.98,
                background_dominance=0.514,
                clip_ics=0.381,
                num_clusters=53.0,
                r_R_score=0.106,
                noise_ratio=0.344,
                duration_sec=None,
                persistence=None,
                norm_avg_velocity=None,
                frames=None
            ),
            TableRow(
                dataset="UCF101",
                imaging_quality=46.798,
                entropy=4.616,
                optical_flow=10.017,
                motion_smoothness=0.987,
                psnr=26.905,
                ssim=0.881,
                temporal_coherence_clip_score=0.983,
                background_dominance=0.416,
                clip_ics=0.384,
                num_clusters=100.0,
                r_R_score=0.09,
                noise_ratio=0.276,
                duration_sec=None,
                persistence=None,
                norm_avg_velocity=None,
                frames=None
            ),
            TableRow(
                dataset="Assembly101",
                imaging_quality=59.173,
                entropy=4.676,
                optical_flow=0.428,
                motion_smoothness=1.000,
                psnr=35.520,
                ssim=0.955,
                temporal_coherence_clip_score=0.995,
                background_dominance=0.277,
                clip_ics=0.157,
                num_clusters=18.0,
                r_R_score=0.086,
                noise_ratio=0.01,
                duration_sec=None,
                persistence=None,
                norm_avg_velocity=None,
                frames=None
            ),
            dataset_metrics
        ]

        self.table.setRowCount(len(data))

        for row_idx, row_data in enumerate(data):
            is_target = "Your Dataset" in row_data.dataset

            for col_idx, val in enumerate(row_data):
                display_val = val

                if isinstance(val, tuple):
                    display_val, trend_str = val

                # Создаем айтем для сортировки
                sort_text = str(display_val).replace(" ", "").replace("h", "")

                # Чтобы избежать наложения текста, в самом айтеме текст НЕ пишем для 1-й колонки и колонок с виджетами
                item = NumericTableWidgetItem("", sort_value=sort_text)
                self.table.setItem(row_idx, col_idx, item)

                if col_idx == 0:
                    clean_name = display_val.replace(" (Your Dataset)", "")
                    widget = self.create_dataset_cell(clean_name, is_target)
                    self.table.setCellWidget(row_idx, col_idx, widget)
                else:
                    # Числовые метрики с трендами
                    widget, label_ref = self.create_metric_widget(display_val)
                    self.table.setCellWidget(row_idx, col_idx, widget)
                    # Сохраняем ссылку на QLabel, чтобы потом покрасить его (Max/Min)
                    item.setData(Qt.UserRole, label_ref)

            if is_target:
                for c in range(self.table.columnCount()):
                    it = self.table.item(row_idx, c)
                    it.setBackground(QColor("#f8faff"))  # Цвет строки Your Dataset

    def highlight_extremes(self):
        def column_metric_map(column_name: str) -> str:
            if column_name == "MUSIQ":
                return "imaging_quality"
            elif column_name == "Entropy":
                return "entropy"
            elif column_name == "Opt. Flow":
                return "optical_flow"
            elif column_name == "Smoothness":
                return "motion_smoothness"
            elif column_name == "PSNR":
                return "psnr"
            elif column_name == "SSIM":
                return "ssim"
            elif column_name == "CLIP Consist.":
                return "temporary_consistency_clip_score"
            elif column_name == "Back. Dominance":
                return "background_dominance"
            elif column_name == "Similarity (CLIP)":
                return "clip_ics"
            elif column_name == "Num Clusters":
                return "num_clusters"
            elif column_name == "r/R score":
                return "r_R_score"
            elif column_name == "Noise Ratio":
                return "noise_ratio"
            elif column_name == "Avg Duration":
                return "duration_sec"
            elif column_name == "Persistence":
                return "persistence"
            elif column_name == "Avg Velocity":
                return "norm_avg_velocity"
            elif column_name == "Frames":
                return "frames_count"
            return column_name  # TODO

        """Красим текст внутри QLabel виджетов"""
        for col in range(1, self.table.columnCount()):
            values = []
            for row in range(self.table.rowCount()):
                item = self.table.item(row, col)
                try:
                    val = float(item.sort_value)
                    values.append((row, val))
                except:
                    continue

            if not values: continue

            spec = THRESHOLDS[column_metric_map(self.headers[col])]

            def find_closest_to_target(array, target):
                return sorted(array, key=lambda v: abs(target - v))[0]

            if spec.mode == "direct":
                max_v = max(values, key=lambda x: x[1])[1]
                min_v = min(values, key=lambda x: x[1])[1]
            elif spec.mode == "inverse":
                max_v = min(values, key=lambda x: x[1])[1]
                min_v = max(values, key=lambda x: x[1])[1]
            else:  # target
                target_value = spec.target
                max_v = max(values, key=lambda x: abs(x[1] - target_value))[1]
                min_v = min(values, key=lambda x: abs(x[1] - target_value))[1]

            for row, val in values:
                label = self.table.item(row, col).data(Qt.UserRole)
                if not label: continue

                if abs(val - max_v) < 0.001:
                    label.setStyleSheet("color: #2ecc71; font-weight: bold;")
                elif abs(val - min_v) < 0.001:
                    label.setStyleSheet("color: #e74c3c; font-weight: bold;")

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
                background-color: white;
                padding: 10px;
                border: none;
                border-bottom: 1px solid #eee;
                font-weight: bold;
                color: #666;
            }
            QHeaderView::section:hover {
                background-color: #fcfcfc;
                color: #000;
                border-bottom: 2px solid #3498db;
            }
            QTableWidget::item:selected {
                background-color: #f0f4f8;
                color: black;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Для одинакового вида на всех ОС
    window = MetricsTable()
    window.show()
    sys.exit(app.exec())