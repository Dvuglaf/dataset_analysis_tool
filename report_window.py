from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel

from metric_table import MetricsTable, TableRow
from model import Model
from widgets import GroupWidgetPlain, GroupWidget


class ReportWindow(QDialog):
    def __init__(self, dataset_row: TableRow, parent):
        super().__init__(parent=parent)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("Dataset analysis report")
        title.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                font-size: 18px;
                color: #2c3e50;
                margin-left: 2px;
            }
        """)
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addSpacing(8)
        table_layout = QVBoxLayout()
        self.metric_table = MetricsTable(dataset_row)
        table_layout.addWidget(self.metric_table)
        table_group = GroupWidget(name="Comparison Table", content_layout=table_layout)
        main_layout.addWidget(table_group)

        # summary_layout = QVBoxLayout()
        # summary_group = GroupWidget(name="Summary & Recommendations", content_layout=summary_layout)
        # main_layout.addWidget(summary_group)

        self.resize(1200, 600)
