from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)


HISTORY_PATH = Path("data/generated_sets.csv")

CSV_COLUMNS = [
    "generation_id",
    "generated_at",
    "target_draw_no",
    "history_window",
    "mask_enabled",
    "mask_strength",
    "mask",
    "numbers",
    "score",
    "g1",
    "g2",
    "g3",
    "conditions",
    "mask_hits",
    "status",
    "actual_hits",
    "checked_at",
]


def _ensure_history_file():
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not HISTORY_PATH.exists():
        with HISTORY_PATH.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
            )
            writer.writeheader()


def _read_history():
    _ensure_history_file()

    with HISTORY_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def _write_history(rows):
    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def append_generated_results(
    results,
    target_draw_no: int,
    history_window: str,
    mask_enabled: bool,
    mask_strength: int,
):
    """
    Zapisuje wszystkie zestawy z jednego generowania.

    Każde kliknięcie GENERUJ otrzymuje własny generation_id.
    """

    if not results:
        return

    rows = _read_history()

    generation_id = uuid4().hex[:12]

    generated_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    for index, result in enumerate(results):
        numbers = ",".join(
            str(number)
            for number in sorted(result.numbers)
        )

        mask = ",".join(
            str(number)
            for number in sorted(result.mask_numbers)
        )

        profile = result.group_profile

        if isinstance(profile, dict):
            g1 = int(profile.get("G1", 0))
            g2 = int(profile.get("G2", 0))
            g3 = int(profile.get("G3", 0))
        else:
            values = list(profile)
            while len(values) < 3:
                values.append(0)

            g1 = int(values[0])
            g2 = int(values[1])
            g3 = int(values[2])

        status = ""

        if index == 0:
            status = "TOP 1"

        rows.append(
            {
                "generation_id": generation_id,
                "generated_at": generated_at,
                "target_draw_no": str(target_draw_no),
                "history_window": history_window,
                "mask_enabled": (
                    "TAK"
                    if mask_enabled
                    else "NIE"
                ),
                "mask_strength": (
                    str(mask_strength)
                    if mask_enabled
                    else "0"
                ),
                "mask": mask,
                "numbers": numbers,
                "score": f"{result.score:.3f}",
                "g1": str(g1),
                "g2": str(g2),
                "g3": str(g3),
                "conditions": (
                    f"{result.condition_count}/6"
                ),
                "mask_hits": str(
                    len(result.mask_numbers)
                ),
                "status": status,
                "actual_hits": "",
                "checked_at": "",
            }
        )

    _write_history(rows)


def mark_historical_repeats(df):
    """
    Zwraca zbiór zestawów, które wystąpiły już
    w historii Lotto.
    """

    if df is None or len(df) == 0:
        return set()

    columns = [
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
        "n6",
    ]

    result = set()

    for row in df[columns].itertuples(
        index=False,
        name=None,
    ):
        result.add(
            tuple(sorted(int(number) for number in row))
        )

    return result


def _calculate_hits(numbers_text, draw_numbers):
    numbers = {
        int(value)
        for value in numbers_text.split(",")
        if value.strip()
    }

    return len(
        numbers.intersection(draw_numbers)
    )


def reconcile_history(df):
    """
    Uzupełnia wyniki wygenerowanych zestawów,
    gdy pojawiło się już odpowiednie losowanie.
    """

    if df is None or len(df) == 0:
        return

    rows = _read_history()

    if not rows:
        return

    draw_map = {}

    for row in df.itertuples(
        index=False
    ):
        draw_no = int(row.draw_no)

        draw_map[draw_no] = {
            int(row.n1),
            int(row.n2),
            int(row.n3),
            int(row.n4),
            int(row.n5),
            int(row.n6),
        }

    changed = False

    for row in rows:
        if row.get("actual_hits", "").strip():
            continue

        target = row.get(
            "target_draw_no",
            "",
        ).strip()

        if not target:
            continue

        try:
            target_draw = int(target)
        except ValueError:
            continue

        draw_numbers = draw_map.get(
            target_draw
        )

        if draw_numbers is None:
            continue

        row["actual_hits"] = str(
            _calculate_hits(
                row["numbers"],
                draw_numbers,
            )
        )

        row["checked_at"] = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        changed = True

    if changed:
        _write_history(rows)


class HistoryTab(QWidget):
    """
    Historia wygenerowanych zestawów.

    Dane są przechowywane w:
        data/generated_sets.csv
    """

    def __init__(
        self,
        df=None,
        parent=None,
    ):
        super().__init__(parent)

        self.df = df
        self.rows = []

        self._build_ui()
        self._apply_style()
        self.set_data(df)

    # ==============================================================
    # UI
    # ==============================================================

    def _build_ui(self):
        root = QVBoxLayout(self)

        root.setContentsMargins(
            14,
            14,
            14,
            14,
        )

        root.setSpacing(10)

        # ==========================================================
        # HEADER
        # ==========================================================

        header = QFrame()
        header.setObjectName(
            "headerFrame"
        )

        header_layout = QHBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            16,
            12,
            16,
            12,
        )

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel(
            "HISTORIA TYPÓW"
        )

        title.setObjectName(
            "titleLabel"
        )

        subtitle = QLabel(
            "Wygenerowane zestawy i ich rzeczywiste wyniki."
        )

        subtitle.setObjectName(
            "subtitleLabel"
        )

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(
            title_box
        )

        header_layout.addStretch()

        self.refresh_button = QPushButton(
            "↻  ODŚWIEŻ"
        )

        self.refresh_button.setObjectName(
            "secondaryButton"
        )

        self.refresh_button.setMinimumHeight(
            40
        )

        self.refresh_button.clicked.connect(
            self.refresh
        )

        header_layout.addWidget(
            self.refresh_button
        )

        root.addWidget(header)

        # ==========================================================
        # STATYSTYKI
        # ==========================================================

        stats_group = QGroupBox(
            "STATYSTYKI"
        )

        stats_layout = QGridLayout(
            stats_group
        )

        stats_layout.setContentsMargins(
            14,
            16,
            14,
            14,
        )

        self.total_label = self._create_stat_label(
            "Wygenerowanych"
        )

        self.checked_label = self._create_stat_label(
            "Rozliczonych"
        )

        self.pending_label = self._create_stat_label(
            "Nierozliczonych"
        )

        self.best_label = self._create_stat_label(
            "Najlepszy wynik"
        )

        self.average_label = self._create_stat_label(
            "Średnia trafień"
        )

        self.hits_2_label = self._create_stat_label(
            "2/6 lub więcej"
        )

        stats_layout.addWidget(
            self.total_label,
            0,
            0,
        )

        stats_layout.addWidget(
            self.checked_label,
            0,
            1,
        )

        stats_layout.addWidget(
            self.pending_label,
            0,
            2,
        )

        stats_layout.addWidget(
            self.best_label,
            0,
            3,
        )

        stats_layout.addWidget(
            self.average_label,
            0,
            4,
        )

        stats_layout.addWidget(
            self.hits_2_label,
            0,
            5,
        )

        root.addWidget(
            stats_group
        )

        # ==========================================================
        # FILTRY
        # ==========================================================

        filter_group = QGroupBox(
            "WIDOK"
        )

        filter_layout = QHBoxLayout(
            filter_group
        )

        filter_layout.setContentsMargins(
            14,
            12,
            14,
            12,
        )

        filter_layout.addWidget(
            QLabel("Pokaż:")
        )

        self.filter_combo = QComboBox()

        self.filter_combo.addItem(
            "Wszystkie",
            "all",
        )

        self.filter_combo.addItem(
            "Nierozliczone",
            "pending",
        )

        self.filter_combo.addItem(
            "0/6",
            "0",
        )

        self.filter_combo.addItem(
            "1/6",
            "1",
        )

        self.filter_combo.addItem(
            "2/6",
            "2",
        )

        self.filter_combo.addItem(
            "3/6",
            "3",
        )

        self.filter_combo.addItem(
            "4/6+",
            "4plus",
        )

        self.filter_combo.addItem(
            "Powtórki",
            "repeat",
        )

        self.filter_combo.currentIndexChanged.connect(
            self._populate_table
        )

        filter_layout.addWidget(
            self.filter_combo
        )

        filter_layout.addStretch()

        self.info_label = QLabel()
        self.info_label.setObjectName(
            "mutedLabel"
        )

        filter_layout.addWidget(
            self.info_label
        )

        root.addWidget(
            filter_group
        )

        # ==========================================================
        # TABELA
        # ==========================================================

        table_group = QGroupBox(
            "WYGENEROWANE ZESTAWY"
        )

        table_layout = QVBoxLayout(
            table_group
        )

        table_layout.setContentsMargins(
            10,
            14,
            10,
            10,
        )

        self.table = QTableWidget()

        self.table.setColumnCount(9)

        self.table.setHorizontalHeaderLabels(
            [
                "DATA",
                "LOSOWANIE",
                "ZESTAW",
                "TRAFIENIA",
                "WYNIK",
                "WARUNKI",
                "MASKA",
                "OKNO",
                "STATUS",
            ]
        )

        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.table.verticalHeader().setVisible(
            False
        )

        self.table.setAlternatingRowColors(
            True
        )

        header = self.table.horizontalHeader()

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Stretch,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            4,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            5,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            6,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            7,
            QHeaderView.ResizeMode.Fixed,
        )

        header.setSectionResizeMode(
            8,
            QHeaderView.ResizeMode.Fixed,
        )

        header.resizeSection(0, 145)
        header.resizeSection(1, 85)
        header.resizeSection(3, 75)
        header.resizeSection(4, 75)
        header.resizeSection(5, 75)
        header.resizeSection(6, 70)
        header.resizeSection(7, 90)
        header.resizeSection(8, 150)

        table_layout.addWidget(
            self.table
        )

        root.addWidget(
            table_group,
            1,
        )

    def _create_stat_label(self, title):
        frame = QFrame()
        frame.setObjectName(
            "statFrame"
        )

        layout = QVBoxLayout(frame)

        layout.setContentsMargins(
            10,
            8,
            10,
            8,
        )

        layout.setSpacing(2)

        title_label = QLabel(title)

        title_label.setObjectName(
            "statTitle"
        )

        value_label = QLabel("0")

        value_label.setObjectName(
            "statValue"
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value_label
        )

        frame.value_label = value_label

        return frame

    # ==============================================================
    # STYL
    # ==============================================================

    def _apply_style(self):
        self.setStyleSheet(
            """
            QGroupBox {
                font-weight: 700;
                border: 1px solid #3a3f48;
                border-radius: 7px;
                margin-top: 10px;
                padding-top: 10px;
                background: #20242b;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 5px;
                color: #d7dce3;
            }

            #headerFrame {
                background: #20242b;
                border: 1px solid #3a3f48;
                border-radius: 8px;
            }

            #titleLabel {
                font-size: 18px;
                font-weight: 800;
                color: #f1f3f5;
            }

            #subtitleLabel {
                color: #9da5b0;
                font-size: 12px;
            }

            #statFrame {
                background: #171a1f;
                border: 1px solid #353b45;
                border-radius: 6px;
            }

            #statTitle {
                color: #9da5b0;
                font-size: 11px;
            }

            #statValue {
                color: #f1f3f5;
                font-size: 18px;
                font-weight: 800;
            }

            #mutedLabel {
                color: #9da5b0;
            }

            QComboBox {
                background: #171a1f;
                border: 1px solid #4a515d;
                border-radius: 5px;
                color: #f1f3f5;
                padding: 5px 8px;
                min-height: 28px;
            }

            QComboBox:focus {
                border: 1px solid #6d8cff;
            }

            QPushButton {
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 700;
            }

            #secondaryButton {
                background: #2a3039;
                color: #dce2ea;
                border: 1px solid #515966;
            }

            #secondaryButton:hover {
                background: #343b47;
            }

            QTableWidget {
                background: #171a1f;
                alternate-background-color: #1d2128;
                border: 1px solid #3a3f48;
                gridline-color: #303640;
                color: #e5e9ee;
                selection-background-color: #3d568f;
                selection-color: white;
            }

            QTableWidget::item {
                padding: 6px 4px;
            }

            QHeaderView::section {
                background: #292f38;
                color: #dce2ea;
                padding: 7px 5px;
                border: 0;
                border-right: 1px solid #3a3f48;
                border-bottom: 1px solid #3a3f48;
                font-weight: 700;
            }

            QScrollBar:vertical {
                background: #171a1f;
                width: 12px;
            }

            QScrollBar::handle:vertical {
                background: #454d59;
                border-radius: 5px;
                min-height: 30px;
            }
            """
        )

    # ==============================================================
    # DANE
    # ==============================================================

    def set_data(self, df):
        self.df = df

        reconcile_history(
            df
        )

        self.refresh()

    def refresh(self):
        self.rows = _read_history()

        self._update_statistics()
        self._populate_table()

    # ==============================================================
    # STATYSTYKI
    # ==============================================================

    def _update_statistics(self):
        total = len(self.rows)

        checked = [
            row
            for row in self.rows
            if row.get("actual_hits", "").strip()
        ]

        pending = total - len(checked)

        values = []

        for row in checked:
            try:
                values.append(
                    int(row["actual_hits"])
                )
            except (ValueError, KeyError):
                pass

        best = (
            max(values)
            if values
            else 0
        )

        average = (
            sum(values) / len(values)
            if values
            else 0.0
        )

        at_least_two = sum(
            value >= 2
            for value in values
        )

        self.total_label.value_label.setText(
            f"{total:,}".replace(",", " ")
        )

        self.checked_label.value_label.setText(
            f"{len(checked):,}".replace(",", " ")
        )

        self.pending_label.value_label.setText(
            f"{pending:,}".replace(",", " ")
        )

        self.best_label.value_label.setText(
            f"{best}/6"
        )

        self.average_label.value_label.setText(
            f"{average:.2f}/6"
        )

        self.hits_2_label.value_label.setText(
            f"{at_least_two:,}".replace(",", " ")
        )

    # ==============================================================
    # TABELA
    # ==============================================================

    def _filtered_rows(self):
        mode = self.filter_combo.currentData()

        if mode == "all":
            return list(
                reversed(self.rows)
            )

        if mode == "pending":
            return list(
                reversed(
                    [
                        row
                        for row in self.rows
                        if not row.get(
                            "actual_hits",
                            "",
                        ).strip()
                    ]
                )
            )

        if mode == "repeat":
            return list(
                reversed(
                    [
                        row
                        for row in self.rows
                        if "POWTÓRKA"
                        in row.get(
                            "status",
                            "",
                        )
                    ]
                )
            )

        if mode == "4plus":
            return list(
                reversed(
                    [
                        row
                        for row in self.rows
                        if row.get(
                            "actual_hits",
                            "",
                        ).isdigit()
                        and int(
                            row["actual_hits"]
                        ) >= 4
                    ]
                )
            )

        return list(
            reversed(
                [
                    row
                    for row in self.rows
                    if row.get(
                        "actual_hits",
                        "",
                    ).strip()
                    == mode
                ]
            )
        )

    def _populate_table(self):
        rows = self._filtered_rows()

        self.table.setRowCount(0)

        for row_index, row in enumerate(rows):
            self.table.insertRow(
                row_index
            )

            actual_hits = row.get(
                "actual_hits",
                "",
            ).strip()

            hits_text = (
                f"{actual_hits}/6"
                if actual_hits
                else "—"
            )

            mask_text = (
                f"{row.get('mask_hits', '0')}/6"
                if row.get(
                    "mask_enabled",
                    "NIE",
                ) == "TAK"
                else "—"
            )

            values = [
                row.get(
                    "generated_at",
                    "",
                )[:16],
                row.get(
                    "target_draw_no",
                    "",
                ),
                row.get(
                    "numbers",
                    "",
                ).replace(
                    ",",
                    ", ",
                ),
                hits_text,
                row.get(
                    "score",
                    "",
                ),
                row.get(
                    "conditions",
                    "",
                ),
                mask_text,
                row.get(
                    "history_window",
                    "",
                ),
                row.get(
                    "status",
                    "",
                ),
            ]

            for column, value in enumerate(
                values
            ):
                item = QTableWidgetItem(
                    value
                )

                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                if column == 2:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    )

                self.table.setItem(
                    row_index,
                    column,
                    item
                )

        self.info_label.setText(
            f"Pokazano {len(rows)} z {len(self.rows)} zapisanych zestawów."
        )


if __name__ == "__main__":
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    window = HistoryTab()

    window.resize(
        1400,
        850,
    )

    window.show()

    sys.exit(
        app.exec()
    )