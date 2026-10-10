from __future__ import annotations

import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from analyzer import (
    build_groups,
    coldest_numbers,
    count_internal_adjacent_pairs,
    group_balance,
    hottest_numbers,
    last_draw_numbers,
    neighbors_from_last_draw,
    number_frequency,
    pair_counter,
    repeated_from_last_draw,
    triple_counter,
)


class AnalysisTab(QWidget):
    """
    Zakładka analizy statystycznej historii Lotto.

    Wszystkie statystyki są liczone wyłącznie z aktualnie wybranego
    okna historii.
    """

    refresh_requested = pyqtSignal()

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)

        self.df = df.copy() if df is not None else pd.DataFrame()
        self.current_range = "50"

        self._build_ui()
        self.refresh_data(self.df)

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        self.setObjectName("AnalysisTab")

        self.setStyleSheet(
            """
            QWidget#AnalysisTab {
                background: #20242b;
                color: #d7dce3;
            }

            QScrollArea {
                background: #20242b;
                border: none;
            }

            QScrollArea > QWidget > QWidget {
                background: #20242b;
            }

            QFrame#HeaderFrame,
            QFrame#SectionFrame,
            QFrame#StatCard,
            QFrame#GroupCard {
                background: #20242b;
                border: 1px solid #303640;
                border-radius: 8px;
            }

            QLabel {
                background: transparent;
            }

            QLabel#Title {
                color: #f1f3f5;
                font-size: 20px;
                font-weight: 700;
            }

            QLabel#Subtitle {
                color: #9da5b0;
                font-size: 11px;
            }

            QLabel#SectionTitle {
                color: #d7dce3;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#CardTitle {
                color: #9da5b0;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#CardValue {
                color: #f1f3f5;
                font-size: 18px;
                font-weight: 700;
            }

            QLabel#CardDetail {
                color: #9da5b0;
                font-size: 10px;
            }

            QLabel#GroupName {
                color: #f1f3f5;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#GroupNumbers {
                color: #dce2ea;
                font-size: 11px;
            }

            QLabel#GroupCount {
                color: #b9c0ca;
                font-size: 11px;
                font-weight: 600;
            }

            QLabel#StatusLabel {
                color: #9da5b0;
                font-size: 10px;
            }

            QPushButton {
                background: #2a3039;
                color: #dce2ea;
                border: 1px solid #343b47;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }

            QPushButton:hover {
                background: #343b47;
            }

            QPushButton:pressed {
                background: #252a32;
            }

            QPushButton#RefreshButton {
                background: #4969d8;
                color: white;
                border: none;
                padding: 7px 15px;
            }

            QPushButton#RefreshButton:hover {
                background: #5877e5;
            }

            QPushButton#RefreshButton:pressed {
                background: #3e5cc5;
            }

            QPushButton#RangeButton {
                background: #2a3039;
                color: #dce2ea;
                border: 1px solid #343b47;
                padding: 6px 13px;
            }

            QPushButton#RangeButton:hover {
                background: #343b47;
            }

            QPushButton#RangeButton:checked {
                background: #4969d8;
                color: white;
                border-color: #4969d8;
            }

            QTableWidget {
                background: #171a1f;
                alternate-background-color: #1d2128;
                border: 1px solid #303640;
                gridline-color: #303640;
                color: #e5e9ee;
                selection-background-color: #3d568f;
                selection-color: white;
                border-radius: 6px;
            }

            QTableWidget::item {
                background: #171a1f;
                color: #dce2ea;
                padding: 4px;
            }

            QTableWidget::item:alternate {
                background: #1d2128;
            }

            QTableWidget::item:selected {
                background: #3d568f;
                color: white;
            }

            QHeaderView::section {
                background: #20242b;
                color: #e5e9ee;
                border: none;
                border-bottom: 1px solid #303640;
                padding: 6px;
                font-weight: 700;
            }

            QScrollBar:vertical {
                background: #171a1f;
                width: 10px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #454d59;
                border-radius: 5px;
                min-height: 25px;
            }

            QScrollBar::handle:vertical:hover {
                background: #596372;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }

            QScrollBar:horizontal {
                background: #171a1f;
                height: 10px;
            }

            QScrollBar::handle:horizontal {
                background: #454d59;
                border-radius: 5px;
                min-width: 25px;
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --------------------------------------------------------------
        # HEADER
        # --------------------------------------------------------------

        header = QFrame()
        header.setObjectName("HeaderFrame")

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel("ANALIZA HISTORII")
        title.setObjectName("Title")

        subtitle = QLabel(
            "Statystyki historyczne — bez predykcji AI"
        )
        subtitle.setObjectName("Subtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        self.refresh_button = QPushButton("ODŚWIEŻ")
        self.refresh_button.setObjectName("RefreshButton")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)

        header_layout.addWidget(self.refresh_button)

        root.addWidget(header)

        # --------------------------------------------------------------
        # RANGE
        # --------------------------------------------------------------

        range_frame = QFrame()
        range_frame.setObjectName("SectionFrame")

        range_layout = QHBoxLayout(range_frame)
        range_layout.setContentsMargins(12, 8, 12, 8)
        range_layout.setSpacing(6)

        range_title = QLabel("ZAKRES:")
        range_title.setObjectName("SectionTitle")
        range_layout.addWidget(range_title)

        self.range_buttons: dict[str, QPushButton] = {}

        for value in ("30", "50", "100", "200", "500", "WSZYSTKIE"):
            button = QPushButton(value)
            button.setObjectName("RangeButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, v=value: self._set_range(v)
            )

            self.range_buttons[value] = button
            range_layout.addWidget(button)

        range_layout.addStretch()

        self.range_info = QLabel("")
        self.range_info.setObjectName("StatusLabel")
        range_layout.addWidget(self.range_info)

        root.addWidget(range_frame)

        # --------------------------------------------------------------
        # SCROLL
        # --------------------------------------------------------------

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        content.setObjectName("AnalysisContent")
        content.setStyleSheet(
            """
            QWidget#AnalysisContent {
                background: #20242b;
            }
            """
        )

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 3, 0)
        content_layout.setSpacing(8)

        scroll.setWidget(content)

        root.addWidget(scroll, 1)

        self.content_layout = content_layout

        # --------------------------------------------------------------
        # TOP STATS
        # --------------------------------------------------------------

        top_grid = QGridLayout()
        top_grid.setHorizontalSpacing(8)
        top_grid.setVerticalSpacing(8)

        self.last_draw_card = self._create_stat_card(
            "OSTATNIE LOSOWANIE",
            "—",
            "Ostatni zestaw w wybranym oknie",
        )

        self.hot_card = self._create_stat_card(
            "HOT",
            "—",
            "Najczęstsze liczby",
        )

        self.cold_card = self._create_stat_card(
            "COLD",
            "—",
            "Najrzadsze liczby",
        )

        top_grid.addWidget(self.last_draw_card, 0, 0)
        top_grid.addWidget(self.hot_card, 0, 1)
        top_grid.addWidget(self.cold_card, 0, 2)

        content_layout.addLayout(top_grid)

        # --------------------------------------------------------------
        # GROUPS
        # --------------------------------------------------------------

        groups_frame = QFrame()
        groups_frame.setObjectName("SectionFrame")

        groups_layout = QVBoxLayout(groups_frame)
        groups_layout.setContentsMargins(12, 10, 12, 10)
        groups_layout.setSpacing(7)

        groups_title = QLabel("GRUPY G1 / G2 / G3")
        groups_title.setObjectName("SectionTitle")
        groups_layout.addWidget(groups_title)

        self.groups_info = QLabel("")
        self.groups_info.setObjectName("Subtitle")
        groups_layout.addWidget(self.groups_info)

        groups_grid = QGridLayout()
        groups_grid.setHorizontalSpacing(8)

        self.group_cards: dict[str, dict[str, QLabel]] = {}

        for column, group_name in enumerate(("G1", "G2", "G3")):
            card, labels = self._create_group_card(group_name)
            self.group_cards[group_name] = labels
            groups_grid.addWidget(card, 0, column)

        groups_layout.addLayout(groups_grid)

        content_layout.addWidget(groups_frame)

        # --------------------------------------------------------------
        # CONDITIONS / STATS
        # --------------------------------------------------------------

        stats_frame = QFrame()
        stats_frame.setObjectName("SectionFrame")

        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(7)

        stats_title = QLabel("STATYSTYKI OKNA")
        stats_title.setObjectName("SectionTitle")
        stats_layout.addWidget(stats_title)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(8)
        stats_grid.setVerticalSpacing(8)

        self.repeated_card = self._create_stat_card(
            "POWTÓRZENIA",
            "—",
            "",
        )

        self.neighbor_card = self._create_stat_card(
            "SĄSIEDZI",
            "—",
            "",
        )

        self.adjacent_card = self._create_stat_card(
            "PARY WEWNĘTRZNE",
            "—",
            "W ostatnim losowaniu",
        )

        self.pairs_card = self._create_stat_card(
            "NAJPOPULARNIEJSZE PARY",
            "—",
            "",
        )

        self.triples_card = self._create_stat_card(
            "NAJPOPULARNIEJSZE TRÓJKI",
            "—",
            "",
        )

        self.draws_card = self._create_stat_card(
            "LOSOWANIA",
            "—",
            "",
        )

        cards = [
            self.repeated_card,
            self.neighbor_card,
            self.adjacent_card,
            self.pairs_card,
            self.triples_card,
            self.draws_card,
        ]

        for index, card in enumerate(cards):
            row = index // 3
            column = index % 3
            stats_grid.addWidget(card, row, column)

        stats_layout.addLayout(stats_grid)

        content_layout.addWidget(stats_frame)

        # --------------------------------------------------------------
        # FREQUENCY + HISTORY
        # --------------------------------------------------------------

        tables_row = QHBoxLayout()
        tables_row.setSpacing(8)

        # ==============================================================
        # FREQUENCY
        # ==============================================================

        frequency_frame = QFrame()
        frequency_frame.setObjectName("SectionFrame")
        frequency_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        frequency_layout = QVBoxLayout(frequency_frame)
        frequency_layout.setContentsMargins(12, 10, 12, 10)
        frequency_layout.setSpacing(6)

        frequency_title = QLabel("CZĘSTOTLIWOŚĆ LICZB")
        frequency_title.setObjectName("SectionTitle")
        frequency_layout.addWidget(frequency_title)

        self.frequency_info = QLabel("")
        self.frequency_info.setObjectName("Subtitle")
        frequency_layout.addWidget(self.frequency_info)

        self.frequency_table = QTableWidget()
        self.frequency_table.setColumnCount(4)
        self.frequency_table.setHorizontalHeaderLabels(
            [
                "MIEJSCE",
                "LICZBA",
                "WYSTĄPIENIA",
                "GRUPA",
            ]
        )

        self._configure_table(
            self.frequency_table,
            stretch_last=False,
        )

        self.frequency_table.setColumnWidth(0, 65)
        self.frequency_table.setColumnWidth(1, 65)
        self.frequency_table.setColumnWidth(2, 90)
        self.frequency_table.setColumnWidth(3, 60)

        self.frequency_table.setMinimumHeight(300)

        frequency_layout.addWidget(
            self.frequency_table
        )

        tables_row.addWidget(
            frequency_frame,
            1,
        )

        # ==============================================================
        # HISTORY
        # ==============================================================

        history_frame = QFrame()
        history_frame.setObjectName("SectionFrame")
        history_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        history_layout = QVBoxLayout(history_frame)
        history_layout.setContentsMargins(12, 10, 12, 10)
        history_layout.setSpacing(6)

        history_title = QLabel("OSTATNIE LOSOWANIA")
        history_title.setObjectName("SectionTitle")
        history_layout.addWidget(history_title)

        self.history_info = QLabel("")
        self.history_info.setObjectName("Subtitle")
        history_layout.addWidget(self.history_info)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(8)
        self.history_table.setHorizontalHeaderLabels(
            [
                "LOSOWANIE",
                "DATA",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
            ]
        )

        self._configure_table(
            self.history_table,
            stretch_last=False,
        )

        self.history_table.setMinimumHeight(300)

        history_layout.addWidget(
            self.history_table
        )

        tables_row.addWidget(
            history_frame,
            1,
        )

        content_layout.addLayout(
            tables_row
        )

        content_layout.addStretch()

        self._set_range("50")

    # ==================================================================
    # WIDGET HELPERS
    # ==================================================================

    def _create_stat_card(
        self,
        title: str,
        value: str,
        detail: str,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("StatCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")

        value_label = QLabel(value)
        value_label.setObjectName("CardValue")
        value_label.setWordWrap(True)

        detail_label = QLabel(detail)
        detail_label.setObjectName("CardDetail")
        detail_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(detail_label)
        layout.addStretch()

        card._title_label = title_label
        card._value_label = value_label
        card._detail_label = detail_label

        return card

    def _create_group_card(
        self,
        group_name: str,
    ) -> tuple[QFrame, dict[str, QLabel]]:
        card = QFrame()
        card.setObjectName("GroupCard")
        card.setMinimumHeight(105)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        name_label = QLabel(group_name)
        name_label.setObjectName("GroupName")

        numbers_label = QLabel("—")
        numbers_label.setObjectName("GroupNumbers")
        numbers_label.setWordWrap(True)

        count_label = QLabel("Wystąpienia: —")
        count_label.setObjectName("GroupCount")

        share_label = QLabel("Udział: —")
        share_label.setObjectName("CardDetail")

        layout.addWidget(name_label)
        layout.addWidget(numbers_label)
        layout.addWidget(count_label)
        layout.addWidget(share_label)
        layout.addStretch()

        return card, {
            "numbers": numbers_label,
            "count": count_label,
            "share": share_label,
        }

    def _configure_table(
        self,
        table: QTableWidget,
        stretch_last: bool = False,
    ):
        table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        table.setAlternatingRowColors(True)

        table.verticalHeader().setVisible(False)

        table.horizontalHeader().setStretchLastSection(
            stretch_last
        )

        table.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

    @staticmethod
    def _center_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)

        item.setTextAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        return item

    @staticmethod
    def _set_card(
        card: QFrame,
        value: str,
        detail: str | None = None,
    ):
        card._value_label.setText(value)

        if detail is not None:
            card._detail_label.setText(detail)

    # ==================================================================
    # DATA
    # ==================================================================

    def refresh_data(self, df: pd.DataFrame | None):
        if df is None:
            self.df = pd.DataFrame()
        else:
            self.df = df.copy()

        self._refresh_analysis()

    def set_data(self, df: pd.DataFrame):
        self.refresh_data(df)

    def update_data(self, df: pd.DataFrame):
        self.refresh_data(df)

    def _select_history_dataframe(self) -> pd.DataFrame:
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        data = self.df.copy()

        if "draw_no" in data.columns:
            data = data.sort_values(
                "draw_no"
            ).reset_index(drop=True)

        if self.current_range == "WSZYSTKIE":
            return data

        try:
            count = int(self.current_range)
        except ValueError:
            return data

        if len(data) <= count:
            return data.reset_index(drop=True)

        return data.tail(count).reset_index(drop=True)

    # ==================================================================
    # RANGE
    # ==================================================================

    def _set_range(self, value: str):
        self.current_range = value

        for key, button in self.range_buttons.items():
            button.setChecked(key == value)

        self._refresh_analysis()

    # ==================================================================
    # REFRESH
    # ==================================================================

    def _on_refresh_clicked(self):
        self._refresh_analysis()
        self.refresh_requested.emit()

    def _refresh_analysis(self):
        data = self._select_history_dataframe()

        if data.empty:
            self._clear_analysis()
            return

        self.range_info.setText(
            f"Analizowane: {len(data)} losowań"
        )

        self._update_last_draw(data)
        self._update_groups(data)
        self._update_stats(data)
        self._update_frequency(data)
        self._update_history(data)

    # ==================================================================
    # LAST DRAW
    # ==================================================================

    def _update_last_draw(self, data: pd.DataFrame):
        numbers = last_draw_numbers(data)

        if not numbers:
            self._set_card(
                self.last_draw_card,
                "—",
                "Brak danych",
            )

            self._set_card(
                self.hot_card,
                "—",
                "Brak danych",
            )

            self._set_card(
                self.cold_card,
                "—",
                "Brak danych",
            )

            return

        last_text = "  ".join(
            str(number)
            for number in numbers
        )

        self._set_card(
            self.last_draw_card,
            last_text,
            "Ostatni zestaw w wybranym oknie",
        )

        hot = hottest_numbers(data)
        cold = coldest_numbers(data)

        hot_text = ", ".join(
            str(number)
            for number in hot[:10]
        )

        cold_text = ", ".join(
            str(number)
            for number in cold[:10]
        )

        self._set_card(
            self.hot_card,
            hot_text or "—",
            "TOP 10 według częstości",
        )

        self._set_card(
            self.cold_card,
            cold_text or "—",
            "BOTTOM 10 według częstości",
        )

    # ==================================================================
    # GROUPS
    # ==================================================================

    def _update_groups(self, data: pd.DataFrame):
        groups_tuple = build_groups(data)

        groups = {
            "G1": set(groups_tuple[0]),
            "G2": set(groups_tuple[1]),
            "G3": set(groups_tuple[2]),
        }

        balance = group_balance(data)

        total_occurrences = sum(
            balance.values()
        )

        self.groups_info.setText(
            "Podział wynika z częstości liczb w wybranym oknie. "
            "Wystąpienia grup pokazują rzeczywisty rozkład "
            "losowań w tym zakresie."
        )

        for group_name in (
            "G1",
            "G2",
            "G3",
        ):
            numbers = sorted(
                groups[group_name]
            )

            count = balance.get(
                group_name,
                0,
            )

            numbers_text = ", ".join(
                str(number)
                for number in numbers
            )

            if total_occurrences:
                share = (
                    count
                    / total_occurrences
                    * 100
                )

                share_text = (
                    f"Udział: {share:.1f}%"
                )
            else:
                share_text = "Udział: —"

            labels = self.group_cards[
                group_name
            ]

            labels["numbers"].setText(
                numbers_text or "—"
            )

            labels["count"].setText(
                f"Wystąpienia: {count}"
            )

            labels["share"].setText(
                share_text
            )

    # ==================================================================
    # STATS
    # ==================================================================

    def _update_stats(self, data: pd.DataFrame):
        last_numbers = last_draw_numbers(data)

        repeated = repeated_from_last_draw(
            last_numbers,
            data,
        )

        neighbors = neighbors_from_last_draw(
            last_numbers,
            data,
        )

        internal_pairs = count_internal_adjacent_pairs(
            last_numbers
        )

        pairs = pair_counter(data)
        triples = triple_counter(data)

        top_pairs = pairs.most_common(3)
        top_triples = triples.most_common(3)

        repeated_text = (
            ", ".join(
                str(number)
                for number in repeated
            )
            if repeated
            else "brak"
        )

        neighbors_text = (
            ", ".join(
                str(number)
                for number in neighbors
            )
            if neighbors
            else "brak"
        )

        pair_text = self._format_combinations(
            top_pairs
        )

        triple_text = self._format_combinations(
            top_triples
        )

        self._set_card(
            self.repeated_card,
            str(len(repeated)),
            repeated_text,
        )

        self._set_card(
            self.neighbor_card,
            str(len(neighbors)),
            neighbors_text,
        )

        self._set_card(
            self.adjacent_card,
            str(internal_pairs),
            "W ostatnim losowaniu",
        )

        self._set_card(
            self.pairs_card,
            pair_text or "—",
            "Liczba w nawiasie = wystąpienia",
        )

        self._set_card(
            self.triples_card,
            triple_text or "—",
            "Liczba w nawiasie = wystąpienia",
        )

        self._set_card(
            self.draws_card,
            str(len(data)),
            "Losowania użyte w analizie",
        )

    # ==================================================================
    # FREQUENCY
    # ==================================================================

    def _update_frequency(self, data: pd.DataFrame):
        frequency = number_frequency(data)

        groups_tuple = build_groups(data)

        groups = {
            "G1": set(groups_tuple[0]),
            "G2": set(groups_tuple[1]),
            "G3": set(groups_tuple[2]),
        }

        rows = []

        for number, count in frequency.items():
            try:
                number_int = int(number)
            except Exception:
                continue

            if number_int in groups["G1"]:
                group = "G1"
            elif number_int in groups["G2"]:
                group = "G2"
            elif number_int in groups["G3"]:
                group = "G3"
            else:
                group = "—"

            rows.append(
                (
                    number_int,
                    int(count),
                    group,
                )
            )

        rows.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        self.frequency_table.setRowCount(
            len(rows)
        )

        for row_index, (
            number,
            count,
            group,
        ) in enumerate(rows):
            self.frequency_table.setItem(
                row_index,
                0,
                self._center_item(
                    str(row_index + 1)
                ),
            )

            self.frequency_table.setItem(
                row_index,
                1,
                self._center_item(
                    str(number)
                ),
            )

            self.frequency_table.setItem(
                row_index,
                2,
                self._center_item(
                    str(count)
                ),
            )

            self.frequency_table.setItem(
                row_index,
                3,
                self._center_item(
                    group
                ),
            )

        self.frequency_table.setColumnWidth(0, 65)
        self.frequency_table.setColumnWidth(1, 65)
        self.frequency_table.setColumnWidth(2, 90)
        self.frequency_table.setColumnWidth(3, 60)

        self.frequency_info.setText(
            f"49 liczb • {len(data)} losowań • "
            "od najczęstszej do najrzadszej"
        )

    # ==================================================================
    # HISTORY
    # ==================================================================

    def _update_history(self, data: pd.DataFrame):
        rows = (
            data
            .iloc[::-1]
            .reset_index(drop=True)
        )

        self.history_table.setRowCount(
            len(rows)
        )

        for row_index, (
            _,
            row,
        ) in enumerate(rows.iterrows()):
            draw_no = row.get(
                "draw_no",
                "",
            )

            draw_date = row.get(
                "date",
                "",
            )

            self.history_table.setItem(
                row_index,
                0,
                self._center_item(
                    str(draw_no)
                ),
            )

            self.history_table.setItem(
                row_index,
                1,
                self._center_item(
                    str(draw_date)
                ),
            )

            for column_index, column_name in enumerate(
                (
                    "n1",
                    "n2",
                    "n3",
                    "n4",
                    "n5",
                    "n6",
                ),
                start=2,
            ):
                value = row.get(
                    column_name,
                    "",
                )

                self.history_table.setItem(
                    row_index,
                    column_index,
                    self._center_item(
                        str(value)
                    ),
                )

        self.history_table.resizeColumnsToContents()

        self.history_info.setText(
            f"Pokazano wszystkie {len(rows)} "
            "losowania z wybranego zakresu"
        )

    # ==================================================================
    # CLEAR
    # ==================================================================

    def _clear_analysis(self):
        self.range_info.setText(
            "Brak danych"
        )

        self._set_card(
            self.last_draw_card,
            "—",
            "Brak danych",
        )

        self._set_card(
            self.hot_card,
            "—",
            "Brak danych",
        )

        self._set_card(
            self.cold_card,
            "—",
            "Brak danych",
        )

        self.groups_info.setText(
            "Brak danych."
        )

        for group_name in (
            "G1",
            "G2",
            "G3",
        ):
            labels = self.group_cards[
                group_name
            ]

            labels["numbers"].setText("—")
            labels["count"].setText("Wystąpienia: —")
            labels["share"].setText("Udział: —")

        for card in (
            self.repeated_card,
            self.neighbor_card,
            self.adjacent_card,
            self.pairs_card,
            self.triples_card,
            self.draws_card,
        ):
            self._set_card(
                card,
                "—",
                "",
            )

        self.frequency_table.setRowCount(0)
        self.frequency_info.setText("Brak danych.")

        self.history_table.setRowCount(0)
        self.history_info.setText("Brak danych.")

    # ==================================================================
    # FORMAT
    # ==================================================================

    @staticmethod
    def _format_combinations(
        combinations,
    ) -> str:
        if not combinations:
            return ""

        parts = []

        for numbers, count in combinations:
            numbers_text = "-".join(
                str(number)
                for number in numbers
            )

            parts.append(
                f"{numbers_text} ({count})"
            )

        return ", ".join(parts)