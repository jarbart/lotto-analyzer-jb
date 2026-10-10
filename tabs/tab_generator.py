from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import (
    QObject,
    QThread,
    pyqtSignal,
    pyqtSlot,
    Qt,
)
from PyQt6.QtGui import QKeySequence, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from generator import (
    CandidateResult,
    GeneratorConfig,
    generate_sets,
)

from tabs.tab_history import (
    append_generated_results,
    mark_historical_repeats,
)


class GeneratorWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(
        self,
        df,
        config: GeneratorConfig,
    ):
        super().__init__()

        self.df = df
        self.config = config

    @pyqtSlot()
    def run(self):
        try:
            self.progress.emit(10)

            results = generate_sets(
                self.df,
                self.config,
            )

            self.progress.emit(100)

            self.finished.emit(
                results
            )

        except Exception as exc:
            self.error.emit(
                str(exc)
            )


class GeneratorTab(QWidget):
    """
    Główna zakładka generatora heurystycznego.

    Ustawienia użytkownika:
        - liczba zestawów,
        - zakres historii,
        - opcjonalna maska,
        - siła maski.

    Parametry techniczne generatora pozostają
    ukryte.

    Maska jest opcjonalna i domyślnie wyłączona.
    Aktualizacja historii jest dostępna bezpośrednio
    z zakładki Generator.
    """

    update_history_requested = pyqtSignal()

    def __init__(
        self,
        df=None,
        parent=None,
    ):
        super().__init__(parent)

        self.df = df
        self.results: list[
            CandidateResult
        ] = []

        self.thread: Optional[
            QThread
        ] = None

        self.worker: Optional[
            GeneratorWorker
        ] = None

        self._build_ui()
        self._apply_style()
        self._update_mask_controls()
        self._refresh_data_info()

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
            "GENERATOR HEURYSTYCZNY"
        )

        title.setObjectName(
            "titleLabel"
        )

        subtitle = QLabel(
            "Zestawy tworzone na podstawie "
            "historii losowań i wybranych warunków."
        )

        subtitle.setObjectName(
            "subtitleLabel"
        )

        title_box.addWidget(
            title
        )

        title_box.addWidget(
            subtitle
        )

        header_layout.addLayout(
            title_box
        )

        header_layout.addStretch()

        self.update_history_button = QPushButton(
            "↻  AKTUALIZUJ HISTORIĘ"
        )

        self.update_history_button.setObjectName(
            "secondaryButton"
        )

        self.update_history_button.setMinimumHeight(
            40
        )

        self.update_history_button.setMinimumWidth(
            190
        )

        self.update_history_button.setToolTip(
            "Pobiera najnowsze wyniki i aktualizuje "
            "lokalną historię losowań."
        )

        self.update_history_button.clicked.connect(
            self._request_history_update
        )

        header_layout.addWidget(
            self.update_history_button
        )

        root.addWidget(
            header
        )

        # ==========================================================
        # SPLITTER
        # ==========================================================

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        splitter.setChildrenCollapsible(
            False
        )

        # ==========================================================
        # LEWY PANEL
        # ==========================================================

        left_panel = QWidget()

        left_layout = QVBoxLayout(
            left_panel
        )

        left_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        left_layout.setSpacing(10)

        # ==========================================================
        # USTAWIENIA
        # ==========================================================

        settings_group = QGroupBox(
            "USTAWIENIA GENEROWANIA"
        )

        settings_layout = QGridLayout(
            settings_group
        )

        settings_layout.setContentsMargins(
            14,
            16,
            14,
            14,
        )

        settings_layout.setHorizontalSpacing(
            12
        )

        settings_layout.setVerticalSpacing(
            12
        )

        sets_label = QLabel(
            "Liczba zestawów:"
        )

        sets_label.setObjectName(
            "fieldLabel"
        )

        self.result_count_spin = QSpinBox()

        self.result_count_spin.setRange(
            1,
            50,
        )

        self.result_count_spin.setValue(
            10
        )

        self.result_count_spin.setSingleStep(
            1
        )

        self.result_count_spin.setButtonSymbols(
            QSpinBox.ButtonSymbols.UpDownArrows
        )

        self.result_count_spin.setMinimumHeight(
            34
        )

        self.result_count_spin.setToolTip(
            "Ile najlepszych, zróżnicowanych "
            "zestawów ma zostać pokazanych."
        )

        settings_layout.addWidget(
            sets_label,
            0,
            0,
        )

        settings_layout.addWidget(
            self.result_count_spin,
            0,
            1,
        )

        history_label = QLabel(
            "Zakres historii:"
        )

        history_label.setObjectName(
            "fieldLabel"
        )

        self.history_combo = QComboBox()

        self.history_combo.setMinimumHeight(
            34
        )

        self.history_combo.addItem(
            "30 ostatnich losowań",
            30,
        )

        self.history_combo.addItem(
            "50 ostatnich losowań",
            50,
        )

        self.history_combo.addItem(
            "100 ostatnich losowań",
            100,
        )

        self.history_combo.addItem(
            "200 ostatnich losowań",
            200,
        )

        self.history_combo.addItem(
            "500 ostatnich losowań",
            500,
        )

        self.history_combo.addItem(
            "Wszystkie losowania",
            None,
        )

        self.history_combo.setCurrentIndex(
            2
        )

        self.history_combo.setToolTip(
            "Tylko ten zakres historii będzie używany "
            "do analizy częstotliwości, grup, HOT/COLD, "
            "par, trójek i warunków."
        )

        settings_layout.addWidget(
            history_label,
            1,
            0,
        )

        settings_layout.addWidget(
            self.history_combo,
            1,
            1,
        )

        left_layout.addWidget(
            settings_group
        )

        # ==========================================================
        # MASKA
        # ==========================================================

        mask_group = QGroupBox(
            "OPCJONALNA MASKA LICZB"
        )

        mask_layout = QVBoxLayout(
            mask_group
        )

        mask_layout.setContentsMargins(
            14,
            16,
            14,
            14,
        )

        mask_layout.setSpacing(8)

        self.mask_enabled_checkbox = QCheckBox(
            "Używaj maski"
        )

        self.mask_enabled_checkbox.setChecked(
            False
        )

        self.mask_enabled_checkbox.setObjectName(
            "maskCheckBox"
        )

        self.mask_enabled_checkbox.setToolTip(
            "Domyślnie maska nie wpływa na generowanie. "
            "Włącz ją tylko wtedy, gdy chcesz preferować "
            "własny zestaw 6 liczb."
        )

        self.mask_enabled_checkbox.toggled.connect(
            self._update_mask_controls
        )

        mask_layout.addWidget(
            self.mask_enabled_checkbox
        )

        mask_info = QLabel(
            "Wpisz 6 liczb z zakresu 1–49. "
            "Kolejność nie ma znaczenia."
        )

        mask_info.setObjectName(
            "mutedLabel"
        )

        self.mask_edit = QLineEdit()

        self.mask_edit.setText(
            "4, 15, 17, 20, 21, 34"
        )

        self.mask_edit.setMinimumHeight(
            38
        )

        self.mask_edit.setPlaceholderText(
            "np. 4, 15, 17, 20, 21, 34"
        )

        self.mask_edit.setToolTip(
            "Własna maska 6 liczb. "
            "Nie jest używana, gdy opcja maski jest wyłączona."
        )

        mask_layout.addWidget(
            mask_info
        )

        mask_layout.addWidget(
            self.mask_edit
        )

        mask_strength_layout = QHBoxLayout()
        mask_strength_layout.setSpacing(10)

        mask_strength_label = QLabel(
            "Siła maski:"
        )

        mask_strength_label.setObjectName(
            "fieldLabel"
        )

        self.mask_strength_spin = QSpinBox()

        self.mask_strength_spin.setRange(
            0,
            100,
        )

        self.mask_strength_spin.setValue(
            45
        )

        self.mask_strength_spin.setSingleStep(
            5
        )

        self.mask_strength_spin.setSuffix(
            "%"
        )

        self.mask_strength_spin.setButtonSymbols(
            QSpinBox.ButtonSymbols.UpDownArrows
        )

        self.mask_strength_spin.setMinimumHeight(
            34
        )

        self.mask_strength_spin.setToolTip(
            "Jak mocno generator ma preferować liczby "
            "z maski. 0% = brak wpływu, "
            "100% = bardzo silna preferencja."
        )

        mask_strength_layout.addWidget(
            mask_strength_label
        )

        mask_strength_layout.addWidget(
            self.mask_strength_spin
        )

        mask_strength_layout.addStretch()

        mask_layout.addLayout(
            mask_strength_layout
        )

        mask_status = QLabel(
            "Domyślnie: wyłączona"
        )

        mask_status.setObjectName(
            "mutedLabel"
        )

        self.mask_status_label = mask_status

        mask_layout.addWidget(
            mask_status
        )

        left_layout.addWidget(
            mask_group
        )

        # ==========================================================
        # AKCJE
        # ==========================================================

        action_group = QGroupBox(
            "AKCJE"
        )

        action_layout = QVBoxLayout(
            action_group
        )

        action_layout.setContentsMargins(
            14,
            16,
            14,
            14,
        )

        action_layout.setSpacing(8)

        self.generate_button = QPushButton(
            "▶  GENERUJ ZESTAWY"
        )

        self.generate_button.setObjectName(
            "primaryButton"
        )

        self.generate_button.setMinimumHeight(
            46
        )

        self.generate_button.setToolTip(
            "Generuje zestawy na podstawie wybranego "
            "zakresu historii i heurystyk."
        )

        self.generate_button.clicked.connect(
            self._start_generation
        )

        action_layout.addWidget(
            self.generate_button
        )

        copy_buttons_layout = QHBoxLayout()
        copy_buttons_layout.setSpacing(8)

        self.copy_set_button = QPushButton(
            "⧉  KOPIUJ ZESTAW"
        )

        self.copy_set_button.setObjectName(
            "secondaryButton"
        )

        self.copy_set_button.setMinimumHeight(
            36
        )

        self.copy_set_button.setToolTip(
            "Kopiuje liczby aktualnie wybranego zestawu."
        )

        self.copy_set_button.clicked.connect(
            self._copy_selected_set
        )

        copy_buttons_layout.addWidget(
            self.copy_set_button
        )

        self.copy_details_button = QPushButton(
            "⧉  KOPIUJ SZCZEGÓŁY"
        )

        self.copy_details_button.setObjectName(
            "secondaryButton"
        )

        self.copy_details_button.setMinimumHeight(
            36
        )

        self.copy_details_button.setToolTip(
            "Kopiuje pełne informacje o aktualnie "
            "wybranym zestawie."
        )

        self.copy_details_button.clicked.connect(
            self._copy_selected_details
        )

        copy_buttons_layout.addWidget(
            self.copy_details_button
        )

        action_layout.addLayout(
            copy_buttons_layout
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100,
        )

        self.progress_bar.setValue(
            0
        )

        self.progress_bar.setTextVisible(
            True
        )

        self.progress_bar.setMinimumHeight(
            22
        )

        action_layout.addWidget(
            self.progress_bar
        )

        self.status_label = QLabel(
            "Gotowy."
        )

        self.status_label.setObjectName(
            "statusLabel"
        )

        action_layout.addWidget(
            self.status_label
        )

        left_layout.addWidget(
            action_group
        )

        # ==========================================================
        # AKTUALNE DANE
        # ==========================================================

        data_group = QGroupBox(
            "AKTUALNE DANE"
        )

        data_layout = QVBoxLayout(
            data_group
        )

        data_layout.setContentsMargins(
            14,
            16,
            14,
            14,
        )

        data_layout.setSpacing(5)

        self.data_info_label = QLabel()

        self.data_info_label.setWordWrap(
            True
        )

        self.data_info_label.setObjectName(
            "mutedLabel"
        )

        data_layout.addWidget(
            self.data_info_label
        )

        left_layout.addWidget(
            data_group
        )

        left_layout.addStretch()

        # ==========================================================
        # PRAWY PANEL
        # ==========================================================

        right_panel = QWidget()

        right_layout = QVBoxLayout(
            right_panel
        )

        right_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        right_layout.setSpacing(10)


        # ==============================================================
        # POWTÓRKI HISTORYCZNE
        # ==============================================================

        def _historical_repeats(self):
            return mark_historical_repeats(
                self.df
            )

        # ==========================================================
        # WYNIKI
        # ==========================================================

        results_group = QGroupBox(
            "WYNIKI"
        )

        results_layout = QVBoxLayout(
            results_group
        )

        results_layout.setContentsMargins(
            10,
            14,
            10,
            10,
        )

        self.results_table = QTableWidget()

        self.results_table.setColumnCount(
            9
        )

        self.results_table.setHorizontalHeaderLabels(
            [
                "#",
                "ZESTAW",
                "WYNIK",
                "G1",
                "G2",
                "G3",
                "WARUNKI",
                "MASKA",
                "STATUS",
            ]
        )

        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        self.results_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )

        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.results_table.verticalHeader().setVisible(
            False
        )

        self.results_table.setAlternatingRowColors(
            True
        )

        self.results_table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )

        self.results_table.customContextMenuRequested.connect(
            self._show_table_context_menu
        )

        header_view = (
            self.results_table.horizontalHeader()
        )

        header_view.setStretchLastSection(
            True
        )

        header_view.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            4,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            5,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            6,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            7,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            8,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )

        header_view.resizeSection(
            0,
            42,
        )

        header_view.resizeSection(
            2,
            85,
        )

        header_view.resizeSection(
            3,
            48,
        )

        header_view.resizeSection(
            4,
            48,
        )

        header_view.resizeSection(
            5,
            48,
        )

        header_view.resizeSection(
            6,
            88,
        )

        header_view.resizeSection(
            7,
            72,
        )

        header_view.resizeSection(
            8,
            70,
        )

        self.results_table.itemSelectionChanged.connect(
            self._show_selected_result
        )

        results_layout.addWidget(
            self.results_table
        )

        right_layout.addWidget(
            results_group,
            3,
        )

        # ==========================================================
        # SZCZEGÓŁY
        # ==========================================================

        details_group = QGroupBox(
            "SZCZEGÓŁY WYBRANEGO ZESTAWU"
        )

        details_layout = QVBoxLayout(
            details_group
        )

        details_layout.setContentsMargins(
            14,
            14,
            14,
            14,
        )

        self.details_label = QLabel(
            "Wybierz zestaw z tabeli, "
            "aby zobaczyć szczegóły."
        )

        self.details_label.setWordWrap(
            True
        )

        self.details_label.setAlignment(
            Qt.AlignmentFlag.AlignTop
        )

        details_layout.addWidget(
            self.details_label
        )

        right_layout.addWidget(
            details_group,
            1,
        )

        splitter.addWidget(
            left_panel
        )

        splitter.addWidget(
            right_panel
        )

        splitter.setStretchFactor(
            0,
            0,
        )

        splitter.setStretchFactor(
            1,
            1,
        )

        splitter.setSizes(
            [
                350,
                900,
            ]
        )

        root.addWidget(
            splitter,
            1,
        )

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

            #fieldLabel {
                color: #d7dce3;
                font-weight: 600;
            }

            #mutedLabel {
                color: #9da5b0;
            }

            #statusLabel {
                color: #b9c0ca;
                padding: 2px 0;
            }

            #maskCheckBox {
                color: #dce2ea;
                font-weight: 700;
                spacing: 8px;
            }

            #maskCheckBox::indicator {
                width: 18px;
                height: 18px;
            }

            #maskCheckBox::indicator:unchecked {
                background: #171a1f;
                border: 1px solid #59616e;
                border-radius: 4px;
            }

            #maskCheckBox::indicator:checked {
                background: #4969d8;
                border: 1px solid #607be0;
                border-radius: 4px;
            }

            QLineEdit,
            QComboBox,
            QSpinBox {
                background: #171a1f;
                border: 1px solid #4a515d;
                border-radius: 5px;
                color: #f1f3f5;
                padding: 5px 8px;
                min-height: 28px;
            }

            QLineEdit:focus,
            QComboBox:focus,
            QSpinBox:focus {
                border: 1px solid #6d8cff;
            }

            QLineEdit:disabled,
            QSpinBox:disabled {
                background: #20242b;
                color: #666e79;
                border-color: #343a43;
            }

            QSpinBox::up-button,
            QSpinBox::down-button {
                width: 24px;
                border-left: 1px solid #4a515d;
                background: #2a2f38;
            }

            QSpinBox::up-button:hover,
            QSpinBox::down-button:hover {
                background: #39414d;
            }

            QComboBox::drop-down {
                width: 28px;
                border-left: 1px solid #4a515d;
            }

            QPushButton {
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 700;
            }

            #primaryButton {
                background: #4969d8;
                color: white;
                border: 1px solid #607be0;
            }

            #primaryButton:hover {
                background: #5877e5;
            }

            #primaryButton:pressed {
                background: #3e5cc5;
            }

            #primaryButton:disabled {
                background: #303641;
                color: #737b87;
                border-color: #414854;
            }

            #secondaryButton {
                background: #2a3039;
                color: #dce2ea;
                border: 1px solid #515966;
            }

            #secondaryButton:hover {
                background: #343b47;
            }

            #secondaryButton:pressed {
                background: #252a32;
            }

            #secondaryButton:disabled {
                background: #232830;
                color: #666e79;
                border-color: #363c46;
            }

            QProgressBar {
                background: #171a1f;
                border: 1px solid #3f4650;
                border-radius: 5px;
                text-align: center;
                color: #dce2ea;
            }

            QProgressBar::chunk {
                background: #4969d8;
                border-radius: 4px;
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

            QMenu {
                background: #20242b;
                color: #e5e9ee;
                border: 1px solid #4a515d;
            }

            QMenu::item {
                padding: 7px 22px 7px 10px;
            }

            QMenu::item:selected {
                background: #3d568f;
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
    # MASKA — UI
    # ==============================================================

    def _update_mask_controls(self):
        enabled = (
            self.mask_enabled_checkbox.isChecked()
        )

        self.mask_edit.setEnabled(
            enabled
        )

        self.mask_strength_spin.setEnabled(
            enabled
        )

        if enabled:
            self.mask_status_label.setText(
                "Maska aktywna."
            )
        else:
            self.mask_status_label.setText(
                "Domyślnie: wyłączona — "
                "maska nie wpływa na generowanie."
            )

    # ==============================================================
    # DANE
    # ==============================================================

    def set_data(self, df):
        self.df = df
        self.results = []

        self.results_table.setRowCount(
            0
        )

        self.details_label.setText(
            "Wybierz zestaw z tabeli, "
            "aby zobaczyć szczegóły."
        )

        self._refresh_data_info()

    def set_dataframe(self, df):
        self.set_data(df)

    def _refresh_data_info(self):
        if self.df is None:
            self.data_info_label.setText(
                "Brak wczytanej historii losowań."
            )
            return

        try:
            draws = len(
                self.df
            )

            if draws:
                first_draw = int(
                    self.df["draw_no"].min()
                )

                last_draw = int(
                    self.df["draw_no"].max()
                )

                self.data_info_label.setText(
                    f"Losowań: {draws}\n"
                    f"Zakres numerów losowań: "
                    f"{first_draw}–{last_draw}"
                )

            else:
                self.data_info_label.setText(
                    "Historia jest pusta."
                )

        except Exception:
            self.data_info_label.setText(
                f"Wczytanych rekordów: "
                f"{len(self.df)}"
            )

    # ==============================================================
    # AKTUALIZACJA HISTORII
    # ==============================================================

    def _request_history_update(self):
        self.update_history_requested.emit()

    # ==============================================================
    # KONFIGURACJA
    # ==============================================================

    def _parse_mask(self):
        raw = self.mask_edit.text().strip()

        if not raw:
            return ()

        raw = raw.replace(
            ";",
            ",",
        )

        parts = [
            part.strip()
            for part in raw.split(",")
            if part.strip()
        ]

        try:
            numbers = [
                int(part)
                for part in parts
            ]

        except ValueError:
            raise ValueError(
                "Maska może zawierać wyłącznie "
                "liczby całkowite."
            )

        if len(numbers) != 6:
            raise ValueError(
                "Maska musi zawierać dokładnie "
                "6 liczb."
            )

        if len(set(numbers)) != 6:
            raise ValueError(
                "Liczby w masce nie mogą się "
                "powtarzać."
            )

        if any(
            number < 1 or number > 49
            for number in numbers
        ):
            raise ValueError(
                "Wszystkie liczby maski muszą "
                "mieścić się w zakresie 1–49."
            )

        return tuple(
            sorted(numbers)
        )

    def _selected_history_window(self):
        return self.history_combo.currentData()

    def _build_config(self):
        config = GeneratorConfig()

        config.result_count = (
            self.result_count_spin.value()
        )

        if self.mask_enabled_checkbox.isChecked():
            mask = self._parse_mask()

            strength = (
                self.mask_strength_spin.value()
                / 100.0
            )

            config.mask = mask

            config.weight_mask = (
                4.5 * strength
            )

        else:
            config.mask = ()

            config.weight_mask = 0.0

        return config

    def _select_history_dataframe(self):
        if self.df is None:
            raise ValueError(
                "Brak historii losowań."
            )

        window = (
            self._selected_history_window()
        )

        if window is None:
            return self.df.copy()

        if len(self.df) <= window:
            return self.df.copy()

        return self.df.tail(
            window
        ).copy()

    # ==============================================================
    # GENEROWANIE
    # ==============================================================

    def _start_generation(self):
        if (
            self.thread is not None
            and self.thread.isRunning()
        ):
            return

        if (
            self.df is None
            or len(self.df) == 0
        ):
            QMessageBox.warning(
                self,
                "Brak danych",
                "Nie ma wczytanej historii losowań.",
            )
            return

        try:
            history_df = (
                self._select_history_dataframe()
            )

            config = self._build_config()

        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Nieprawidłowe ustawienia",
                str(exc),
            )
            return

        self.results = []

        self.results_table.setRowCount(
            0
        )

        self.details_label.setText(
            "Generowanie zestawów..."
        )

        self.generate_button.setEnabled(
            False
        )

        self.update_history_button.setEnabled(
            False
        )

        self.copy_set_button.setEnabled(
            False
        )

        self.copy_details_button.setEnabled(
            False
        )

        self.progress_bar.setValue(
            0
        )

        mask_status = (
            " z aktywną maską"
            if self.mask_enabled_checkbox.isChecked()
            else ""
        )

        self.status_label.setText(
            "Generowanie na podstawie "
            f"{len(history_df)} losowań"
            f"{mask_status}..."
        )

        self.thread = QThread(
            self
        )

        self.worker = GeneratorWorker(
            history_df,
            config,
        )

        self.worker.moveToThread(
            self.thread
        )

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.progress.connect(
            self.progress_bar.setValue
        )

        self.worker.finished.connect(
            self._generation_finished
        )

        self.worker.error.connect(
            self._generation_error
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.error.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self._generation_thread_finished
        )

        self.thread.start()

    # ==============================================================
    # PROFIL GRUP
    # ==============================================================

    @staticmethod
    def _normalize_group_profile(
        profile,
    ):
        if isinstance(
            profile,
            dict,
        ):
            return {
                "G1": int(
                    profile.get(
                        "G1",
                        0,
                    )
                ),
                "G2": int(
                    profile.get(
                        "G2",
                        0,
                    )
                ),
                "G3": int(
                    profile.get(
                        "G3",
                        0,
                    )
                ),
            }

        if isinstance(
            profile,
            (tuple, list),
        ):
            values = list(
                profile
            )

            while len(values) < 3:
                values.append(0)

            return {
                "G1": int(
                    values[0]
                ),
                "G2": int(
                    values[1]
                ),
                "G3": int(
                    values[2]
                ),
            }

        return {
            "G1": 0,
            "G2": 0,
            "G3": 0,
        }

    # ==============================================================
    # WYNIKI
    # ==============================================================

    def _generation_finished(
        self,
        results,
    ):
        try:
            self.results = list(
                results
            )

        except Exception:
            self.results = []

        self._populate_results()

        try:
            target_draw_no = (
                int(
                    self.df["draw_no"].max()
                )
                + 1
            )

            window = (
                self._selected_history_window()
            )

            if window is None:
                history_window_text = (
                    "Wszystkie"
                )
            else:
                history_window_text = (
                    f"{window}"
                )

            append_generated_results(
                results=self.results,
                target_draw_no=target_draw_no,
                history_window=history_window_text,
                mask_enabled=(
                    self.mask_enabled_checkbox.isChecked()
                ),
                mask_strength=(
                    self.mask_strength_spin.value()
                ),
            )

        except Exception as exc:
            QMessageBox.warning(
                self,
                "Historia typów",
                (
                    "Zestawy zostały wygenerowane, "
                    "ale nie udało się ich zapisać "
                    "do historii.\n\n"
                    f"{exc}"
                ),
            )

        self.progress_bar.setValue(
            100
        )

        if self.results:
            self.status_label.setText(
                "Gotowe. Wygenerowano "
                f"{len(self.results)} zestawów."
            )

        else:
            self.status_label.setText(
                "Generator nie zwrócił zestawów."
            )

    def _generation_error(
        self,
        message,
    ):
        self.status_label.setText(
            "Błąd podczas generowania."
        )

        QMessageBox.critical(
            self,
            "Błąd generatora",
            message,
        )

    def _generation_thread_finished(self):
        self.generate_button.setEnabled(
            True
        )

        self.update_history_button.setEnabled(
            True
        )

        self.copy_set_button.setEnabled(
            bool(self.results)
        )

        self.copy_details_button.setEnabled(
            bool(self.results)
        )

        if self.thread is not None:
            self.thread.deleteLater()

        self.thread = None
        self.worker = None

    def _historical_repeats(self):
        """Zwraca wszystkie zestawy z historycznych losowań."""
        return mark_historical_repeats(self.df)

    def _populate_results(self):
        self.results_table.setRowCount(
            0
        )

        historical_sets = (
            self._historical_repeats()
        )

        for row, result in enumerate(
            self.results
        ):
            self.results_table.insertRow(
                row
            )

            profile = (
                self._normalize_group_profile(
                    result.group_profile
                )
            )

            numbers_text = ", ".join(
                str(number)
                for number in result.numbers
            )

            conditions_text = (
                f"{result.condition_count} / 6"
            )

            mask_text = (
                f"{len(result.mask_numbers)} / 6"
            )

            candidate_key = tuple(
                sorted(
                    int(number)
                    for number in result.numbers
                )
            )

            is_repeat = (
                candidate_key
                in historical_sets
            )

            status_parts = []

            if row == 0:
                status_parts.append(
                    "★  TOP 1"
                )

            if is_repeat:
                status_parts.append(
                    "↻  POWTÓRKA"
                )

            status = " • ".join(
                status_parts
            )

            values = [
                str(row + 1),
                numbers_text,
                f"{result.score:.3f}",
                str(profile["G1"]),
                str(profile["G2"]),
                str(profile["G3"]),
                conditions_text,
                mask_text,
                status,
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

                if row == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self.results_table.setItem(
                    row,
                    column,
                    item
                )

        header_view = (
            self.results_table.horizontalHeader()
        )

        header_view.setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            4,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            5,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            6,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            7,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            8,
            QHeaderView.ResizeMode.Fixed,
        )

        header_view.setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.Stretch,
        )

        header_view.resizeSection(
            0,
            42,
        )

        header_view.resizeSection(
            2,
            85,
        )

        header_view.resizeSection(
            3,
            48,
        )

        header_view.resizeSection(
            4,
            48,
        )

        header_view.resizeSection(
            5,
            48,
        )

        header_view.resizeSection(
            6,
            88,
        )

        header_view.resizeSection(
            7,
            72,
        )

        header_view.resizeSection(
            8,
            70,
        )

        if self.results:
            self.results_table.selectRow(
                0
            )

    # ==============================================================
    # WYBRANY WYNIK
    # ==============================================================

    def _selected_result(self):
        selected = (
            self.results_table.selectedItems()
        )

        if not selected:
            return None

        row = selected[0].row()

        if (
            row < 0
            or row >= len(self.results)
        ):
            return None

        return self.results[row]

    def _show_selected_result(self):
        result = (
            self._selected_result()
        )

        if result is None:
            self.copy_set_button.setEnabled(
                False
            )

            self.copy_details_button.setEnabled(
                False
            )

            return

        self.copy_set_button.setEnabled(
            True
        )

        self.copy_details_button.setEnabled(
            True
        )

        self.details_label.setText(
            self._build_details_html(
                result
            )
        )

    # ==============================================================
    # SZCZEGÓŁY
    # ==============================================================

    def _build_details_html(
        self,
        result,
    ):
        profile = (
            self._normalize_group_profile(
                result.group_profile
            )
        )

        numbers = ", ".join(
            str(number)
            for number in result.numbers
        )

        mask_numbers = ", ".join(
            str(number)
            for number in result.mask_numbers
        )

        conditions = [
            (
                "✓"
                if result.repeated
                else "✗",
                "Powtórzenie z ostatniego",
            ),
            (
                "✓"
                if result.neighbors
                else "✗",
                "Sąsiad ostatniego",
            ),
            (
                "✓"
                if result.internal_pair
                else "✗",
                "Para wewnętrzna",
            ),
            (
                "✓"
                if result.hot
                else "✗",
                "HOT",
            ),
            (
                "✓"
                if result.cold
                else "✗",
                "COLD",
            ),
            (
                "✓"
                if result.hot_cold_together
                else "✗",
                "HOT + COLD",
            ),
        ]

        condition_lines = "<br>".join(
            f"{mark}  {name}"
            for mark, name in conditions
        )

        mask_state = (
            "AKTYWNA"
            if self.mask_enabled_checkbox.isChecked()
            else "WYŁĄCZONA"
        )

        mask_value = (
            f"{len(result.mask_numbers)} / 6"
            if self.mask_enabled_checkbox.isChecked()
            else "—"
        )

        mask_hits_text = (
            mask_numbers or "brak"
            if self.mask_enabled_checkbox.isChecked()
            else "maska wyłączona"
        )

        return (
            f"<b>ZESTAW:</b> {numbers}"
            f"<br><br>"
            f"<b>WYNIK HEURYSTYCZNY:</b> "
            f"{result.score:.3f}"
            f"<br><br>"
            f"<b>PROFIL GRUP:</b>"
            f"<br>"
            f"&nbsp;&nbsp;G1: {profile['G1']}"
            f"&nbsp;&nbsp;•&nbsp;&nbsp;"
            f"G2: {profile['G2']}"
            f"&nbsp;&nbsp;•&nbsp;&nbsp;"
            f"G3: {profile['G3']}"
            f"<br><br>"
            f"<b>WARUNKI:</b> "
            f"{result.condition_count} / 6"
            f"<br>"
            f"{condition_lines}"
            f"<br><br>"
            f"<b>MASKA:</b> "
            f"{mask_state}"
            f"<br>"
            f"Trafienia: "
            f"{mask_value}"
            f"<br>"
            f"Trafione liczby: "
            f"{mask_hits_text}"
            f"<br><br>"
            f"<b>JAKOŚĆ BALANSU GRUP:</b> "
            f"{result.balance_quality:.3f}"
            f"<br>"
            f"<b>JAKOŚĆ WARUNKÓW:</b> "
            f"{result.condition_quality:.3f}"
            f"<br>"
            f"<b>JAKOŚĆ HISTORII:</b> "
            f"{result.history_quality:.3f}"
        )

    # ==============================================================
    # TEKST DO KOPIOWANIA
    # ==============================================================

    def _build_result_text(
        self,
        result,
    ):
        profile = (
            self._normalize_group_profile(
                result.group_profile
            )
        )

        numbers = ", ".join(
            str(number)
            for number in result.numbers
        )

        mask_numbers = ", ".join(
            str(number)
            for number in result.mask_numbers
        )

        mask_enabled = (
            self.mask_enabled_checkbox.isChecked()
        )

        return (
            f"ZESTAW: {numbers}\n"
            f"WYNIK HEURYSTYCZNY: "
            f"{result.score:.3f}\n"
            f"GRUPY: "
            f"G1={profile['G1']} | "
            f"G2={profile['G2']} | "
            f"G3={profile['G3']}\n"
            f"WARUNKI: "
            f"{result.condition_count}/6\n"
            f"POWTÓRZENIE Z OSTATNIEGO: "
            f"{'TAK' if result.repeated else 'NIE'}\n"
            f"SĄSIAD OSTATNIEGO: "
            f"{'TAK' if result.neighbors else 'NIE'}\n"
            f"PARA WEWNĘTRZNA: "
            f"{'TAK' if result.internal_pair else 'NIE'}\n"
            f"HOT: "
            f"{'TAK' if result.hot else 'NIE'}\n"
            f"COLD: "
            f"{'TAK' if result.cold else 'NIE'}\n"
            f"HOT + COLD: "
            f"{'TAK' if result.hot_cold_together else 'NIE'}\n"
            f"MASKA: "
            f"{'AKTYWNA' if mask_enabled else 'WYŁĄCZONA'}\n"
            f"TRAFIENIA MASKI: "
            f"{len(result.mask_numbers)}/6"
            f"{' — ' + mask_numbers if mask_enabled and mask_numbers else ''}\n"
            f"JAKOŚĆ BALANSU GRUP: "
            f"{result.balance_quality:.3f}\n"
            f"JAKOŚĆ WARUNKÓW: "
            f"{result.condition_quality:.3f}\n"
            f"JAKOŚĆ HISTORII: "
            f"{result.history_quality:.3f}"
        )

    # ==============================================================
    # KOPIOWANIE
    # ==============================================================

    def _copy_selected_set(self):
        result = (
            self._selected_result()
        )

        if result is None:
            return

        numbers = ", ".join(
            str(number)
            for number in result.numbers
        )

        clipboard = (
            QApplication.clipboard()
        )

        clipboard.setText(
            numbers
        )

        self.status_label.setText(
            "Zestaw skopiowany do schowka."
        )

    def _copy_selected_details(self):
        result = (
            self._selected_result()
        )

        if result is None:
            return

        clipboard = (
            QApplication.clipboard()
        )

        clipboard.setText(
            self._build_result_text(
                result
            )
        )

        self.status_label.setText(
            "Szczegóły skopiowane do schowka."
        )

    # ==============================================================
    # MENU KONTEKSTOWE TABELI
    # ==============================================================

    def _show_table_context_menu(
        self,
        position,
    ):
        result = (
            self._selected_result()
        )

        if result is None:
            return

        menu = QMenu(
            self
        )

        copy_set_action = QAction(
            "Kopiuj zestaw",
            self,
        )

        copy_set_action.triggered.connect(
            self._copy_selected_set
        )

        copy_details_action = QAction(
            "Kopiuj szczegóły",
            self,
        )

        copy_details_action.triggered.connect(
            self._copy_selected_details
        )

        menu.addAction(
            copy_set_action
        )

        menu.addAction(
            copy_details_action
        )

        menu.exec(
            self.results_table.viewport().mapToGlobal(
                position
            )
        )

    # ==============================================================
    # SKRÓT KLAWISZOWY
    # ==============================================================

    def keyPressEvent(self, event):
        if event.matches(
            QKeySequence.StandardKey.Copy
        ):
            self._copy_selected_set()
            event.accept()
            return

        super().keyPressEvent(
            event
        )

    # ==============================================================
    # ZAMKNIĘCIE
    # ==============================================================

    def closeEvent(self, event):
        if (
            self.thread is not None
            and self.thread.isRunning()
        ):
            self.thread.quit()
            self.thread.wait(
                2000
            )

        event.accept()


if __name__ == "__main__":
    import sys
    import pandas as pd

    app = QApplication(
        sys.argv
    )

    demo_df = pd.DataFrame(
        columns=[
            "draw_no",
            "date",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
        ]
    )

    window = GeneratorTab(
        demo_df
    )

    window.resize(
        1400,
        850,
    )

    window.show()

    sys.exit(
        app.exec()
    )