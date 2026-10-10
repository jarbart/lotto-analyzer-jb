from __future__ import annotations

import sys

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from loader import load_data
from tabs.tab_analysis import AnalysisTab
from tabs.tab_generator import GeneratorTab
from tabs.tab_history import HistoryTab
from updater import UpdateResult, update_and_load


DATA_PATH = "data/lotto.txt"


class UpdateWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            result, df = update_and_load(self.path)
            self.finished.emit((result, df))
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lotto Analyzer JB")
        self.resize(1450, 1050)

        self.data = None

        self.update_thread: QThread | None = None
        self.update_worker: UpdateWorker | None = None
        self.progress_dialog: QProgressDialog | None = None

        self._build_ui()
        self._apply_style()
        self._load_initial_data()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Najpierw próbujemy wczytać dane.
        try:
            self.data = load_data(DATA_PATH)
        except Exception:
            self.data = None

        self.generator_tab = GeneratorTab(
            df=self.data,
            parent=self,
        )

        self.analysis_tab = AnalysisTab(
            df=self.data,
            parent=self,
        )

        self.history_tab = HistoryTab(
            df=self.data,
            parent=self,
        )

        self.tabs.addTab(
            self.generator_tab,
            "Generator",
        )

        self.tabs.addTab(
            self.analysis_tab,
            "Analiza",
        )

        self.tabs.addTab(
            self.history_tab,
            "Historia typów",
        )

        central_layout.addWidget(self.tabs)

        self.setCentralWidget(central)

        # Przycisk aktualizacji znajduje się teraz bezpośrednio
        # w zakładce Generator.
        self.generator_tab.update_history_requested.connect(
            self._start_history_update
        )

    # ------------------------------------------------------------------
    # Styl
    # ------------------------------------------------------------------

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #171a1f;
            }

            QWidget {
                background: #171a1f;
                color: #e5e9ee;
                font-size: 13px;
            }

            QTabWidget::pane {
                border: 1px solid #353b45;
                background: #171a1f;
                top: -1px;
            }

            QTabBar::tab {
                background: #232830;
                color: #9da5b0;
                padding: 10px 24px;
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }

            QTabBar::tab:hover {
                background: #2d333d;
                color: #dce2ea;
            }

            QTabBar::tab:selected {
                background: #303844;
                color: #ffffff;
                font-weight: 700;
            }

            QMessageBox {
                background: #20242b;
            }

            QProgressDialog {
                background: #20242b;
            }
            """
        )

    # ------------------------------------------------------------------
    # Dane początkowe
    # ------------------------------------------------------------------

    def _load_initial_data(self):
        if self.data is None:
            QMessageBox.warning(
                self,
                "Brak historii",
                (
                    "Nie udało się wczytać historii losowań z:\n\n"
                    f"{DATA_PATH}\n\n"
                    "Sprawdź, czy plik istnieje."
                ),
            )
            return

        self._refresh_tabs()

        # Przy uruchomieniu aplikacji wykonujemy automatyczną
        # aktualizację historii.
        self._start_history_update(
            automatic=True
        )

    # ------------------------------------------------------------------
    # Aktualizacja historii
    # ------------------------------------------------------------------

    def _start_history_update(self, automatic: bool = False):
        if (
            self.update_thread is not None
            and self.update_thread.isRunning()
        ):
            return

        self.generator_tab.update_history_button.setEnabled(False)

        if automatic:
            self.generator_tab.status_label.setText(
                "Sprawdzanie aktualizacji historii..."
            )
        else:
            self.generator_tab.status_label.setText(
                "Aktualizowanie historii losowań..."
            )

        self.progress_dialog = QProgressDialog(
            "Aktualizowanie historii losowań...",
            None,
            0,
            0,
            self,
        )

        self.progress_dialog.setWindowTitle(
            "Aktualizacja historii"
        )

        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.show()

        self.update_thread = QThread(self)

        self.update_worker = UpdateWorker(
            DATA_PATH
        )

        self.update_worker.moveToThread(
            self.update_thread
        )

        self.update_thread.started.connect(
            self.update_worker.run
        )

        self.update_worker.finished.connect(
            self._update_finished
        )

        self.update_worker.error.connect(
            self._update_error
        )

        self.update_worker.finished.connect(
            self.update_thread.quit
        )

        self.update_worker.error.connect(
            self.update_thread.quit
        )

        self.update_thread.finished.connect(
            self._update_thread_finished
        )

        self.update_thread.start()

    def _update_finished(self, payload):
        try:
            if (
                not isinstance(payload, tuple)
                or len(payload) != 2
            ):
                raise ValueError(
                    "Updater zwrócił nieprawidłowy wynik."
                )

            result, df = payload

            if not isinstance(result, UpdateResult):
                raise ValueError(
                    "Updater zwrócił nieprawidłowy obiekt wyniku."
                )

            if df is None or not hasattr(df, "copy"):
                raise ValueError(
                    "Updater nie zwrócił poprawnej historii."
                )

            self.data = df.copy()

            self._refresh_tabs()

            self._show_update_result(result)

        except Exception as exc:
            self._update_error(str(exc))

    def _show_update_result(self, result: UpdateResult):
        added = getattr(
            result,
            "added",
            None,
        )

        message = getattr(
            result,
            "message",
            "",
        )

        if added is None:
            added = getattr(
                result,
                "added_count",
                0,
            )

        if not message:
            if added:
                message = (
                    f"Dodano {added} nowych losowań."
                )
            else:
                message = (
                    "Historia jest aktualna."
                )

        self.generator_tab.status_label.setText(
            message
        )

        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None

        # Aktualizację pokazujemy tylko wtedy, gdy użytkownik
        # kliknął przycisk ręcznie.
        #
        # Przy automatycznym starcie aplikacji nie wyskakujemy
        # z komunikatem za każdym uruchomieniem.
        if added:
            QMessageBox.information(
                self,
                "Historia zaktualizowana",
                message,
            )

    def _update_error(self, message: str):
        if self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog.deleteLater()
            self.progress_dialog = None

        self.generator_tab.status_label.setText(
            "Nie udało się zaktualizować historii."
        )

        QMessageBox.critical(
            self,
            "Błąd aktualizacji",
            (
                "Nie udało się zaktualizować historii "
                "losowań.\n\n"
                f"{message}"
            ),
        )

    def _update_thread_finished(self):
        self.generator_tab.update_history_button.setEnabled(
            True
        )

        if self.update_thread is not None:
            self.update_thread.deleteLater()

        self.update_thread = None
        self.update_worker = None

    # ------------------------------------------------------------------
    # Odświeżanie zakładek
    # ------------------------------------------------------------------

    def _refresh_tabs(self):
        if self.data is None:
            return

        self.generator_tab.set_data(
            self.data
        )

        self.analysis_tab.set_data(
            self.data
        )

        self.history_tab.set_data(
            self.data
        )

    # ------------------------------------------------------------------
    # Zamknięcie aplikacji
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if (
            self.update_thread is not None
            and self.update_thread.isRunning()
        ):
            self.update_thread.quit()
            self.update_thread.wait(3000)

        event.accept()


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Lotto Analyzer JB")

    window = MainWindow()
    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()