from __future__ import annotations

import html as html_module
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from loader import NUMBER_COLUMNS, load_data, save_data


# ============================================================
# ŹRÓDŁA DANYCH
# ============================================================

OFFICIAL_URL = (
    "https://www.lotto.pl/lotto/wyniki-i-wygrane"
)

# Publiczne archiwum wyników Lotto.
# Używane jako fallback, gdy oficjalne źródło
# nie pozwala pobrać lub poprawnie odczytać wyników.
FALLBACK_URL = (
    "https://megalotto.pl/wyniki/lotto/100-ostatnich-losowan"
)

REQUEST_TIMEOUT = 15

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


# ============================================================
# MODELE
# ============================================================

@dataclass
class LottoDraw:
    """
    Pojedyncze losowanie klasycznego Lotto 6/49.
    """

    draw_no: int
    date: str
    numbers: list[int]


@dataclass
class UpdateResult:
    """
    Wynik aktualizacji lokalnej historii.
    """

    success: bool
    added_count: int
    latest_draw_no: int
    latest_date: str
    message: str
    added_draws: list[LottoDraw]


# ============================================================
# HTTP
# ============================================================

def _download_page(url: str) -> str:
    """
    Pobiera stronę HTTP.

    Funkcja nie próbuje obchodzić zabezpieczeń serwera.
    Jeżeli serwer zwróci np. 403, wyjątek trafia wyżej
    i updater może przejść do źródła zapasowego.
    """

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "pl-PL,pl;q=0.9,en;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            content = response.read()

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Serwer odrzucił żądanie HTTP "
            f"(kod {exc.code})."
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Nie można połączyć się ze źródłem: "
            f"{exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "Przekroczono czas oczekiwania "
            "na odpowiedź serwera."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Nie udało się pobrać strony: {exc}"
        ) from exc

    try:
        return content.decode("utf-8")

    except UnicodeDecodeError:
        return content.decode(
            "utf-8",
            errors="replace",
        )


# ============================================================
# CZYSZCZENIE HTML
# ============================================================

def _clean_html(html: str) -> str:
    """
    Zamienia HTML na możliwie prosty tekst.

    Nie wymaga BeautifulSoup.
    """

    text = re.sub(
        r"<script\b[^>]*>.*?</script>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<style\b[^>]*>.*?</style>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<noscript\b[^>]*>.*?</noscript>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = html_module.unescape(
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# WALIDACJA LOSOWANIA
# ============================================================

def _validate_draw(
    draw_no: int,
    date: str,
    numbers: list[int],
) -> bool:
    """
    Sprawdza, czy rekord wygląda jak poprawne
    losowanie Lotto 6/49.
    """

    if draw_no <= 0:
        return False

    if len(numbers) != 6:
        return False

    if len(set(numbers)) != 6:
        return False

    if not all(
        1 <= number <= 49
        for number in numbers
    ):
        return False

    if not re.fullmatch(
        r"\d{2}\.\d{2}\.\d{4}",
        date,
    ):
        return False

    return True


# ============================================================
# PARSER OFICJALNEGO ŹRÓDŁA
# ============================================================

def _parse_official_page(
    html: str,
) -> list[LottoDraw]:
    """
    Próbuje odczytać klasyczne Lotto z oficjalnej strony.

    Parser szuka:
        Lotto Nr losowania: N
        + sześciu liczb.

    Lotto Plus i inne gry nie są akceptowane.
    """

    text = _clean_html(
        html
    )

    draws: dict[int, LottoDraw] = {}

    pattern = re.compile(
        r"Lotto\s+"
        r"Nr\s+losowania:\s*(\d+)\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})",
        flags=re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        draw_no = int(
            match.group(1)
        )

        numbers = [
            int(match.group(index))
            for index in range(2, 8)
        ]

        # Szukamy daty bezpośrednio przed rekordem.
        before = text[
            max(
                0,
                match.start() - 300,
            ):match.start()
        ]

        date_match = re.search(
            r"(\d{2}[.-]\d{2}[.-]\d{4})",
            before,
        )

        if not date_match:
            continue

        raw_date = date_match.group(1)

        date = raw_date.replace(
            "-",
            ".",
        )

        numbers.sort()

        if not _validate_draw(
            draw_no,
            date,
            numbers,
        ):
            continue

        draws[draw_no] = LottoDraw(
            draw_no=draw_no,
            date=date,
            numbers=numbers,
        )

    result = list(
        draws.values()
    )

    result.sort(
        key=lambda item: item.draw_no
    )

    return result


# ============================================================
# PARSER FALLBACKU
# ============================================================

def _parse_fallback_page(
    html: str,
) -> list[LottoDraw]:
    """
    Parser publicznego archiwum wyników.

    Przykładowy rekord:

        7414.
        06-10-2026
        8
        26
        27
        35
        42
        46
    """

    text = _clean_html(
        html
    )

    draws: dict[int, LottoDraw] = {}

    pattern = re.compile(
        r"(\d{3,6})\.\s+"
        r"(\d{2})-(\d{2})-(\d{4})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})\s+"
        r"(\d{1,2})",
    )

    for match in pattern.finditer(text):
        draw_no = int(
            match.group(1)
        )

        day = match.group(2)
        month = match.group(3)
        year = match.group(4)

        date = (
            f"{day}."
            f"{month}."
            f"{year}"
        )

        numbers = [
            int(match.group(index))
            for index in range(5, 11)
        ]

        numbers.sort()

        if not _validate_draw(
            draw_no,
            date,
            numbers,
        ):
            continue

        draws[draw_no] = LottoDraw(
            draw_no=draw_no,
            date=date,
            numbers=numbers,
        )

    result = list(
        draws.values()
    )

    result.sort(
        key=lambda item: item.draw_no
    )

    return result


# ============================================================
# POBIERANIE AKTUALNYCH WYNIKÓW
# ============================================================

def fetch_latest_draws() -> list[LottoDraw]:
    """
    Pobiera aktualne wyniki Lotto.

    Kolejność:
        1. oficjalna strona LOTTO,
        2. publiczne archiwum jako fallback.

    Zwracane są wyłącznie poprawne losowania Lotto 6/49.
    """

    errors: list[str] = []

    # --------------------------------------------------------
    # 1. Oficjalne źródło
    # --------------------------------------------------------

    try:
        html = _download_page(
            OFFICIAL_URL
        )

        draws = _parse_official_page(
            html
        )

        if draws:
            return draws

        errors.append(
            "Oficjalne źródło: pobrano stronę, "
            "ale parser nie znalazł wyników Lotto."
        )

    except Exception as exc:
        errors.append(
            f"Oficjalne źródło: {exc}"
        )

    # --------------------------------------------------------
    # 2. Fallback
    # --------------------------------------------------------

    try:
        html = _download_page(
            FALLBACK_URL
        )

        draws = _parse_fallback_page(
            html
        )

        if draws:
            return draws

        errors.append(
            "Fallback: pobrano stronę, "
            "ale parser nie znalazł wyników Lotto."
        )

    except Exception as exc:
        errors.append(
            f"Fallback: {exc}"
        )

    # --------------------------------------------------------
    # Brak źródła
    # --------------------------------------------------------

    details = "\n".join(
        f"- {error}"
        for error in errors
    )

    raise RuntimeError(
        "Nie udało się pobrać aktualnych wyników Lotto.\n\n"
        f"{details}"
    )


# ============================================================
# BACKUP
# ============================================================

def _create_backup(
    path: Path,
) -> Path:
    """
    Tworzy kopię bezpieczeństwa lotto.txt.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = path.with_name(
        f"{path.stem}_backup_{timestamp}{path.suffix}"
    )

    shutil.copy2(
        path,
        backup_path,
    )

    return backup_path


# ============================================================
# KONWERSJA
# ============================================================

def _draw_to_row(
    draw: LottoDraw,
) -> dict[str, object]:
    """
    Zamienia LottoDraw na rekord DataFrame.
    """

    if len(draw.numbers) != 6:
        raise ValueError(
            f"Losowanie {draw.draw_no} "
            "nie zawiera dokładnie 6 liczb."
        )

    return {
        "draw_no": draw.draw_no,
        "date": draw.date,
        "n1": draw.numbers[0],
        "n2": draw.numbers[1],
        "n3": draw.numbers[2],
        "n4": draw.numbers[3],
        "n5": draw.numbers[4],
        "n6": draw.numbers[5],
    }


# ============================================================
# AKTUALIZACJA PLIKU
# ============================================================

def update_lotto_file(
    path: str | Path = "data/lotto.txt",
) -> UpdateResult:
    """
    Aktualizuje lokalny lotto.txt.

    Dodawane są wyłącznie losowania, których numeru
    nie ma jeszcze w lokalnym pliku.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: "
            f"{path.resolve()}"
        )

    local_df = load_data(
        path
    )

    if local_df.empty:
        raise RuntimeError(
            "Lokalny plik lotto.txt jest pusty."
        )

    remote_draws = fetch_latest_draws()

    if not remote_draws:
        raise RuntimeError(
            "Źródło zewnętrzne nie zwróciło "
            "żadnych poprawnych losowań Lotto."
        )

    local_draw_numbers = {
        int(value)
        for value in local_df["draw_no"].tolist()
    }

    missing_draws = [
        draw
        for draw in remote_draws
        if draw.draw_no not in local_draw_numbers
    ]

    missing_draws.sort(
        key=lambda draw: draw.draw_no
    )

    # --------------------------------------------------------
    # Brak zmian
    # --------------------------------------------------------

    if not missing_draws:
        latest_row = local_df.iloc[-1]

        latest_draw_no = int(
            latest_row["draw_no"]
        )

        latest_date = str(
            latest_row["date"]
        )

        return UpdateResult(
            success=True,
            added_count=0,
            latest_draw_no=latest_draw_no,
            latest_date=latest_date,
            message=(
                "Plik lotto.txt jest już aktualny."
            ),
            added_draws=[],
        )

    # --------------------------------------------------------
    # Backup
    # --------------------------------------------------------

    backup_path = _create_backup(
        path
    )

    # --------------------------------------------------------
    # Dodanie nowych losowań
    # --------------------------------------------------------

    new_rows = [
        _draw_to_row(draw)
        for draw in missing_draws
    ]

    new_df = pd.DataFrame(
        new_rows,
        columns=[
            "draw_no",
            "date",
            *NUMBER_COLUMNS,
        ],
    )

    combined = pd.concat(
        [
            local_df,
            new_df,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["draw_no"],
            keep="last",
        )
        .sort_values(
            "draw_no"
        )
        .reset_index(
            drop=True
        )
    )

    save_data(
        combined,
        path,
    )

    latest_row = combined.iloc[-1]

    latest_draw_no = int(
        latest_row["draw_no"]
    )

    latest_date = str(
        latest_row["date"]
    )

    added_text = ", ".join(
        str(draw.draw_no)
        for draw in missing_draws
    )

    return UpdateResult(
        success=True,
        added_count=len(
            missing_draws
        ),
        latest_draw_no=latest_draw_no,
        latest_date=latest_date,
        message=(
            f"Dodano {len(missing_draws)} "
            f"nowych losowań: {added_text}. "
            f"Kopia bezpieczeństwa: "
            f"{backup_path.name}"
        ),
        added_draws=missing_draws,
    )


# ============================================================
# AKTUALIZACJA + WCZYTANIE
# ============================================================

def update_and_load(
    path: str | Path = "data/lotto.txt",
) -> tuple[UpdateResult, pd.DataFrame]:
    """
    Aktualizuje lotto.txt i zwraca świeży DataFrame.
    """

    result = update_lotto_file(
        path
    )

    df = load_data(
        path
    )

    return result, df


# ============================================================
# TEST / URUCHOMIENIE RĘCZNE
# ============================================================

if __name__ == "__main__":
    base_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    data_path = (
        base_dir
        / "data"
        / "lotto.txt"
    )

    print(
        "LOTTO - aktualizacja wyników"
    )

    print(
        f"Plik: {data_path}"
    )

    print(
        f"Źródło główne: {OFFICIAL_URL}"
    )

    print(
        f"Fallback: {FALLBACK_URL}"
    )

    print()

    try:
        result = update_lotto_file(
            data_path
        )

        print(
            result.message
        )

        print(
            f"Najnowsze losowanie: "
            f"{result.latest_draw_no}"
        )

        print(
            f"Data: "
            f"{result.latest_date}"
        )

        if result.added_draws:
            print()
            print(
                "Dodane losowania:"
            )

            for draw in result.added_draws:
                numbers_text = ", ".join(
                    str(number)
                    for number in draw.numbers
                )

                print(
                    f"  {draw.draw_no}. "
                    f"{draw.date} "
                    f"{numbers_text}"
                )

    except Exception as exc:
        print()
        print(
            "BŁĄD AKTUALIZACJI:"
        )
        print(
            str(exc)
        )