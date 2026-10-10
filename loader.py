from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# STAŁE
# ============================================================

NUMBER_COLUMNS = [
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
]

COLUMNS = [
    "draw_no",
    "date",
    *NUMBER_COLUMNS,
]

DEFAULT_DATA_PATH = Path(
    "data/lotto.txt"
)


# ============================================================
# WALIDACJA POJEDYNCZEGO LOSOWANIA
# ============================================================

def _validate_draw(
    draw_no: int,
    date: str,
    numbers: list[int],
    line_no: int | None = None,
) -> None:
    """
    Waliduje pojedyncze losowanie Lotto 6/49.

    Format:
        numer losowania
        data
        dokładnie 6 różnych liczb 1-49
    """

    location = (
        f" w linii {line_no}"
        if line_no is not None
        else ""
    )

    if draw_no <= 0:
        raise ValueError(
            f"Nieprawidłowy numer losowania"
            f"{location}: {draw_no}"
        )

    if not date:
        raise ValueError(
            f"Brak daty dla losowania {draw_no}"
            f"{location}."
        )

    if len(numbers) != 6:
        raise ValueError(
            f"Losowanie {draw_no}{location} "
            f"nie zawiera dokładnie 6 liczb: "
            f"{numbers}"
        )

    if len(set(numbers)) != 6:
        raise ValueError(
            f"Losowanie {draw_no}{location} "
            f"zawiera powtarzającą się liczbę: "
            f"{numbers}"
        )

    invalid_numbers = [
        number
        for number in numbers
        if not 1 <= number <= 49
    ]

    if invalid_numbers:
        raise ValueError(
            f"Losowanie {draw_no}{location} "
            f"zawiera liczby spoza zakresu 1-49: "
            f"{invalid_numbers}"
        )


# ============================================================
# WCZYTYWANIE
# ============================================================

def load_data(
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Wczytuje historię Lotto z pliku lotto.txt.

    Oczekiwany format:

        1. 27.01.1957 8,12,31,39,43,45

    Zwracany DataFrame:

        draw_no | date | n1 | n2 | n3 | n4 | n5 | n6

    Liczby każdego losowania są sortowane rosnąco.
    Losowania są sortowane według numeru losowania.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku z wynikami Lotto: "
            f"{path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"Podana ścieżka nie jest plikiem: "
            f"{path.resolve()}"
        )

    rows: list[list[object]] = []

    try:
        content = path.read_text(
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Nie można odczytać pliku {path.resolve()} "
            "jako UTF-8."
        ) from exc

    for line_no, raw_line in enumerate(
        content.splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        # Puste linie ignorujemy.
        if not line:
            continue

        parts = line.split()

        if len(parts) < 3:
            raise ValueError(
                f"Nieprawidłowy zapis w linii "
                f"{line_no}: {raw_line}"
            )

        # ----------------------------------------------------
        # Numer losowania
        # ----------------------------------------------------

        draw_no_text = parts[0].rstrip(".")

        try:
            draw_no = int(
                draw_no_text
            )
        except ValueError as exc:
            raise ValueError(
                f"Nieprawidłowy numer losowania "
                f"w linii {line_no}: "
                f"{parts[0]}"
            ) from exc

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        date = parts[1]

        # ----------------------------------------------------
        # Liczby
        # ----------------------------------------------------

        number_text = "".join(
            parts[2:]
        )

        raw_numbers = [
            value.strip()
            for value in number_text.split(",")
            if value.strip()
        ]

        try:
            numbers = [
                int(value)
                for value in raw_numbers
            ]
        except ValueError as exc:
            raise ValueError(
                f"Nieprawidłowe liczby w linii "
                f"{line_no}: {raw_line}"
            ) from exc

        # ----------------------------------------------------
        # Walidacja
        # ----------------------------------------------------

        _validate_draw(
            draw_no=draw_no,
            date=date,
            numbers=numbers,
            line_no=line_no,
        )

        rows.append(
            [
                draw_no,
                date,
                *sorted(numbers),
            ]
        )

    if not rows:
        raise ValueError(
            f"Plik {path.resolve()} "
            "nie zawiera żadnych losowań."
        )

    df = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    # --------------------------------------------------------
    # Typy
    # --------------------------------------------------------

    df["draw_no"] = df[
        "draw_no"
    ].astype(int)

    for column in NUMBER_COLUMNS:
        df[column] = df[
            column
        ].astype(int)

    # --------------------------------------------------------
    # Duplikaty numerów losowań
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        subset=["draw_no"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_numbers = sorted(
            set(
                df.loc[
                    duplicate_mask,
                    "draw_no",
                ].astype(int)
            )
        )

        raise ValueError(
            "Plik zawiera powtarzające się "
            "numery losowań: "
            f"{duplicate_numbers}"
        )

    # --------------------------------------------------------
    # Sortowanie
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            "draw_no"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# KOMPATYBILNOŚĆ ZE STARSZYM KODEM
# ============================================================

def load_history(
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Alias dla kompatybilności ze starszym kodem.
    """

    return load_data(
        path
    )


# ============================================================
# ZAPIS
# ============================================================

def save_data(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_DATA_PATH,
) -> None:
    """
    Zapisuje DataFrame do formatu lotto.txt.

    Przed zapisem każdy rekord jest ponownie walidowany.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    missing = [
        column
        for column in COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Brak wymaganych kolumn: "
            f"{missing}"
        )

    rows = (
        df[
            COLUMNS
        ]
        .copy()
        .sort_values(
            "draw_no"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Duplikaty numerów losowań
    # --------------------------------------------------------

    duplicate_mask = rows.duplicated(
        subset=["draw_no"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_numbers = sorted(
            set(
                rows.loc[
                    duplicate_mask,
                    "draw_no",
                ].astype(int)
            )
        )

        raise ValueError(
            "Nie można zapisać pliku, ponieważ "
            "DataFrame zawiera powtarzające się "
            "numery losowań: "
            f"{duplicate_numbers}"
        )

    # --------------------------------------------------------
    # Budowanie pliku
    # --------------------------------------------------------

    lines: list[str] = []

    for row_index, row in rows.iterrows():
        try:
            draw_no = int(
                row["draw_no"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Nieprawidłowy numer losowania "
                f"w rekordzie {row_index}."
            ) from exc

        date = str(
            row["date"]
        ).strip()

        try:
            numbers = [
                int(row[column])
                for column in NUMBER_COLUMNS
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Nieprawidłowe liczby "
                f"dla losowania {draw_no}."
            ) from exc

        _validate_draw(
            draw_no=draw_no,
            date=date,
            numbers=numbers,
        )

        numbers.sort()

        lines.append(
            f"{draw_no}. {date} "
            f"{','.join(str(number) for number in numbers)}"
        )

    # --------------------------------------------------------
    # Zapis
    # --------------------------------------------------------

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# KOMPATYBILNOŚĆ ZE STARSZYM KODEM
# ============================================================

def save_history(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_DATA_PATH,
) -> None:
    """
    Alias dla kompatybilności ze starszym kodem.
    """

    save_data(
        df,
        path,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    data_path = (
        Path(__file__).resolve().parent
        / DEFAULT_DATA_PATH
    )

    print(
        "LOTTO - test loader.py"
    )
    print(
        f"Plik: {data_path}"
    )
    print()

    try:
        df = load_data(
            data_path
        )

        print(
            f"Wczytano losowań: {len(df)}"
        )

        if not df.empty:
            first = df.iloc[0]
            last = df.iloc[-1]

            print(
                f"Pierwsze: "
                f"{int(first['draw_no'])}. "
                f"{first['date']}"
            )

            print(
                f"Ostatnie: "
                f"{int(last['draw_no'])}. "
                f"{last['date']}"
            )

            last_numbers = [
                int(last[column])
                for column in NUMBER_COLUMNS
            ]

            print(
                "Ostatnie liczby: "
                + ",".join(
                    str(number)
                    for number in last_numbers
                )
            )

    except Exception as exc:
        print(
            "BŁĄD:"
        )
        print(
            str(exc)
        )