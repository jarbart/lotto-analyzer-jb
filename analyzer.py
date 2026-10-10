from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

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


# ============================================================
# POMOCNICZE
# ============================================================

def _number_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Zwraca kolumny zawierające liczby losowania.

    Standardowo używane są:
        n1 ... n6

    Funkcja zachowuje pewną kompatybilność ze starszymi
    DataFrame, które mogą mieć inaczej nazwane kolumny
    zaczynające się od 'n'.
    """

    if all(
        column in df.columns
        for column in NUMBER_COLUMNS
    ):
        return NUMBER_COLUMNS.copy()

    candidates = [
        column
        for column in df.columns
        if str(column).lower().startswith("n")
    ]

    if len(candidates) >= 6:
        return candidates[:6]

    raise ValueError(
        "Nie znaleziono 6 kolumn z liczbami "
        "losowania. "
        f"Dostępne kolumny: {list(df.columns)}"
    )


def _draw_rows(
    df: pd.DataFrame,
) -> Iterable[list[int]]:
    """
    Zwraca kolejne losowania jako listy sześciu liczb.

    Analizowane są wyłącznie rekordy przekazanego DataFrame.
    """

    columns = _number_columns(
        df
    )

    for _, row in df.iterrows():
        numbers: list[int] = []

        for column in columns:
            value = row[column]

            if pd.isna(value):
                continue

            numbers.append(
                int(value)
            )

        if len(numbers) == 6:
            yield numbers


def _as_frequency_series(
    data: (
        pd.DataFrame
        | pd.Series
        | dict
        | Counter
    ),
) -> pd.Series:
    """
    Normalizuje różne źródła częstotliwości do Series
    zawierającego liczby 1-49.
    """

    if isinstance(data, pd.Series):
        frequency = data.copy()

    elif isinstance(data, Counter):
        frequency = pd.Series(
            dict(data)
        )

    elif isinstance(data, dict):
        frequency = pd.Series(
            data
        )

    elif isinstance(data, pd.DataFrame):
        frequency = number_frequency(
            data
        )

    else:
        raise TypeError(
            "Oczekiwano DataFrame, Series, "
            "dict albo Counter."
        )

    # Normalizacja indeksów do int.
    try:
        frequency.index = (
            frequency.index.astype(int)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Indeks częstotliwości zawiera "
            "nieprawidłowe wartości."
        ) from exc

    # Jeżeli ten sam numer wystąpił w indeksie więcej niż raz,
    # sumujemy jego wartości.
    if not frequency.index.is_unique:
        frequency = (
            frequency
            .groupby(level=0)
            .sum()
        )

    # Normalizacja wartości do liczb całkowitych.
    try:
        frequency = frequency.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Częstotliwość zawiera "
            "nieprawidłowe wartości."
        ) from exc

    # Zawsze zwracamy pełny zakres 1-49.
    frequency = frequency.reindex(
        range(1, 50),
        fill_value=0,
    )

    return frequency.astype(int)


# ============================================================
# CZĘSTOTLIWOŚĆ LICZB
# ============================================================

def number_frequency(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Oblicza częstotliwość występowania liczb 1-49.

    UWAGA:
    Funkcja analizuje tylko rekordy znajdujące się
    w przekazanym DataFrame.
    """

    counter: Counter[int] = Counter()

    for row in _draw_rows(df):
        counter.update(row)

    return pd.Series(
        {
            number: counter.get(
                number,
                0,
            )
            for number in range(1, 50)
        },
        dtype=int,
    )


# ============================================================
# GRUPY G1 / G2 / G3
# ============================================================

def build_groups(
    data: (
        pd.DataFrame
        | pd.Series
        | dict
        | Counter
    ),
) -> tuple[
    set[int],
    set[int],
    set[int],
]:
    """
    Buduje trzy grupy liczb na podstawie częstotliwości:

        G1 = 16 najczęstszych
        G2 = kolejne 16
        G3 = pozostałe 17

    Przy remisie decyduje niższy numer.

    Dzięki temu podział jest deterministyczny.
    """

    frequency = _as_frequency_series(
        data
    )

    ordered = sorted(
        range(1, 50),
        key=lambda number: (
            -int(frequency.get(number, 0)),
            number,
        ),
    )

    g1 = set(
        ordered[:16]
    )

    g2 = set(
        ordered[16:32]
    )

    g3 = set(
        ordered[32:49]
    )

    return (
        g1,
        g2,
        g3,
    )


def groups_as_dict(
    data: (
        pd.DataFrame
        | pd.Series
        | dict
        | Counter
    ),
) -> dict[str, set[int]]:
    """
    Zwraca grupy G1/G2/G3 jako słownik.
    """

    g1, g2, g3 = build_groups(
        data
    )

    return {
        "G1": g1,
        "G2": g2,
        "G3": g3,
    }


# ============================================================
# HOT / COLD
# ============================================================

def hottest_numbers(
    data: (
        pd.DataFrame
        | pd.Series
        | dict
        | Counter
    ),
) -> list[int]:
    """
    Zwraca wszystkie liczby o najwyższej częstotliwości.

    Remisy są zachowane.
    """

    frequency = _as_frequency_series(
        data
    )

    max_value = int(
        frequency.max()
    )

    return [
        number
        for number in range(1, 50)
        if int(
            frequency.get(
                number,
                0,
            )
        ) == max_value
    ]


def coldest_numbers(
    data: (
        pd.DataFrame
        | pd.Series
        | dict
        | Counter
    ),
) -> list[int]:
    """
    Zwraca wszystkie liczby o najniższej częstotliwości.

    Remisy są zachowane.
    """

    frequency = _as_frequency_series(
        data
    )

    min_value = int(
        frequency.min()
    )

    return [
        number
        for number in range(1, 50)
        if int(
            frequency.get(
                number,
                0,
            )
        ) == min_value
    ]


# ============================================================
# OSTATNIE LOSOWANIE
# ============================================================

def last_draw_numbers(
    df: pd.DataFrame,
) -> list[int]:
    """
    Zwraca liczby z ostatniego losowania
    znajdującego się w przekazanym DataFrame.

    Jeżeli DataFrame zawiera kolumnę draw_no,
    rekord o największym numerze losowania jest używany
    jako ostatni.

    W przeciwnym przypadku używany jest ostatni rekord.
    """

    columns = _number_columns(
        df
    )

    if df.empty:
        return []

    if "draw_no" in df.columns:
        valid_df = df.dropna(
            subset=["draw_no"]
        )

        if valid_df.empty:
            row = df.iloc[-1]
        else:
            row = valid_df.loc[
                valid_df["draw_no"].idxmax()
            ]
    else:
        row = df.iloc[-1]

    numbers = [
        int(row[column])
        for column in columns
        if not pd.isna(row[column])
    ]

    return sorted(
        numbers
    )


# ============================================================
# SĄSIEDZI
# ============================================================

def adjacent_numbers(
    data: pd.DataFrame | Iterable[int],
) -> list[int]:
    """
    Zwraca liczby sąsiadujące z podanymi liczbami.

    Dla n:
        n-1
        n+1

    Zakres zostaje ograniczony do 1-49.
    """

    if isinstance(
        data,
        pd.DataFrame,
    ):
        numbers = last_draw_numbers(
            data
        )
    else:
        numbers = [
            int(number)
            for number in data
        ]

    result: set[int] = set()

    for number in numbers:
        if number > 1:
            result.add(
                number - 1
            )

        if number < 49:
            result.add(
                number + 1
            )

    return sorted(
        result
    )


# ============================================================
# BILANS GRUP
# ============================================================

def group_balance(
    df: pd.DataFrame,
) -> dict[str, int]:
    """
    Oblicza rzeczywistą liczbę wystąpień G1/G2/G3
    w przekazanym zakresie historii.

    To jest ważne dla generatora:
    nie ma tutaj żadnego założenia 2/2/2 ani 3/2/1.

    Przykład:
        G1 = 138
        G2 = 92
        G3 = 70

    Generator może następnie potraktować ten rozkład
    jako miękką informację o niedoreprezentowaniu grup.
    """

    groups = groups_as_dict(
        df
    )

    result = {
        "G1": 0,
        "G2": 0,
        "G3": 0,
    }

    for row in _draw_rows(df):
        for number in row:
            if number in groups["G1"]:
                result["G1"] += 1

            elif number in groups["G2"]:
                result["G2"] += 1

            elif number in groups["G3"]:
                result["G3"] += 1

    return result


# ============================================================
# PARY
# ============================================================

def pair_counter(
    df: pd.DataFrame,
) -> Counter[tuple[int, int]]:
    """
    Zlicza współwystępowanie wszystkich par liczb
    w przekazanym zakresie historii.

    Każde losowanie wnosi maksymalnie 15 par.
    """

    counter: Counter[
        tuple[int, int]
    ] = Counter()

    for row in _draw_rows(df):
        counter.update(
            combinations(
                sorted(row),
                2,
            )
        )

    return counter


# ============================================================
# TRÓJKI
# ============================================================

def triple_counter(
    df: pd.DataFrame,
) -> Counter[
    tuple[int, int, int]
]:
    """
    Zlicza współwystępowanie wszystkich trójek liczb
    w przekazanym zakresie historii.

    Każde losowanie wnosi maksymalnie 20 trójek.
    """

    counter: Counter[
        tuple[int, int, int]
    ] = Counter()

    for row in _draw_rows(df):
        counter.update(
            combinations(
                sorted(row),
                3,
            )
        )

    return counter


# ============================================================
# PARY WEWNĘTRZNE
# ============================================================

def count_internal_adjacent_pairs(
    numbers: Iterable[int],
) -> int:
    """
    Liczy sąsiadujące liczby znajdujące się
    jednocześnie w jednym kandydacie.

    Przykład:

        [4, 15, 16, 20, 21, 34]

    zawiera dwie pary:
        15-16
        20-21

    """

    values = sorted(
        set(
            int(number)
            for number in numbers
        )
    )

    return sum(
        1
        for first, second in zip(
            values,
            values[1:],
        )
        if second == first + 1
    )


# ============================================================
# POWTÓRZENIA Z OSTATNIEGO LOSOWANIA
# ============================================================

def repeated_from_last_draw(
    candidate: Iterable[int],
    df: pd.DataFrame,
) -> list[int]:
    """
    Zwraca liczby kandydata, które wystąpiły
    w ostatnim losowaniu wybranego DataFrame.
    """

    candidate_set = {
        int(number)
        for number in candidate
    }

    last_set = set(
        last_draw_numbers(
            df
        )
    )

    return sorted(
        candidate_set & last_set
    )


# ============================================================
# SĄSIEDZI OSTATNIEGO LOSOWANIA
# ============================================================

def neighbors_from_last_draw(
    candidate: Iterable[int],
    df: pd.DataFrame,
) -> list[int]:
    """
    Zwraca liczby kandydata, które są sąsiadami
    liczb z ostatniego losowania.
    """

    candidate_set = {
        int(number)
        for number in candidate
    }

    neighbors = set(
        adjacent_numbers(
            df
        )
    )

    return sorted(
        candidate_set & neighbors
    )


# ============================================================
# PROFIL GRUPOWY KANDYDATA
# ============================================================

def candidate_group_counts(
    candidate: Iterable[int],
    groups: dict[str, set[int]],
) -> dict[str, int]:
    """
    Zwraca liczbę liczb kandydata należących do G1/G2/G3.

    Funkcja pomocnicza dla GUI i generatora.
    """

    result = {
        "G1": 0,
        "G2": 0,
        "G3": 0,
    }

    for number in candidate:
        number = int(number)

        if number in groups["G1"]:
            result["G1"] += 1

        elif number in groups["G2"]:
            result["G2"] += 1

        elif number in groups["G3"]:
            result["G3"] += 1

    return result


# ============================================================
# PROFIL GRUPOWY — KROTKOWY
# ============================================================

def candidate_group_profile(
    candidate: Iterable[int],
    groups: dict[str, set[int]],
) -> tuple[int, int, int]:
    """
    Zwraca profil kandydata:

        (G1, G2, G3)

    Przykład:
        (2, 3, 1)
    """

    counts = candidate_group_counts(
        candidate,
        groups,
    )

    return (
        counts["G1"],
        counts["G2"],
        counts["G3"],
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    from loader import load_data

    data_path = (
        pd.Path("data/lotto.txt")
        if False
        else "data/lotto.txt"
    )

    df = load_data(
        data_path
    )

    print(
        "LOTTO - test analyzer.py"
    )
    print()

    print(
        f"Liczba losowań: {len(df)}"
    )

    frequencies = number_frequency(
        df
    )

    groups = groups_as_dict(
        df
    )

    balance = group_balance(
        df
    )

    hottest = hottest_numbers(
        df
    )

    coldest = coldest_numbers(
        df
    )

    last_draw = last_draw_numbers(
        df
    )

    neighbors = adjacent_numbers(
        df
    )

    print(
        f"G1: {sorted(groups['G1'])}"
    )

    print(
        f"G2: {sorted(groups['G2'])}"
    )

    print(
        f"G3: {sorted(groups['G3'])}"
    )

    print()

    print(
        f"Bilans grup: {balance}"
    )

    print(
        f"Hot: {hottest}"
    )

    print(
        f"Cold: {coldest}"
    )

    print(
        f"Ostatnie losowanie: {last_draw}"
    )

    print(
        f"Sąsiedzi: {neighbors}"
    )

    print()

    print(
        "Top 10 częstotliwości:"
    )

    top_numbers = sorted(
        range(1, 50),
        key=lambda number: (
            -int(
                frequencies[number]
            ),
            number,
        ),
    )[:10]

    for number in top_numbers:
        print(
            f"  {number:2d}: "
            f"{int(frequencies[number])}"
        )