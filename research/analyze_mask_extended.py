
from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loader import load_data


BASE_DIR = PROJECT_ROOT
DATA_PATH = BASE_DIR / "data" / "lotto.txt"

MASK = (4, 15, 17, 20, 21, 34)
START_DRAW = 6000
END_DRAW = None

SAMPLE_SIZE = 10_000
SEED = 20261008
BATCH_SIZE = 200

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def fmt(value):
    return f"{value:.3f}"


def hit_counts(draws, numbers):
    target = set(numbers)
    return np.array(
        [len(target.intersection(row)) for row in draws],
        dtype=np.uint8,
    )


def summary(hits):
    return {
        "mean": float(hits.mean()),
        "two": int(np.count_nonzero(hits >= 2)),
        "three": int(np.count_nonzero(hits >= 3)),
        "four": int(np.count_nonzero(hits >= 4)),
        "max": int(hits.max()),
    }


def adjacent_pair_count(numbers):
    ordered = sorted(numbers)
    return sum(
        b == a + 1
        for a, b in zip(ordered, ordered[1:])
    )


def structural_profile(numbers):
    numbers = tuple(sorted(numbers))
    low = sum(n <= 24 for n in numbers)
    even = sum(n % 2 == 0 for n in numbers)
    adjacent = adjacent_pair_count(numbers)
    return low, even, adjacent


def make_sample(
    rng,
    size,
    mode,
    excluded,
):
    """
    Wylosowane zestawy są stałe dla wszystkich losowań.

    mode:
      all       - dowolne 6/49
      low5      - dokładnie 5 liczb z 1-24
      adjacent  - dokładnie jedna para sąsiadująca
      matched   - dokładnie 5 niskich, 3 parzyste,
                  dokładnie jedna para sąsiadująca
    """
    result = set()
    all_numbers = np.arange(1, 50)
    low_numbers = np.arange(1, 25)
    high_numbers = np.arange(25, 50)

    while len(result) < size:
        if mode in ("low5", "matched"):
            low = rng.choice(
                low_numbers,
                size=5,
                replace=False,
            )
            high = rng.choice(
                high_numbers,
                size=1,
                replace=False,
            )
            candidate = tuple(
                sorted(
                    np.concatenate([low, high]).tolist()
                )
            )
        else:
            candidate = tuple(
                sorted(
                    rng.choice(
                        all_numbers,
                        size=6,
                        replace=False,
                    ).tolist()
                )
            )

        if candidate == excluded:
            continue

        low, even, adjacent = structural_profile(candidate)

        if mode == "adjacent" and adjacent != 1:
            continue

        if mode == "matched":
            if low != 5 or even != 3 or adjacent != 1:
                continue

        result.add(candidate)

    return np.array(
        sorted(result),
        dtype=np.uint8,
    )


def evaluate_sample(draws, candidates):
    """
    Ocena wielu stałych zestawów w partiach.
    """
    incidence = np.zeros(
        (len(draws), 50),
        dtype=np.uint8,
    )

    row_index = np.arange(len(draws))[:, None]
    incidence[row_index, draws] = 1

    rows = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start:start + BATCH_SIZE]

        # Kształt: [losowania, zestawy w partii]
        hits = incidence[:, batch].sum(axis=2)

        rows.extend(
            summary(hits[:, i])
            for i in range(hits.shape[1])
        )

    return pd.DataFrame(rows)


def compare(label, actual, reference):
    print(f"\n{label}")
    print("-" * 66)

    names = {
        "mean": "Średnia trafień",
        "two": "Losowania 2/6+",
        "three": "Losowania 3/6+",
        "four": "Losowania 4/6+",
        "max": "Najwyższy wynik",
    }

    for key, title in names.items():
        values = reference[key]
        value = actual[key]

        lower = 100 * (values < value).mean()
        equal = 100 * (values == value).mean()
        higher = 100 * (values > value).mean()

        print(
            f"{title:<22}"
            f" Twój: {value:>7.3f}"
            f" | Śr.: {values.mean():>7.3f}"
        )
        print(
            f"  Gorsze: {lower:5.1f}%"
            f" | Równe: {equal:5.1f}%"
            f" | Lepsze: {higher:5.1f}%"
        )


def analyze_numbers(df, draws):
    print("\n" + "=" * 66)
    print("1. CZĘSTOTLIWOŚĆ POSZCZEGÓLNYCH LICZB")
    print("=" * 66)

    expected = len(draws) * 6 / 49

    for number in MASK:
        count = int(np.count_nonzero(draws == number))
        print(
            f"{number:>2}: {count:>4} wystąpień"
            f" | oczekiwane {expected:.1f}"
            f" | różnica {count - expected:+.1f}"
        )

    print("\n" + "=" * 66)
    print("2. WSZYSTKIE PARY Z TWOJEGO ZESTAWU")
    print("=" * 66)

    expected_pair = (
        len(draws) * comb(47, 4) / comb(49, 6)
    )

    pair_results = []

    for a, b in combinations(MASK, 2):
        count = int(
            np.count_nonzero(
                np.any(draws == a, axis=1)
                & np.any(draws == b, axis=1)
            )
        )

        pair_results.append((a, b, count))

    for a, b, count in sorted(
        pair_results,
        key=lambda x: x[2],
        reverse=True,
    ):
        print(
            f"{a:>2} + {b:>2}: {count:>3}"
            f" | oczekiwane {expected_pair:.1f}"
        )

    print("\n" + "=" * 66)
    print("3. TRÓJKI Z TWOJEGO ZESTAWU")
    print("=" * 66)

    triple_results = []

    for triple in combinations(MASK, 3):
        present = np.ones(
            len(draws),
            dtype=bool,
        )

        for number in triple:
            present &= np.any(
                draws == number,
                axis=1,
            )

        count = int(present.sum())
        triple_results.append((triple, count))

    for triple, count in sorted(
        triple_results,
        key=lambda x: x[1],
        reverse=True,
    )[:10]:
        print(
            f"{str(triple):<17} {count:>3} razy"
        )


def analyze_halves(df, draws):
    print("\n" + "=" * 66)
    print("4. PIERWSZA I DRUGA POŁOWA")
    print("=" * 66)

    midpoint = len(draws) // 2

    for label, part in (
        ("Pierwsza połowa", draws[:midpoint]),
        ("Druga połowa", draws[midpoint:]),
    ):
        hits = hit_counts(part, MASK)
        stats = summary(hits)

        print(
            f"{label}: {len(part)} losowań"
            f" | średnia {stats['mean']:.3f}"
            f" | 2+ {stats['two']}"
            f" | 3+ {stats['three']}"
            f" | 4+ {stats['four']}"
        )

    print(
        "\nUwaga: różnice między połowami nie są"
        " dowodem trendu ani zmiany przyszłych szans."
    )


def analyze_recent_periods(draws):
    print("\n" + "=" * 66)
    print("5. OSTATNIE 50, 100, 200 I 500 LOSOWAŃ")
    print("=" * 66)

    for window in (50, 100, 200, 500):
        part = draws[-window:]
        hits = hit_counts(part, MASK)
        stats = summary(hits)

        print(
            f"Ostatnie {len(part):>3}:"
            f" średnia {stats['mean']:.3f}"
            f" | 2+ {stats['two']:>3}"
            f" | 3+ {stats['three']:>3}"
            f" | 4+ {stats['four']:>3}"
        )


def main():
    df = load_data(DATA_PATH)
    df = df.sort_values("draw_no").copy()

    df = df[df["draw_no"] >= START_DRAW]

    if END_DRAW is not None:
        df = df[df["draw_no"] <= END_DRAW]

    if df.empty:
        raise ValueError("Brak danych w wybranym zakresie.")

    draws = df[NUMBER_COLUMNS].to_numpy(
        dtype=np.uint8
    )

    print("=" * 66)
    print("ROZSZERZONA ANALIZA MASKI")
    print("=" * 66)
    print("Zestaw:", MASK)
    print(
        "Losowania:",
        int(df["draw_no"].min()),
        "–",
        int(df["draw_no"].max()),
    )
    print("Liczba losowań:", len(draws))
    print("Profil (niskie, parzyste, pary sąsiadujące):",
          structural_profile(MASK))

    analyze_numbers(df, draws)
    analyze_halves(df, draws)
    analyze_recent_periods(draws)

    actual = summary(hit_counts(draws, MASK))

    rng = np.random.default_rng(SEED)

    groups = [
        ("Wszystkie zestawy 6/49", "all"),
        ("Dokładnie 5 niskich + 1 wysoka", "low5"),
        ("Dokładnie jedna para sąsiadująca", "adjacent"),
        (
            "Identyczna struktura: 5 niskich,"
            " 3 parzyste, 1 para sąsiadująca",
            "matched",
        ),
    ]

    print("\n" + "=" * 66)
    print("6. PORÓWNANIA ZE STAŁYMI ZESTAWAMI")
    print("=" * 66)

    for label, mode in groups:
        print(f"\nPrzygotowywanie: {label}...")

        candidates = make_sample(
            rng,
            SAMPLE_SIZE,
            mode,
            MASK,
        )

        reference = evaluate_sample(
            draws,
            candidates,
        )

        compare(
            label,
            actual,
            reference,
        )

    print("\n" + "=" * 66)
    print("WNIOSKI METODOLOGICZNE")
    print("=" * 66)
    print(
        "Porównania pokazują wyłącznie wyniki historyczne."
    )
    print(
        "Wszystkie zestawy mają takie same szanse"
        " w przyszłym uczciwym losowaniu."
    )
    print(
        "Nie należy zwiększać stawek z powodu"
        " wcześniejszych braków trafień."
    )


if __name__ == "__main__":
    main()
