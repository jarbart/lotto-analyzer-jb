
from __future__ import annotations

from collections import Counter
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
OUTPUT_PATH = BASE_DIR / "data" / "mask_analysis_draws.csv"

MASK = (4, 15, 17, 20, 21, 34)

START_DRAW = 6000
END_DRAW = None  # None = ostatnie dostępne losowanie

SIMULATIONS = 10_000
SEED = 20261008

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def probability_exact_hits(k: int) -> float:
    """Prawdopodobieństwo dokładnie k trafień w Lotto 6/49."""
    return comb(6, k) * comb(43, 6 - k) / comb(49, 6)


def longest_streak(values, predicate) -> int:
    """Najdłuższa seria kolejnych losowań spełniających warunek."""
    longest = 0
    current = 0

    for value in values:
        if predicate(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def generate_random_sets(count: int, seed: int) -> np.ndarray:
    """
    Tworzy unikalne losowe zestawy 6/49.
    Każdy zestaw pozostaje stały przez cały badany okres.
    """
    rng = np.random.default_rng(seed)
    unique_sets = set()

    while len(unique_sets) < count:
        numbers = tuple(
            sorted(
                rng.choice(
                    np.arange(1, 50),
                    size=6,
                    replace=False,
                ).tolist()
            )
        )
        unique_sets.add(numbers)

    return np.array(sorted(unique_sets), dtype=np.int16)


def simulate_fixed_sets(
    draws: np.ndarray,
    random_sets: np.ndarray,
) -> pd.DataFrame:
    """
    Każdy losowy zestaw jest sprawdzany przeciwko
    wszystkim historycznym losowaniom.

    Przetwarzanie partiami ogranicza zużycie pamięci.
    """
    incidence = np.zeros(
        (len(draws), 50),
        dtype=np.uint8,
    )

    row_indices = np.arange(len(draws))[:, None]
    incidence[row_indices, draws] = 1

    results = []
    batch_size = 250

    for start in range(0, len(random_sets), batch_size):
        batch = random_sets[start:start + batch_size]

        # Wynik: liczba trafień dla każdego losowania
        # i każdego stałego zestawu z partii.
        hits = incidence[:, batch].sum(axis=2)

        for column in range(len(batch)):
            values = hits[:, column]

            results.append(
                {
                    "mean_hits": float(values.mean()),
                    "hits_2plus": int((values >= 2).sum()),
                    "hits_3plus": int((values >= 3).sum()),
                    "hits_4plus": int((values >= 4).sum()),
                    "max_hits": int(values.max()),
                }
            )

    return pd.DataFrame(results)


def show_comparison(
    label: str,
    actual: float,
    simulated: pd.Series,
):
    average = float(simulated.mean())

    below = 100 * float((simulated < actual).mean())
    equal = 100 * float((simulated == actual).mean())
    above = 100 * float((simulated > actual).mean())

    print(f"\n{label}")
    print(f"  Twój zestaw:          {actual:.4f}")
    print(f"  Średnia porównawcza:  {average:.4f}")
    print(f"  Zestawy z wynikiem niższym: {below:.1f}%")
    print(f"  Zestawy z wynikiem równym:  {equal:.1f}%")
    print(f"  Zestawy z wynikiem wyższym: {above:.1f}%")


def main():
    print("=" * 66)
    print("ANALIZA STAŁEGO ZESTAWU LOTTO")
    print("=" * 66)

    print("Zestaw:", ", ".join(map(str, MASK)))
    print("Początek:", START_DRAW)

    df = load_data(DATA_PATH)
    df = df.sort_values("draw_no").copy()

    df = df[df["draw_no"] >= START_DRAW]

    if END_DRAW is not None:
        df = df[df["draw_no"] <= END_DRAW]

    if df.empty:
        raise ValueError(
            "Brak losowań w wybranym zakresie."
        )

    draws = df[NUMBER_COLUMNS].to_numpy(dtype=np.int16)

    mask_set = set(MASK)

    hits = np.array(
        [
            len(mask_set.intersection(row))
            for row in draws
        ],
        dtype=np.int16,
    )

    total = len(hits)
    distribution = Counter(hits.tolist())

    print(f"Ostatnie losowanie: {int(df['draw_no'].max())}")
    print(f"Liczba losowań: {total}")

    print("\n" + "=" * 66)
    print("1. RZECZYWISTE TRAFIENIA")
    print("=" * 66)

    print(
        f"{'Trafienia':<12}"
        f"{'Rzeczywiste':>14}"
        f"{'Oczekiwane':>16}"
        f"{'Szansa/los.':>17}"
    )

    for k in range(7):
        probability = probability_exact_hits(k)
        expected = total * probability
        actual = distribution.get(k, 0)

        print(
            f"{k}/6{'':<9}"
            f"{actual:>14}"
            f"{expected:>16.2f}"
            f"{100 * probability:>16.4f}%"
        )

    actual_max = int(hits.max())

    print(f"\nNajlepszy wynik: {actual_max}/6")
    print(f"Średnia trafień: {hits.mean():.4f}/6")
    print(f"Oczekiwana średnia: {36 / 49:.4f}/6")

    print("\n" + "=" * 66)
    print("2. NAJLEPSZE LOSOWANIA")
    print("=" * 66)

    best_rows = df.loc[hits == actual_max]

    date_column = next(
        (
            name
            for name in ("date", "draw_date", "data")
            if name in df.columns
        ),
        None,
    )

    for _, row in best_rows.iterrows():
        numbers = {
            int(row[column])
            for column in NUMBER_COLUMNS
        }

        matching = sorted(mask_set.intersection(numbers))

        date_text = (
            f" | data: {row[date_column]}"
            if date_column is not None
            else ""
        )

        print(
            f"Losowanie {int(row['draw_no'])}"
            f"{date_text}"
            f" | {actual_max}/6"
            f" | trafione: {matching}"
        )

    print("\n" + "=" * 66)
    print("3. NAJDŁUŻSZE SERIE")
    print("=" * 66)

    print(
        "Najdłuższa seria bez żadnego trafienia:",
        longest_streak(hits, lambda x: x == 0),
    )

    print(
        "Najdłuższa seria bez 2/6 lub więcej:",
        longest_streak(hits, lambda x: x < 2),
    )

    print(
        "Najdłuższa seria bez 3/6 lub więcej:",
        longest_streak(hits, lambda x: x < 3),
    )

    print(
        "Najdłuższa seria bez 4/6 lub więcej:",
        longest_streak(hits, lambda x: x < 4),
    )

    print("\n" + "=" * 66)
    print("4. WYNIKI W BLOKACH PO 100 LOSOWAŃ")
    print("=" * 66)

    for start in range(0, total, 100):
        block = hits[start:start + 100]

        first_no = int(df.iloc[start]["draw_no"])
        last_no = int(
            df.iloc[min(start + 99, total - 1)]["draw_no"]
        )

        print(
            f"{first_no}–{last_no}: "
            f"średnia {block.mean():.3f}, "
            f"2+ = {int((block >= 2).sum())}, "
            f"3+ = {int((block >= 3).sum())}, "
            f"4+ = {int((block >= 4).sum())}"
        )

    print("\n" + "=" * 66)
    print("5. PORÓWNANIE Z 10 000 STAŁYCH ZESTAWÓW")
    print("=" * 66)

    print("Trwa porównanie historyczne...")

    random_sets = generate_random_sets(
        SIMULATIONS,
        SEED,
    )

    simulation = simulate_fixed_sets(
        draws,
        random_sets,
    )

    show_comparison(
        "Średnia trafień na losowanie",
        float(hits.mean()),
        simulation["mean_hits"],
    )

    show_comparison(
        "Liczba losowań z 2/6 lub więcej",
        int((hits >= 2).sum()),
        simulation["hits_2plus"],
    )

    show_comparison(
        "Liczba losowań z 3/6 lub więcej",
        int((hits >= 3).sum()),
        simulation["hits_3plus"],
    )

    show_comparison(
        "Liczba losowań z 4/6 lub więcej",
        int((hits >= 4).sum()),
        simulation["hits_4plus"],
    )

    show_comparison(
        "Najlepszy osiągnięty wynik",
        actual_max,
        simulation["max_hits"],
    )

    print("\n" + "=" * 66)
    print("6. PRAWDOPODOBIEŃSTWO BRAKU 4/6+")
    print("=" * 66)

    p_4plus = sum(
        probability_exact_hits(k)
        for k in (4, 5, 6)
    )

    p_no_4plus = (1 - p_4plus) ** total

    print(
        "Prawdopodobieństwo braku 4/6+ "
        f"przez {total} losowań: "
        f"{100 * p_no_4plus:.2f}%"
    )

    # Zapis szczegółów do osobnego CSV.
    export = df.copy()

    export["mask_hits"] = hits

    export["matching_numbers"] = [
        ",".join(
            str(number)
            for number in sorted(
                mask_set.intersection(row)
            )
        )
        for row in draws
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    export.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 66)
    print("ZAKOŃCZONO")
    print("=" * 66)

    print(f"Szczegóły zapisano do: {OUTPUT_PATH}")
    print("Pliki i algorytmy LottoAI nie zostały zmienione.")


if __name__ == "__main__":
    main()
