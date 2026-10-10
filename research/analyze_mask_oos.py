
from __future__ import annotations

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

TRAIN_START = 6000
TRAIN_END = 6699
TEST_START = 6700
TEST_END = 7414

SAMPLE_SIZE = 20_000
SEED = 20261008
BATCH_SIZE = 250

NUMBER_COLUMNS = ["n1", "n2", "n3", "n4", "n5", "n6"]


def profile(numbers):
    numbers = sorted(numbers)

    low = sum(n <= 24 for n in numbers)
    even = sum(n % 2 == 0 for n in numbers)

    adjacent = sum(
        b == a + 1
        for a, b in zip(numbers, numbers[1:])
    )

    return low, even, adjacent


def generate_matching_sets(count, seed):
    rng = np.random.default_rng(seed)

    low_numbers = np.arange(1, 25)
    high_numbers = np.arange(25, 50)

    result = set()
    mask = tuple(sorted(MASK))

    while len(result) < count:
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
            sorted(np.concatenate([low, high]).tolist())
        )

        if candidate == mask:
            continue

        if profile(candidate) == (5, 3, 1):
            result.add(candidate)

    return np.array(
        sorted(result),
        dtype=np.uint8,
    )


def calculate_metrics(draws, candidates):
    """
    Zwraca wyniki dla wszystkich kandydatów:
    średnia trafień, 2/6+, 3/6+, 4/6+.

    Obliczenia partiami, bez dużych macierzy 3D.
    """
    incidence = np.zeros(
        (len(draws), 50),
        dtype=np.uint8,
    )

    rows = np.arange(len(draws))[:, None]
    incidence[rows, draws] = 1

    results = []

    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[start:start + BATCH_SIZE]

        hits = incidence[:, batch].sum(axis=2)

        results.extend(
            {
                "mean": float(hits[:, i].mean()),
                "two": int((hits[:, i] >= 2).sum()),
                "three": int((hits[:, i] >= 3).sum()),
                "four": int((hits[:, i] >= 4).sum()),
            }
            for i in range(hits.shape[1])
        )

    return pd.DataFrame(results)


def mask_metrics(draws):
    target = set(MASK)

    hits = np.array(
        [
            len(target.intersection(row))
            for row in draws
        ],
        dtype=np.uint8,
    )

    return {
        "mean": float(hits.mean()),
        "two": int((hits >= 2).sum()),
        "three": int((hits >= 3).sum()),
        "four": int((hits >= 4).sum()),
    }


def print_metrics(title, metrics, total):
    print(f"\n{title}")
    print("-" * 62)
    print(f"Liczba losowań:       {total}")
    print(f"Średnia trafień:      {metrics['mean']:.4f}")
    print(f"Trafienia 2/6+:       {metrics['two']}")
    print(f"Trafienia 3/6+:       {metrics['three']}")
    print(f"Trafienia 4/6+:       {metrics['four']}")


def main():
    df = load_data(DATA_PATH)
    df = df.sort_values("draw_no").copy()

    train_df = df[
        (df["draw_no"] >= TRAIN_START)
        & (df["draw_no"] <= TRAIN_END)
    ]

    test_df = df[
        (df["draw_no"] >= TEST_START)
        & (df["draw_no"] <= TEST_END)
    ]

    if len(train_df) != 700:
        raise ValueError(
            f"Okres treningowy ma {len(train_df)} losowań, "
            "oczekiwano 700."
        )

    if len(test_df) != 715:
        raise ValueError(
            f"Okres testowy ma {len(test_df)} losowań, "
            "oczekiwano 715."
        )

    train_draws = train_df[
        NUMBER_COLUMNS
    ].to_numpy(dtype=np.uint8)

    test_draws = test_df[
        NUMBER_COLUMNS
    ].to_numpy(dtype=np.uint8)

    print("=" * 66)
    print("TEST OUT-OF-SAMPLE — STAŁE ZESTAWY LOTTO")
    print("=" * 66)

    print("Maska:", MASK)
    print("Profil maski:", profile(MASK))
    print("Trening: 6000–6699")
    print("Test:    6700–7414")
    print("Kandydaci:", SAMPLE_SIZE)

    print("\nGenerowanie zestawów porównawczych...")

    candidates = generate_matching_sets(
        SAMPLE_SIZE,
        SEED,
    )

    print("Obliczanie wyników treningowych...")
    train = calculate_metrics(
        train_draws,
        candidates,
    )

    print("Obliczanie wyników testowych...")
    test = calculate_metrics(
        test_draws,
        candidates,
    )

    actual_train = mask_metrics(train_draws)
    actual_test = mask_metrics(test_draws)

    print_metrics(
        "TWÓJ ZESTAW — TRENING",
        actual_train,
        len(train_draws),
    )

    print_metrics(
        "TWÓJ ZESTAW — TEST",
        actual_test,
        len(test_draws),
    )

    print("\n" + "=" * 66)
    print("1. WYBÓR NAJLEPSZYCH 1% W TRENINGU")
    print("=" * 66)

    top_count = max(1, SAMPLE_SIZE // 100)

    top_indices = train.sort_values(
        by=["two", "mean"],
        ascending=[False, False],
        kind="mergesort",
    ).head(top_count).index

    top_train = train.loc[top_indices]
    top_test = test.loc[top_indices]

    print(
        f"Wybrano {len(top_indices)} zestawów "
        "na podstawie trafień 2/6+ w treningu."
    )

    print("\nŚREDNIE WYNIKI TRENINGOWE")
    print(
        f"Wszystkie zestawy — 2/6+: "
        f"{train['two'].mean():.2f}"
    )
    print(
        f"Najlepszy 1% — 2/6+:     "
        f"{top_train['two'].mean():.2f}"
    )

    print("\nŚREDNIE WYNIKI TESTOWE")
    print(
        f"Wszystkie zestawy — 2/6+: "
        f"{test['two'].mean():.2f}"
    )
    print(
        f"Najlepszy 1% — 2/6+:     "
        f"{top_test['two'].mean():.2f}"
    )

    print(
        f"Różnica w teście: "
        f"{top_test['two'].mean() - test['two'].mean():+.2f}"
    )

    print("\n" + "=" * 66)
    print("2. CZY PRZEWAGA PRZESZŁA DO TESTU?")
    print("=" * 66)

    for key, label in (
        ("mean", "Średnia trafień"),
        ("two", "Trafienia 2/6+"),
        ("three", "Trafienia 3/6+"),
        ("four", "Trafienia 4/6+"),
    ):
        train_advantage = (
            top_train[key].mean()
            - train[key].mean()
        )

        test_advantage = (
            top_test[key].mean()
            - test[key].mean()
        )

        print(f"\n{label}")
        print(
            f"Przewaga w treningu: "
            f"{train_advantage:+.4f}"
        )
        print(
            f"Przewaga w teście:    "
            f"{test_advantage:+.4f}"
        )

    print("\n" + "=" * 66)
    print("3. KORELACJA TRENING–TEST")
    print("=" * 66)

    for key, label in (
        ("mean", "Średnia trafień"),
        ("two", "Trafienia 2/6+"),
        ("three", "Trafienia 3/6+"),
        ("four", "Trafienia 4/6+"),
    ):
        correlation = train[key].corr(test[key])

        print(
            f"{label:<22}: "
            f"{correlation:+.4f}"
        )

    print("\n" + "=" * 66)
    print("4. GDZIE JEST TWOJA MASKA?")
    print("=" * 66)

    for period, reference, actual in (
        ("Trening", train, actual_train),
        ("Test", test, actual_test),
    ):
        below = 100 * (
            reference["two"] < actual["two"]
        ).mean()

        equal = 100 * (
            reference["two"] == actual["two"]
        ).mean()

        above = 100 * (
            reference["two"] > actual["two"]
        ).mean()

        print(f"\n{period}")
        print(f"2/6+ maski: {actual['two']}")
        print(f"Gorsze zestawy: {below:.1f}%")
        print(f"Równe zestawy:  {equal:.1f}%")
        print(f"Lepsze zestawy: {above:.1f}%")

    print("\n" + "=" * 66)
    print("KONIEC TESTU")
    print("=" * 66)

    print(
        "To test historyczny. Nie dowodzi "
        "przewidywalności przyszłych losowań."
    )


if __name__ == "__main__":
    main()
