
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

START_DRAW = 6000
END_DRAW = 7414
TRAIN_SIZE = 700

CANDIDATES = 20_000
PERMUTATIONS = 1_000
SEED = 20261008

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


def generate_candidates(count, rng):
    low_pool = np.arange(1, 25)
    high_pool = np.arange(25, 50)

    result = set()

    while len(result) < count:
        low = rng.choice(
            low_pool,
            size=5,
            replace=False,
        )

        high = rng.choice(
            high_pool,
            size=1,
            replace=False,
        )

        candidate = tuple(
            sorted(np.concatenate([low, high]).tolist())
        )

        if candidate == MASK:
            continue

        if profile(candidate) == (5, 3, 1):
            result.add(candidate)

    return np.array(
        sorted(result),
        dtype=np.uint8,
    )


def build_hit_matrix(draws, candidates):
    """
    Macierz:
      wiersze = losowania
      kolumny = stałe zestawy
      wartości = liczba trafień 0..6
    """
    incidence = np.zeros(
        (len(draws), 50),
        dtype=np.uint8,
    )

    rows = np.arange(len(draws))[:, None]
    incidence[rows, draws] = 1

    matrix = np.empty(
        (len(draws), len(candidates)),
        dtype=np.uint8,
    )

    batch_size = 250

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]

        matrix[:, start:start + len(batch)] = (
            incidence[:, batch].sum(axis=2)
        )

    return matrix


def correlation(x, y):
    """
    Korelacja Pearsona dla dwóch wektorów.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denominator = np.sqrt(
        np.dot(x, x) * np.dot(y, y)
    )

    if denominator == 0:
        return float("nan")

    return float(np.dot(x, y) / denominator)


def metrics_for_split(matrix, train_indices):
    """
    Wydajnie wyznacza wyniki w obu okresach.

    Wykorzystuje sumy po wszystkich losowaniach,
    zamiast za każdym razem przeliczać test od zera.
    """
    train_hits = matrix[train_indices]

    total_mean = matrix.sum(axis=0, dtype=np.int32)
    train_mean = train_hits.sum(axis=0, dtype=np.int32)

    total_two = (matrix >= 2).sum(
        axis=0,
        dtype=np.int32,
    )
    train_two = (train_hits >= 2).sum(
        axis=0,
        dtype=np.int32,
    )

    total_three = (matrix >= 3).sum(
        axis=0,
        dtype=np.int32,
    )
    train_three = (train_hits >= 3).sum(
        axis=0,
        dtype=np.int32,
    )

    test_size = len(matrix) - len(train_indices)

    return {
        "mean": (
            train_mean / len(train_indices),
            (total_mean - train_mean) / test_size,
        ),
        "two": (
            train_two,
            total_two - train_two,
        ),
        "three": (
            train_three,
            total_three - train_three,
        ),
    }


def evaluate_split(matrix, train_indices):
    metrics = metrics_for_split(
        matrix,
        train_indices,
    )

    results = {}

    for key, (train, test) in metrics.items():
        results[key] = correlation(train, test)

    return results


def report(label, observed, simulated):
    simulated = np.asarray(
        simulated,
        dtype=np.float64,
    )

    lower = np.mean(simulated <= observed)
    upper = np.mean(simulated >= observed)

    # Dwustronne empiryczne p:
    # porównanie odległości od mediany permutacji.
    center = float(np.median(simulated))

    distances = np.abs(simulated - center)
    observed_distance = abs(observed - center)

    p_two_sided = (
        1 + np.count_nonzero(
            distances >= observed_distance
        )
    ) / (len(simulated) + 1)

    print(f"\n{label}")
    print("-" * 60)
    print(f"Korelacja rzeczywista:     {observed:+.4f}")
    print(f"Średnia permutacji:        {simulated.mean():+.4f}")
    print(f"Mediana permutacji:        {center:+.4f}")
    print(
        "Przedział 2,5–97,5%:       "
        f"{np.quantile(simulated, 0.025):+.4f}"
        " do "
        f"{np.quantile(simulated, 0.975):+.4f}"
    )
    print(
        f"Permutacje <= wyniku:      {100 * lower:.1f}%"
    )
    print(
        f"Permutacje >= wyniku:      {100 * upper:.1f}%"
    )
    print(
        f"Empiryczne p dwustronne:   {p_two_sided:.4f}"
    )


def main():
    df = load_data(DATA_PATH)
    df = df.sort_values("draw_no")

    df = df[
        (df["draw_no"] >= START_DRAW)
        & (df["draw_no"] <= END_DRAW)
    ]

    expected_count = END_DRAW - START_DRAW + 1

    if len(df) != expected_count:
        raise ValueError(
            f"Oczekiwano {expected_count} losowań, "
            f"otrzymano {len(df)}."
        )

    draws = df[NUMBER_COLUMNS].to_numpy(
        dtype=np.uint8
    )

    rng = np.random.default_rng(SEED)

    print("=" * 66)
    print("TEST PERMUTACYJNY — LOTTO")
    print("=" * 66)

    print("Losowania:", START_DRAW, "–", END_DRAW)
    print("Liczba losowań:", len(draws))
    print("Trening:", TRAIN_SIZE)
    print("Test:", len(draws) - TRAIN_SIZE)
    print("Kandydaci:", CANDIDATES)
    print("Permutacje:", PERMUTATIONS)

    print("\nGenerowanie zestawów...")
    candidates = generate_candidates(
        CANDIDATES,
        rng,
    )

    print("Budowanie macierzy trafień...")
    matrix = build_hit_matrix(
        draws,
        candidates,
    )

    observed_indices = np.arange(TRAIN_SIZE)

    observed = evaluate_split(
        matrix,
        observed_indices,
    )

    simulated = {
        "mean": [],
        "two": [],
        "three": [],
    }

    print("\nUruchamianie permutacji...")

    for iteration in range(PERMUTATIONS):
        shuffled = rng.permutation(len(draws))
        train_indices = shuffled[:TRAIN_SIZE]

        result = evaluate_split(
            matrix,
            train_indices,
        )

        for key in simulated:
            simulated[key].append(result[key])

        if (iteration + 1) % 100 == 0:
            print(
                f"Wykonano {iteration + 1}"
                f"/{PERMUTATIONS}"
            )

    print("\n" + "=" * 66)
    print("WYNIKI")
    print("=" * 66)

    report(
        "ŚREDNIA LICZBA TRAFIEŃ",
        observed["mean"],
        simulated["mean"],
    )

    report(
        "TRAFIENIA 2/6+",
        observed["two"],
        simulated["two"],
    )

    report(
        "TRAFIENIA 3/6+",
        observed["three"],
        simulated["three"],
    )

    print("\n" + "=" * 66)
    print("INTERPRETACJA")
    print("=" * 66)

    print(
        "Małe p sugeruje, że rzeczywisty podział "
        "czasowy daje nietypową korelację "
        "względem losowych podziałów."
    )
    print(
        "Nie oznacza to automatycznie "
        "możliwości przewidywania Lotto."
    )


if __name__ == "__main__":
    main()
