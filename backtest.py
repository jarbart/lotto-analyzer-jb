from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import pandas as pd

from analyzer import (
    candidate_group_profile,
    groups_as_dict,
)
from generator import (
    GeneratorConfig,
    LottoGenerator,
)
from loader import load_data


# ============================================================
# KONFIGURACJA
# ============================================================

DATA_PATH = Path("data/lotto.txt")

# Szybki, reprezentatywny test.
TEST_DRAWS = 300

# Sprawdzamy wszystkie trzy interesujące okna.
HISTORY_WINDOWS = [30, 50, 100]

RESULT_COUNT = 10

GENERATOR_SEED = 42
RANDOM_SEED = 123456

# Generator:
#
# 1500 zamiast 5000 kandydatów
# 50 zamiast 100 refinement
#
# Logika generatora pozostaje dokładnie taka sama.
CANDIDATE_POOL = 1500
REFINEMENT = 50


# ============================================================
# WYNIKI
# ============================================================

@dataclass
class HitStats:
    hits: list[int]

    @property
    def mean(self) -> float:
        if not self.hits:
            return 0.0

        return float(
            np.mean(self.hits)
        )

    @property
    def best(self) -> int:
        if not self.hits:
            return 0

        return max(self.hits)

    def percentage_at_least(
        self,
        threshold: int,
    ) -> float:
        if not self.hits:
            return 0.0

        return (
            100.0
            * sum(
                hit >= threshold
                for hit in self.hits
            )
            / len(self.hits)
        )


@dataclass
class BacktestResult:
    top1: HitStats
    best10: HitStats
    numbers_used: Counter
    group_profiles: Counter


# ============================================================
# POMOCNICZE
# ============================================================

def draw_numbers(
    row: pd.Series,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(row[column])
            for column in (
                "n1",
                "n2",
                "n3",
                "n4",
                "n5",
                "n6",
            )
        )
    )


def count_hits(
    candidate: tuple[int, ...],
    actual: tuple[int, ...],
) -> int:
    return len(
        set(candidate)
        & set(actual)
    )


def random_set(
    rng: Random,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            rng.sample(
                range(1, 50),
                6,
            )
        )
    )


def get_group_profile(
    candidate: tuple[int, ...],
    history: pd.DataFrame,
) -> tuple[int, int, int]:
    groups = groups_as_dict(
        history
    )

    return candidate_group_profile(
        candidate,
        groups,
    )


# ============================================================
# RANDOM BASELINE
# ============================================================

def generate_random_results(
    actual: tuple[int, ...],
    rng: Random,
) -> tuple[int, list[tuple[int, ...]]]:
    candidates: list[
        tuple[int, ...]
    ] = []

    seen: set[
        tuple[int, ...]
    ] = set()

    while len(candidates) < RESULT_COUNT:
        candidate = random_set(
            rng
        )

        if candidate in seen:
            continue

        seen.add(candidate)
        candidates.append(candidate)

    hits = [
        count_hits(
            candidate,
            actual,
        )
        for candidate in candidates
    ]

    return (
        max(hits),
        candidates,
    )


# ============================================================
# JEDNO OKNO HISTORII
# ============================================================

def run_window_test(
    data: pd.DataFrame,
    history_window: int,
    test_draws: int,
    generator_seed: int,
    random_seed: int,
) -> tuple[
    BacktestResult,
    BacktestResult,
]:
    total_draws = len(data)

    start_index = max(
        history_window,
        total_draws - test_draws,
    )

    actual_test_count = (
        total_draws
        - start_index
    )

    print()
    print("=" * 78)
    print(
        f"OKNO HISTORII: {history_window} | "
        f"TEST: {actual_test_count} LOSOWAŃ"
    )
    print("=" * 78)

    generator_hits_top1: list[int] = []
    generator_hits_best10: list[int] = []

    random_hits_top1: list[int] = []
    random_hits_best10: list[int] = []

    generator_number_usage: Counter = Counter()
    random_number_usage: Counter = Counter()

    generator_profiles: Counter = Counter()
    random_profiles: Counter = Counter()

    random_rng = Random(
        random_seed
    )

    for test_number, index in enumerate(
        range(
            start_index,
            total_draws,
        ),
        start=1,
    ):
        history_start = (
            index
            - history_window
        )

        history = data.iloc[
            history_start:index
        ].copy()

        actual = draw_numbers(
            data.iloc[index]
        )

        # ----------------------------------------------------
        # LOTTO AI
        # ----------------------------------------------------

        config = GeneratorConfig(
            candidate_pool=CANDIDATE_POOL,
            result_count=RESULT_COUNT,
            refinement=REFINEMENT,
            seed=generator_seed,
            weight_mask=0.0,
        )

        generator = LottoGenerator(
            data=history,
            config=config,
        )

        results = (
            generator.generate_sets()
        )

        if not results:
            raise RuntimeError(
                "Generator nie zwrócił żadnych wyników."
            )

        top1 = results[0]

        top1_hit = count_hits(
            top1.numbers,
            actual,
        )

        best10_hit = max(
            count_hits(
                result.numbers,
                actual,
            )
            for result in results
        )

        generator_hits_top1.append(
            top1_hit
        )

        generator_hits_best10.append(
            best10_hit
        )

        for result in results:
            for number in result.numbers:
                generator_number_usage[
                    number
                ] += 1

            generator_profiles[
                result.group_profile
            ] += 1

        # ----------------------------------------------------
        # RANDOM
        # ----------------------------------------------------

        random_top1, random_candidates = (
            generate_random_results(
                actual,
                random_rng,
            )
        )

        random_top1_hit = count_hits(
            random_candidates[0],
            actual,
        )

        random_best10_hit = max(
            count_hits(
                candidate,
                actual,
            )
            for candidate in random_candidates
        )

        random_hits_top1.append(
            random_top1_hit
        )

        random_hits_best10.append(
            random_best10_hit
        )

        for candidate in random_candidates:
            for number in candidate:
                random_number_usage[
                    number
                ] += 1

            profile = get_group_profile(
                candidate,
                history,
            )

            random_profiles[
                profile
            ] += 1

        # ----------------------------------------------------
        # PROSTY POSTĘP
        # ----------------------------------------------------

        if (
            test_number % 25 == 0
            or test_number
            == actual_test_count
        ):
            print(
                f"  postęp: "
                f"{test_number}/"
                f"{actual_test_count}"
            )

    return (
        BacktestResult(
            top1=HitStats(
                generator_hits_top1
            ),
            best10=HitStats(
                generator_hits_best10
            ),
            numbers_used=(
                generator_number_usage
            ),
            group_profiles=(
                generator_profiles
            ),
        ),
        BacktestResult(
            top1=HitStats(
                random_hits_top1
            ),
            best10=HitStats(
                random_hits_best10
            ),
            numbers_used=(
                random_number_usage
            ),
            group_profiles=(
                random_profiles
            ),
        ),
    )


# ============================================================
# STATYSTYKI
# ============================================================

def print_stats(
    name: str,
    stats: HitStats,
) -> None:
    print()
    print(name)

    print(
        f"  średnia trafień : "
        f"{stats.mean:.3f}"
    )

    print(
        f"  najlepszy wynik : "
        f"{stats.best}/6"
    )

    print(
        f"  >=1             : "
        f"{stats.percentage_at_least(1):.1f}%"
    )

    print(
        f"  >=2             : "
        f"{stats.percentage_at_least(2):.1f}%"
    )

    print(
        f"  >=3             : "
        f"{stats.percentage_at_least(3):.1f}%"
    )

    print(
        f"  >=4             : "
        f"{stats.percentage_at_least(4):.1f}%"
    )


def print_comparison(
    generator: BacktestResult,
    random: BacktestResult,
) -> None:
    delta_top1 = (
        generator.top1.mean
        - random.top1.mean
    )

    delta_best10 = (
        generator.best10.mean
        - random.best10.mean
    )

    delta_top1_2 = (
        generator.top1.percentage_at_least(2)
        - random.top1.percentage_at_least(2)
    )

    delta_best10_2 = (
        generator.best10.percentage_at_least(2)
        - random.best10.percentage_at_least(2)
    )

    delta_top1_3 = (
        generator.top1.percentage_at_least(3)
        - random.top1.percentage_at_least(3)
    )

    delta_best10_3 = (
        generator.best10.percentage_at_least(3)
        - random.best10.percentage_at_least(3)
    )

    print()
    print("-" * 78)
    print("LOTTO AI vs RANDOM")
    print("-" * 78)

    print(
        f"TOP 1       : "
        f"{generator.top1.mean:.3f} "
        f"vs "
        f"{random.top1.mean:.3f} "
        f" | Δ {delta_top1:+.3f}"
    )

    print(
        f"BEST TOP 10 : "
        f"{generator.best10.mean:.3f} "
        f"vs "
        f"{random.best10.mean:.3f} "
        f" | Δ {delta_best10:+.3f}"
    )

    print(
        f"TOP 1 >=2   : "
        f"{generator.top1.percentage_at_least(2):.1f}% "
        f"vs "
        f"{random.top1.percentage_at_least(2):.1f}% "
        f" | Δ {delta_top1_2:+.1f} pp"
    )

    print(
        f"TOP10 >=2   : "
        f"{generator.best10.percentage_at_least(2):.1f}% "
        f"vs "
        f"{random.best10.percentage_at_least(2):.1f}% "
        f" | Δ {delta_best10_2:+.1f} pp"
    )

    print(
        f"TOP 1 >=3   : "
        f"{generator.top1.percentage_at_least(3):.1f}% "
        f"vs "
        f"{random.top1.percentage_at_least(3):.1f}% "
        f" | Δ {delta_top1_3:+.1f} pp"
    )

    print(
        f"TOP10 >=3   : "
        f"{generator.best10.percentage_at_least(3):.1f}% "
        f"vs "
        f"{random.best10.percentage_at_least(3):.1f}% "
        f" | Δ {delta_best10_3:+.1f} pp"
    )


# ============================================================
# KONCENTRACJA
# ============================================================

def print_concentration(
    name: str,
    usage: Counter,
    total_slots: int,
) -> None:
    expected = (
        total_slots
        / 49.0
    )

    ranking = sorted(
        range(1, 50),
        key=lambda number: (
            -usage[number],
            number,
        ),
    )

    print()
    print("-" * 78)
    print(
        f"KONCENTRACJA — {name}"
    )
    print("-" * 78)

    print(
        f"Oczekiwane użycia jednej liczby: "
        f"{expected:.1f}"
    )

    print()
    print("TOP 5:")

    for number in ranking[:5]:
        count = usage[number]

        ratio = (
            count / expected
            if expected > 0
            else 0.0
        )

        print(
            f"  {number:2d}: "
            f"{count:4d} "
            f"({ratio:.2f}x)"
        )

    print()
    print("BOTTOM 5:")

    for number in ranking[-5:]:
        count = usage[number]

        ratio = (
            count / expected
            if expected > 0
            else 0.0
        )

        print(
            f"  {number:2d}: "
            f"{count:4d} "
            f"({ratio:.2f}x)"
        )


# ============================================================
# PROFILE G1/G2/G3
# ============================================================

def print_profiles(
    name: str,
    profiles: Counter,
    total_sets: int,
) -> None:
    print()
    print(
        f"PROFILE G1/G2/G3 — {name}"
    )

    ordered = sorted(
        profiles.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for profile, count in ordered:
        percentage = (
            100.0
            * count
            / total_sets
            if total_sets > 0
            else 0.0
        )

        print(
            f"  {profile[0]}/"
            f"{profile[1]}/"
            f"{profile[2]} : "
            f"{count:5d} "
            f"({percentage:5.1f}%)"
        )


# ============================================================
# PODSUMOWANIE
# ============================================================

def print_summary(
    results: dict[
        int,
        tuple[
            BacktestResult,
            BacktestResult,
        ],
    ],
) -> None:
    print()
    print()
    print("=" * 78)
    print("PODSUMOWANIE 30 / 50 / 100")
    print("=" * 78)

    print()
    print(
        "OKNO | AI TOP1 | RND TOP1 | "
        "Δ TOP1 | AI TOP10 | RND TOP10 | Δ TOP10"
    )

    print("-" * 78)

    for window in sorted(
        results
    ):
        generator, random = (
            results[window]
        )

        delta_top1 = (
            generator.top1.mean
            - random.top1.mean
        )

        delta_best10 = (
            generator.best10.mean
            - random.best10.mean
        )

        print(
            f"{window:3d} | "
            f"{generator.top1.mean:7.3f} | "
            f"{random.top1.mean:8.3f} | "
            f"{delta_top1:+7.3f} | "
            f"{generator.best10.mean:8.3f} | "
            f"{random.best10.mean:9.3f} | "
            f"{delta_best10:+7.3f}"
        )

    print()
    print("-" * 78)

    positive_top1 = sum(
        generator.top1.mean
        > random.top1.mean
        for generator, random
        in results.values()
    )

    positive_best10 = sum(
        generator.best10.mean
        > random.best10.mean
        for generator, random
        in results.values()
    )

    print(
        f"TOP1 lepsze od RANDOM: "
        f"{positive_top1}/"
        f"{len(results)} okien"
    )

    print(
        f"BEST10 lepsze od RANDOM: "
        f"{positive_best10}/"
        f"{len(results)} okien"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("LOTTO AI — SZYBKI FINALNY BACKTEST")
    print("=" * 78)

    data = load_data(
        DATA_PATH
    )

    print()
    print(
        f"Wszystkich losowań: "
        f"{len(data)}"
    )

    print(
        f"Zakres: "
        f"{int(data['draw_no'].min())} - "
        f"{int(data['draw_no'].max())}"
    )

    print(
        f"Test: ostatnie {TEST_DRAWS} losowań"
    )

    print(
        f"Okna historii: "
        f"{HISTORY_WINDOWS}"
    )

    print(
        f"Pula generatora: "
        f"{CANDIDATE_POOL}"
    )

    print(
        f"Refinement: "
        f"{REFINEMENT}"
    )

    print(
        "Maska: OFF"
    )

    results: dict[
        int,
        tuple[
            BacktestResult,
            BacktestResult,
        ],
    ] = {}

    for window in HISTORY_WINDOWS:
        generator_result, random_result = (
            run_window_test(
                data=data,
                history_window=window,
                test_draws=TEST_DRAWS,
                generator_seed=(
                    GENERATOR_SEED
                ),
                random_seed=(
                    RANDOM_SEED
                    + window
                ),
            )
        )

        results[window] = (
            generator_result,
            random_result,
        )

        print_stats(
            "LOTTO AI — TOP 1",
            generator_result.top1,
        )

        print_stats(
            "LOTTO AI — BEST TOP 10",
            generator_result.best10,
        )

        print_stats(
            "RANDOM — TOP 1",
            random_result.top1,
        )

        print_stats(
            "RANDOM — BEST TOP 10",
            random_result.best10,
        )

        print_comparison(
            generator_result,
            random_result,
        )

        total_slots = (
            len(
                generator_result.best10.hits
            )
            * RESULT_COUNT
            * 6
        )

        print_concentration(
            "LOTTO AI",
            generator_result.numbers_used,
            total_slots,
        )

        print_concentration(
            "RANDOM",
            random_result.numbers_used,
            total_slots,
        )

        total_sets = (
            len(
                generator_result.best10.hits
            )
            * RESULT_COUNT
        )

        print_profiles(
            "LOTTO AI",
            generator_result.group_profiles,
            total_sets,
        )

        print_profiles(
            "RANDOM",
            random_result.group_profiles,
            total_sets,
        )

    print_summary(
        results
    )

    print()
    print("=" * 78)
    print("KONIEC")
    print("=" * 78)


if __name__ == "__main__":
    main()