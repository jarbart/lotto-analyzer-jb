from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Iterable

import numpy as np
import pandas as pd

from analyzer import (
    adjacent_numbers,
    candidate_group_profile,
    coldest_numbers,
    count_internal_adjacent_pairs,
    groups_as_dict,
    hottest_numbers,
    last_draw_numbers,
    number_frequency,
    pair_counter,
    triple_counter,
)


# ============================================================
# KONFIGURACJA
# ============================================================

@dataclass
class GeneratorConfig:
    candidate_pool: int = 5000
    result_count: int = 10
    refinement: int = 100

    seed: int = 42
    max_overlap: int = 3
    max_same_group_profile: int = 3

    # --------------------------------------------------------
    # GŁÓWNY SCORING
    # --------------------------------------------------------
    #
    # Wszystkie heurystyki są pomocnicze.
    # Generator nie próbuje wymusić konkretnego profilu
    # ani konkretnej liczby warunków.

    weight_conditions: float = 1.25

    # Balans nadal jest ważny, ale słabszy niż wcześniej.
    weight_group_balance: float = 1.75

    # Maska użytkownika jest domyślnie wyłączona.
    weight_mask: float = 0.0

    # Częstotliwość ma charakter pomocniczy.
    weight_frequency: float = 0.65

    # Pary i triple są tylko lekkim sygnałem historycznym.
    weight_pairs: float = 0.50
    weight_triples: float = 0.10

    # --------------------------------------------------------
    # POJEDYNCZE CECHY
    # --------------------------------------------------------

    condition_repeat: float = 0.60
    condition_neighbor: float = 0.60
    condition_internal_pair: float = 0.45
    condition_hot: float = 0.30
    condition_cold: float = 0.20
    condition_hot_cold: float = 0.20

    # --------------------------------------------------------
    # REFINEMENT
    # --------------------------------------------------------

    refinement_balance: float = 0.55
    refinement_conditions: float = 0.25
    refinement_mask: float = 0.25
    refinement_history: float = 0.40

    # --------------------------------------------------------
    # GENEROWANIE
    # --------------------------------------------------------

    # Większość kandydatów pochodzi z neutralnego losowania.
    structured_share: float = 0.10

    repeat_share: float = 0.15
    neighbor_share: float = 0.15
    hot_share: float = 0.10
    cold_share: float = 0.10
    pair_share: float = 0.10

    # --------------------------------------------------------
    # MASKA UŻYTKOWNIKA
    # --------------------------------------------------------

    mask: tuple[int, ...] = (
        4,
        15,
        17,
        20,
        21,
        34,
    )

    # 0 = OFF
    weight_mask: float = 0.0

    # --------------------------------------------------------
    # RÓŻNORODNOŚĆ TOP10
    # --------------------------------------------------------

    # Większa niż wcześniej ochrona przed nadmiernym
    # powtarzaniem tych samych liczb.
    diversity_number_penalty: float = 0.70

    # Po przekroczeniu tego poziomu kara rośnie.
    diversity_soft_limit: int = 2


# ============================================================
# WYNIK
# ============================================================

@dataclass(frozen=True)
class CandidateResult:
    numbers: tuple[int, ...]
    score: float

    condition_count: int

    repeated: bool
    neighbors: bool
    internal_pair: bool
    hot: bool
    cold: bool
    hot_cold_together: bool

    condition_hits: tuple[str, ...]
    mask_numbers: tuple[int, ...]

    group_profile: tuple[int, int, int]

    balance_quality: float
    history_quality: float
    condition_quality: float


# ============================================================
# DANE ANALITYCZNE
# ============================================================

@dataclass
class AnalysisData:
    data: pd.DataFrame

    frequencies: pd.Series
    groups: dict[str, set[int]]
    group_totals: dict[str, int]

    last_draw: list[int]
    neighbors: list[int]

    hottest: list[int]
    coldest: list[int]

    pairs: Counter
    triples: Counter

    historical_profiles: Counter


# ============================================================
# POMOCNICZE
# ============================================================

def _clamp01(value: float) -> float:
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def _safe_float(
    value: object,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# GENERATOR
# ============================================================

class LottoGenerator:
    """
    Heurystyczny generator zestawów Lotto.

    Nie jest predyktorem kolejnego losowania.

    Wszystkie statystyki historyczne są liczone wyłącznie
    na przekazanym DataFrame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: GeneratorConfig | None = None,
    ) -> None:
        self.data = self._prepare_data(data)
        self.config = config or GeneratorConfig()

        self.rng = Random(
            self.config.seed
        )

        self.analysis = self._build_analysis(
            self.data
        )

    # --------------------------------------------------------
    # DANE
    # --------------------------------------------------------

    @staticmethod
    def _prepare_data(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if data is None:
            raise ValueError(
                "DataFrame historii nie może być None."
            )

        if data.empty:
            raise ValueError(
                "Historia losowań jest pusta."
            )

        required = {
            "draw_no",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
        }

        missing = required - set(
            data.columns
        )

        if missing:
            raise ValueError(
                "Brak wymaganych kolumn: "
                + ", ".join(sorted(missing))
            )

        result = data.copy()

        result = result.sort_values(
            "draw_no"
        )

        result = result.reset_index(
            drop=True
        )

        return result

    # --------------------------------------------------------
    # ANALIZA
    # --------------------------------------------------------

    def _build_analysis(
        self,
        data: pd.DataFrame,
    ) -> AnalysisData:
        frequencies = number_frequency(
            data
        )

        groups = groups_as_dict(
            data
        )

        groups = {
            "G1": set(groups.get("G1", set())),
            "G2": set(groups.get("G2", set())),
            "G3": set(groups.get("G3", set())),
        }

        group_totals = {
            "G1": 0,
            "G2": 0,
            "G3": 0,
        }

        for group_name, group_numbers in groups.items():
            total = 0

            for number in group_numbers:
                total += int(
                    frequencies.get(
                        number,
                        0,
                    )
                )

            group_totals[group_name] = total

        last_draw = last_draw_numbers(
            data
        )

        neighbors = adjacent_numbers(
            data
        )

        hottest = hottest_numbers(
            data
        )

        coldest = coldest_numbers(
            data
        )

        pairs = pair_counter(
            data
        )

        triples = triple_counter(
            data
        )

        historical_profiles = (
            self._historical_profiles(
                data,
                groups,
            )
        )

        return AnalysisData(
            data=data,
            frequencies=frequencies,
            groups=groups,
            group_totals=group_totals,
            last_draw=last_draw,
            neighbors=neighbors,
            hottest=hottest,
            coldest=coldest,
            pairs=pairs,
            triples=triples,
            historical_profiles=historical_profiles,
        )

    @staticmethod
    def _historical_profiles(
        data: pd.DataFrame,
        groups: dict[str, set[int]],
    ) -> Counter:
        profiles: Counter = Counter()

        for _, row in data.iterrows():
            numbers = (
                int(row["n1"]),
                int(row["n2"]),
                int(row["n3"]),
                int(row["n4"]),
                int(row["n5"]),
                int(row["n6"]),
            )

            profile = candidate_group_profile(
                numbers,
                groups,
            )

            profiles[profile] += 1

        return profiles

    # --------------------------------------------------------
    # NEUTRALNE LOSOWANIE
    # --------------------------------------------------------

    def _random_candidate(
        self,
    ) -> tuple[int, ...]:
        """
        Całkowicie neutralne losowanie.

        Historia nie wpływa na samą pulę kandydatów.
        """

        return tuple(
            sorted(
                self.rng.sample(
                    range(1, 50),
                    6,
                )
            )
        )

    # --------------------------------------------------------
    # LOSOWANIE Z PULI
    # --------------------------------------------------------

    def _pick_from_pool(
        self,
        pool: Iterable[int],
        selected: set[int],
    ) -> int | None:
        available = [
            int(number)
            for number in pool
            if int(number) not in selected
        ]

        if not available:
            return None

        return self.rng.choice(
            available
        )

    # --------------------------------------------------------
    # KANDYDAT STRUKTURALNY
    # --------------------------------------------------------

    def _structured_candidate(
        self,
        analysis: AnalysisData,
    ) -> tuple[int, ...]:
        selected: set[int] = set()

        feature_pools = [
            (
                analysis.last_draw,
                self.config.repeat_share,
            ),
            (
                analysis.neighbors,
                self.config.neighbor_share,
            ),
            (
                analysis.hottest,
                self.config.hot_share,
            ),
            (
                analysis.coldest,
                self.config.cold_share,
            ),
        ]

        for pool, share in feature_pools:
            if self.rng.random() > share:
                continue

            number = self._pick_from_pool(
                pool,
                selected,
            )

            if number is not None:
                selected.add(number)

        # Para jest tylko lekką inspiracją.
        if (
            len(selected) < 6
            and self.rng.random()
            < self.config.pair_share
        ):
            pair_candidates = [
                pair
                for pair, count
                in analysis.pairs.items()
                if count > 0
                and all(
                    number not in selected
                    for number in pair
                )
            ]

            if pair_candidates:
                pair = self.rng.choice(
                    pair_candidates
                )

                for number in pair:
                    if len(selected) < 6:
                        selected.add(
                            int(number)
                        )

        while len(selected) < 6:
            selected.add(
                self.rng.randint(
                    1,
                    49,
                )
            )

        return tuple(
            sorted(selected)
        )

    # --------------------------------------------------------
    # WARUNKI
    # --------------------------------------------------------

    def _condition_details(
        self,
        candidate: Iterable[int],
    ) -> dict[str, object]:
        numbers = tuple(
            sorted(
                int(x)
                for x in candidate
            )
        )

        candidate_set = set(numbers)

        last_set = set(
            self.analysis.last_draw
        )

        neighbor_set = set(
            self.analysis.neighbors
        )

        hot_set = set(
            self.analysis.hottest
        )

        cold_set = set(
            self.analysis.coldest
        )

        repeated_numbers = sorted(
            candidate_set & last_set
        )

        neighbor_numbers = sorted(
            candidate_set & neighbor_set
        )

        hot_numbers = sorted(
            candidate_set & hot_set
        )

        cold_numbers = sorted(
            candidate_set & cold_set
        )

        internal_pairs = (
            count_internal_adjacent_pairs(
                numbers
            )
        )

        repeated = bool(
            repeated_numbers
        )

        neighbors = bool(
            neighbor_numbers
        )

        internal_pair = (
            internal_pairs > 0
        )

        hot = bool(
            hot_numbers
        )

        cold = bool(
            cold_numbers
        )

        hot_cold_together = (
            hot and cold
        )

        condition_hits: list[str] = []

        if repeated:
            condition_hits.append(
                "POWTÓRKA"
            )

        if neighbors:
            condition_hits.append(
                "SĄSIAD"
            )

        if internal_pair:
            condition_hits.append(
                "PARA"
            )

        if hot:
            condition_hits.append(
                "HOT"
            )

        if cold:
            condition_hits.append(
                "COLD"
            )

        if hot_cold_together:
            condition_hits.append(
                "HOT+COLD"
            )

        return {
            "powtórka": repeated,
            "sąsiad": neighbors,
            "para": internal_pair,
            "hot": hot,
            "cold": cold,
            "HOT+COLD": hot_cold_together,
            "repeated_numbers": tuple(
                repeated_numbers
            ),
            "neighbor_numbers": tuple(
                neighbor_numbers
            ),
            "hot_numbers": tuple(
                hot_numbers
            ),
            "cold_numbers": tuple(
                cold_numbers
            ),
            "internal_pairs": internal_pairs,
            "condition_hits": tuple(
                condition_hits
            ),
        }

    # --------------------------------------------------------
    # BALANS GRUP
    # --------------------------------------------------------

    def _balance_quality(
        self,
        candidate: Iterable[int],
    ) -> float:
        """
        Miękki, adaptacyjny balans G1/G2/G3.

        NIE używamy sztywnego 2/2/2 ani 3/2/1.

        Najpierw wyznaczamy rzeczywisty udział G1/G2/G3
        w przekazanym oknie historii.

        Następnie wyznaczamy oczekiwaną liczbę elementów
        każdej grupy w zestawie 6 liczb.

        Im bliżej tego profilu znajduje się kandydat,
        tym wyższa jakość.

        Historyczna częstość konkretnych profili jest używana
        tylko jako bardzo lekki dodatek, a nie jako główny
        mechanizm wymuszający profil.
        """

        profile = candidate_group_profile(
            candidate,
            self.analysis.groups,
        )

        total_group_occurrences = sum(
            self.analysis.group_totals.values()
        )

        if total_group_occurrences <= 0:
            return 0.5

        target_counts: dict[str, float] = {}

        for group_name in ("G1", "G2", "G3"):
            group_total = float(
                self.analysis.group_totals.get(
                    group_name,
                    0,
                )
            )

            share = (
                group_total
                / total_group_occurrences
            )

            target_counts[group_name] = (
                6.0 * share
            )

        deviations = (
            abs(
                profile[0]
                - target_counts["G1"]
            ),
            abs(
                profile[1]
                - target_counts["G2"]
            ),
            abs(
                profile[2]
                - target_counts["G3"]
            ),
        )

        # Normalizujemy odchylenie.
        #
        # Nie chcemy ostro karać profili odbiegających
        # o jeden element od optimum.
        distance = (
            deviations[0] ** 2
            + deviations[1] ** 2
            + deviations[2] ** 2
        )

        adaptive_quality = float(
            np.exp(
                -0.45 * distance
            )
        )

        adaptive_quality = _clamp01(
            adaptive_quality
        )

        # Historyczny profil ma tylko 15% wpływu.
        historical = (
            self.analysis.historical_profiles
        )

        total_profiles = sum(
            historical.values()
        )

        if total_profiles > 0:
            observed = (
                historical.get(
                    profile,
                    0,
                )
                / total_profiles
            )

            max_probability = (
                max(historical.values())
                / total_profiles
            )

            if max_probability > 0:
                historical_quality = (
                    observed
                    / max_probability
                )
            else:
                historical_quality = 0.5
        else:
            historical_quality = 0.5

        historical_quality = _clamp01(
            historical_quality
        )

        quality = (
            0.85 * adaptive_quality
            + 0.15 * historical_quality
        )

        return _clamp01(
            quality
        )

    # --------------------------------------------------------
    # CZĘSTOTLIWOŚĆ
    # --------------------------------------------------------

    def _frequency_quality(
        self,
        candidate: Iterable[int],
    ) -> float:
        """
        Nie nagradzamy ekstremalnego HOT.

        Preferujemy zestawy, których średnia
        częstotliwość nie jest skrajnie odchylona
        od średniej historycznej.
        """

        values = [
            _safe_float(
                self.analysis.frequencies.get(
                    number,
                    0,
                )
            )
            for number in candidate
        ]

        if not values:
            return 0.5

        frequencies = np.asarray(
            values,
            dtype=float,
        )

        mean_frequency = float(
            self.analysis.frequencies.mean()
        )

        if mean_frequency <= 0:
            return 0.5

        deviations = (
            np.abs(
                frequencies
                - mean_frequency
            )
            / mean_frequency
        )

        average_deviation = float(
            np.mean(deviations)
        )

        quality = np.exp(
            -1.5
            * average_deviation
        )

        return _clamp01(
            float(quality)
        )

    # --------------------------------------------------------
    # PARY
    # --------------------------------------------------------

    def _pair_quality(
        self,
        candidate: Iterable[int],
    ) -> float:
        pairs = list(
            combinations(
                sorted(candidate),
                2,
            )
        )

        if not pairs:
            return 0.0

        counts = [
            self.analysis.pairs.get(
                pair,
                0,
            )
            for pair in pairs
        ]

        maximum = max(
            self.analysis.pairs.values(),
            default=0,
        )

        if maximum <= 0:
            return 0.0

        average = float(
            np.mean(counts)
        )

        return _clamp01(
            average / maximum
        )

    # --------------------------------------------------------
    # TRIPLE
    # --------------------------------------------------------

    def _triple_quality(
        self,
        candidate: Iterable[int],
    ) -> float:
        triples = list(
            combinations(
                sorted(candidate),
                3,
            )
        )

        if not triples:
            return 0.0

        counts = [
            self.analysis.triples.get(
                triple,
                0,
            )
            for triple in triples
        ]

        maximum = max(
            self.analysis.triples.values(),
            default=0,
        )

        if maximum <= 0:
            return 0.0

        average = float(
            np.mean(counts)
        )

        return _clamp01(
            average / maximum
        )

    # --------------------------------------------------------
    # WARUNKI
    # --------------------------------------------------------

    def _condition_quality(
        self,
        details: dict[str, object],
    ) -> float:
        """
        Warunki są cechą pomocniczą.

        Brak warunku nie jest dużą karą.
        Nie próbujemy wymusić 6/6.
        """

        weights = [
            (
                bool(details["powtórka"]),
                self.config.condition_repeat,
            ),
            (
                bool(details["sąsiad"]),
                self.config.condition_neighbor,
            ),
            (
                bool(details["para"]),
                self.config.condition_internal_pair,
            ),
            (
                bool(details["hot"]),
                self.config.condition_hot,
            ),
            (
                bool(details["cold"]),
                self.config.condition_cold,
            ),
            (
                bool(details["HOT+COLD"]),
                self.config.condition_hot_cold,
            ),
        ]

        if not weights:
            return 0.5

        active_weight = sum(
            weight
            for active, weight in weights
            if active
        )

        total_weight = sum(
            weight
            for _, weight in weights
        )

        if total_weight <= 0:
            return 0.5

        raw = (
            active_weight
            / total_weight
        )

        # Spłaszczamy różnice.
        quality = 0.35 + (
            0.65 * raw
        )

        return _clamp01(
            quality
        )

    # --------------------------------------------------------
    # HISTORIA
    # --------------------------------------------------------

    def _history_quality(
        self,
        candidate: Iterable[int],
    ) -> float:
        frequency = (
            self._frequency_quality(
                candidate
            )
        )

        pair = (
            self._pair_quality(
                candidate
            )
        )

        triple = (
            self._triple_quality(
                candidate
            )
        )

        return _clamp01(
            0.60 * frequency
            + 0.30 * pair
            + 0.10 * triple
        )

    # --------------------------------------------------------
    # MASKA
    # --------------------------------------------------------

    def _mask_quality(
        self,
        candidate: Iterable[int],
    ) -> tuple[float, tuple[int, ...]]:
        mask = set(
            int(number)
            for number in self.config.mask
            if 1 <= int(number) <= 49
        )

        if not mask:
            return 0.0, ()

        selected = tuple(
            sorted(
                set(candidate)
                & mask
            )
        )

        if not selected:
            return 0.0, ()

        quality = (
            len(selected)
            / len(mask)
        )

        return (
            _clamp01(quality),
            selected,
        )

    # --------------------------------------------------------
    # SCORING
    # --------------------------------------------------------

    def score_candidate(
        self,
        candidate: Iterable[int],
        return_details: bool = False,
    ) -> float | CandidateResult:
        numbers = tuple(
            sorted(
                int(x)
                for x in candidate
            )
        )

        if len(numbers) != 6:
            raise ValueError(
                "Kandydat musi zawierać dokładnie 6 liczb."
            )

        if len(set(numbers)) != 6:
            raise ValueError(
                "Liczby w kandydacie muszą być unikalne."
            )

        if any(
            number < 1 or number > 49
            for number in numbers
        ):
            raise ValueError(
                "Liczby muszą znajdować się w zakresie 1-49."
            )

        details = self._condition_details(
            numbers
        )

        balance_quality = (
            self._balance_quality(
                numbers
            )
        )

        condition_quality = (
            self._condition_quality(
                details
            )
        )

        frequency_quality = (
            self._frequency_quality(
                numbers
            )
        )

        pair_quality = (
            self._pair_quality(
                numbers
            )
        )

        triple_quality = (
            self._triple_quality(
                numbers
            )
        )

        history_quality = (
            self._history_quality(
                numbers
            )
        )

        mask_quality, mask_numbers = (
            self._mask_quality(
                numbers
            )
        )

        group_profile = (
            candidate_group_profile(
                numbers,
                self.analysis.groups,
            )
        )

        score = (
            self.config.weight_conditions
            * condition_quality
            + self.config.weight_group_balance
            * balance_quality
            + self.config.weight_mask
            * mask_quality
            + self.config.weight_frequency
            * frequency_quality
            + self.config.weight_pairs
            * pair_quality
            + self.config.weight_triples
            * triple_quality
        )

        condition_hits = tuple(
            details["condition_hits"]
        )

        condition_count = len(
            condition_hits
        )

        result = CandidateResult(
            numbers=numbers,
            score=float(score),
            condition_count=condition_count,
            repeated=bool(
                details["powtórka"]
            ),
            neighbors=bool(
                details["sąsiad"]
            ),
            internal_pair=bool(
                details["para"]
            ),
            hot=bool(
                details["hot"]
            ),
            cold=bool(
                details["cold"]
            ),
            hot_cold_together=bool(
                details["HOT+COLD"]
            ),
            condition_hits=condition_hits,
            mask_numbers=mask_numbers,
            group_profile=group_profile,
            balance_quality=balance_quality,
            history_quality=history_quality,
            condition_quality=condition_quality,
        )

        if return_details:
            return result

        return result.score

    # --------------------------------------------------------
    # REFINEMENT
    # --------------------------------------------------------

    def refine_candidate(
        self,
        candidate: CandidateResult,
    ) -> CandidateResult:
        mask_quality, mask_numbers = (
            self._mask_quality(
                candidate.numbers
            )
        )

        refined_score = (
            candidate.score
            + self.config.refinement_balance
            * candidate.balance_quality
            + self.config.refinement_conditions
            * candidate.condition_quality
            + self.config.refinement_mask
            * mask_quality
            + self.config.refinement_history
            * candidate.history_quality
        )

        return CandidateResult(
            numbers=candidate.numbers,
            score=float(
                refined_score
            ),
            condition_count=(
                candidate.condition_count
            ),
            repeated=candidate.repeated,
            neighbors=candidate.neighbors,
            internal_pair=candidate.internal_pair,
            hot=candidate.hot,
            cold=candidate.cold,
            hot_cold_together=(
                candidate.hot_cold_together
            ),
            condition_hits=(
                candidate.condition_hits
            ),
            mask_numbers=mask_numbers,
            group_profile=(
                candidate.group_profile
            ),
            balance_quality=(
                candidate.balance_quality
            ),
            history_quality=(
                candidate.history_quality
            ),
            condition_quality=(
                candidate.condition_quality
            ),
        )

    # --------------------------------------------------------
    # NAKŁADANIE KAR ZA POWTARZANIE LICZB
    # --------------------------------------------------------

    def _diversity_adjusted_score(
        self,
        candidate: CandidateResult,
        selected: list[CandidateResult],
    ) -> float:
        """
        Ocena kandydata względem już wybranych TOP10.

        Im częściej liczby kandydata występują już
        w wybranych zestawach, tym większa kara.

        To NIE jest twardy zakaz.
        """

        if not selected:
            return candidate.score

        counts: Counter = Counter()

        for result in selected:
            for number in result.numbers:
                counts[number] += 1

        penalty = 0.0

        for number in candidate.numbers:
            occurrences = counts[number]

            if occurrences <= 0:
                continue

            soft_limit = (
                self.config.diversity_soft_limit
            )

            excess = max(
                0,
                occurrences - soft_limit + 1,
            )

            # Podstawowa kara za każde ponowne użycie.
            penalty += (
                occurrences
                * self.config.diversity_number_penalty
                * 0.40
            )

            # Po przekroczeniu miękkiego limitu
            # kara staje się wyraźniejsza.
            if excess > 0:
                penalty += (
                    excess
                    * self.config.diversity_number_penalty
                    * 0.80
                )

        # Dodatkowo niewielka kara za bardzo duże
        # podobieństwo całego zestawu.
        for existing in selected:
            overlap = self._overlap(
                candidate.numbers,
                existing.numbers,
            )

            if overlap > 2:
                penalty += (
                    (overlap - 2)
                    * self.config.diversity_number_penalty
                    * 1.10
                )

        return (
            candidate.score
            - penalty
        )

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    @staticmethod
    def _overlap(
        first: Iterable[int],
        second: Iterable[int],
    ) -> int:
        return len(
            set(first)
            & set(second)
        )

    # --------------------------------------------------------
    # WYBÓR TOP10
    # --------------------------------------------------------

    def _select_diverse_results(
        self,
        candidates: list[CandidateResult],
    ) -> list[CandidateResult]:
        """
        Selekcja zachłanna.

        Ranking jest dynamicznie korygowany o różnorodność
        liczb oraz profilu G1/G2/G3.

        Nie blokujemy żadnego profilu całkowicie.
        """

        remaining = list(
            candidates
        )

        selected: list[
            CandidateResult
        ] = []

        profile_counts: Counter = Counter()

        while (
            remaining
            and len(selected)
            < self.config.result_count
        ):
            best_index = None
            best_score = None

            for index, candidate in enumerate(
                remaining
            ):
                profile = (
                    candidate.group_profile
                )

                profile_penalty = 0.0

                if (
                    profile_counts[profile]
                    >= self.config.max_same_group_profile
                ):
                    profile_penalty = 1.00

                adjusted = (
                    self._diversity_adjusted_score(
                        candidate,
                        selected,
                    )
                    - profile_penalty
                )

                if (
                    best_score is None
                    or adjusted > best_score
                ):
                    best_score = adjusted
                    best_index = index

            if best_index is None:
                break

            chosen = remaining.pop(
                best_index
            )

            selected.append(
                chosen
            )

            profile_counts[
                chosen.group_profile
            ] += 1

        return selected

    # --------------------------------------------------------
    # GENEROWANIE
    # --------------------------------------------------------

    def generate_sets(
        self,
        result_count: int | None = None,
    ) -> list[CandidateResult]:
        original_count = None

        if result_count is not None:
            original_count = (
                self.config.result_count
            )

            self.config.result_count = int(
                result_count
            )

        try:
            pool_size = max(
                int(
                    self.config.candidate_pool
                ),
                self.config.result_count
                * 100,
            )

            structured_target = int(
                pool_size
                * _clamp01(
                    self.config.structured_share
                )
            )

            candidates: dict[
                tuple[int, ...],
                CandidateResult,
            ] = {}

            # ------------------------------------------------
            # PULA NEUTRALNA
            # ------------------------------------------------

            for _ in range(pool_size):
                candidate = (
                    self._random_candidate()
                )

                if candidate in candidates:
                    continue

                result = self.score_candidate(
                    candidate,
                    return_details=True,
                )

                candidates[
                    candidate
                ] = result

            # ------------------------------------------------
            # PULA STRUKTURALNA
            # ------------------------------------------------

            for _ in range(
                structured_target
            ):
                candidate = (
                    self._structured_candidate(
                        self.analysis
                    )
                )

                if candidate in candidates:
                    continue

                result = self.score_candidate(
                    candidate,
                    return_details=True,
                )

                candidates[
                    candidate
                ] = result

            all_candidates = list(
                candidates.values()
            )

            all_candidates.sort(
                key=lambda item: (
                    -item.score,
                    item.numbers,
                )
            )

            # ------------------------------------------------
            # REFINEMENT
            # ------------------------------------------------

            refinement_count = min(
                int(
                    self.config.refinement
                ),
                len(all_candidates),
            )

            refined = [
                self.refine_candidate(
                    candidate
                )
                for candidate in all_candidates[
                    :refinement_count
                ]
            ]

            remaining = all_candidates[
                refinement_count:
            ]

            combined = (
                refined
                + remaining
            )

            combined.sort(
                key=lambda item: (
                    -item.score,
                    item.numbers,
                )
            )

            # ------------------------------------------------
            # DYNAMICZNY WYBÓR TOP10
            # ------------------------------------------------

            selected = (
                self._select_diverse_results(
                    combined
                )
            )

            return selected

        finally:
            if original_count is not None:
                self.config.result_count = (
                    original_count
                )

    # --------------------------------------------------------
    # TOP 1
    # --------------------------------------------------------

    def top1_result(
        self,
        results: list[CandidateResult] | None = None,
    ) -> CandidateResult:
        if results is None:
            results = self.generate_sets()

        if not results:
            raise RuntimeError(
                "Nie udało się wygenerować żadnego zestawu."
            )

        return results[0]

    # --------------------------------------------------------
    # FORMATOWANIE
    # --------------------------------------------------------

    @staticmethod
    def format_result(
        result: CandidateResult,
    ) -> str:
        numbers = ", ".join(
            str(number)
            for number in result.numbers
        )

        conditions = (
            ", ".join(
                result.condition_hits
            )
            if result.condition_hits
            else "brak"
        )

        g1, g2, g3 = (
            result.group_profile
        )

        return (
            f"Zestaw: {numbers}\n"
            f"Score: {result.score:.3f}\n"
            f"Warunki: "
            f"{result.condition_count}/6"
            f" ({conditions})\n"
            f"G1/G2/G3: "
            f"{g1}/{g2}/{g3}\n"
            f"Balans: "
            f"{result.balance_quality:.3f}\n"
            f"Historia: "
            f"{result.history_quality:.3f}"
        )


# ============================================================
# KOMPATYBILNOŚĆ
# ============================================================

Generator = LottoGenerator


def generate_sets(
    data: pd.DataFrame,
    config: GeneratorConfig | None = None,
) -> list[CandidateResult]:
    generator = LottoGenerator(
        data=data,
        config=config,
    )

    return generator.generate_sets()


def top1_result(
    data: pd.DataFrame,
    config: GeneratorConfig | None = None,
) -> CandidateResult:
    generator = LottoGenerator(
        data=data,
        config=config,
    )

    return generator.top1_result()


def format_result(
    result: CandidateResult,
) -> str:
    return LottoGenerator.format_result(
        result
    )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    from loader import load_data

    data = load_data(
        "data/lotto.txt"
    )

    history = data.tail(50).copy()

    config = GeneratorConfig(
        result_count=10,

        # Maska domyślnie OFF.
        weight_mask=0.0,

        # Tylko niewielka część kandydatów
        # korzysta ze struktury historycznej.
        structured_share=0.10,
    )

    generator = LottoGenerator(
        data=history,
        config=config,
    )

    results = generator.generate_sets()

    print()
    print("=" * 70)
    print("LOTTO AI — DEMO")
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):
        marker = (
            "★  TOP 1"
            if index == 1
            else ""
        )

        print()
        print(
            f"{index:>2}. "
            f"{', '.join(map(str, result.numbers))}"
            f"  |  score={result.score:.3f}"
            f"  |  warunki="
            f"{result.condition_count}/6"
            f"  |  G1/G2/G3="
            f"{result.group_profile[0]}/"
            f"{result.group_profile[1]}/"
            f"{result.group_profile[2]}"
            f"  {marker}"
        )