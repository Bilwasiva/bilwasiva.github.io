#!/usr/bin/env python3
"""Illustrative test-time-compute experiment; NOT a benchmark of an LLM.

Demonstrates the difference between oracle pass@N and selected-answer success
with an imperfect verifier, including simple per-difficulty compute routing.

Requires Python 3.10+, standard library only.
Run: python3 reasoning_test_time_compute_demo.py --trials 20000 --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Candidate:
    is_correct: bool
    verifier_score: float


@dataclass(frozen=True)
class TaskType:
    name: str
    single_sample_accuracy: float
    adaptive_budget: int


TASK_TYPES = (
    TaskType("easy", 0.82, 1),
    TaskType("moderate", 0.43, 4),
    TaskType("hard", 0.16, 8),
)
MAX_CANDIDATES = 8


def select_best(candidates: list[Candidate]) -> Candidate:
    """Use ONLY the verifier score; hidden truth must not affect selection."""
    if not candidates:
        raise ValueError("Cannot select from an empty candidate set")
    return max(candidates, key=lambda candidate: candidate.verifier_score)


def oracle_has_solution(candidates: list[Candidate]) -> bool:
    """Evaluation-only; not an executable strategy without a perfect oracle."""
    return any(candidate.is_correct for candidate in candidates)


def sample_candidates(rng: Random, p_correct: float, count: int) -> list[Candidate]:
    """Independent Bernoulli correctness; noisy Gaussian verifier scores.

    This synthetic data model is deliberately simplistic and transparent.
    Correlation, domain shift, prompt effects and real model score distributions
    are *not* represented.
    """
    if not (0.0 <= p_correct <= 1.0):
        raise ValueError("p_correct must be within [0, 1]")
    if count < 1:
        raise ValueError("count must be >= 1")

    outcomes = []
    for _ in range(count):
        correct = rng.random() < p_correct
        # The overlap makes the verifier imperfect. It never sees true labels.
        score = rng.gauss(1.0 if correct else 0.0, 1.05)
        outcomes.append(Candidate(is_correct=correct, verifier_score=score))
    return outcomes


@dataclass
class Counters:
    attempts: int = 0
    successful: int = 0
    oracle_successful: int = 0
    samples: int = 0

    def record(self, sample_list: list[Candidate]) -> None:
        self.attempts += 1
        self.successful += int(select_best(sample_list).is_correct)
        self.oracle_successful += int(oracle_has_solution(sample_list))
        self.samples += len(sample_list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.attempts

    @property
    def oracle_rate(self) -> float:
        return self.oracle_successful / self.attempts

    @property
    def mean_samples(self) -> float:
        return self.samples / self.attempts


def run(trials_per_type: int, seed: int) -> dict[str, Counters]:
    if trials_per_type < 1:
        raise ValueError("trials_per_type must be >= 1")

    rng = Random(seed)
    names = ("fixed_1", "fixed_4", "fixed_8", "adaptive")
    results = {name: Counters() for name in names}

    for task_type in TASK_TYPES:
        for _ in range(trials_per_type):
            # All strategies use *prefixes of the same candidate pool* for a
            # fair comparison on each simulated task.
            candidates = sample_candidates(
                rng, task_type.single_sample_accuracy, MAX_CANDIDATES
            )
            budgets = {
                "fixed_1": 1,
                "fixed_4": 4,
                "fixed_8": 8,
                "adaptive": task_type.adaptive_budget,
            }
            for name, budget in budgets.items():
                results[name].record(candidates[:budget])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20000,
                        help="Tasks per difficulty class (default: 20000)")
    parser.add_argument("--seed", type=int, default=7,
                        help="Deterministic random seed (default: 7)")
    args = parser.parse_args()
    results = run(args.trials, args.seed)

    p = 0.4
    n = 5
    print(f"Independent-candidate oracle example: 1 - (1 - {p})^{n}"
          f" = {1 - (1 - p) ** n:.5f}")
    print("\nSynthetic tasks: 3 difficulty classes, equally weighted")
    print("Verifier: overlapping Gaussian scores; higher on average if correct")
    print("All results are simulation outputs, NOT real-model benchmarks.\n")
    print(f"{'Policy':<14} {'Chosen success':>15} {'Oracle pass@N':>15}"
          f" {'Mean samples':>14} {'Tasks':>10}")
    print("-" * 73)
    for name, value in results.items():
        assert value.successful <= value.oracle_successful
        print(f"{name:<14} {value.success_rate:>14.2%}"
              f" {value.oracle_rate:>14.2%}"
              f" {value.mean_samples:>14.2f}"
              f" {value.attempts:>10,d}")

    print("\nKey distinction: Oracle pass@N reveals if a correct candidate"
          " existed; selected success shows if the imperfect verifier chose it.")


if __name__ == "__main__":
    main()