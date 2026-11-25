import unittest
from random import Random
from reasoning_test_time_compute_demo import (
    Candidate, oracle_has_solution, select_best, sample_candidates, run
)


class ReasoningDemoTests(unittest.TestCase):
    def test_verifier_selection_uses_score(self):
        xs = [Candidate(True, -1.0), Candidate(False, 4.5)]
        self.assertEqual(select_best(xs), xs[1])
        self.assertTrue(oracle_has_solution(xs))

    def test_empty_selection_rejected(self):
        with self.assertRaises(ValueError):
            select_best([])

    def test_zero_and_full_probabilities(self):
        rng = Random(5)
        self.assertTrue(all(x.is_correct for x in sample_candidates(rng, 1, 20)))
        self.assertFalse(any(x.is_correct for x in sample_candidates(rng, 0, 20)))

    def test_oracle_is_upper_bound(self):
        results = run(50, 11)
        for result in results.values():
            self.assertLessEqual(result.successful, result.oracle_successful)

    def test_reproducible(self):
        self.assertEqual(run(40, 9), run(40, 9))


if __name__ == "__main__":
    unittest.main()