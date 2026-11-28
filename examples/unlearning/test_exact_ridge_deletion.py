"""Run: python -m unittest discover -s examples/unlearning -v"""
import unittest

from exact_ridge_deletion import Example, ExactRidgeUnlearner, example_rows, train_from_scratch


class ExactRemovalTests(unittest.TestCase):
    def test_exact_removal_matches_clean_retraining_for_each_record(self):
        rows = example_rows()
        for target in rows:
            with self.subTest(target=target.record_id):
                m = train_from_scratch(rows)
                self.assertTrue(m.forget(target.record_id, "req-" + target.record_id))
                clean = train_from_scratch([r for r in rows if r.record_id != target.record_id])
                for actual, expected in zip(m.coefficients(), clean.coefficients()):
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_replay_is_idempotent(self):
        model = train_from_scratch(example_rows())
        self.assertTrue(model.forget("forget-me", "request-1"))
        first = model.coefficients()
        self.assertFalse(model.forget("forget-me", "request-1"))
        self.assertFalse(model.forget("forget-me", "request-2"))
        self.assertEqual(model.coefficients(), first)

    def test_request_id_cannot_be_reused_for_another_record(self):
        m = train_from_scratch(example_rows())
        m.forget("forget-me", "request-one")
        with self.assertRaises(ValueError):
            m.forget("r0", "request-one")

    def test_tombstone_blocks_reinsertion(self):
        m = train_from_scratch(example_rows())
        m.forget("r0", "request-zero")
        with self.assertRaises(ValueError):
            m.add(Example("r0", 0.0, 0.0))

    def test_sequential_deletions_match_retraining(self):
        rows = example_rows()
        m = train_from_scratch(rows)
        for id_ in ("forget-me", "r1", "r4"):
            m.forget(id_, "req-" + id_)
        clean = train_from_scratch([r for r in rows if r.record_id not in {"forget-me", "r1", "r4"}])
        for a, b in zip(m.coefficients(), clean.coefficients()):
            self.assertAlmostEqual(a, b, places=10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            ExactRidgeUnlearner(ridge=0)
        m = ExactRidgeUnlearner()
        with self.assertRaises(ValueError):
            m.add(Example("bad", float("nan"), 3))
        with self.assertRaises(KeyError):
            m.forget("unknown", "request-id")


if __name__ == "__main__":
    unittest.main()