"""Unit tests for the educational closed-loop simulator."""
import unittest
from toy_safe_mpc import State, GOAL, run, is_safe, safety_shield, candidate_plan

class SafeMPCTests(unittest.TestCase):
    def test_safe_trace(self):
        self.assertTrue(all(is_safe(s) for s in run()))
    def test_goal(self):
        self.assertLess(abs(run()[-1].x - GOAL), 0.22)
    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):
            safety_shield(State(0.0, 0.0), 99)
    def test_candidate_action_in_allowed_set(self):
        self.assertIn(candidate_plan(State(0.0, 0.0)), (-1.0,0.0,1.0))
    def test_does_not_cross_limit(self):
        self.assertLessEqual(max(s.x for s in run()), 1.50)

if __name__ == '__main__':
    unittest.main()