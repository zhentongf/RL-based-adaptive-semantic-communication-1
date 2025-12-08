import unittest

from fuzzy_logic import decide_use_nn

class TestFuzzyLogic(unittest.TestCase):
    def test_direct_high_snr_near_slow(self):
        self.assertFalse(decide_use_nn(40.0, 5.0, 0.5))

    def test_semantic_low_snr(self):
        self.assertTrue(decide_use_nn(10.0, 5.0, 0.5))

    def test_semantic_far_distance(self):
        self.assertTrue(decide_use_nn(30.0, 120.0, 2.0))

    def test_semantic_fast_speed(self):
        self.assertTrue(decide_use_nn(30.0, 10.0, 30.0))

    def test_boundary_transition(self):
        self.assertTrue(decide_use_nn(26.0, 45.0, 12.0))

if __name__ == '__main__':
    unittest.main()

