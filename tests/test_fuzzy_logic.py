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

    def test_semantic_medium_conditions(self):
        # Medium SNR (28dB), Medium Dist (50m), Medium Speed (12.5m/s) -> Semantic
        # Rules: Medium SNR + Medium Dist -> Semantic
        self.assertTrue(decide_use_nn(28.0, 50.0, 12.5))

    def test_direct_high_snr_medium_dist(self):
        # High SNR (35dB), Medium Dist (50m), Slow Speed (5m/s) -> Direct
        # Rules: High SNR + Medium Dist + Slow Speed -> Direct
        self.assertFalse(decide_use_nn(35.0, 50.0, 5.0))

    def test_boundary_transition(self):
        # Was previously: 26.0, 45.0, 12.0
        # 26.0 dB -> Highish Medium
        # 45.0 m -> Medium
        # 12.0 m/s -> Medium
        # Rule: Medium SNR + Medium Dist -> Semantic
        self.assertTrue(decide_use_nn(26.0, 45.0, 12.0))

if __name__ == '__main__':
    unittest.main()

