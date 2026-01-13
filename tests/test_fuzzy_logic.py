import unittest

from fuzzy_logic import decide_use_nn

class TestFuzzyLogic(unittest.TestCase):
    def test_direct_high_snr_near_slow(self):
        # High SNR (1.0), Near Dist (0.0), Slow Speed (0.0) -> Direct (False)
        self.assertFalse(decide_use_nn(1.0, 0.0, 0.0))

    def test_semantic_low_snr(self):
        # Low SNR (0.0), Near Dist (0.0), Slow Speed (0.0) -> Semantic (True)
        # Low SNR is critical failure
        self.assertTrue(decide_use_nn(0.0, 0.0, 0.0))

    def test_semantic_far_distance(self):
        # High SNR (1.0), Far Dist (1.0), Slow Speed (0.0) -> Semantic (True)
        # Far Dist is critical failure
        self.assertTrue(decide_use_nn(1.0, 1.0, 0.0))

    def test_semantic_fast_speed(self):
        # High SNR (1.0), Near Dist (0.0), Fast Speed (1.0) -> Semantic (True)
        # Fast Speed is critical failure
        self.assertTrue(decide_use_nn(1.0, 0.0, 1.0))

    def test_semantic_medium_conditions(self):
        # Medium SNR (0.5), Medium Dist (0.5), Medium Speed (0.5) -> Semantic (True)
        # Medium SNR + Medium Dist/Speed -> Semantic
        self.assertTrue(decide_use_nn(0.5, 0.5, 0.5))

    def test_direct_high_snr_medium_dist(self):
        # High SNR (1.0), Medium Dist (0.5), Slow Speed (0.0) -> Direct (False)
        # High SNR compensates for Medium Dist
        self.assertFalse(decide_use_nn(1.0, 0.5, 0.0))

    def test_boundary_transition(self):
        # 0.4 SNR (Med/Lowish), 0.6 Dist (Med/Farish), 0.2 Speed (Slow/Medish)
        # SNR 0.4: Low=0.2, Med=0.8, High=0
        # Dist 0.6: Near=0, Med=0.8, Far=0.2
        # Speed 0.2: Slow=0.6, Med=0.4, Fast=0
        
        # Semantic Support:
        # Base: max(LowSNR(0.2), FarDist(0.2), FastSpeed(0)) = 0.2
        # Interm: min(MedSNR(0.8), MedDist(0.8)) = 0.8
        # Max Semantic = 0.8
        
        # Direct Support:
        # Base: min(High(0), Near(0), Slow(0.6)) = 0
        # Interm1: min(High(0), MedDist(0.8), Slow(0.6)) = 0
        # Interm2: min(MedSNR(0.8), Near(0), Slow(0.6)) = 0
        # Max Direct = 0
        
        # Result: Semantic (True)
        self.assertTrue(decide_use_nn(0.4, 0.6, 0.2))

    def test_out_of_bounds(self):
        # Test input validation (should clamp and work)
        # 1.5 -> 1.0 (High/Bad/Bad)
        # -0.5 -> 0.0 (Low/Good/Good)
        
        # SNR 1.5 -> 1.0 (High)
        # Dist -0.5 -> 0.0 (Near)
        # Speed -0.5 -> 0.0 (Slow)
        # Result: Direct (False)
        self.assertFalse(decide_use_nn(1.5, -0.5, -0.5))

if __name__ == '__main__':
    unittest.main()
