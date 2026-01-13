import math
from typing import Dict, Tuple

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    Returns membership in [0, 1] for x given parameters (a <= b <= c).
    """
    if x == b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0

class BaseMembership:
    """Base class for fuzzy membership functions with normalized inputs [0, 1]."""
    def __init__(self):
        # Normalized triangular parameters as requested
        self.low_params = (0.0, 0.0, 0.5)
        self.med_params = (0.0, 0.5, 1.0)
        self.high_params = (0.5, 1.0, 1.0)

    def _validate(self, x: float) -> float:
        """Ensure input is within [0, 1] range."""
        if not (0.0 <= x <= 1.0):
            # Clamp or raise error? User asked for input validation. 
            # We'll print a warning and clamp for robustness, or raise ValueError.
            # Given it's a simulation, clamping is safer to avoid crash, but explicit validation implies check.
            # I will clamp but warn if outside expected tolerance, or just clamp.
            # Let's clamp to be safe for float inaccuracies.
            return max(0.0, min(1.0, x))
        return x

    def get_memberships(self, x: float) -> Dict[str, float]:
        raise NotImplementedError

class SNRMembership(BaseMembership):
    """Membership function for Normalized SNR (0=Low/Bad, 1=High/Good)."""
    def get_memberships(self, x: float) -> Dict[str, float]:
        x = self._validate(x)
        return {
            "low": _trimf(x, *self.low_params),      # Low SNR (Bad)
            "medium": _trimf(x, *self.med_params),   # Medium SNR
            "high": _trimf(x, *self.high_params)     # High SNR (Good)
        }

class DistanceMembership(BaseMembership):
    """Membership function for Normalized Distance (0=Near/Good, 1=Far/Bad)."""
    def get_memberships(self, x: float) -> Dict[str, float]:
        x = self._validate(x)
        return {
            "near": _trimf(x, *self.low_params),     # Near (Good)
            "medium": _trimf(x, *self.med_params),   # Medium
            "far": _trimf(x, *self.high_params)      # Far (Bad)
        }

class SpeedMembership(BaseMembership):
    """Membership function for Normalized Speed (0=Slow/Good, 1=Fast/Bad)."""
    def get_memberships(self, x: float) -> Dict[str, float]:
        x = self._validate(x)
        return {
            "slow": _trimf(x, *self.low_params),     # Slow (Good)
            "medium": _trimf(x, *self.med_params),   # Medium
            "fast": _trimf(x, *self.high_params)     # Fast (Bad)
        }

def evaluate_rules(mu_snr: Dict[str, float],
                   mu_dist: Dict[str, float],
                   mu_speed: Dict[str, float]) -> Tuple[float, float]:
    """
    Evaluate fuzzy rules and return (semantic_score, direct_score).
    
    Rules:
    - Direct (Good channel):
        * High SNR AND Near Dist AND Slow Speed (Ideal)
        * High SNR AND (Medium Dist OR Medium Speed) (SNR compensates)
        * Medium SNR AND Near Dist AND Slow Speed (Conditions compensate)
        
    - Semantic (Poor channel):
        * Low SNR OR Far Dist OR Fast Speed (Critical failures)
        * Medium SNR AND (Medium/Far Dist OR Medium/Fast Speed) (Risky)
    """
    # 1. Base Scores from extreme conditions
    # Direct: Good conditions (High SNR, Near Dist, Slow Speed)
    direct = min(mu_snr.get("high", 0.0), mu_dist.get("near", 0.0), mu_speed.get("slow", 0.0))
    
    # Semantic: Bad conditions (Low SNR, Far Dist, Fast Speed)
    semantic = max(mu_snr.get("low", 0.0), mu_dist.get("far", 0.0), mu_speed.get("fast", 0.0))
    
    # 2. Add intermediate support for Direct (Good conditions compensating for medium ones)
    # High SNR allows for Medium Distance OR Medium Speed
    direct = max(direct, min(mu_snr.get("high", 0.0), mu_dist.get("medium", 0.0), mu_speed.get("slow", 0.0)))
    direct = max(direct, min(mu_snr.get("high", 0.0), mu_dist.get("near", 0.0), mu_speed.get("medium", 0.0)))
    # Medium SNR is okay if Distance is Near AND Speed is Slow
    direct = max(direct, min(mu_snr.get("medium", 0.0), mu_dist.get("near", 0.0), mu_speed.get("slow", 0.0)))

    # 3. Add intermediate support for Semantic (Medium conditions leaning to semantic)
    # If SNR is Medium, any other Medium condition pushes towards Semantic to be safe
    semantic = max(semantic, min(mu_snr.get("medium", 0.0), mu_dist.get("medium", 0.0)))
    semantic = max(semantic, min(mu_snr.get("medium", 0.0), mu_speed.get("medium", 0.0)))
    
    return float(semantic), float(direct)

def decide_use_nn(snr_norm: float, dist_norm: float, speed_norm: float) -> bool:
    """
    Decide transmission mode using fuzzy logic with normalized inputs.
    
    Inputs (Normalized [0, 1]):
    - snr_norm: 0 (Low/Bad) to 1 (High/Good)
    - dist_norm: 0 (Near/Good) to 1 (Far/Bad)
    - speed_norm: 0 (Slow/Good) to 1 (Fast/Bad)
    
    Output:
    - True for semantic (NN-based) transmission, False for direct transmission.
    """
    snr_mem = SNRMembership()
    dist_mem = DistanceMembership()
    speed_mem = SpeedMembership()
    
    mu_snr = snr_mem.get_memberships(snr_norm)
    mu_dist = dist_mem.get_memberships(dist_norm)
    mu_speed = speed_mem.get_memberships(speed_norm)
    
    semantic, direct = evaluate_rules(mu_snr, mu_dist, mu_speed)
    return semantic >= direct
