import math
from typing import Dict, Tuple

def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.
    Returns membership in [0, 1] for x given parameters (a <= b <= c <= d).
    """
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.
    Returns membership in [0, 1] for x given parameters (a <= b <= c).
    """
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0

def membership_snr_db(snr_db: float,
                      low_params: Tuple[float, float, float, float] = (0.0, 0.0, 15.0, 25.0),
                      med_params: Tuple[float, float, float] = (18.0, 28.0, 38.0),
                      high_params: Tuple[float, float, float, float] = (30.0, 40.0, 60.0, 60.0)) -> Dict[str, float]:
    """
    Memberships for SNR in dB: low, medium, high.
    Parameter tuples define trapezoids/triangle and can be tuned without code changes.
    """
    low = _trapmf(snr_db, *low_params)
    med = _trimf(snr_db, *med_params)
    high = _trapmf(snr_db, *high_params)
    return {"low": low, "medium": med, "high": high}

def membership_distance_m(dist_m: float,
                          near_params: Tuple[float, float, float, float] = (0.0, 0.0, 20.0, 40.0),
                          med_params: Tuple[float, float, float] = (30.0, 50.0, 70.0),
                          far_params: Tuple[float, float, float, float] = (60.0, 80.0, 200.0, 200.0)) -> Dict[str, float]:
    """
    Memberships for distance in meters: near, medium, far.
    Uses overlapping trapezoids and triangles to provide smooth transitions.
    """
    near = _trapmf(dist_m, *near_params)
    med = _trimf(dist_m, *med_params)
    far = _trapmf(dist_m, *far_params)
    return {"near": near, "medium": med, "far": far}

def membership_rel_speed_ms(rel_speed_ms: float,
                            slow_params: Tuple[float, float, float, float] = (0.0, 0.0, 5.0, 10.0),
                            med_params: Tuple[float, float, float] = (5.0, 12.5, 20.0),
                            fast_params: Tuple[float, float, float, float] = (15.0, 25.0, 50.0, 50.0)) -> Dict[str, float]:
    """
    Memberships for relative speed in m/s: slow, medium, fast.
    """
    slow = _trapmf(rel_speed_ms, *slow_params)
    med = _trimf(rel_speed_ms, *med_params)
    fast = _trapmf(rel_speed_ms, *fast_params)
    return {"slow": slow, "medium": med, "fast": fast}

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
    direct = min(mu_snr.get("high", 0.0), mu_dist.get("near", 0.0), mu_speed.get("slow", 0.0))
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

def decide_use_nn(snr_trad_db: float, distance_m: float, rel_speed_ms: float) -> bool:
    """
    Decide transmission mode using fuzzy logic.
    Inputs:
    - snr_trad_db: traditional SNR in dB
    - distance_m: distance in meters
    - rel_speed_ms: relative speed in m/s
    Output:
    - True for semantic (NN-based) transmission, False for direct transmission.
    """
    mu_snr = membership_snr_db(snr_trad_db)
    mu_dist = membership_distance_m(distance_m)
    mu_speed = membership_rel_speed_ms(rel_speed_ms)
    semantic, direct = evaluate_rules(mu_snr, mu_dist, mu_speed)
    return semantic >= direct

