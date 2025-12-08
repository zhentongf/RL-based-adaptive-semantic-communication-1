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
                          near_params: Tuple[float, float, float, float] = (0.0, 0.0, 20.0, 50.0),
                          far_params: Tuple[float, float, float, float] = (30.0, 60.0, 200.0, 200.0)) -> Dict[str, float]:
    """
    Memberships for distance in meters: near, far.
    Uses overlapping trapezoids to provide smooth transitions.
    """
    near = _trapmf(dist_m, *near_params)
    far = _trapmf(dist_m, *far_params)
    return {"near": near, "far": far}

def membership_rel_speed_ms(rel_speed_ms: float,
                            slow_params: Tuple[float, float, float, float] = (0.0, 0.0, 5.0, 15.0),
                            fast_params: Tuple[float, float, float, float] = (10.0, 20.0, 50.0, 50.0)) -> Dict[str, float]:
    """
    Memberships for relative speed in m/s: slow, fast.
    """
    slow = _trapmf(rel_speed_ms, *slow_params)
    fast = _trapmf(rel_speed_ms, *fast_params)
    return {"slow": slow, "fast": fast}

def evaluate_rules(mu_snr: Dict[str, float],
                   mu_dist: Dict[str, float],
                   mu_speed: Dict[str, float]) -> Tuple[float, float]:
    """
    Evaluate fuzzy rules and return (semantic_score, direct_score).
    Rules:
    - Direct when SNR high AND distance near AND speed slow.
    - Semantic when SNR low OR distance far OR speed fast.
    Smooth transitions by adding medium SNR variants to both sides.
    """
    direct = min(mu_snr.get("high", 0.0), mu_dist.get("near", 0.0), mu_speed.get("slow", 0.0))
    semantic = max(mu_snr.get("low", 0.0), mu_dist.get("far", 0.0), mu_speed.get("fast", 0.0))
    direct = max(direct, min(mu_snr.get("medium", 0.0), mu_dist.get("near", 0.0), mu_speed.get("slow", 0.0)))
    semantic = max(semantic, min(mu_snr.get("medium", 0.0), max(mu_dist.get("far", 0.0), mu_speed.get("fast", 0.0))))
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

