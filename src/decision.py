# src/decision.py
import numpy as np

def classify_segments(segments, cfg):
    """
    Convert edge segments into hazards with type, severity, confidence, distance, etc.
    Uses sign of the height jump to pick Rise (ascending) vs Drop (descending).
    """
    hazards = []
    step_lo, step_hi = [v / 1000.0 for v in cfg["step_mm"]]      # meters
    curb_lo, curb_hi = [v / 1000.0 for v in cfg["curb_mm"]]
    uneven_lo, _     = [v / 1000.0 for v in cfg["uneven_mm"]]
    min_conf = float(cfg.get("min_confidence", 0.35))
    walk_v   = float(cfg.get("assumed_walk_speed_mps", 1.2))

    for s in segments:
        d  = float(s.get("delta_m", 0.0))
        sd = float(s.get("signed_delta_m", 0.0))
        dist = float(s.get("distance_m", np.nan))
        if not np.isfinite(dist) or dist <= 0:
            # conservative fallback if not provided
            dist = 0.8

        if d >= step_lo:
            sev = clamp01((d - step_lo) / max(1e-6, (step_hi - step_lo)))
            typ = "Rise" if sd > 0 else "Drop"
            conf = max(min_conf, 0.6 + 0.4 * sev)
        elif d >= curb_lo:
            sev = clamp01((d - curb_lo) / max(1e-6, (curb_hi - curb_lo)))
            typ = "Rise" if sd > 0 else "Drop"
            conf = max(min_conf, 0.5 + 0.4 * sev)
        elif d >= uneven_lo:
            sev = 0.35
            typ = "Rough"
            conf = max(min_conf, 0.35)
        else:
            continue

        hazards.append({
            "type": typ,
            "severity": float(sev if typ != "Rough" else 0.35),
            "distance_m": float(dist),
            "bearing_deg": float(s.get("bearing_deg", 0.0)),
            "confidence": float(conf),
            "ttc_s": float(dist / max(1e-6, walk_v)),
        })

    return hazards

def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1.0 else float(x))
