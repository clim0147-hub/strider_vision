from dataclasses import dataclass
import numpy as np

@dataclass
class DepthFrame:
    depth_m: np.ndarray    # HxW float32 depth in meters
    intrinsics: tuple      # (fx, fy, cx, cy)
    pose: np.ndarray       # 4x4 camera pose matrix

@dataclass
class Hazard:
    type: str              # "Rise", "Drop", "Rough"
    severity: float        # 0..1
    distance_m: float
    bearing_deg: float
    confidence: float
