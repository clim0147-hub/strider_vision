import numpy as np
from io_types import DepthFrame, Hazard

def run_inference():
    # Placeholder: load depth frame (here we just mock a 480x640 with random values)
    depth = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
    intrinsics = (585, 585, 320, 240)
    pose = np.eye(4)

    frame = DepthFrame(depth, intrinsics, pose)

    # Fake hazard for testing
    hazard = Hazard(type="Rise", severity=0.7, distance_m=1.2,
                    bearing_deg=-5.0, confidence=0.8)

    print("Detected hazards:", [hazard])

if __name__ == "__main__":
    run_inference()


