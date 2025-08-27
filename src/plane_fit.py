import numpy as np

def backproject(depth_m: np.ndarray, fx, fy, cx, cy):
    """
    Back-project per-pixel depth (meters) into camera-space 3D points.
    Returns [H,W,3] float32 array (x,y,z).
    """
    h, w = depth_m.shape
    ys, xs = np.indices((h, w))
    z = depth_m.astype(np.float32, copy=False)
    x = ((xs - cx) * z / fx).astype(np.float32, copy=False)
    y = ((ys - cy) * z / fy).astype(np.float32, copy=False)
    pts = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    return pts

def ransac_plane(points_xyz, valid_mask, iters=300, sample_n=3, thresh=0.02, rng=None):
    """
    Fit plane n·X + d = 0 with RANSAC.
    thresh in meters (point-to-plane distance).
    Returns (n: [3], d: float, inlier_mask: [H,W] bool)
    """
    if rng is None:
        rng = np.random.default_rng(123)

    pts = points_xyz[valid_mask]  # [N,3]
    N = pts.shape[0]
    if N < 1000:
        return None, None, np.zeros(points_xyz.shape[:2], dtype=bool)

    best_inliers = None
    best_n, best_d = None, None
    idxs = np.arange(N)

    for _ in range(iters):
        sample = pts[rng.choice(idxs, size=sample_n, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = (n / norm).astype(np.float32)
        d = -np.dot(n, sample[0].astype(np.float32))

        dist = np.abs(pts @ n + d)
        inliers = dist < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_n, best_d = n, float(d)

    if best_n is None:
        return None, None, np.zeros(points_xyz.shape[:2], dtype=bool)

    # build full-res inlier mask
    H, W = points_xyz.shape[:2]
    dist_full = np.abs(points_xyz.reshape(-1, 3) @ best_n + best_d).reshape(H, W)
    inlier_mask_full = (dist_full < thresh) & valid_mask
    return best_n.astype(np.float32), float(best_d), inlier_mask_full

def height_above_plane(points_xyz, n, d):
    """
    Signed distance to plane in meters for each pixel.
    Positive values are in +n direction.
    """
    h = (points_xyz @ n + d).astype(np.float32, copy=False)
    return h
