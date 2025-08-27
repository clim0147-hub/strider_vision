# src/edges.py
import numpy as np
import cv2

def sobel_edges(h, ksize=3):
    """Sobel magnitude on height map; NaNs -> 0 to keep it robust."""
    h = np.nan_to_num(h, nan=0.0).astype(np.float32)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag

def project_forward_mask(max_bearing_deg, fx, cx, W):
    """
    Compute per-column bearing (deg) and a mask that keeps only columns
    within +/- max_bearing_deg of the optical axis.
    """
    cols = np.arange(W, dtype=np.float32)
    bearings = np.degrees(np.arctan((cols - cx) / fx))
    mask = (np.abs(bearings) <= float(max_bearing_deg))
    return mask, bearings

def rgb_edges_from_file(rgb_path, blur=3, t1=50, t2=150):
    """Optional RGB edge map to fuse with depth edges when depth is too smooth."""
    g = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if blur and blur > 0:
        g = cv2.GaussianBlur(g, (blur, blur), 0)
    e = cv2.Canny(g, t1, t2)
    return (e.astype(np.float32) / 255.0)

def _pick_peak_rows(col_mag, rel_thr=0.25, min_gap=3):
    """
    Simple 1D peak picking in a column: threshold by relativity to max, then
    coalesce close indices into a single representative row (mean).
    """
    if not np.isfinite(col_mag).any():
        return []
    m = float(np.nanmax(col_mag))
    if m <= 0.0:
        return []
    thr = rel_thr * m
    peaks = np.where(col_mag > thr)[0]
    if peaks.size == 0:
        return []
    groups, cur = [], [int(peaks[0])]
    for r in peaks[1:]:
        if int(r) - cur[-1] <= min_gap:
            cur.append(int(r))
        else:
            groups.append(int(np.mean(cur)))
            cur = [int(r)]
    groups.append(int(np.mean(cur)))
    return groups

def find_edge_segments(height,
                       depth,
                       mag,
                       bearings_deg,
                       cone_mask,
                       min_delta_m=0.03,
                       sample_px=4,
                       peak_rel=0.25,
                       row_skip_top=10,
                       row_skip_bottom=10):
    """
    Scan each forward-facing column; detect step edges in height map.
    For each edge row r:
      near = r + sample_px (toward bottom; closer to camera)
      far  = r - sample_px (toward top; farther from camera)
      signed_delta_m = h_far - h_near
        > 0  => surface ahead is higher  (Rise / ascending)
        < 0  => surface ahead is lower  (Drop / descending)
    Returns a list of dicts with keys:
      row, col, bearing_deg, delta_m, signed_delta_m, distance_m
    """
    H, W = height.shape
    segs = []

    # Bounds to avoid the extreme top/bottom
    r_lo = int(max(0, row_skip_top))
    r_hi = int(min(H, H - row_skip_bottom))

    for c in range(W):
        if not cone_mask[c]:
            continue

        col_mag = mag[:, c].copy()
        # zero out top/bottom strips
        if r_lo > 0:
            col_mag[:r_lo] = 0
        if r_hi < H:
            col_mag[r_hi:] = 0

        peak_rows = _pick_peak_rows(col_mag, rel_thr=peak_rel, min_gap=3)
        if not peak_rows:
            continue

        for r in peak_rows:
            r_near = min(H - 1, r + sample_px)  # towards bottom (closer)
            r_far  = max(0,       r - sample_px)  # towards top (farther)

            h_near = height[r_near, c]
            h_far  = height[r_far,  c]
            if not (np.isfinite(h_near) and np.isfinite(h_far)):
                continue

            signed_delta = float(h_far - h_near)
            delta = abs(signed_delta)
            if delta < float(min_delta_m):
                continue

            # Distance estimate: use depth at the near sample if finite, else center
            d_est = np.nan
            if depth is not None and depth.shape == height.shape:
                cand = [depth[idx, c] for idx in (r, r_near, r_far) if 0 <= idx < H]
                cand = [float(x) for x in cand if np.isfinite(x) and x > 0]
                if cand:
                    d_est = min(cand)  # prefer closest valid reading
            if not np.isfinite(d_est):
                # rough fallback mapping row->distance
                y_norm = 1.0 - (r / max(1, H - 1))
                d_est = 0.3 + 1.7 * y_norm

            segs.append({
                "row": int(r),
                "col": int(c),
                "bearing_deg": float(bearings_deg[c]),
                "delta_m": float(delta),
                "signed_delta_m": float(signed_delta),
                "distance_m": float(d_est),
            })

    return segs
