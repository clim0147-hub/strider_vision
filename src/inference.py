# src/inference.py
import argparse, json, yaml, cv2, numpy as np, os

from io_types import DepthFrame, Hazard  # kept for future expansion
from plane_fit import backproject, ransac_plane, height_above_plane
from edges import sobel_edges, project_forward_mask, find_edge_segments
from decision import classify_segments

# ----------------- helpers -----------------
def percent_safe(a, q, mask=None, default=1.0):
    if mask is not None:
        a = a[mask]
    if a.size == 0 or not np.isfinite(a).any():
        return default
    return float(np.nanpercentile(a, q))

def valid_mask_from_depth(depth, near=0.25, far=8.0):
    return np.isfinite(depth) & (depth > near) & (depth < far)

def estimate_intrinsics(width, height, fx=None, fy=None, cx=None, cy=None, fov_deg=60.0):
    # approximate intrinsics from image size + FOV if not provided
    if fx is None or fy is None:
        f = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fx or f
        fy = fy or f
    cx = cx if cx is not None else (width  * 0.5)
    cy = cy if cy is not None else (height * 0.5)
    return float(fx), float(fy), float(cx), float(cy)

def load_depth(path, uint8_max_m=8.0):
    """
    Loads a depth PNG/JPG that may be:
      - uint16 millimeters
      - float32 meters
      - uint8 normalized [0..255] -> mapped to [0..uint8_max_m] meters
    Zeros are treated as invalid (set to NaN).
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)

    if d.dtype == np.uint8:
        d = d.astype(np.float32) / 255.0 * float(uint8_max_m)
    elif d.dtype == np.uint16:
        d = d.astype(np.float32) / 1000.0
    else:
        d = d.astype(np.float32)

    d[d <= 0.0] = np.nan
    return d

def try_predict_midas(rgb_path, model_type="DPT_Hybrid", device=None):
    try:
        import torch
        from midas_infer import load_midas, predict_depth
    except Exception as e:
        raise RuntimeError(
            "MiDaS not available. Install PyTorch + timm and add src/midas_infer.py.\n"
            "pip install torch torchvision timm\n"
            f"Import error: {e}"
        )
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = load_midas(device=device, model_type=model_type)
    depth = predict_depth(model, tfm, rgb_path, device=device)
    return depth

def autoscale_monocular_depth(depth, fx, fy, cx, cy, camera_height_m=1.4):
    """
    Primary: scale using ground-plane |d| ~= camera_height_m (RANSAC on relative depth).
    Fallback: if RANSAC fails, return original depth with scale=1.
    """
    valid_rel = valid_mask_from_depth(depth, near=0.05, far=20.0)
    if not valid_rel.any():
        return depth, 1.0, False

    pts_rel = backproject(depth, fx, fy, cx, cy)
    n0, d0, _ = ransac_plane(pts_rel, valid_rel, iters=600, thresh=0.03)
    if n0 is None or abs(d0) < 1e-6:
        return depth, 1.0, False

    k = float(camera_height_m) / float(abs(d0))
    depth_scaled = (depth * k).astype(np.float32, copy=False)
    return depth_scaled, k, True

def save_debug_image(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--depth", help="path to depth image (m/mm/uint8)")
    ap.add_argument("--rgb", help="path to RGB image (monocular depth will be estimated)")
    # Intrinsics
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--fov_deg", type=float, default=60.0, help="FOV if fx/fy not provided")
    # Ranges / scaling
    ap.add_argument("--camera_height_m", type=float, default=1.4, help="Assumed camera height for monocular autoscale")
    ap.add_argument("--near_m", type=float, default=0.25)
    ap.add_argument("--far_m",  type=float, default=8.0)
    ap.add_argument("--uint8_max_m", type=float, default=8.0, help="If depth PNG is uint8, map 0..255 -> 0..uint8_max_m meters")
    # Config + model + output
    ap.add_argument("--config", default="configs/thresholds.yaml")
    ap.add_argument("--out_vis", default="out/vis.png")
    ap.add_argument("--midas_type", default="DPT_Hybrid", help="DPT_Hybrid, DPT_Large, MiDaS_small")
    # Visualization / debug
    ap.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    # Post-processing
    ap.add_argument("--conf_floor", type=float, default=0.40, help="drop hazards with confidence below this")
    ap.add_argument("--nms_min_sep_m", type=float, default=0.30, help="NMS min separation (m) between hazards by distance")
    ap.add_argument("--nms_max_keep", type=int, default=2, help="NMS keep only this many nearest hazards")
    # RGB-edge fusion
    ap.add_argument("--fuse_rgb_edges", action="store_true", help="Fuse Canny edges from RGB to help on smooth depth")
    # View filtering
    ap.add_argument("--view_mode", default="auto",
                    choices=["auto", "ascending", "descending"],
                    help="Filter hazards to match the view. auto=choose dominant sign; ascending=keep only Rise; descending=keep only Drop")
    ap.add_argument("--view_margin", type=float, default=0.0,
                    help="Optional severity margin when allowing Rough with chosen view")
    args = ap.parse_args()

    if not args.depth and not args.rgb:
        raise SystemExit("Provide --depth OR --rgb")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1) get depth (meters)
    used_midas = False
    if args.depth:
        depth = load_depth(args.depth, uint8_max_m=args.uint8_max_m)
    else:
        depth = try_predict_midas(args.rgb, model_type=args.midas_type)
        used_midas = True

    H, W = depth.shape
    fx, fy, cx, cy = estimate_intrinsics(
        W, H, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, fov_deg=args.fov_deg
    )

    # 2) if monocular, autoscale so plane height ≈ camera_height_m
    scale_mode = "none"
    if used_midas:
        depth, k, ok = autoscale_monocular_depth(depth, fx, fy, cx, cy, camera_height_m=args.camera_height_m)
        scale_mode = "plane" if ok else "none"

    # 3) valid mask
    valid = valid_mask_from_depth(depth, near=args.near_m, far=args.far_m)

    # Debug: save depth visualization
    if args.debug:
        v95 = percent_safe(depth, 95, mask=valid, default=1.0)
        vis_depth = (np.clip(np.nan_to_num(depth, nan=v95*0.5) / max(1e-6, v95), 0, 1) * 255).astype(np.uint8)
        vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_TURBO)
        save_debug_image("out/debug_depth.png", vis_depth)

    if not valid.any():
        print("[]")
        os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
        cv2.imwrite(args.out_vis, np.zeros((H, W, 3), np.uint8))
        return

    # 4) back-project & plane fit
    pts = backproject(depth, fx, fy, cx, cy)
    n, d, inliers = ransac_plane(pts, valid, iters=700, thresh=0.03)
    if n is None:
        print("[]")
        os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
        v95 = percent_safe(depth, 95, mask=valid, default=1.0)
        vis = (np.clip(np.nan_to_num(depth, nan=v95*0.5) / max(1e-6, v95), 0, 1) * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(args.out_vis, vis)
        return

    # 5) height map
    height = height_above_plane(pts, n, d)
    height[~valid] = np.nan
    height = height.astype(np.float32, copy=False)
    height = np.ascontiguousarray(height)

    if args.debug:
        hq = np.nan_to_num(height, nan=0.0)
        h_vis = np.clip((hq + 0.2) / 0.4, 0, 1)  # visualize ±0.2m
        h_vis = (h_vis * 255).astype(np.uint8)
        h_vis = cv2.applyColorMap(h_vis, cv2.COLORMAP_VIRIDIS)
        save_debug_image("out/debug_height.png", h_vis)

    # 6) edges & optional RGB fusion
    mag = sobel_edges(height)
    if args.fuse_rgb_edges and args.rgb:
        from edges import rgb_edges_from_file  # helper defined in edges.py
        e_rgb = rgb_edges_from_file(args.rgb)
        if e_rgb is not None and e_rgb.shape == mag.shape:
            m = (mag / (mag.max() + 1e-6)).astype(np.float32)
            mag = (0.7 * m + 0.3 * e_rgb).astype(np.float32)

    if args.debug:
        m_vis = (np.clip(mag / (mag.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
        save_debug_image("out/debug_edges.png", m_vis)

    # 7) forward cone bearings
    cone_mask, bearings_deg = project_forward_mask(cfg["max_bearing_deg"], fx, cx, W)

    # 8) segments → hazards (signed delta inside segments)
    segments = find_edge_segments(
        height, depth, mag, bearings_deg, cone_mask,
        min_delta_m=min(cfg["uneven_mm"]) / 1000.0
    )

    # --- infer view mode from signed deltas (if auto) ---
    view_mode = args.view_mode
    if view_mode == "auto":
        signed = [s.get("signed_delta_m", 0.0) for s in segments if np.isfinite(s.get("signed_delta_m", np.nan))]
        if len(signed) >= 3:
            med = float(np.median(signed))
            if med > 0.0:
                view_mode = "ascending"
            elif med < 0.0:
                view_mode = "descending"
            else:
                view_mode = "auto"  # inconclusive; keep both
        else:
            view_mode = "auto"

    hazards = classify_segments(segments, cfg)

    # --- enforce the chosen view on hazards (keep one type only) ---
    if view_mode in ("ascending", "descending"):
        keep_type = "Rise" if view_mode == "ascending" else "Drop"
        margin = float(args.view_margin)
        hazards = [
            h for h in hazards
            if h["type"] == keep_type
               or (h["type"] == "Rough" and h["severity"] >= (0.35 + margin))
        ]

    # 9) Post-filters: confidence + NMS (by distance)
    hazards = [h for h in hazards if h["confidence"] >= args.conf_floor]

    filtered = []
    for h in hazards:
        if h["distance_m"] < 0.55 and h["confidence"] < 0.7:
            continue
        filtered.append(h)
    hazards = filtered

    def nms_by_distance(hzs, min_sep_m=0.30, max_keep=2):
        kept = []
        for h in sorted(hzs, key=lambda x: x["distance_m"]):
            if all(abs(h["distance_m"] - k["distance_m"]) >= min_sep_m for k in kept):
                kept.append(h)
            if len(kept) >= max_keep:
                break
        return kept

    hazards = nms_by_distance(hazards, min_sep_m=args.nms_min_sep_m, max_keep=args.nms_max_keep)

    # 10) output JSON
    print(json.dumps(hazards, indent=2))

    # 11) visualization
    v95 = percent_safe(depth, 95, mask=valid, default=1.0)
    depth_vis = np.nan_to_num(depth, nan=v95 * 0.5)
    scale = v95 if v95 > 1e-6 else float(depth_vis.max() if depth_vis.max() > 1e-6 else 1.0)
    vis = (np.clip(depth_vis / scale, 0, 1) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if inliers is not None:
        gmask = (inliers & valid).astype(np.uint8) * 80
        vis[:, :, 1] = np.maximum(vis[:, :, 1], gmask)

    for h in hazards:
        bearing = h["bearing_deg"]
        col = int(round(cx + np.tan(np.radians(bearing)) * fx))
        col = max(0, min(W - 1, col))
        color = (0, 0, 255) if h["type"] == "Rise" else (255, 0, 0) if h["type"] == "Drop" else (0, 165, 255)
        cv2.line(vis, (col, 0), (col, H - 1), color, 1)
        txt = f'{h["type"]} {h["distance_m"]:.1f}m c{h["confidence"]:.2f}'
        cv2.putText(vis, txt, (min(col + 5, W - 160), int(H * 0.60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
    cv2.imwrite(args.out_vis, vis)

if __name__ == "__main__":
    main()
# src/inference.py
import argparse, json, yaml, cv2, numpy as np, os

from io_types import DepthFrame, Hazard  # kept for future expansion
from plane_fit import backproject, ransac_plane, height_above_plane
from edges import sobel_edges, project_forward_mask, find_edge_segments
from decision import classify_segments

# ----------------- helpers -----------------
def percent_safe(a, q, mask=None, default=1.0):
    if mask is not None:
        a = a[mask]
    if a.size == 0 or not np.isfinite(a).any():
        return default
    return float(np.nanpercentile(a, q))

def valid_mask_from_depth(depth, near=0.25, far=8.0):
    return np.isfinite(depth) & (depth > near) & (depth < far)

def estimate_intrinsics(width, height, fx=None, fy=None, cx=None, cy=None, fov_deg=60.0):
    # approximate intrinsics from image size + FOV if not provided
    if fx is None or fy is None:
        f = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
        fx = fx or f
        fy = fy or f
    cx = cx if cx is not None else (width  * 0.5)
    cy = cy if cy is not None else (height * 0.5)
    return float(fx), float(fy), float(cx), float(cy)

def load_depth(path, uint8_max_m=8.0):
    """
    Loads a depth PNG/JPG that may be:
      - uint16 millimeters
      - float32 meters
      - uint8 normalized [0..255] -> mapped to [0..uint8_max_m] meters
    Zeros are treated as invalid (set to NaN).
    """
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)

    if d.dtype == np.uint8:
        d = d.astype(np.float32) / 255.0 * float(uint8_max_m)
    elif d.dtype == np.uint16:
        d = d.astype(np.float32) / 1000.0
    else:
        d = d.astype(np.float32)

    d[d <= 0.0] = np.nan
    return d

def try_predict_midas(rgb_path, model_type="DPT_Hybrid", device=None):
    try:
        import torch
        from midas_infer import load_midas, predict_depth
    except Exception as e:
        raise RuntimeError(
            "MiDaS not available. Install PyTorch + timm and add src/midas_infer.py.\n"
            "pip install torch torchvision timm\n"
            f"Import error: {e}"
        )
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = load_midas(device=device, model_type=model_type)
    depth = predict_depth(model, tfm, rgb_path, device=device)
    return depth

def autoscale_monocular_depth(depth, fx, fy, cx, cy, camera_height_m=1.4):
    """
    Primary: scale using ground-plane |d| ~= camera_height_m (RANSAC on relative depth).
    Fallback: if RANSAC fails, return original depth with scale=1.
    """
    valid_rel = valid_mask_from_depth(depth, near=0.05, far=20.0)
    if not valid_rel.any():
        return depth, 1.0, False

    pts_rel = backproject(depth, fx, fy, cx, cy)
    n0, d0, _ = ransac_plane(pts_rel, valid_rel, iters=600, thresh=0.03)
    if n0 is None or abs(d0) < 1e-6:
        return depth, 1.0, False

    k = float(camera_height_m) / float(abs(d0))
    depth_scaled = (depth * k).astype(np.float32, copy=False)
    return depth_scaled, k, True

def save_debug_image(path, img):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # Inputs
    ap.add_argument("--depth", help="path to depth image (m/mm/uint8)")
    ap.add_argument("--rgb", help="path to RGB image (monocular depth will be estimated)")
    # Intrinsics
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--fov_deg", type=float, default=60.0, help="FOV if fx/fy not provided")
    # Ranges / scaling
    ap.add_argument("--camera_height_m", type=float, default=1.4, help="Assumed camera height for monocular autoscale")
    ap.add_argument("--near_m", type=float, default=0.25)
    ap.add_argument("--far_m",  type=float, default=8.0)
    ap.add_argument("--uint8_max_m", type=float, default=8.0, help="If depth PNG is uint8, map 0..255 -> 0..uint8_max_m meters")
    # Config + model + output
    ap.add_argument("--config", default="configs/thresholds.yaml")
    ap.add_argument("--out_vis", default="out/vis.png")
    ap.add_argument("--midas_type", default="DPT_Hybrid", help="DPT_Hybrid, DPT_Large, MiDaS_small")
    # Visualization / debug
    ap.add_argument("--debug", action="store_true", help="Save intermediate debug images")
    # Post-processing
    ap.add_argument("--conf_floor", type=float, default=0.40, help="drop hazards with confidence below this")
    ap.add_argument("--nms_min_sep_m", type=float, default=0.30, help="NMS min separation (m) between hazards by distance")
    ap.add_argument("--nms_max_keep", type=int, default=2, help="NMS keep only this many nearest hazards")
    # RGB-edge fusion
    ap.add_argument("--fuse_rgb_edges", action="store_true", help="Fuse Canny edges from RGB to help on smooth depth")
    # View filtering
    ap.add_argument("--view_mode", default="auto",
                    choices=["auto", "ascending", "descending"],
                    help="Filter hazards to match the view. auto=choose dominant sign; ascending=keep only Rise; descending=keep only Drop")
    ap.add_argument("--view_margin", type=float, default=0.0,
                    help="Optional severity margin when allowing Rough with chosen view")
    args = ap.parse_args()

    if not args.depth and not args.rgb:
        raise SystemExit("Provide --depth OR --rgb")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 1) get depth (meters)
    used_midas = False
    if args.depth:
        depth = load_depth(args.depth, uint8_max_m=args.uint8_max_m)
    else:
        depth = try_predict_midas(args.rgb, model_type=args.midas_type)
        used_midas = True

    H, W = depth.shape
    fx, fy, cx, cy = estimate_intrinsics(
        W, H, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, fov_deg=args.fov_deg
    )

    # 2) if monocular, autoscale so plane height ≈ camera_height_m
    scale_mode = "none"
    if used_midas:
        depth, k, ok = autoscale_monocular_depth(depth, fx, fy, cx, cy, camera_height_m=args.camera_height_m)
        scale_mode = "plane" if ok else "none"

    # 3) valid mask
    valid = valid_mask_from_depth(depth, near=args.near_m, far=args.far_m)

    # Debug: save depth visualization
    if args.debug:
        v95 = percent_safe(depth, 95, mask=valid, default=1.0)
        vis_depth = (np.clip(np.nan_to_num(depth, nan=v95*0.5) / max(1e-6, v95), 0, 1) * 255).astype(np.uint8)
        vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_TURBO)
        save_debug_image("out/debug_depth.png", vis_depth)

    if not valid.any():
        print("[]")
        os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
        cv2.imwrite(args.out_vis, np.zeros((H, W, 3), np.uint8))
        return

    # 4) back-project & plane fit
    pts = backproject(depth, fx, fy, cx, cy)
    n, d, inliers = ransac_plane(pts, valid, iters=700, thresh=0.03)
    if n is None:
        print("[]")
        os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
        v95 = percent_safe(depth, 95, mask=valid, default=1.0)
        vis = (np.clip(np.nan_to_num(depth, nan=v95*0.5) / max(1e-6, v95), 0, 1) * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(args.out_vis, vis)
        return

    # 5) height map
    height = height_above_plane(pts, n, d)
    height[~valid] = np.nan
    height = height.astype(np.float32, copy=False)
    height = np.ascontiguousarray(height)

    if args.debug:
        hq = np.nan_to_num(height, nan=0.0)
        h_vis = np.clip((hq + 0.2) / 0.4, 0, 1)  # visualize ±0.2m
        h_vis = (h_vis * 255).astype(np.uint8)
        h_vis = cv2.applyColorMap(h_vis, cv2.COLORMAP_VIRIDIS)
        save_debug_image("out/debug_height.png", h_vis)

    # 6) edges & optional RGB fusion
    mag = sobel_edges(height)
    if args.fuse_rgb_edges and args.rgb:
        from edges import rgb_edges_from_file  # helper defined in edges.py
        e_rgb = rgb_edges_from_file(args.rgb)
        if e_rgb is not None and e_rgb.shape == mag.shape:
            m = (mag / (mag.max() + 1e-6)).astype(np.float32)
            mag = (0.7 * m + 0.3 * e_rgb).astype(np.float32)

    if args.debug:
        m_vis = (np.clip(mag / (mag.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
        save_debug_image("out/debug_edges.png", m_vis)

    # 7) forward cone bearings
    cone_mask, bearings_deg = project_forward_mask(cfg["max_bearing_deg"], fx, cx, W)

    # 8) segments → hazards (signed delta inside segments)
    segments = find_edge_segments(
        height, depth, mag, bearings_deg, cone_mask,
        min_delta_m=min(cfg["uneven_mm"]) / 1000.0
    )

    # --- infer view mode from signed deltas (if auto) ---
    view_mode = args.view_mode
    if view_mode == "auto":
        signed = [s.get("signed_delta_m", 0.0) for s in segments if np.isfinite(s.get("signed_delta_m", np.nan))]
        if len(signed) >= 3:
            med = float(np.median(signed))
            if med > 0.0:
                view_mode = "ascending"
            elif med < 0.0:
                view_mode = "descending"
            else:
                view_mode = "auto"  # inconclusive; keep both
        else:
            view_mode = "auto"

    hazards = classify_segments(segments, cfg)

    # --- enforce the chosen view on hazards (keep one type only) ---
    if view_mode in ("ascending", "descending"):
        keep_type = "Rise" if view_mode == "ascending" else "Drop"
        margin = float(args.view_margin)
        hazards = [
            h for h in hazards
            if h["type"] == keep_type
               or (h["type"] == "Rough" and h["severity"] >= (0.35 + margin))
        ]

    # 9) Post-filters: confidence + NMS (by distance)
    hazards = [h for h in hazards if h["confidence"] >= args.conf_floor]

    filtered = []
    for h in hazards:
        if h["distance_m"] < 0.55 and h["confidence"] < 0.7:
            continue
        filtered.append(h)
    hazards = filtered

    def nms_by_distance(hzs, min_sep_m=0.30, max_keep=2):
        kept = []
        for h in sorted(hzs, key=lambda x: x["distance_m"]):
            if all(abs(h["distance_m"] - k["distance_m"]) >= min_sep_m for k in kept):
                kept.append(h)
            if len(kept) >= max_keep:
                break
        return kept

    hazards = nms_by_distance(hazards, min_sep_m=args.nms_min_sep_m, max_keep=args.nms_max_keep)

    # 10) output JSON
    print(json.dumps(hazards, indent=2))

    # 11) visualization
    v95 = percent_safe(depth, 95, mask=valid, default=1.0)
    depth_vis = np.nan_to_num(depth, nan=v95 * 0.5)
    scale = v95 if v95 > 1e-6 else float(depth_vis.max() if depth_vis.max() > 1e-6 else 1.0)
    vis = (np.clip(depth_vis / scale, 0, 1) * 255).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if inliers is not None:
        gmask = (inliers & valid).astype(np.uint8) * 80
        vis[:, :, 1] = np.maximum(vis[:, :, 1], gmask)

    for h in hazards:
        bearing = h["bearing_deg"]
        col = int(round(cx + np.tan(np.radians(bearing)) * fx))
        col = max(0, min(W - 1, col))
        color = (0, 0, 255) if h["type"] == "Rise" else (255, 0, 0) if h["type"] == "Drop" else (0, 165, 255)
        cv2.line(vis, (col, 0), (col, H - 1), color, 1)
        txt = f'{h["type"]} {h["distance_m"]:.1f}m c{h["confidence"]:.2f}'
        cv2.putText(vis, txt, (min(col + 5, W - 160), int(H * 0.60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(os.path.dirname(args.out_vis) or ".", exist_ok=True)
    cv2.imwrite(args.out_vis, vis)

if __name__ == "__main__":
    main()
