# src/fusion_depth_yolo_stairs.py
import os, sys, json
import cv2, torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ----------------- Config -----------------
HAZARD_LABELS = {
    0: "Step Down / Curb Down",
    1: "Flat Surface (safe)",
    2: "Step Up / Curb Up",
    3: "Rough / Uneven Surface",   # not used yet
    4: "Obstacle / Barrier",
}

# COCO classes we consider "obstacles" (expand as desired)
COCO_OBSTACLE_IDS = {
    0, 1,2,3,5,7, 9, 11,12,13,14,15,16,17,18,19,20,21,
    24,25,26,27,28,31,32,33,34,36,41,44,46,47,49,50,52,
    56,57,58,60,62,63,64,67,73,75,76
}

# >>> Set this to your fine-tuned YOLOv8 stairs/curb model if you have one:
STAIR_CURB_WEIGHTS = None
# e.g.:
# STAIR_CURB_WEIGHTS = r"C:\Users\User\Desktop\strider_vision\assets\stairs_curbs_yolov8n.pt"

# Frame processing
FRAME_SKIP = 5
SCORE_THRESH_OBS = 0.35
SCORE_THRESH_STAIRS = 0.35
MAX_DETS_PER_FRAME = 20

# Depth scaling (MiDaS inverse depth → meters)
NEAR_M, FAR_M = 0.3, 3.0

# Heuristic curb detector params (used if no YOLO stairs weights)
HEUR_EDGE_LOW, HEUR_EDGE_HIGH = 60, 160      # Canny
HEUR_HOUGH_THRESH = 60                       # Hough line votes
HEUR_MIN_LINE_LEN = 60                       # pixels
HEUR_MAX_LINE_GAP = 8
HEUR_MAX_TILT_DEG = 12                       # keep near-horizontal lines
HEUR_SAMPLE_BAND = 6                         # px band above/below the line for depth sampling
HEUR_MIN_DEPTH_STEP_M = 0.15                 # classify only if depth delta >= 15 cm
# ------------------------------------------


# -------- MiDaS depth (small) -------------
def load_midas(device="cuda"):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")   # downloads on 1st run
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    model.to(device).eval()
    return model, transform

def depth_map_to_meters(inv_depth: np.ndarray, near_m=NEAR_M, far_m=FAR_M) -> np.ndarray:
    d = inv_depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)  # 0..1, 1=closest
    meters = far_m - d * (far_m - near_m)           # 1->near, 0->far
    return np.clip(meters, near_m, far_m)

def depth_for_box(meters: np.ndarray, box, quantile=0.30):
    x1,y1,x2,y2 = [int(v) for v in box]
    h, w = meters.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return float(FAR_M)
    patch = meters[y1:y2, x1:x2]
    if patch.size == 0:
        return float(FAR_M)
    return float(np.quantile(patch, quantile))

def normalized_offset(frame_w, cx):
    return float(np.clip((cx - frame_w/2) / (frame_w/2), -1.0, 1.0))

def surface_quality_proxy(gray, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    h, w = gray.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return 1.0
    roi = gray[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 50, 150)
    edge_frac = edges.mean() / 255.0
    return float(np.clip(1.0 - edge_frac, 0.0, 1.0))

def severity_from(distance_m, base_score):
    near_term = max(0.0, (FAR_M - distance_m) / (FAR_M - NEAR_M))
    return float(np.clip(0.6*near_term + 0.4*base_score, 0.0, 1.0))
# ------------------------------------------


# -------- YOLO helpers --------------------
def load_yolo_models():
    yolo_obs = YOLO("yolov8n.pt")  # COCO obstacles
    yolo_st = YOLO(STAIR_CURB_WEIGHTS) if (STAIR_CURB_WEIGHTS and os.path.exists(STAIR_CURB_WEIGHTS)) else None
    return yolo_obs, yolo_st

def detect_obstacles(yolo_obs, frame_bgr, W, H, gray, depth_m, score_thresh=SCORE_THRESH_OBS):
    out = []
    r = yolo_obs.predict(source=frame_bgr, verbose=False)[0]
    if r.boxes is None:
        return out
    for b in r.boxes:
        cls = int(b.cls.item())
        if cls not in COCO_OBSTACLE_IDS:
            continue
        score = float(b.conf.item())
        if score < score_thresh:
            continue
        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        dist = depth_for_box(depth_m, (x1,y1,x2,y2))
        cx = (x1+x2)/2.0
        off = normalized_offset(W, cx)
        surf_q = surface_quality_proxy(gray, (x1,y1,x2,y2))
        sev = severity_from(dist, score)

        out.append({
            "score": score,
            "hazard_type": 4,
            "hazard_label": HAZARD_LABELS[4],
            "distance_m": dist,
            "offset": off,
            "severity": sev,
            "surface_quality": surf_q,
            "box": [x1,y1,x2,y2],
        })
    return out

def detect_stairs_curbs_yolo(yolo_st, frame_bgr, W, H, gray, depth_m, score_thresh=SCORE_THRESH_STAIRS):
    """
    EXPECTED class mapping in your fine-tuned model (edit if different):
      0 -> Step Down / Curb Down  (hazard_type 0)
      1 -> Flat / Other (optional) (ignored or map to 4)
      2 -> Step Up / Curb Up      (hazard_type 2)
    """
    out = []
    r = yolo_st.predict(source=frame_bgr, verbose=False)[0]
    if r.boxes is None:
        return out
    for b in r.boxes:
        score = float(b.conf.item())
        if score < score_thresh:
            continue
        cls = int(b.cls.item())
        if cls == 0: htype = 0
        elif cls == 2: htype = 2
        else:
            # you can map '1' to obstacle or ignore
            htype = 4

        x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        dist = depth_for_box(depth_m, (x1,y1,x2,y2))
        cx = (x1+x2)/2.0
        off = normalized_offset(W, cx)
        surf_q = surface_quality_proxy(gray, (x1,y1,x2,y2))
        sev = severity_from(dist, score)

        out.append({
            "score": score,
            "hazard_type": htype,
            "hazard_label": HAZARD_LABELS[htype],
            "distance_m": dist,
            "offset": off,
            "severity": sev,
            "surface_quality": surf_q,
            "box": [x1,y1,x2,y2],
        })
    return out
# ------------------------------------------


# -------- Heuristic stairs/curb (fallback) --------
def detect_stairs_curbs_heuristic(frame_bgr, depth_m):
    """
    Find near-horizontal edges with a clear depth step across them.
    Classify:
      - Step Down (0): depth BELOW the line is GREATER (farther) than above (drop away)
      - Step Up   (2): depth BELOW is SMALLER (nearer) than above (rise/curb up)
    Return list of pseudo-box detections with modest scores.
    """
    H, W = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, HEUR_EDGE_LOW, HEUR_EDGE_HIGH)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180.0,
        threshold=HEUR_HOUGH_THRESH,
        minLineLength=HEUR_MIN_LINE_LEN,
        maxLineGap=HEUR_MAX_LINE_GAP
    )
    out = []
    if lines is None:
        return out

    for l in lines[:,0,:]:
        x1,y1,x2,y2 = map(int, l)
        # reject strongly tilted lines
        dx, dy = x2-x1, y2-y1
        angle = np.degrees(np.arctan2(dy, dx + 1e-6))
        if abs(angle) > HEUR_MAX_LINE_TILT(angle):
            continue

        # sample shallow bands just above/below the line for depth comparison
        # define a small rectangle around the line
        y_up1 = max(0, min(H-1, int(min(y1,y2) - HEUR_SAMPLE_BAND)))
        y_up2 = max(0, min(H-1, int(min(y1,y2))))
        y_dn1 = max(0, min(H-1, int(max(y1,y2))))
        y_dn2 = max(0, min(H-1, int(max(y1,y2) + HEUR_SAMPLE_BAND)))

        x_left  = max(0, min(W-1, min(x1,x2)))
        x_right = max(0, min(W-1, max(x1,x2)))
        if x_right <= x_left:
            continue

        up_patch = depth_m[y_up1:y_up2, x_left:x_right]
        dn_patch = depth_m[y_dn1:y_dn2, x_left:x_right]
        if up_patch.size == 0 or dn_patch.size == 0:
            continue

        # robust compare (median)
        up_m = float(np.median(up_patch))
        dn_m = float(np.median(dn_patch))
        delta = dn_m - up_m  # positive => farther below (drop / step down)

        if abs(delta) < HEUR_MIN_DEPTH_STEP_M:
            continue  # ignore tiny depth steps

        # create a pseudo box from the line extent
        pad_y = int(0.02 * H) + HEUR_SAMPLE_BAND
        pad_x = int(0.03 * W)
        bx1 = max(0, min(W-1, x_left - pad_x))
        bx2 = max(0, min(W-1, x_right + pad_x))
        by1 = max(0, min(H-1, min(y1,y2) - pad_y))
        by2 = max(0, min(H-1, max(y1,y2) + pad_y))
        if bx2 <= bx1 or by2 <= by1:
            continue

        # classify & score
        if delta > 0:   # below is farther -> drop
            htype = 0   # step down / curb down
        else:
            htype = 2   # step up / curb up

        # heuristic score: normalized by depth change and width
        base = float(np.clip(abs(delta) / 0.5, 0.2, 0.9))  # 0.2..0.9
        dist_est = depth_for_box(depth_m, (bx1,by1,bx2,by2), quantile=0.3)
        cx = (bx1 + bx2) / 2.0
        off = normalized_offset(W, cx)
        sev = severity_from(dist_est, base)
        # surface quality proxy
        sq = surface_quality_proxy(gray, (bx1,by1,bx2,by2))

        out.append({
            "score": base,
            "hazard_type": htype,
            "hazard_label": HAZARD_LABELS[htype],
            "distance_m": dist_est,
            "offset": off,
            "severity": sev,
            "surface_quality": sq,
            "box": [bx1,by1,bx2,by2],
        })

    # Merge overlapping heuristic boxes (simple NMS)
    if not out:
        return out
    boxes = np.array([o["box"] for o in out], dtype=np.float32)
    scores = np.array([o["score"] for o in out], dtype=np.float32)
    keep = nms_numpy(boxes, scores, iou_thresh=0.4)
    return [out[i] for i in keep]

def HEUR_MAX_LINE_TILT(angle_deg):
    # keep near-horizontal only (±HEUR_MAX_LINE_TILT degrees)
    return HEUR_MAX_LINE_TILT_DEG if 'HEUR_MAX_LINE_TILT_DEG' in globals() else 12

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, (ax2-ax1)) * max(0, (ay2-ay1))
    area_b = max(0, (bx2-bx1)) * max(0, (by2-by1))
    return inter / (area_a + area_b - inter + 1e-6)

def nms_numpy(boxes, scores, iou_thresh=0.5):
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idx = rest[ious <= iou_thresh]
    return keep
# --------------------------------------------------


# -------- Drawing / video -------------------------
def open_writer(path, fps, w, h):
    trials = [
        ('mp4v', path),
        ('avc1', path),
        ('XVID', os.path.splitext(path)[0]+'.avi'),
        ('MJPG', os.path.splitext(path)[0]+'_mjpg.avi')
    ]
    for cc, p in trials:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        wr = cv2.VideoWriter(p, fourcc, fps, (w,h))
        if wr.isOpened():
            return wr, p, cc
    return None, None, None

def draw_box(frame, box, color, label):
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, max(12, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
# --------------------------------------------------


def run(video_path, out_json, out_video,
        frame_skip=FRAME_SKIP,
        score_obs=SCORE_THRESH_OBS,
        score_stairs=SCORE_THRESH_STAIRS,
        max_dets=MAX_DETS_PER_FRAME):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load models
    midas, midas_tf = load_midas(device=device)
    yolo_obs, yolo_stairs = load_yolo_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    writer, out_vid_path, codec = open_writer(out_video, fps, W, H)
    if writer is None:
        cap.release()
        raise RuntimeError("Could not open a VideoWriter; install codecs or use AVI fallback.")

    results_json = []
    frame_idx = 0
    last_boxes = []

    pbar = tqdm(total=total, desc=f"Fusion+Stairs (skip={frame_skip})", unit="frame")
    with torch.inference_mode():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            ts = frame_idx / fps if fps > 0 else 0.0

            if frame_idx % frame_skip == 0:
                # ----- Depth -----
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                inp = midas_tf(rgb).to(device)
                inv_depth = midas(inp).squeeze().cpu().numpy()
                depth_m = depth_map_to_meters(inv_depth, NEAR_M, FAR_M)

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # ----- Obstacles (YOLO COCO) -----
                dets = detect_obstacles(yolo_obs, frame_bgr, W, H, gray, depth_m, score_thresh=score_obs)

                # ----- Stairs / Curbs -----
                if yolo_stairs is not None:
                    dets += detect_stairs_curbs_yolo(yolo_stairs, frame_bgr, W, H, gray, depth_m,
                                                     score_thresh=score_stairs)
                else:
                    dets += detect_stairs_curbs_heuristic(frame_bgr, depth_m)

                # ----- Sort by distance (nearest first), cap -----
                dets.sort(key=lambda d: d["distance_m"])
                dets = dets[:max_dets]
                last_boxes = dets

                # ----- Record JSON (only reasonable detections) -----
                valid = [d for d in dets if d["distance_m"] <= FAR_M and d["score"] >= min(score_obs, score_stairs, 0.25)]
                if valid:
                    results_json.append({
                        "frame_index": frame_idx,
                        "time_sec": round(ts, 3),
                        "detections": valid
                    })

            # ----- Draw overlay (carry between skips) -----
            for d in last_boxes:
                color = (0,0,255) if d["hazard_type"] in (0,2,4) else (255,255,0)
                label = f'{d["hazard_label"]} | {d["distance_m"]:.2f}m | {d["score"]:.2f}'
                draw_box(frame_bgr, d["box"], color, label)

            # header
            cv2.rectangle(frame_bgr, (8,8), (W-8, 40), (0,0,0), -1)
            cv2.putText(frame_bgr, f"t={ts:.2f}s  dets={len(last_boxes)}  codec={codec}",
                        (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            writer.write(frame_bgr)
            frame_idx += 1
            pbar.update(1)
    pbar.close()
    cap.release()
    writer.release()

    # Save JSON
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "video_path": os.path.abspath(video_path),
            "output_video": os.path.abspath(out_vid_path),
            "fps_used": fps,
            "frame_skip": frame_skip,
            "score_thresholds": {"obstacle": score_obs, "stairs": score_stairs},
            "stairs_mode": "yolo" if yolo_stairs is not None else "heuristic",
            "detections_by_frame": results_json
        }, f, indent=2)

    print(f"\nSaved JSON:  {out_json}")
    print(f"Saved video: {out_vid_path}  (codec={codec})")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    default_in  = os.path.join(root, "data", "walk.mp4")
    default_js  = os.path.join(root, "out", "fusion_stairs_detections.json")
    default_vid = os.path.join(root, "out", "fusion_stairs_annotated.mp4")

    import argparse
    ap = argparse.ArgumentParser(description="Fusion: MiDaS depth + YOLO obstacles + stairs/curbs (YOLO or heuristic)")
    ap.add_argument("--video", type=str, default=default_in, help="input video path")
    ap.add_argument("--json",  type=str, default=default_js, help="output JSON path")
    ap.add_argument("--out",   type=str, default=default_vid, help="output video path")
    ap.add_argument("--skip",  type=int, default=FRAME_SKIP, help="process every Nth frame")
    ap.add_argument("--score_obs", type=float, default=SCORE_THRESH_OBS, help="obstacle score threshold")
    ap.add_argument("--score_stairs", type=float, default=SCORE_THRESH_STAIRS, help="stairs/curb score threshold")
    args = ap.parse_args()

    run(args.video, args.json, args.out, frame_skip=args.skip,
        score_obs=args.score_obs, score_stairs=args.score_stairs)
