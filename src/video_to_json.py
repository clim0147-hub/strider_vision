# src/video_multi_to_json.py
import os, sys, json
import cv2, torch
import numpy as np
from tqdm import tqdm
from src.models import ElevationNet

HAZARD_LABELS = {
    0: "Step Down / Curb Down",
    1: "Flat Surface (safe)",
    2: "Step Up / Curb Up",
    3: "Rough / Uneven Surface",
    4: "Obstacle / Barrier",
}

def preprocess(frame_bgr, W=320, H=192):
    """Convert a BGR frame to the model's RGB 320x192 tensor in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
    return t.unsqueeze(0)  # [1,3,H,W]

def _open_writer(path, fps, w, h):
    """Try multiple codecs for Windows compatibility."""
    trials = [
        ('mp4v', path),                                     # MP4 (MPEG-4)
        ('avc1', path),                                     # MP4 (H.264 if available)
        ('XVID', os.path.splitext(path)[0] + '.avi'),       # AVI (XVID)
        ('MJPG', os.path.splitext(path)[0] + '_mjpg.avi'),  # AVI (Motion-JPEG)
    ]
    for fourcc_str, out_path in trials:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return writer, out_path, fourcc_str
    return None, None, None

def draw_header(frame, ts, n_boxes, codec):
    H, W = frame.shape[:2]
    cv2.rectangle(frame, (8,8), (W-8, 40), (0,0,0), -1)
    txt = f"t={ts:.2f}s   dets={n_boxes}   codec={codec}"
    cv2.putText(frame, txt, (16,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

def run(
    video_path,
    out_json,
    out_video,
    frame_skip=5,
    score_thresh=0.30,
    bypass_gate=True,       # while heads are un/under-trained, keep True to visualize
    max_dets=20
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ElevationNet(in_ch=3, pretrained=True, max_dets=max_dets).to(device).eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360

    # open writer
    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    writer, actual_out, used_codec = _open_writer(out_video, fps, W, H)
    if writer is None:
        cap.release()
        raise RuntimeError("Failed to open any VideoWriter (mp4v/avc1/XVID/MJPG). Install codecs or try AVI fallback.")

    # model-input → video size scaling
    sx = W / 320.0
    sy = H / 192.0

    results = []
    frame_idx = 0
    last_boxes = []   # carry scaled boxes & meta for skipped frames

    with torch.inference_mode():
        pbar = tqdm(total=total_frames, desc=f"Processing video (skip={frame_skip})", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = frame_idx / fps if fps > 0 else 0.0

            if frame_idx % frame_skip == 0:
                # inference on this frame (model expects 320x192 tensor inside preprocess)
                x = preprocess(frame).to(device)
                single, _, dets = net(x, return_logits=False)     # dets: [1,K,10] in 320x192 space
                dets = dets[0].cpu().numpy()                      # [K,10]

                # select & scale boxes
                picked = []
                scaled_for_draw = []
                for d in dets:
                    score, cls, dist, off, sev, surf, x1, y1, x2, y2 = d
                    cls = int(round(cls))
                    if score < score_thresh:
                        continue
                    if not bypass_gate and cls == 1:
                        continue

                    # scale 320x192 → WxH
                    X1 = int(round(x1 * sx)); Y1 = int(round(y1 * sy))
                    X2 = int(round(x2 * sx)); Y2 = int(round(y2 * sy))
                    # clamp
                    X1 = max(0, min(W-1, X1)); X2 = max(0, min(W-1, X2))
                    Y1 = max(0, min(H-1, Y1)); Y2 = max(0, min(H-1, Y2))
                    if X2 <= X1 or Y2 <= Y1:
                        continue  # skip degenerate boxes

                    picked.append({
                        "score": float(score),
                        "hazard_type": cls,
                        "hazard_label": HAZARD_LABELS.get(cls, "Unknown"),
                        "distance_m": float(dist),
                        "offset": float(off),
                        "severity": float(sev),
                        "surface_quality": float(surf),
                        "box_input_320x192": [int(x1), int(y1), int(x2), int(y2)],  # debug
                        "box": [X1, Y1, X2, Y2]  # scaled → draw this
                    })
                    scaled_for_draw.append((X1, Y1, X2, Y2, cls, float(score), float(dist)))

                # sort by ascending distance (nearest first)
                picked.sort(key=lambda z: z["distance_m"])
                scaled_for_draw.sort(key=lambda t: t[6])  # sort by dist value
                last_boxes = scaled_for_draw

                # record JSON only for processed frames
                if picked:
                    results.append({
                        "frame_index": frame_idx,
                        "time_sec": round(ts, 3),
                        "detections": picked
                    })

            # draw the latest detections on every frame (carry over during skips)
            for (X1, Y1, X2, Y2, cls, score, dist) in last_boxes:
                color = (0, 0, 255) if cls != 1 else (255, 255, 0)
                cv2.rectangle(frame, (X1, Y1), (X2, Y2), color, 2)
                tag = f'{HAZARD_LABELS.get(cls,"Unknown")} ({dist:.2f}m | {score:.2f})'
                cv2.putText(frame, tag, (X1, max(12, Y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            draw_header(frame, ts, len(last_boxes), used_codec)
            writer.write(frame)

            frame_idx += 1
            pbar.update(1)
        pbar.close()

    cap.release()
    writer.release()

    # write JSON
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "video_path": os.path.abspath(video_path),
            "output_video": os.path.abspath(actual_out),
            "fps_used": fps,
            "frame_skip": frame_skip,
            "score_threshold": score_thresh,
            "bypass_gate": bypass_gate,
            "detections_by_frame": results
        }, f, indent=2)

    print(f"\nSaved JSON: {out_json}")
    print(f"Saved video: {actual_out} (codec={used_codec})")

if __name__ == "__main__":
    import argparse
    root = os.path.dirname(os.path.dirname(__file__))
    default_in  = os.path.join(root, "data", "walk.mp4")
    default_js  = os.path.join(root, "out", "hazard_multi.json")
    default_vid = os.path.join(root, "out", "hazard_multi.mp4")

    ap = argparse.ArgumentParser(description="Multi-instance hazard to JSON + annotated video")
    ap.add_argument("--video", type=str, default=default_in, help="input video path")
    ap.add_argument("--json",  type=str, default=default_js, help="output JSON path")
    ap.add_argument("--out",   type=str, default=default_vid, help="output video path")
    ap.add_argument("--skip",  type=int, default=5, help="frame skip (process every Nth frame)")
    ap.add_argument("--score", type=float, default=0.30, help="score threshold for boxes")
    ap.add_argument("--bypass", action="store_true", help="bypass flat-surface gate (show more boxes)")
    ap.add_argument("--maxd",  type=int, default=20, help="max detections per frame (cap)")
    args = ap.parse_args()

    run(
        video_path=args.video,
        out_json=args.json,
        out_video=args.out,
        frame_skip=args.skip,
        score_thresh=args.score,
        bypass_gate=args.bypass,
        max_dets=args.maxd
    )
