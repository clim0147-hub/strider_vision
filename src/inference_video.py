# src/inference_video.py
import cv2, torch, time
import numpy as np
from src.models import ElevationNet

W, H = 320, 192  # your spec

def preprocess(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.0  # [3,H,W]
    return t.unsqueeze(0)  # [1,3,H,W]

def draw_overlay(vis, y, fps):
    conf, type_id, dist, off, sev, surf = y
    h, w = vis.shape[:2]
    txt = f"conf:{conf:.2f}  type:{int(round(type_id))}  dist:{dist:.2f}m  off:{off:.2f}  sev:{sev:.2f}  surf:{surf:.2f}  FPS:{fps:.1f}"
    cv2.rectangle(vis, (5, 5), (w-5, 60), (0,0,0), -1)
    cv2.putText(vis, txt, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
    # simple center line for offset intuition
    cx = int(w/2 + off*(w/2-10))
    cv2.line(vis, (int(w/2), 70), (int(w/2), h-10), (100,100,100), 1)
    cv2.line(vis, (cx, 70), (cx, h-10), (255,255,255), 2)

def main(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ElevationNet(in_ch=3, pretrained=True).to(device).eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open:", video_path)
        return

    # temporal smoothing + simple hysteresis
    ema = None
    alpha = 0.6
    k_on, k_off = 3, 3         # k-of-N frames to start/stop alert
    N = 8
    recent = [0]*N
    alert_on = False
    last_change = 0.0

    t0 = time.time()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        tic = time.time()

        x = preprocess(frame).to(device)
        with torch.inference_mode():
             result = net(x, return_logits=False)  # or net(x) if you prefer
        # handle 1- or 2-item returns
        y = result[0] if isinstance(result, (tuple, list)) else result
        y = y[0].detach().cpu().numpy()

        # EMA smoothing
        ema = y if ema is None else alpha*ema + (1-alpha)*y
        conf, type_id, dist, off, sev, surf = ema

        # Debounce/hysteresis using confidence gate
        thresh = 0.55
        recent.pop(0); recent.append(1 if conf >= thresh else 0)
        if not alert_on and sum(recent) >= k_on:
            alert_on = True; last_change = time.time()
        elif alert_on and (N - sum(recent)) >= k_off:
            alert_on = False; last_change = time.time()

        # Visualize
        vis = frame.copy()
        draw_overlay(vis, (conf, type_id, dist, off, sev, surf), fps=frames / max(1e-6, (time.time() - t0)))
        if alert_on:
            cv2.putText(vis, "ALERT", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Strider Vision — Inference", vis)

        # ~15ms wait; press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        _ = time.time() - tic

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys, os
    root = os.path.dirname(os.path.dirname(__file__))
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "walk.mp4")
    main(video_path)
