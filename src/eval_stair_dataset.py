# src/eval_stair_dataset_rgb.py
import os, json, glob, subprocess, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", default="data/stair_dataset/val/images",
                    help="Folder containing RGB images")
    ap.add_argument("--out_dir", default="out/stair_eval_rgb",
                    help="Where to save visualizations and results.json")
    ap.add_argument("--midas_type", default="DPT_Hybrid",
                    help="MiDaS_small | DPT_Hybrid | DPT_Large")
    ap.add_argument("--camera_height_m", type=float, default=1.4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect all candidate RGB images
    all_files = []
    patterns = ["color*.jpg", "color*.png", "color*.jpeg",
                "park*.jpg", "park*.png", "park*.jpeg",
                "sh_color*.jpg", "sh_color*.png", "sh_color*.jpeg"]
    for pat in patterns:
        all_files += glob.glob(os.path.join(args.rgb_dir, pat))

    if not all_files:
        print(f"❌ No RGB files found in {args.rgb_dir}.")
        return

    results = []

    for rgb_path in sorted(all_files):
        base = os.path.splitext(os.path.basename(rgb_path))[0]
        out_vis = os.path.join(args.out_dir, base + "_vis.png")

        cmd = [
            "python", "src/inference.py",
            "--rgb", rgb_path,
            "--midas_type", args.midas_type,
            "--camera_height_m", str(args.camera_height_m),
            "--out_vis", out_vis
        ]
        print(f"Running {base}...")
        try:
            output = subprocess.check_output(cmd, text=True)
            hazards = json.loads(output)
        except Exception as e:
            hazards = {"error": str(e)}
        results.append({"file": base, "hazards": hazards})

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Processed {len(results)} images. Results saved to {args.out_dir}/results.json")

if __name__ == "__main__":
    main()
