import os
from PIL import Image

# === Configuration ===
ROOT_DIR = r"C:\Users\User\Desktop\strider_vision\data\stair_dataset"
SETS = ['train', 'val']  # Modify if needed
IMG_SIZE = {
    'train': (512, 512),
    'val': (512, 512)
}
LINE_WIDTH_PIXELS = 6  # Thickness of bounding box around lines

def line_to_box(x1, y1, x2, y2, padding):
    xmin = min(x1, x2) - padding
    ymin = min(y1, y2) - padding
    xmax = max(x1, x2) + padding
    ymax = max(y1, y2) + padding
    return max(0, xmin), max(0, ymin), min(512, xmax), min(512, ymax)

def to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def convert_label_file(label_path, image_path, img_w, img_h):
    yolo_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"❌ Skipping malformed line: {line.strip()} in {label_path}")
                continue

            cls, x1, y1, x2, y2 = map(float, parts)
            xmin, ymin, xmax, ymax = line_to_box(x1, y1, x2, y2, padding=LINE_WIDTH_PIXELS)
            xc, yc, w, h = to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)

            if 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1:
                yolo_lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            else:
                print(f"⚠️ Invalid normalized box in {label_path}, skipping line.")

    with open(label_path, "w") as f:
        for line in yolo_lines:
            f.write(line + "\n")
    print(f"✅ Converted: {os.path.basename(label_path)}")

def process_set(split):
    print(f"\n🔄 Processing {split.upper()} set...")
    label_dir = os.path.join(ROOT_DIR, split, "labels")
    image_dir = os.path.join(ROOT_DIR, split, "images")

    img_w, img_h = IMG_SIZE[split]

    if not os.path.exists(label_dir) or not os.path.exists(image_dir):
        print(f"❌ Missing folders for {split} set, skipping.")
        return

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found for {label_file}, skipping.")
            continue

        convert_label_file(label_path, image_path, img_w, img_h)

if __name__ == "__main__":
    for s in SETS:
        process_set(s)

    print("\n✅ All labels converted to YOLO format.")
