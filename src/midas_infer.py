import torch, cv2
import numpy as np

def load_midas(device="cpu", model_type="MiDaS_small"):
    """
    model_type: 'MiDaS_small', 'DPT_Hybrid', 'DPT_Large'
    """
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval().to(device)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    tfm = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform
    return model, tfm

def predict_depth(model, tfm, rgb_path, device="cpu"):
    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {rgb_path}. Check the path.")
    img = img_bgr[:, :, ::-1]  # BGR->RGB
    input_batch = tfm(img).to(device)
    with torch.no_grad():
        pred = model(input_batch)
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().cpu().numpy()
    # normalize to ~0..3m for our pipeline (relative scale)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = depth * 3.0
    return depth.astype(np.float32)
