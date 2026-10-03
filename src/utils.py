"""Torch-free helpers: dataset folder discovery and Grad-CAM heat-map overlay."""
import os
import numpy as np
from PIL import Image
import matplotlib

SPLIT_NAMES = [("Training", "Testing"), ("train", "test"), ("Train", "Test")]

def find_split_dirs(root, depth=0):
    """Return (train_dir, test_dir). Searches a few levels down because Kaggle zips often nest folders."""
    for tr, te in SPLIT_NAMES:
        a, b = os.path.join(root, tr), os.path.join(root, te)
        if os.path.isdir(a) and os.path.isdir(b):
            return a, b
    if depth < 3:
        for sub in sorted(os.listdir(root)):
            p = os.path.join(root, sub)
            if os.path.isdir(p):
                try:
                    return find_split_dirs(p, depth + 1)
                except FileNotFoundError:
                    pass
    raise FileNotFoundError(f"Could not find train/test folders under {root}. "
                            f"Expected e.g. Training/ and Testing/ (or train/ and test/), each with one sub-folder per class.")

def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Blend a [0,1] Grad-CAM map (any size) on top of a PIL image."""
    img = pil_img.convert("RGB")
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    heat = (matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[:, :, :3] * 255).astype(np.float32)
    blend = (1 - alpha) * np.asarray(img, dtype=np.float32) + alpha * heat
    return Image.fromarray(blend.astype(np.uint8))
