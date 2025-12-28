import numpy as np
import cv2

def build_heatmap(local_scores, grid, frame_shape):
    """
    Heatmap that highlights ONLY true local instability.
    Stable regions stay blue.
    """

    heat = np.array(local_scores, dtype=np.float32).reshape(grid)

    # --- remove global background ---
    baseline = np.median(heat)
    heat = np.clip(heat - baseline, 0.0, None)

    # --- normalize by strongest anomaly ---
    max_val = heat.max()
    if max_val > 1e-6:
        heat /= max_val
    else:
        heat[:] = 0.0

    # --- VERY mild gamma (preserve contrast) ---
    heat = heat ** 0.85

    # --- upscale ---
    heat = cv2.resize(
        heat,
        (frame_shape[1], frame_shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    heat = (heat * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return heatmap