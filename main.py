import cv2
import numpy as np
from ultralytics import YOLO

from escalation import (
    velocity_variance,
    directional_entropy,
    density_rate,
    escalation_contrast,
    squash_escalation
)
from local_physics import local_motion_metrics
from smoothing import EMASmoother, HysteresisFilter
from rate_limiter import RateLimiter
from heatmap_smoothing import HeatmapSmoother
from heatmap import build_heatmap

VIDEO_PATH = "videos/v5.webm"
MODEL_PATH = "models/yolov8n.pt"
GRID = (3, 3)

def risk_color(score):
    if score < 0.3:
        return (0, 200, 0)
    elif score < 0.6:
        return (0, 200, 200)
    else:
        return (0, 0, 255)

def draw_grid(frame, grid):
    h, w = frame.shape[:2]
    rows, cols = grid
    for r in range(1, rows):
        cv2.line(frame, (0, r*h//rows), (w, r*h//rows), (255,255,255), 1)
    for c in range(1, cols):
        cv2.line(frame, (c*w//cols, 0), (c*w//cols, h), (255,255,255), 1)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    model = YOLO(MODEL_PATH)

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    ema_fast = EMASmoother(0.15)
    ema_slow = EMASmoother(0.05)
    hyst = HysteresisFilter(0.05, 0.03)
    limiter = RateLimiter(0.015)

    heat_smoother = HeatmapSmoother(GRID)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        results = model(frame, conf=0.3, classes=[0], verbose=False)
        people = len(results[0].boxes)
        d_rate = density_rate(people)

        local_metrics = local_motion_metrics(flow, frame.shape, GRID)

        local_energies = []
        for v, h in local_metrics:
            e = (
                0.4 * np.clip(d_rate / 5.0, 0, 1) +
                0.35 * np.clip(v / 10.0, 0, 1) +
                0.25 * np.clip(h / 3.0, 0, 1)
            )
            local_energies.append(float(e))

        # ---- FINAL ESCALATION PIPELINE ----
        # Contrast-based escalation from spatial heterogeneity
        contrast_energy = escalation_contrast(local_energies)
        raw = squash_escalation(contrast_energy)

        s = ema_fast.update(raw)
        s = ema_slow.update(s)
        s = hyst.update(s)
        s = limiter.update(s)

        heat_smoother.update(local_energies)
        stable_local = heat_smoother.spatial_smooth()

        heatmap = build_heatmap(stable_local.flatten(), GRID, frame.shape)

        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        draw_grid(overlay, GRID)

        cv2.rectangle(overlay, (10, 10), (360, 85), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"Escalation Risk: {s:.2f}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            risk_color(s),
            2
        )

        cv2.imshow("Escalation Monitor", overlay)

        prev_gray = gray
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()