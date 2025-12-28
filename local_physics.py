from escalation import velocity_variance, directional_entropy


def local_motion_metrics(flow, frame_shape, grid=(3, 3)):
    h, w = frame_shape[:2]
    rows, cols = grid
    metrics = []

    for r in range(rows):
        for c in range(cols):
            y1 = r * h // rows
            y2 = (r + 1) * h // rows
            x1 = c * w // cols
            x2 = (c + 1) * w // cols

            cell_flow = flow[y1:y2, x1:x2]

            if cell_flow.size == 0:
                metrics.append((0.0, 0.0))
                continue

            v = velocity_variance(cell_flow)
            hdir = directional_entropy(cell_flow)
            metrics.append((v, hdir))

    return metrics