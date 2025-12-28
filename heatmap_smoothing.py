import numpy as np
import cv2


class HeatmapSmoother:
    def __init__(self, grid=(3, 3), alpha=0.25, persist_thresh=0.04):
        self.grid = grid
        self.alpha = alpha
        self.persist_thresh = persist_thresh
        self.state = np.zeros(grid, dtype=np.float32)

    def update(self, values):
        new_map = np.array(values).reshape(self.grid)

        self.state = self.alpha * new_map + (1 - self.alpha) * self.state

        diff = abs(self.state - new_map)
        self.state[diff < self.persist_thresh] = self.state[diff < self.persist_thresh]

        return self.state.copy()

    def spatial_smooth(self):
        return cv2.GaussianBlur(self.state, (3, 3), sigmaX=0.8)