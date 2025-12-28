class EMASmoother:
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class HysteresisFilter:
    def __init__(self, up_thresh=0.05, down_thresh=0.03):
        self.up = up_thresh
        self.down = down_thresh
        self.last = None

    def update(self, x):
        if self.last is None:
            self.last = x
            return x

        if x > self.last + self.up:
            self.last = x
        elif x < self.last - self.down:
            self.last = x

        return self.last