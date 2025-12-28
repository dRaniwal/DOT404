class RateLimiter:
    def __init__(self, max_delta=0.02):
        self.max_delta = max_delta
        self.last = None

    def update(self, x):
        if self.last is None:
            self.last = x
            return x

        delta = x - self.last
        delta = max(-self.max_delta, min(self.max_delta, delta))
        self.last += delta
        return self.last