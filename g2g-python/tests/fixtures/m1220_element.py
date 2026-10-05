import time

RUN_SECONDS = 0.03


class SlowTransform:
    def __init__(self):
        self.calls = 0

    def g2g_process(self, buf, width, height, fmt, meta):
        self.calls += 1
        time.sleep(RUN_SECONDS)
        meta.add_blob("calls", bytes([self.calls]))
