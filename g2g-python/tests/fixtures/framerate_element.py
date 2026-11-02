FRAMERATES_SEEN = []


class FramerateRecorder:
    def g2g_process(self, buf, width, height, fmt, meta):
        FRAMERATES_SEEN.append(meta.framerate)
