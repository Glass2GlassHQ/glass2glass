# Hosted elements for the M1241 text_src tests.

EMIT_ERRORS = []


class Captioner:
    def __init__(self):
        self.frames = 0

    def g2g_process(self, buf, width, height, fmt, meta):
        meta.emit(("caption %d" % self.frames).encode("utf-8"), output="text_src")
        self.frames += 1


class RecordingCaptioner:
    def g2g_process(self, buf, width, height, fmt, meta):
        try:
            meta.emit(b"caption", output="text_src")
        except ValueError as error:
            EMIT_ERRORS.append(str(error))
            raise
