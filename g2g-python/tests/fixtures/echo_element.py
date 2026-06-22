# Minimal stdlib-only hosted element for the M198 step-2 zero-copy test.
# Stands in for a gst-python-ml `backend/g2g` element shell: it receives the
# frame as a writable buffer-protocol object and proves an in-place write
# reaches the Rust Frame's memory, with no numpy / cv2 dependency.


class EchoTransform:
    """Bumps the first byte in place and echoes geometry/format as a blob."""

    def g2g_process(self, buf, width, height, fmt, meta):
        mv = memoryview(buf)
        assert not mv.readonly, "frame buffer must be writable"
        assert mv.nbytes == width * height * 4, "expected RGBA geometry"
        # In-place mutation: this write lands directly in the Rust frame buffer.
        mv[0] = (mv[0] + 1) % 256
        # Attach an analytics result through the sink (the AnalyticsBackend
        # mirror): label id 7, a box, and a confidence.
        meta.add_object(7, 1.0, 2.0, 3.0, 4.0, 0.9)
