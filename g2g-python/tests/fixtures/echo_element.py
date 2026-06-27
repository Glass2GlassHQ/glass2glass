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
        # Attach opaque tagged side-data (the FrameIO.append_blob mirror), e.g.
        # an embedding's serialized bytes.
        meta.add_blob("embedding", bytes([1, 2, 3, 4]))

    def g2g_process_batch(self, buffers, width, height, fmt, meta):
        # Stand-in for batched inference: sum the batch's first bytes into the
        # anchor (buffers[0]), and attach one detection whose label is the
        # batch size, proving N inputs reached one Python call.
        total = 0
        for b in buffers:
            total = (total + memoryview(b)[0]) % 256
        memoryview(buffers[0])[0] = total
        meta.add_object(len(buffers), 0.0, 0.0, 1.0, 1.0, 1.0)


class RetainingTransform:
    """Misbehaves: stashes the frame buffer view on self, so it outlives the
    g2g_process call. The host must reject this (the retained pointer would
    dangle once the frame is freed downstream) rather than risk a use-after-free.
    """

    def g2g_process(self, buf, width, height, fmt, meta):
        # Retain a writable view past return: this is the contract violation the
        # host's export-counter guard catches.
        self.saved = memoryview(buf)


class CounterSource:
    """A source: writes its frame index into byte 0, ends after three frames."""

    def __init__(self):
        self.n = 0

    def g2g_produce(self, buf, width, height, fmt, meta):
        if self.n >= 3:
            return False  # end of stream
        memoryview(buf)[0] = self.n
        self.n += 1
        return True


class PropEcho:
    """Echoes forwarded element properties back as metadata, proving the host
    set them on the instance: a gst-style `model-name` reaches `self.model_name`,
    and an int property keeps its int type (used directly as a detection label,
    which the native sink requires to be an integer)."""

    def g2g_process(self, buf, width, height, fmt, meta):
        model = getattr(self, "model_name", "<unset>")
        device = getattr(self, "device", "<unset>")
        # Forwarded as a Python int; add_object's label requires an integer, so
        # passing it straight through fails loudly if it arrived as a string.
        batch = getattr(self, "batch_size", 0)
        meta.add_blob("model_name", model.encode("utf-8"))
        meta.add_blob("device", device.encode("utf-8"))
        meta.add_object(batch, 0.0, 0.0, 1.0, 1.0, 1.0)
