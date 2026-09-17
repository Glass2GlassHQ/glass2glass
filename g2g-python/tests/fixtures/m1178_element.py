# Hosted elements for the M1178 tests: what a native detector upstream attached
# reaches the hosted class through the `meta` sink, and `only-on` decides whether
# the class is called at all.


class MetaReader:
    """Echoes back what the incoming frame carried, as blobs the Rust side reads.

    The `read_objects` / `read_blobs` shape gst-python-ml's g2g backend calls.
    """

    def g2g_process(self, buf, width, height, fmt, meta):
        objects = meta.objects()
        blobs = meta.blobs()
        names = meta.class_names()
        tracking = meta.tracking_ids()
        meta.add_blob("seen-count", bytes([len(objects)]))
        for obj in objects[:1]:
            meta.add_blob("seen-label", obj["label"].encode("utf-8"))
            box = "%d,%d,%d,%d" % (obj["x"], obj["y"], obj["w"], obj["h"])
            meta.add_blob("seen-box", box.encode("utf-8"))
            meta.add_blob("seen-score", ("%.3f" % obj["score"]).encode("utf-8"))
        meta.add_blob("seen-names", ",".join(names).encode("utf-8"))
        meta.add_blob("seen-headers", ",".join(sorted(blobs)).encode("utf-8"))
        meta.add_blob("seen-alert", blobs.get("alert", b"<none>"))
        meta.add_blob(
            "seen-tracking",
            ",".join("none" if t is None else str(t) for t in tracking).encode("utf-8"),
        )


class AppendingTransform:
    """Stages a detection, a tracking identity and a blob of its own, so the
    merge with what upstream attached is observable downstream."""

    LABEL = 3
    OBJECT_ID = 99

    def g2g_process(self, buf, width, height, fmt, meta):
        meta.add_blob("verdict", b"ok")
        detection = meta.add_object(self.LABEL, 0.0, 0.0, width, height, 0.5)
        tracking = meta.add_tracking(self.OBJECT_ID)
        meta.relate(detection, tracking)


class CountingTransform:
    """Counts the calls it received, so a frame `only-on` skipped is visible as a
    call that did not happen."""

    def __init__(self):
        self.calls = 0

    def g2g_process(self, buf, width, height, fmt, meta):
        self.calls += 1
        meta.add_blob("calls", bytes([self.calls]))
