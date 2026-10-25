class UpstreamTagger:
    OBJECT_ID = 7

    def g2g_process(self, buf, width, height, fmt, meta):
        meta.track_upstream(1, self.OBJECT_ID)


class StreamLabeller:
    def g2g_process(self, buf, width, height, fmt, meta):
        meta.set_class_names(["stream_0_person"])
        meta.add_object(0, 0.0, 0.0, width, height, 0.5)
