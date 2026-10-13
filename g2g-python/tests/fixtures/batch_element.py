BATCH_SIZES = []


class BatchDetector:
    def g2g_process_batch(self, buffers, width, height, fmt, meta):
        BATCH_SIZES.append(len(buffers))
        meta.set_class_names(["stream_%d_thing" % i for i in range(len(buffers))])
        for i in range(len(buffers)):
            meta.add_object(i, i + 1, 0, 1, 1, 1.0)
