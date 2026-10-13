# Hosted sinks for the M1232 pysink tests. Each writes what it saw to the file
# its `location` property names, one key=value per line.


class CountingSink:
    def __init__(self):
        self.location = None
        self.frames = 0
        self.payloads = []
        self.eos = False

    def g2g_properties(self):
        return ["location"]

    def g2g_process(self, buf, width, height, fmt, meta):
        self.frames += 1
        self.write()

    def g2g_process_payload(self, buffers, caps, meta):
        self.frames += 1
        self.payloads.append((caps, bytes(memoryview(buffers[0])).decode("utf-8")))
        self.write()

    def g2g_eos(self):
        self.eos = True
        self.write()

    def write(self):
        lines = ["frames=%d" % self.frames, "eos=%s" % self.eos]
        for caps, text in self.payloads:
            lines.append("payload=%s %s" % (caps, text))
        with open(self.location, "w") as out:
            out.write("\n".join(lines) + "\n")


class ReturningSink(CountingSink):
    def g2g_process(self, buf, width, height, fmt, meta):
        super().g2g_process(buf, width, height, fmt, meta)
        return bytes(memoryview(buf))


class EmittingSink(CountingSink):
    def g2g_process_payload(self, buffers, caps, meta):
        super().g2g_process_payload(buffers, caps, meta)
        meta.emit(b"a sink has nowhere to send this")
