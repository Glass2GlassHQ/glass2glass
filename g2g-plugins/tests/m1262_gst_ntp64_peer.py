import sys
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtp", "1.0")
from gi.repository import Gst, GstRtp  # noqa: E402

NTP_64_URI = "urn:ietf:params:rtp-hdrext:ntp-64"
NTP_CAPS = "timestamp/x-ntp"
PAYLOAD_TYPE = 96
CLOCK_RATE = 90000
FRAME_DURATION_NS = 33_333_333
SEND_SPACING_S = 0.005
PULL_TIMEOUT_NS = 10_000_000_000
REQUIRED_ELEMENTS = ["rtphdrextntp64", "rtph264pay", "rtph264depay", "udpsrc", "udpsink", "appsrc", "appsink"]
SKIP_EXIT_CODE = 77


def rtp_caps(ntp_id):
    structure = Gst.Structure.new_empty("application/x-rtp")
    structure.set_value("media", "video")
    structure.set_value("encoding-name", "H264")
    structure.set_value("clock-rate", CLOCK_RATE)
    structure.set_value("payload", PAYLOAD_TYPE)
    structure.set_value(f"extmap-{ntp_id}", NTP_64_URI)
    caps = Gst.Caps.new_empty()
    caps.append_structure(structure)
    return caps


def check():
    missing = [name for name in REQUIRED_ELEMENTS if Gst.ElementFactory.find(name) is None]
    if missing:
        print("missing", " ".join(missing), flush=True)
        sys.exit(SKIP_EXIT_CODE)


def receive(ntp_id, abs_id, count):
    pipeline = Gst.parse_launch(
        "udpsrc name=src address=127.0.0.1 port=0 ! rtph264depay ! appsink name=sink sync=false"
    )
    source = pipeline.get_by_name("src")
    source.set_property("caps", rtp_caps(ntp_id))

    def read_abs_capture_time(_pad, info):
        ok, rtp = GstRtp.RTPBuffer.map(info.get_buffer(), Gst.MapFlags.READ)
        if ok:
            found, data = rtp.get_extension_onebyte_header(abs_id, 0)
            rtp.unmap()
            if found:
                print("abs", bytes(data).hex(), flush=True)
        return Gst.PadProbeReturn.OK

    source.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, read_abs_capture_time)
    pipeline.set_state(Gst.State.PLAYING)
    print("ready", source.get_property("port"), flush=True)
    sink = pipeline.get_by_name("sink")
    reference = Gst.Caps.from_string(NTP_CAPS)
    for _ in range(count):
        sample = sink.emit("try-pull-sample", PULL_TIMEOUT_NS)
        if sample is None:
            break
        meta = sample.get_buffer().get_reference_timestamp_meta(reference)
        print("ntp", meta.timestamp if meta else "none", flush=True)
    pipeline.set_state(Gst.State.NULL)
    print("done", flush=True)


def send(port, ntp_id, access_unit, ntp_nanos):
    pipeline = Gst.parse_launch(
        "appsrc name=src format=time "
        "caps=video/x-h264,stream-format=byte-stream,alignment=au "
        "! rtph264pay config-interval=-1 ! capsfilter name=filter "
        "! udpsink name=sink host=127.0.0.1 sync=false"
    )
    pipeline.get_by_name("filter").set_property("caps", rtp_caps(ntp_id))
    pipeline.get_by_name("sink").set_property("port", port)
    pipeline.set_state(Gst.State.PLAYING)
    source = pipeline.get_by_name("src")
    reference = Gst.Caps.from_string(NTP_CAPS)
    for index, timestamp in enumerate(ntp_nanos):
        buffer = Gst.Buffer.new_wrapped(access_unit)
        buffer.pts = index * FRAME_DURATION_NS
        buffer.duration = FRAME_DURATION_NS
        buffer.add_reference_timestamp_meta(reference, timestamp, Gst.CLOCK_TIME_NONE)
        source.emit("push-buffer", buffer)
        time.sleep(SEND_SPACING_S)
    source.emit("end-of-stream")
    pipeline.get_bus().timed_pop_filtered(PULL_TIMEOUT_NS, Gst.MessageType.EOS | Gst.MessageType.ERROR)
    pipeline.set_state(Gst.State.NULL)
    print("done", flush=True)


def main():
    Gst.init(None)
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "receive":
        receive(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif mode == "send":
        send(int(sys.argv[2]), int(sys.argv[3]), bytes.fromhex(sys.argv[4]), [int(value) for value in sys.argv[5:]])


main()
