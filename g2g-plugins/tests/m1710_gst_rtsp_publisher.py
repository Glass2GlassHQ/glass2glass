import socket
import sys
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

NTP_64_URI = "urn:ietf:params:rtp-hdrext:ntp-64"
NTP_CAPS = "timestamp/x-ntp"
PAYLOAD_TYPE = 96
CLOCK_RATE = 90000
FRAME_DURATION_NS = 33_333_333
SEND_SPACING_S = 0.005
PULL_TIMEOUT_NS = 10_000_000_000
SOCKET_TIMEOUT_S = 10
RTP_CHANNEL = 0
RTCP_CHANNEL = 1
INTERLEAVED_MARKER = b"$"
REQUIRED_ELEMENTS = ["rtphdrextntp64", "rtph264pay", "appsrc", "appsink"]
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


# GStreamer's payloader with its ntp-64 writer, collected to send over the RTSP connection
def payload_with_ntp64(ntp_id, access_unit, ntp_nanos):
    pipeline = Gst.parse_launch(
        "appsrc name=src format=time "
        "caps=video/x-h264,stream-format=byte-stream,alignment=au "
        "! rtph264pay config-interval=-1 ! capsfilter name=filter "
        "! appsink name=sink sync=false"
    )
    pipeline.get_by_name("filter").set_property("caps", rtp_caps(ntp_id))
    pipeline.set_state(Gst.State.PLAYING)
    source = pipeline.get_by_name("src")
    reference = Gst.Caps.from_string(NTP_CAPS)
    for index, timestamp in enumerate(ntp_nanos):
        buffer = Gst.Buffer.new_wrapped(access_unit)
        buffer.pts = index * FRAME_DURATION_NS
        buffer.duration = FRAME_DURATION_NS
        buffer.add_reference_timestamp_meta(reference, timestamp, Gst.CLOCK_TIME_NONE)
        source.emit("push-buffer", buffer)
    source.emit("end-of-stream")
    sink = pipeline.get_by_name("sink")
    packets = []
    # appsink holds its EOS behind the samples still queued, so pull until it arrives
    while True:
        sample = sink.emit("try-pull-sample", PULL_TIMEOUT_NS)
        if sample is None:
            break
        buffer = sample.get_buffer()
        packets.append(buffer.extract_dup(0, buffer.get_size()))
    pipeline.set_state(Gst.State.NULL)
    return packets


class RtspConnection:
    def __init__(self, port):
        self.url = f"rtsp://127.0.0.1:{port}/stream"
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=SOCKET_TIMEOUT_S)
        self.cseq = 0
        self.session = None

    def request(self, method, url, headers=(), body=b""):
        self.cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {self.cseq}"]
        if self.session:
            lines.append(f"Session: {self.session}")
        lines.extend(f"{name}: {value}" for name, value in headers)
        if body:
            lines.append(f"Content-Length: {len(body)}")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + body)
        return self.response()

    def response(self):
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1)
            if not chunk:
                raise ConnectionError("server closed the control connection")
            head += chunk
        text = head.decode()
        status = text.split("\r\n", 1)[0]
        if " 200 " not in status:
            raise ConnectionError(status)
        for line in text.split("\r\n"):
            name, _, value = line.partition(":")
            if name.lower() == "session":
                self.session = value.strip().split(";")[0]
            if name.lower() == "content-length" and int(value):
                self.sock.recv(int(value))
        return text

    def send_rtp(self, packet):
        header = INTERLEAVED_MARKER + bytes([RTP_CHANNEL]) + len(packet).to_bytes(2, "big")
        self.sock.sendall(header + packet)


def publish(port, ntp_id, access_unit, ntp_nanos):
    packets = payload_with_ntp64(ntp_id, access_unit, ntp_nanos)
    connection = RtspConnection(port)
    connection.request("OPTIONS", connection.url)
    sdp = (
        "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=gst\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n"
        f"m=video 0 RTP/AVP {PAYLOAD_TYPE}\r\n"
        f"a=rtpmap:{PAYLOAD_TYPE} H264/{CLOCK_RATE}\r\n"
        f"a=extmap:{ntp_id} {NTP_64_URI}\r\n"
        "a=control:streamid=0\r\n"
    ).encode()
    connection.request("ANNOUNCE", connection.url, [("Content-Type", "application/sdp")], sdp)
    connection.request(
        "SETUP",
        f"{connection.url}/streamid=0",
        [("Transport", f"RTP/AVP/TCP;unicast;interleaved={RTP_CHANNEL}-{RTCP_CHANNEL};mode=record")],
    )
    connection.request("RECORD", connection.url, [("Range", "npt=0.000-")])
    for packet in packets:
        connection.send_rtp(packet)
        time.sleep(SEND_SPACING_S)
    print("sent", len(packets), flush=True)
    connection.request("TEARDOWN", connection.url)
    print("done", flush=True)


def main():
    Gst.init(None)
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "publish":
        publish(int(sys.argv[2]), int(sys.argv[3]), bytes.fromhex(sys.argv[4]), [int(value) for value in sys.argv[5:]])


main()
