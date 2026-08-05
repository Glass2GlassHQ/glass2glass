"""Independent WebTransport peer for the M901 tests: an aioquic (Python) server
that speaks the g2g remote-transform protocol.

It reads the u32-LE length-framed wire stream off the client's bidirectional
WebTransport stream with its own parser, treats the first message as the leading
CapsChanged (configuration, no reply), and echoes every later message back
verbatim. Nothing here shares code with g2g, so a passing round trip means our
QUIC / HTTP-3 CONNECT handshake and our framing are right by the protocol, not by
agreement with ourselves.

usage: peer.py <cert.pem> <key.pem> <port> <ready-file>
"""

import asyncio
import struct
import sys

from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import HeadersReceived, WebTransportStreamDataReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ConnectionTerminated, ProtocolNegotiated

# Set once the client's session ends, so the peer exits on its own rather than
# leaving a process behind if the test cannot reap it.
DONE = None
# Upper bound on a run, in case no client ever arrives.
WATCHDOG_SECONDS = 120


class WirePeer(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http = None
        self._buf = bytearray()
        self._seen = 0
        self._echoed = 0

    def quic_event_received(self, event):
        if isinstance(event, ProtocolNegotiated):
            self._http = H3Connection(self._quic, enable_webtransport=True)
        elif isinstance(event, ConnectionTerminated):
            print(f"closed after {self._echoed} echoes", flush=True)
            if DONE is not None:
                DONE.set()
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                self._h3_event_received(h3_event)

    def _h3_event_received(self, event):
        if isinstance(event, HeadersReceived):
            headers = dict(event.headers)
            connect = (
                headers.get(b":method") == b"CONNECT"
                and headers.get(b":protocol") == b"webtransport"
            )
            if connect:
                print("session accepted", flush=True)
                self._http.send_headers(
                    stream_id=event.stream_id,
                    headers=[
                        (b":status", b"200"),
                        (b"sec-webtransport-http3-draft", b"draft02"),
                    ],
                )
            else:
                self._http.send_headers(
                    stream_id=event.stream_id,
                    headers=[(b":status", b"404")],
                    end_stream=True,
                )
            self.transmit()
        elif isinstance(event, WebTransportStreamDataReceived):
            self._buf += event.data
            self._drain(event.stream_id)
            self.transmit()

    def _drain(self, stream_id):
        # One message is a u32 LE length then that many bytes. Anything shorter
        # stays buffered: QUIC delivers a byte stream, not messages.
        while len(self._buf) >= 4:
            (length,) = struct.unpack_from("<I", self._buf, 0)
            if len(self._buf) < 4 + length:
                return
            message = bytes(self._buf[: 4 + length])
            del self._buf[: 4 + length]
            self._seen += 1
            if self._seen == 1:
                print(f"caps message {length} bytes", flush=True)
                continue
            self._quic.send_stream_data(stream_id, message)
            self._echoed += 1
            print(f"echoed {self._echoed}", flush=True)


async def main(cert, key, port, ready):
    global DONE
    DONE = asyncio.Event()
    config = QuicConfiguration(
        is_client=False,
        alpn_protocols=H3_ALPN,
        max_datagram_frame_size=65536,
    )
    config.load_cert_chain(cert, key)
    await serve("127.0.0.1", port, configuration=config, create_protocol=WirePeer)
    with open(ready, "w") as f:
        f.write("ready\n")
    print(f"listening on {port}", flush=True)
    try:
        await asyncio.wait_for(DONE.wait(), timeout=WATCHDOG_SECONDS)
    except asyncio.TimeoutError:
        print("watchdog expired", flush=True)


if __name__ == "__main__":
    cert, key, port, ready = sys.argv[1:5]
    try:
        asyncio.run(main(cert, key, int(port), ready))
    except KeyboardInterrupt:
        pass
