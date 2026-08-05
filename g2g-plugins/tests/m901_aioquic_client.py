"""Independent WebTransport client for the M901 tests: an aioquic (Python) client
that pushes a prepared wire stream at our `RemoteWtSrc`.

It performs its own QUIC handshake, HTTP-3 SETTINGS exchange and CONNECT
(`:protocol = webtransport`), opens one bidirectional WebTransport stream, and
writes the file it was given verbatim. The file holds the g2g wire codec, already
length-framed by the Rust side (the codec is g2g's; the layers under it are what
this validates), so a passing run means our server accepts a foreign client's
session and reads its stream boundaries correctly.

usage: client.py <port> <stream-file>
"""

import asyncio
import ssl
import sys

from aioquic.asyncio import QuicConnectionProtocol, connect
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import HeadersReceived
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ProtocolNegotiated


class WtClient(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._http = None
        self._session_id = None
        self.accepted = asyncio.Event()

    def quic_event_received(self, event):
        if isinstance(event, ProtocolNegotiated):
            self._http = H3Connection(self._quic, enable_webtransport=True)
        if self._http is not None:
            for h3_event in self._http.handle_event(event):
                if isinstance(h3_event, HeadersReceived):
                    status = dict(h3_event.headers).get(b":status")
                    print(f"connect status {status.decode()}", flush=True)
                    if status == b"200":
                        self.accepted.set()

    async def open_session(self, authority):
        self._session_id = self._quic.get_next_available_stream_id()
        self._http.send_headers(
            stream_id=self._session_id,
            headers=[
                (b":method", b"CONNECT"),
                (b":scheme", b"https"),
                (b":authority", authority.encode()),
                (b":path", b"/g2g"),
                (b":protocol", b"webtransport"),
                (b"origin", b"https://" + authority.encode()),
            ],
        )
        self.transmit()
        await asyncio.wait_for(self.accepted.wait(), timeout=20)

    def push(self, blob):
        stream_id = self._http.create_webtransport_stream(self._session_id)
        self._quic.send_stream_data(stream_id, blob, end_stream=True)
        self.transmit()


async def main(port, blob_path):
    config = QuicConfiguration(
        is_client=True,
        alpn_protocols=H3_ALPN,
        max_datagram_frame_size=65536,
    )
    # The server's certificate is a throwaway self-signed one; this peer is
    # validating the transport, not the PKI.
    config.verify_mode = ssl.CERT_NONE
    async with connect(
        "127.0.0.1", port, configuration=config, create_protocol=WtClient
    ) as client:
        await client.open_session(f"127.0.0.1:{port}")
        with open(blob_path, "rb") as f:
            blob = f.read()
        client.push(blob)
        print(f"pushed {len(blob)} bytes", flush=True)
        # Hold the connection open long enough for the server to drain the
        # stream: closing here would tear down the QUIC connection under it.
        await asyncio.sleep(3)
    print("done", flush=True)


if __name__ == "__main__":
    port, blob_path = sys.argv[1:3]
    asyncio.run(main(int(port), blob_path))
