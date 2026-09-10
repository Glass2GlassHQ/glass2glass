# Transports

Network and cross-process carriers: the native WebRTC stack, the
distributed-graph elements that cut a graph edge across a process or machine
boundary, MoQ Transport, SMPTE ST 2110 media transport, and zero-copy local IPC
for GPU memory.
Part of the design in [DESIGN.md](DESIGN.md).

## WebRTC

The WebRTC elements are built on [str0m](https://github.com/algesten/str0m), a
sans-IO WebRTC stack implementing ICE, DTLS, SRTP and RTP as a pure state
machine. g2g owns the `UdpSocket` and the timer and drives str0m's `poll_output`
/ `handle_input` loop, the contract the `srt` and `rtspserver` modules follow.
str0m's pure-Rust `rust-crypto` backend is selected, so there is no OpenSSL or
libnice dependency. Everything is behind the opt-in `webrtc` feature, off by
default, so the no_std baseline is unaffected. This is the native, server-grade
counterpart of the browser-only data-channel `WebRtcSrc` (the browser sandbox in
[DESIGN.md](DESIGN.md)).

### Element family

One PeerConnection carries one track per element or N tracks in a session
element. The trait an element implements selects the shape, and each maps to a
terminal runner from the fan-in / fan-out family
([DESIGN-caps.md](DESIGN-caps.md), fan-out and fan-in).

| Element | Tracks | Direction | Trait | Runner |
| :--- | :--- | :--- | :--- | :--- |
| `WebRtcSink` | 1 | send (WHIP) | `AsyncElement` (sink) | linear |
| `WebRtcWhepSrc` | 1 | recv (WHEP) | `SourceLoop` | linear |
| `WebRtcSessionSink` | N | send (WHIP) | `MultiInputElement` | `run_fanin_session` |
| `WebRtcWhepSessionSrc` | N | recv (WHEP) | `MultiOutputSource` | `run_fanout_session` |
| `WebRtcDuplexSession` | N | sendrecv | `MultiDuplexSession` | `run_duplex_session` |

The one-track sink and source keep the `Rtc` on a spawned task and hand access
units over a bounded channel, so the element never touches the `Rtc` and stays
`Send`. The session sink is a terminal `MultiInputElement`, the network being
the destination, and `run_fanin_session` fans N sources into it over one tagged
`(input, packet)` channel. The session source is the mirror: a terminal
`MultiOutputSource`, 0 inputs and N outputs, driven by `run_fanout_session`.

### Simulcast

Send-side simulcast is in `webrtc_simulcast.rs`, shared by `LiveKitSink` and
`WebRtcSessionSink`: `SimulcastPads` (the video-layer plus audio pad model, pad
0 = highest resolution), rid assignment `f` / `h` / `q` high to low, the
one-m-line `a=rid` / `a=simulcast` offer, per-`(mid,rid)` `KeyframeRoutes`, and
the BWE `LayerAllocator`, which switches whole layers on and off with time
hysteresis and holds per-layer targets. The LiveKit path is browser-validated
end to end. The WHIP path is validated against Broadcast Box, the reference peer
for client-simulcast ingest, since mediamtx cannot ingest it and LiveKit's WHIP
ingress transcodes one layer. A three-layer publish appears server-side as three
rids with independently growing packet counts.

### Session sources as graph nodes

`NodeKind::FanoutSrc(n)`, added by `Graph::add_fanout_src`, runs a terminal
`MultiOutputSource` as a graph node with 0 inputs and N outputs it generates
itself. Its ports are solved from `output_caps`, the demux constraint shape with
the input half inert, and its arm runs the element into per-edge senders. The
element owes every port an `Eos`. `FanoutSrcFactory` plus named output pads make
the node reachable from `parse_launch`
(`livekitsrc name=s url=...  s. ! ...  s. ! ...`), and `MultiOutputSource`
carries a properties surface for the launch settings. Validated live with
`LiveKitSrc` as a graph node subscribing on a real server.

### Session sinks as graph nodes

A terminal fan-in is also a graph node, `NodeKind::FaninSink(n)` via
`Graph::add_fanin_sink`, so a transform chain can feed each session pad instead
of a bare source: the live encoder fan graph
`src -> tee -> videoscale -> ffmpegenc` per simulcast layer ends on
`LiveKitSink` inside one `run_graph`, cooperative or threaded. The node reuses
the muxer's `GraphNodeRef::Muxer` payload and negotiation shape with the output
half inert, no output edge existing. Its arm is the `run_fanin_session`
discipline over the DAG's per-edge channels: round-robin drain, per-input `Eos`
flush, end on all-`Eos`, and each pad's `reverse_channel` relayed onto its own
in-edge, so a per-`(mid,rid)` PLI reaches exactly the encoder feeding that layer
as a `PushOutcome::Reconfigure(ForceKeyframe)`.
`MultiInputElement::is_terminal` marks a session element as legal to end a
graph. In `parse_launch` a fan-in name with nothing downstream builds this node,
while a merging muxer without a downstream stays a `MuxerWithoutOutput` parse
error, its merged output otherwise being silently discarded.

### Duplex sessions

Bidirectional sendrecv needs an element that is at once a sink for the tracks it
publishes and a source for the tracks it receives over one connection, which
neither session runner can express. `MultiDuplexSession` is that union: N send
inputs and M recv outputs, driven by `run_duplex_session`. A single
`run(inbound, out)` owns the connection and `select`s over the inbound send
packets (`DuplexInbound`) and the network, pushing received frames to `out`. The
send and recv halves share `&mut self` directly with no detached task, unlike
the send-only session, which spawns the `Rtc` onto its own task to avoid
`process` and run-loop aliasing.

### Growing the pad count live

`run_duplex_session_dynamic` is the renegotiating sibling: its arms run under a
`dynamic_join`, so pads are not fixed at build time. A local track enters
through `DynamicDuplexHandle::add_send_track`, which reserves the index and
enqueues it under one lock, so the session learns pads in order. The runner
fixates the source alone and announces its caps on the new index before any
frame, which is how a session that never declared the pad learns it exists, and
`DuplexInbound::reverse_channel` hands it the PLI / BWE route back.

A remote track with no free pad is taken by the session calling
`MultiOutputSink::add_port`, default `None` so fixed runners refuse growth. The
runner mints the port's link and asks an app-supplied sink factory for the
element that drains it, a factory of `None` leaving the port counted as drops.
Backpressure on the runner's internal add channels delays the attach rather than
failing the run.

### Signaling

WHIP for egress and WHEP for ingress are the same wire move: an
`application/sdp` POST of the local offer that returns the remote answer, over
reqwest in `webrtc_util::post_sdp`. The media server is the relay in the middle,
so there is no peer-to-peer mode for WHIP/WHEP.

WHIP and WHEP are unidirectional by spec, so sendrecv cannot use them. The
duplex session exchanges SDP directly between two peers over an `SdpChannel`, an
in-process offer/answer transport for a P2P loopback, and a real SFU signaller
such as LiveKit plugs into the same seam. For mid-session renegotiation a
cloneable `DuplexControl` toggles a track, batching direction changes between
SendRecv and Inactive into one re-offer over the `SdpChannel`, and the peer
answers it in its loop. Typed `offer\n` / `answer\n` prefixes distinguish the
exchange, and on glare the answerer role yields.

### Spare pads for mid-session tracks

The fixed-arity pad model has no pad to grow into, so a session reserves them up
front. `with_spare_tracks(video, audio)` appends declared-but-inactive pads
after the active ones, and they carry no m-line at the handshake. Each
negotiated m-line is a binding holding its `Mid`, its kind, and the input and
output pads it serves. Recv routing, PLI, BWE and the direction toggles all
resolve through the binding.

A spare binds two ways. Its send pad gets its first frame, and the session
offers the peer a new sendrecv m-line, one exchange at a time, the frame itself
dropped like any frame before its m-line exists. Or the peer's re-offer lands
and `MediaAdded` fires for an unknown mid, which claims the first free pad of
that kind on both sides. A pad bound mid-session emits its `CapsChanged` before
its first frame, and the active pads are announced at session start. A track
whose kind has no free pad left is rejected, staying unbound with its media
skipped, because the pad count is fixed at graph build time.

`DuplexControl::remove_track` is the inverse. It calls `stop_media` on the
m-line, setting port 0 and taking it out of the BUNDLE group, batched into the
same re-offer as the direction toggles. Both peers then drop that binding by
walking their media after each SDP application. A stopped m-line stays in the
session with `Media::stopped()` set, which also retracts an add that lost a
glare race. Dropping the binding frees its pads with no `Eos` on the output pad,
since a later track may claim it and the end of the run EOSes every pad anyway.
Reuse always negotiates a new m-line, a stopped one cannot be reactivated, and
the freed output pad re-announces its caps before the new track's first frame.

The two roles discover their m-line `Mid`s differently and the asymmetry
matters: the offerer captures its `Mid`s from the return of
`SdpApi::add_media`, while the answerer learns them from `Event::MediaAdded`
after `accept_offer`, because str0m does not emit `MediaAdded` for media the
local side added.

### LiveKit signaller

`livekit_signal` is the protocol seam: an HS256 JWT access-token mint and a
hand-rolled protobuf codec for the `livekit_rtc.proto` subset, over a
tokio-tungstenite WebSocket, `ws://` and, via native-tls, `wss://`, the TLS
stack the WHIP reqwest client already links.

`LiveKitSink` publishes, the client offering as in WHIP. `LiveKitSrc`
subscribes, and there the offer direction reverses: the SFU offers the
subscriber PeerConnection over the signalling socket and re-offers on every
track-set change, and the element answers each with `accept_offer`, learning its
mids from `MediaAdded` per the answerer rule above. The source is a terminal
`MultiOutputSource` with video and audio ports run by `run_fanout_session`,
gates video until the first keyframe, repeats a PLI until one arrives, and takes
the first video and audio m-line offered, making it a one-subscription element.
Both are validated against a real LiveKit server, including an in-room
sink-to-src A/V loopback and the same loopback over a TLS-terminated `wss://`
proxy.

`LiveKitDuplex` is the full participant. LiveKit has no sendrecv m-lines, so it
runs both PeerConnections, a publisher it offers and a subscriber the server
offers, in one loop over one socket, routing trickle by `SignalTarget`, exposed
as a `MultiDuplexSession` for the duplex runner. Two participants exchanging A/V
are validated live.

### ICE and NAT traversal

`webrtc_util::add_ice_candidates` always adds the socket's host candidate and,
when a STUN server is configured, a server-reflexive candidate discovered by a
hand-rolled RFC 5389 Binding on the ICE socket. Candidates ride in the SDP, so a
same-host P2P pair connects over localhost with no STUN.

Where a reflexive candidate cannot punch through, a hand-rolled TURN client in
`turn.rs` provides a relay: RFC 5766/8656 Allocate with long-term auth, channel
binding, and periodic Refresh. str0m only offers `Candidate::relayed` and leaves
the data plane to the run loop. A relayed pair's transmits all carry
`source == relay_addr`, the routing signal to wrap the datagram for the relay,
and direct host and srflx paths are untouched. The first transmit to a new peer
sends a ChannelBind, which installs the peer permission and, once its success
lands, upgrades that peer from 36-byte Send / Data indications to 4-byte-header
ChannelData frames both ways. A `438 Stale Nonce` on any authenticated request
adopts the error response's nonce and un-caches the affected state so the lazy
paths retry with it.

The client-to-server leg also runs over TCP and TLS, the
`turn:...?transport=tcp` and `turns:` RFC 7065 forms. A local bridge task
tunnels the client's datagrams over one stream connection, re-delimiting
messages, STUN being self-describing by length and ChannelData padded to 4 bytes
on the stream, so `TurnClient` and every element run loop stay
transport-agnostic and the allocation still relays UDP toward peers. The codec
is address-family agnostic: XOR addresses encode and decode IPv6 with the cookie
plus transaction id, and a v6-bound client requests a v6 relayed address per RFC
6156. Validated against a real coturn on all three transports and over IPv6,
covering allocate, bind, and a ChannelData round trip both directions.

An element takes a comma-separated server list, each entry optionally carrying
GStreamer-style `turn://user:pass@host` credentials. A `TurnSet` allocates on
every server and contributes one relayed candidate each, and the data plane
routes by which relay a transmit's `source` names. The duplex session has the
same STUN/TURN surface, its relayed candidates riding in the offer/answer SDP
with no trickle channel.

### RTCP feedback

RTCP feedback rides the reverse channel of [DESIGN-caps.md](DESIGN-caps.md)
[DESIGN-caps.md](DESIGN-caps.md). A remote PLI (`Event::KeyframeRequest`) becomes a
`Reconfigure::ForceKeyframe` walked upstream via
`AsyncElement::take_reconfigure` to the encoder, where `Av1Enc` forces a rav1e
IDR. Ingress originates a PLI on a mid-GOP join. str0m's BWE
(`Event::EgressBitrateEstimate`, TWCC/REMB) becomes `PushOutcome::Bitrate` via
`take_bitrate`, and the encoder retargets, rav1e by a hysteresis-gated context
rebuild. `OpusEnc` retargets live through `OPUS_SET_BITRATE` with no rebuild,
and `FfmpegH264Enc` by a hysteresis-gated reopen, zerolatency with nothing in
flight.

Both signals hop past intervening transforms. An element that does not consume
them has its output adapter relay the pending `ForceKeyframe` or bitrate onto
its input link, so `enc ! h264parse ! webrtc-sink` reaches the encoder.
Consumption is declared by `AsyncElement::handles_keyframe_requests` and
`handles_bitrate_requests`, false by default and overridden by encoders.
`Propose` and `Renegotiate` never relay, concerning the adjacent element's own
caps.

Simulcast splits the aggregate BWE estimate per layer. The allocator hands each
active layer its nominal share on that layer's reverse channel, and a shed layer
gets the `Bitrate(0)` idle hint, on which the encoder skips frames unencoded
except a sparse 1-in-32 keep-alive. The resume signal rides push outcomes, so
the cadence must not fully stop, and resume forces an IDR. The allocator is
re-ticked with the last estimate once a second, because BWE only emits deltas
and retargeted encoders settle exactly on the estimate, which would otherwise
freeze the drop and restore hysteresis windows.

### Codec plumbing

A `Track` enum unifies the per-track facts WebRTC must agree on: the codec
(H.264 or Opus), the m-line `MediaKind`, and the RTP clock, 90 kHz or 48 kHz.
`media_time` maps a nanosecond PTS onto the track's RTP timestamp.

H.264 crosses the boundary as Annex-B, the pipeline convention of
[DESIGN-decode.md](DESIGN-decode.md): str0m's packetizer splits NAL units and
its depayloader emits start-code framing. A receive-side video element
advertises a `Dim::Range` / `Rate::Range` placeholder rather than `Dim::Any`,
because geometry is only known from the in-band SPS and `fixate()`
([DESIGN-caps.md](DESIGN-caps.md)) rejects `Any` at negotiation. A
downstream `H264Parse` recovers the real dimensions.

### Validation status

On-network validated against a local mediamtx, single-track WHIP/WHEP and
multi-track A/V, and by in-process P2P loopbacks on localhost, video and full
A/V sendrecv. Structurally the stack covers one connection with N tracks,
BUNDLE, sendrecv, PLI and BWE, which is `webrtcbin` parity. What remains is
maturity rather than architecture, and `DESIGN_TODO.md`'s "WebRTC" item carries
the tiered list.

## Distributed graphs

A graph is normally one process, but a pipeline stage is not bound to the
machine that produced its input. The distributed-graph primitive lets any edge
be cut so the downstream subgraph runs in another process or on another machine,
without rewriting the graph: replace the edge with
`... ! remotesink host=H port=P` on the near side and `remotesrc port=P ! ...`
on the far side. This is the general form of a browser-to-server offload, the
same "move a stage across a boundary by swapping one element" thesis as the
portability story, applied to the network axis rather than the target axis.

### Wire codec

The foundation is a target-agnostic wire codec in `g2g-core` (`wire.rs`,
`no_std + alloc`, no dependency). `encode_packet` and `decode_packet` serialize
an entire `PipelinePacket` to a self-contained, versioned, little-endian byte
buffer and back, covering every variant, every `Caps` shape, the frame timing
and sequence, and, with the `metadata` feature, the `AnalyticsMeta` detection
graph and `BlobMeta` side-data in band. Because it is pure computation it
compiles on every target the core does, `wasm32` included, so a browser client
and a native peer speak the identical format.

Only CPU memory crosses the wire: `MemoryDomain::System` bytes verbatim,
`SystemView` materialized to dense bytes. A device-resident domain, CUDA, wgpu,
D3D11 or DMABUF, returns `WireError::UnsupportedDomain`, so a GPU frame must
pass an explicit download element first, exactly as reaching any CPU sink
already requires.

### TCP carrier

`RemoteSink` (the `remote` feature, a `std + tokio` element pair) is the TCP
client. It accepts any caps (`caps_constraint_as_sink` = `AcceptsAny`), connects
in `configure_pipeline`, and forwards each packet length-framed as a `u32`
length then the wire body, emitting the negotiated caps as the first packet so
the receiver learns the media type from the stream.

`RemoteSrc` is the TCP listener. It accepts one connection and discovers its
output caps from that first `CapsChanged`, the async caps-discovery pattern
`RtspSrc` uses, then re-emits the leading caps and every subsequent packet
downstream, ending on the sender's `Eos` or a clean close. A `metadata`-off
receiver ignores a `metadata`-on sender's meta payload, the last field of a
`DataFrame` body, rather than mis-parsing it, so a mixed-feature deployment
degrades to no metadata and never to corruption.

### WebSocket carrier

`RemoteWsSink` / `RemoteWsSrc` (the `remote-ws` feature) carry the identical
wire-codec stream over a WebSocket connection via `tokio-tungstenite`. WebSocket
is already message-framed, so one `encode_packet` body is one binary WebSocket
message with no `u32` length prefix. The protocol is otherwise identical, caps
as the first message, discovered by the server in `intercept_caps`.
`RemoteWsSink` is the client and `RemoteWsSrc` the listening server, matching
the TCP roles. The one behavioural difference is that the WebSocket handshake is
async, so the sink connects on its first `process` rather than in
`configure_pipeline`.

This carrier exists for reach: a browser peer can speak only WebSocket, so it is
the transport that lets a `g2g-web` graph join the primitive. `WsWireSink`
(`g2g-plugins`, the `web` feature) is the wasm send half, wrapping the browser
`WebSocket` API around the same `encode_packet`, so a browser graph
`... -> WsWireSink` ships an edge to a native `RemoteWsSrc -> ...`. Because the
wire codec compiles unchanged on `wasm32`, the browser and the native server
share the serializer.

### Remote transform

A one-way edge cut runs the whole downstream subgraph remotely, but some stages
must stay put around the offloaded one: a browser detection offload can move
only inference, decode and the overlay plus canvas present being browser-bound.
A remote transform ships each input packet to a peer over one WebSocket and
emits the processed packet the peer returns, keeping the graph shape. Caps are
identity. The remote stage may attach `metadata` such as `AnalyticsMeta`
detections, which crosses in band, but it does not change the format.

The protocol is strictly FIFO so each per-frame read pairs with its frame: the
leading `CapsChanged` is config and gets no reply, then one `DataFrame` per
frame with one processed reply each, then `Eos`. `Segment` and `Flush` pass
through locally. The native `RemoteWsTransform` is a tokio-tungstenite client
that offloads a middle stage to another machine, and the browser
`WsWireTransform` is its wasm twin.

One generic, detection-agnostic element covers what a hand-rolled RGBA-up /
boxes-down protocol would. The browser graph
`WebSocketSrc -> WebCodecsDecode -> WsWireTransform -> AnalyticsOverlay ->
CanvasSink` moves inference to a native peer running the real
`OrtInference -> DetectionPostprocess` chain, attaching the boxes as
`AnalyticsMeta`. The tradeoff against a bespoke protocol is bandwidth: the
transform round-trips the whole frame both ways, the honest cost of a generic
packet-in / packet-out stage, which is fine on a LAN.

### WebTransport carrier

`RemoteWtSink` / `RemoteWtSrc` / `RemoteWtTransform` (the `webtransport`
feature) are the third carrier of the same wire codec, over one reliable
bidirectional WebTransport stream per connection, HTTP/3 CONNECT over QUIC via
`web-transport-quinn`. A WebTransport stream is a QUIC stream, a byte stream
with no message boundaries, so the framing is the TCP pair's `u32` length
prefix, shared verbatim, not the WebSocket pair's one message per packet. The
protocol above that is identical, including the transform's FIFO frame-out /
processed-frame-back round trip.

The QUIC connection adds per-stream head-of-line blocking, a 1-RTT handshake,
and a browser peer that reaches it with `new WebTransport(url)` and no
TLS-terminating proxy in front. QUIC is always TLS, so unlike the other two
servers this one cannot start without a `certificate` / `private-key` PEM pair,
and a client that will not trust a system root names the certificate by SHA-256
digest in `server-certificate-hashes`, the browser API's
`serverCertificateHashes`.

### Shared carrier machinery

The three carriers share their machinery rather than repeating it. `RemoteClient`
on the send side and `RemoteSource` on the receive side are generic over a
transport, and `RemoteTransform<T>` is generic over a `PacketDuplex` transport.
Each carrier file supplies only what is transport-specific, how a connection is
dialed or a listener bound and how one packet is written and read, plus its
element identity and properties.

### Reconnection

`RemoteSink` / `RemoteWsSink` / `RemoteWtSink` take `with_reconnect(attempts)`
and a `reconnect-attempts` property. The initial connect is deferred and retried
with a short backoff, and a mid-stream send failure drops the dead socket,
reconnects, and re-sends the current caps, the far side's required first packet,
before retrying, so a peer that starts late or restarts is tolerated up to the
attempt budget.

`RemoteSrc` / `RemoteWsSrc` / `RemoteWtSrc` take `with_reconnect()` and a
`keep-listening` property. A client that drops without a clean `Eos` is not the
stream's end: the source keeps its listener open and accepts a replacement
client, which re-sends its leading caps to be forwarded downstream so it
re-negotiates if changed. Only an explicit `Eos` or a frame limit ends a
keep-listening source. Both directions are validated over loopback, the sink
retrying until a late-binding server appears and the source stitching a stream
across a sender that drops and is replaced.

## MoQ transport

The distributed-graph carriers move g2g's own packet stream between g2g peers.
MoQ Transport is the other use of the same WebTransport carrier: a published
IETF media protocol, so the far peer is a relay and a player that know nothing
about g2g. `moqt` implements it in-tree, both directions.

### Dialect and version

The dialect is the IETF draft, not moq-lite, which is a single-vendor dialect
with its own ALPN that cannot talk to IETF endpoints. The versions are draft-16,
`0xff000010`, which Cloudflare's `moq-relay-ietf` runs in production, and
draft-18, which moq-dev, imquic, moqxr and Meta's public moxygen relay speak.
Nothing on crates.io implements the IETF draft, so the wire layer is written
here the way the SRT and ST 2110 stacks were: read the draft, read the reference
implementation `cloudflare/moq-rs`, and validate against the reference peer.

From draft-16 the version is not negotiated in the SETUP payload. The QUIC ALPN
for WebTransport is always `h3`, so the version rides the HTTP/3 CONNECT request
as the WebTransport subprotocol `moqt-16`, and CLIENT_SETUP / SERVER_SETUP carry
parameters only.

The elements offer every version in their `versions` property, default `18,16`
in preference order, as WebTransport subprotocols on one CONNECT, and the
server's pick selects the codec for the session. `moq-relay-ietf` echoes
`moqt-16` when offered it. A server that echoes no subprotocol predates
multi-version offers and every such server is a draft-16 peer, so the fallback
is draft-16 when it was offered. The SETUP handshake that follows validates the
choice either way.

### Draft-18

Between draft-16 and draft-18 the wire was restructured, so `moqt::v18` is a
sibling module rather than a flag on the draft-16 one. Its differences:

- its own `vi64` integer, a leading-ones length prefix of 1 to 9 bytes holding a
  full `u64`, where non-minimal encodings are legal.
- a single SETUP message `0x2F00` on a pair of unidirectional control streams.
- one bidirectional stream per request, whose response carries no request id,
  the stream being the correlation.
- typed control-message parameters in place of KVP parameters, where an unknown
  type cannot be skipped and is a session error.
- bit-table SUBGROUP_HEADER and OBJECT_DATAGRAM types, plus PADDING streams and
  datagrams to discard.
- cancellation by stream reset. Draft-18 has no UNSUBSCRIBE, FETCH_CANCEL or
  MAX_REQUEST_ID.

What is version-agnostic is shared, not copied: track namespaces,
Key-Value-Pairs and the object-id delta rule are in the draft-16 coding module
with a varint flavour on the shared `Reader`, and the reorder policy, catalog
and WebTransport carrier are reused as they are. The publisher answers each
request on its own stream and sends PUBLISH_DONE there at EOS. The subscriber
drains PUBLISH_DONE's stream count before ending a subscription, because the
message races the data streams it is counting. FETCH is refused with
`NOT_SUPPORTED` on both drafts.

### Layering

`moqt::coding` holds varints, byte strings, track namespaces and names, and the
delta-coded Key-Value-Pair sequences. `moqt::message` holds the control message
set and its `type / 16-bit length / payload` framing. `moqt::data` holds the
subgroup stream header and per-object header, `moqt::datagram` the datagram
object, `moqt::reassembly` the decoding of a subgroup stream plus the ordering
policy below, and `moqt::catalog` the JSON track list, written and read in one
place so the two cannot drift. All are pure `alloc` with no I/O, so the wire
layer is unit-testable on byte vectors. The layouts are asserted against the
byte sequences the reference implementation asserts for itself, because a round
trip alone cannot catch two fields swapped with each other.

`moqt::session` adds the live session over the WebTransport carrier. It reuses
that carrier's dial, `remotewtio::dial`, which takes a subprotocol argument so
the certificate-hash handling exists in one place, opens the control stream as
the session's first bidirectional stream, and runs the control stream's read
half in its own task, so a SUBSCRIBE is decoded as it arrives instead of when
the element next has a frame to push. A subscriber also starts a data reader:
one task accepting unidirectional streams, one task per stream decoding it, all
funnelling whole objects into a single channel.

Everything a peer sends is bounded before use. Counts and lengths are checked
against the draft's limits, the KVP running key is a checked add, nothing is
preallocated from a peer-supplied count, a single object is capped by
`max-object-size` so one stream cannot allocate without limit, and a message
that does not consume exactly its declared length is a protocol violation.

### moqtsink

The publisher takes an ISO-BMFF byte stream
(`... ! mp4mux ! moqtsink location=https://relay:4443/ namespace=live/cam`), so
the muxer stays a separate element and the sink carries no second fragmenter. It
walks the boxes with the same helpers the HLS segmenter uses, and
`fmp4::trun_first_sample_is_sync` is shared between them. The object mapping:

- `ftyp`+`moov` is one object in group 0 on the init track (`0.mp4`), which is
  what a subscriber fetches first.
- each `moof`+`mdat` pair, with the `styp` / `prft` that open its segment, is one
  object on the media track `{track_id}.m4s`. CMAF requires an object to hold at
  least one whole chunk, and a `moof`+`mdat` pair is exactly that.
- a fragment whose first sample is a sync sample starts a new group, so a
  group is a GOP and each group is one subgroup on its own unidirectional
  stream. A subscriber that joins mid-group is served from the next keyframe,
  which is the only point it could start decoding anyway.
- a `.catalog` track carries the JSON track list a player reads to learn the
  track names and codec parameters.

Subgroup streams carry the header type `0x15`, an explicit subgroup id plus an
extension-header block, and objects whose id delta is measured per stream, the
first object of a stream taking the delta as its absolute id. That is
byte-identical to what the reference publisher emits when one stream carries the
whole group. `subgroups` spreads a group's objects across that many concurrent
subgroup streams round-robin, so one object's loss does not hold up the next
inside a GOP. The reference relay cannot carry that: it renumbers each subgroup's
objects from zero, discarding the delta, so the subgroups collide on one set of
ids and all but one are dropped as duplicates. A publisher aimed at that relay
leaves `subgroups` at 1.

`SUBSCRIBE_OK` reuses the request id as the track alias, §10.1 asking only for
uniqueness within the session, and a stream is opened only after that
acknowledgement, so the subscriber can resolve the alias in the stream header. A
subscriber-side request the publisher does not serve, FETCH, TRACK_STATUS or
REQUEST_UPDATE, gets an explicit `REQUEST_ERROR NOT_SUPPORTED` rather than
silence. Draft-16 SUBSCRIBE carries neither a group order nor a filter field, so
`priority`, the publisher-priority byte in every subgroup header, is the only
delivery setting and there is no group-order property to expose.

The control plane runs without frames. `configure_pipeline` dials the relay and
publishes the namespace in the background, a sync caller without a runtime
falling back to dialling on the first frame, and a failed dial is retried per
frame while the relay comes up. A pump task owns the control stream's inbound
half, answering each message as it lands, and the pump and frame publishing take
turns on one lock, so a subscription never changes hands inside a frame. A media
SUBSCRIBE that arrives before the `moov` names any track is held in a bounded
queue, request ids being peer-controlled, and resolved with `SUBSCRIBE_OK` or
`DOES_NOT_EXIST` once the `moov` arrives. Init and catalog subscriptions are
served the moment their single object exists.

### Datagram objects

`datagrams=true` carries each media object in a QUIC datagram instead of on a
subgroup stream: unreliable, bounded by the path MTU, and free of head-of-line
blocking, which is what a live path wants for droppable media. It is off by
default because it changes the delivery guarantee.

The layout is in `moqt::datagram`, from
`moq-transport/src/data/datagram.rs`, a type table like the stream header saying
which of the object id, the extension block and the object status are present,
whether a payload follows, and whether the object ends its group. The payload
has no length prefix, since the datagram boundary ends it. A datagram is a whole
message that will never be continued, so a short one is a protocol violation
rather than something to wait for, and a datagram that does not decode is
dropped rather than failing the session: an unreliable carriage already loses
objects, and killing the session over one bad datagram would lose the rest.

Three consequences shape the publisher. An object larger than the path MTU
cannot be a datagram, so it falls back to a subgroup stream rather than being
dropped, which the delta coding already handles, a stream opened mid-group
starting from an absolute object id. The init and catalog tracks always ride
streams, because losing either loses the whole broadcast. And a group carried
only by datagrams has no stream whose close says it is finished, so the
publisher sends an end-of-group status datagram at each group boundary, without
which the subscriber would hold the group until a buffering bound moved it on.
The subscriber feeds datagram objects into the same `Reassembler` as stream
objects, so ordering, the bounds and the never-stall policy are one
implementation and not two.

### moqtsrc

The subscriber is the inverse and emits a `ByteStream{IsoBmff}` a demuxer takes
unchanged (`moqtsrc location=... namespace=live/cam ! fmp4demux ! ...`). It
reads the `.catalog` track to learn the media tracks and their init track,
falling back to the reference defaults `0.mp4` and `{track_id}.m4s` when the
publisher publishes none, emits the init object as the first frame, then each
media object as it comes into order. `track-name` picks a track other than the
catalog's first.

### Reassembly

A subgroup is its own unidirectional QUIC stream, so a track's streams arrive
concurrently and its objects are ordered by group id then object id across
streams, not by arrival. Object ids are delta-coded per stream: the first
object's delta is its absolute id, and each later one is `previous + delta + 1`.
`Reassembler` holds a cursor and a fixed memory budget.

- Playback starts at the first group whose object 0 arrives, so joining
  mid-group skips to the next group rather than emitting a partial one.
- Objects are emitted strictly in cursor order, and anything below the cursor, a
  late stream or a duplicate, is dropped and counted.
- A group is done when every stream that carried it has finished, or an
  `EndOfGroup` object closed it. Then the cursor moves to the next group id.
- A hole in a group that is already done can never be filled, so the cursor
  jumps to the lowest object still buffered in it.
- A group that never completes is bounded, not waited on. Past `max-groups` or
  `max-buffer-bytes` the oldest group is dropped whole and the cursor moves past
  it, so buffering never grows and the stream resumes at the next group boundary
  instead of stalling. Objects for a group already left behind are refused
  rather than reordered backwards.

A data stream can outrun the control stream that names its track alias, so
events for an alias no subscription claims yet are held, under the same byte
budget, and replayed when SUBSCRIBE_OK arrives.

### Reference-peer validation

Validation is against the reference peers rather than a loopback, in both
directions. `moqtsink` publishes through a locally spawned `moq-relay-ietf` and
the bytes `moq-sub` writes on the far side are compared to the bytes that went
in. `moq-pub` publishes through the same relay into `moqtsrc`. And the g2g round
trip `mp4mux ! moqtsink` to relay to `moqtsrc` is compared byte for byte, over a
run long enough to span a group boundary.

The datagram and multi-subgroup paths cannot be validated that way, because
`moq-relay-ietf` has no datagram code at all and drops all but one subgroup of a
group. They are validated `moqtsink` to `moqtsrc` directly over a real QUIC
connection, through a test peer that answers each side's CLIENT_SETUP and then
copies control messages, streams and datagrams byte for byte with no track state
of its own, so what the subscriber decodes is what the publisher encoded. The
datagram byte layouts are asserted against vectors the reference
implementation's own encoder produced. The relay is still exercised on those
settings for what it proves: that a subscriber keeps playing, in order and a
whole fragment at a time, when a peer delivers only part of every group.

### Browser leg

`tools/moqt-demo/` is the independent leg, since everything above validates
against Cloudflare's Rust or our own and a JS client is neither. The page
subscribes with [MOQtail](https://github.com/moqtail/moqtail), a third-party
draft-16 implementation, reads the `.catalog`, fetches the init track and
appends the `moof`+`mdat` objects to one MSE `SourceBuffer` unchanged, so the
browser's own demuxer and H.264 decoder consume the exact bytes `moqtsink`
wrote. `headless/run-moqt-play.mjs` drives it in headless Chromium against a
locally spawned relay and asserts on what the decoder produced: frame count,
decoded size, and the seven SMPTE bars sampled off a canvas. `watch-live.mjs` is
the same plumbing with `libcamerasrc` as the source and a real browser window.

Two constraints shape it. A browser accepts a self-signed relay only through
`serverCertificateHashes`, which requires an ECDSA P-256 certificate valid at
most 14 days, and a certificate that signs itself cannot be the leaf
(`CaUsedAsEndEntity`), so the harness mints a CA and a leaf under it. And the
catalog and init tracks each hold one object published before any subscriber
exists, so they must be subscribed with an absolute start at group 0. A
latest-object filter delivers nothing.

## Local zero-copy IPC

Every carrier above ships CPU bytes, the wire codec refusing device memory, so a
GPU producer feeding a GPU consumer in another process pays a full
device-to-host-to-device round trip to cross. On the same machine that copy is
avoidable, because two processes can map the same VRAM.

### CUDA IPC handles

`localipc` (the `local-ipc` feature, NVIDIA-only via the `cuda` gate) is the
CUDA path. `ipc_export` turns a `CUdeviceptr` into a 64-byte `CudaIpcHandle` via
`cuIpcGetMemHandle`, which another process passes to `ipc_open`
(`cuIpcOpenMemHandle`) to obtain a pointer to the same allocation, reading the
producer's VRAM with no copy.

The design point that makes this cheap is that a CUDA IPC handle is plain bytes,
unlike a DMABUF file descriptor, which needs `SCM_RIGHTS` fd-passing over a Unix
socket to be meaningful in another process. So the handle rides any byte
transport already in the tree, even the wire codec itself. The constraints: the
two ends share a machine and a GPU, a handle from device 0 being meaningless on
device 1, the exporting allocation stays live until the importer opens it, which
the producer frame's keep-alive covers, and the importer closes before the
exporter frees. The `cuda_ipc_smoke` example validates the whole path
cross-process on an RTX 3060: a parent fills a device allocation, exports the
handle, and spawns a child that maps and reads it back byte-for-byte, with only
the 64 bytes crossing between processes.

### LocalCudaSink and LocalCudaSrc

`LocalCudaSink` / `LocalCudaSrc` are the GPU-resident analog of `RemoteSink` /
`RemoteSrc` and carry a `MemoryDomain::Cuda` NV12 frame across a Unix socket.
The sink exports the frame's allocation and sends a descriptor holding the
handle, plane offsets, pitches, dims and timing. The source maps it, and here
makes the one pragmatic concession to lifetime: the producer's allocation must
stay valid until the consumer is done, and coupling two processes' whole
pipelines is fragile, so the source takes a single on-GPU device-to-device copy
(`cuMemcpyDtoD`, still no PCIe) into its own buffer and then acks. The sink
holds the source frame only until that ack, one frame in flight, so the two
lifetimes decouple cleanly and the design is independent of the runner's
frame-drop timing. The `local_cuda_transport` example validates the full element
path cross-process on an RTX 3060, NV12 frames verified pixel-exact in the
receiving process.

`LocalCudaSrc::zero_copy()` removes even that receive-side copy: the source
emits the producer's mapped VRAM directly, so the consumer reads the producer's
memory in place, NVENC-from-mapped with no copy anywhere. The lifetime handshake
the copy mode trades away is explicit here instead. The emitted frame's
keep-alive closes the IPC mapping and signals the run loop on drop, and the
source acks the producer only once that fires, meaning the frame is fully
consumed downstream, so the producer holds the source allocation exactly until
the consumer is done. That is real backpressure, one frame in flight across the
boundary with the producer stalling for the consumer, which is why it is opt-in.
The default copy mode decouples the two pipelines and suits a slow or fan-out
consumer, and `zero_copy()` suits a prompt, single-in-flight consumer where
eliminating the copy matters. Both modes are validated cross-process on the
3060, the example running either via `G2G_ZEROCOPY=1`.

### Vendor-neutral DMABUF transport

The GPU-agnostic counterpart is `DmaBufSink` / `DmaBufSrc`, the `local-dmabuf`
feature on Linux. A dma-buf is not plain bytes but a file descriptor, so the
byte-handle model does not apply. The sink passes the frame's dma-buf fd to the
source as `SCM_RIGHTS` ancillary data of a `sendmsg` over a Unix socket, using
hand-rolled `sendmsg` / `recvmsg` FFI in `scmfd` with no crate dependency, Linux
LP64 only, and the kernel installs a dup of the fd in the receiver.

That makes the transport both simpler and safer than the CUDA path. The
underlying buffer is kernel-refcounted across both processes' fds, so once the
sink's `sendmsg` returns the receiver's dup already keeps the buffer alive and
the sink may drop its frame immediately, with no per-frame ack. Backpressure
still comes from the graph's bounded channel upstream. Every message is a
fixed-size record sent and received with a single `sendmsg` / `recvmsg`, so a
frame record's fd is never separated from its bytes and a plain read never
crosses, and thus discards, a pending fd.

The transport is GPU-agnostic and carries any dma-buf: a GPU-exported texture, a
V4L2 or CSI capture buffer, a `dma_heap` or `udmabuf` allocation. Importing the
received fd into a wgpu buffer is the separate `DmaBufToWgpu` element
(`dmabuf-wgpu`) on the receive side. The `local_dmabuf_transport` example
validates the whole path cross-process with a genuine `udmabuf`, a CPU-mappable
dma-buf built from a sealed memfd, so it needs no GPU, and each frame's bytes
are mmap-verified in the receiving process.

### Exporting a GPU frame to a dma-buf

`WgpuToDmaBuf` (the `dmabuf-wgpu` feature) is the GPU producer that pairs with
`DmaBufToWgpu` across the boundary. It consumes a GPU-resident
`MemoryDomain::WgpuBuffer` and emits a `MemoryDomain::DmaBuf` referencing the
same pixels, so a rendered or decoded GPU frame leaves the process with no CPU
copy. Feed the output to `DmaBufSink`.

A wgpu-allocated buffer is not itself exportable, so the element allocates its
own Vulkan buffer backed by `VkExportMemoryAllocateInfo` with the dma-buf handle
type, copies the input into it on the GPU, and exports the memory as a dma-buf
fd with `vkGetMemoryFdKHR`. The exported fd is an independent reference to the
underlying buffer under dma-buf refcounting, so the element frees its own Vulkan
handles immediately and the fd keeps the memory alive, as does the receiver's
`SCM_RIGHTS` dup once `DmaBufSink` sends it. The input and the exportable buffer
must share one `wgpu::Device` for the copy, so a producer feeds this element on
its device, exposed via `gpu()` / `wrap_buffer`. By default the element waits
for the copy to finish with `device.poll(Wait)` before exporting, so a consumer
sees complete pixels, and `with_external_semaphore(true)` replaces that stall
with an exported timeline semaphore. Validated on the RTX 3060: a buffer
exported to a dma-buf and re-imported by `DmaBufToWgpu` on a separate wgpu
device reads back byte-exact (`m559_wgpu_dmabuf_export`), which also confirms
dma-buf export and import work on this NVIDIA driver.

Packed RGBA/BGRA/YUYV and 8-bit NV12 are supported. The plane-aware frame size,
where a packed format is one plane and NV12 / I420 add the half-height chroma
region, and the row stride are `RawVideoFormat::frame_bytes` and `row_stride`,
which both the export and the `DmaBufToWgpu` import use, so they always agree on
the buffer size.

The whole GPU-egress stack composes end to end across a process boundary:
`WgpuToDmaBuf -> DmaBufSink -> [process] -> DmaBufSrc -> DmaBufToWgpu` moves a
GPU-resident frame from one process to another with only a dma-buf fd crossing
by `SCM_RIGHTS` and no CPU copy on either side. The lifetimes compose too, the
export freeing its Vulkan handles at once and the sink's fd send keeping the
buffer alive via the receiver's dup. The `gpu_dmabuf_ipc` example proves this
cross-process on the 3060, frames re-imported GPU-resident in the child and
every pixel verified.

### Cross-device synchronisation

Cross-device and cross-process synchronisation has two modes. The default is the
producer-side `device.poll(Wait)`, a small stall that is correct because the
copy is fully flushed before the fd is handed off.

The zero-stall mode, `WgpuToDmaBuf::with_external_semaphore(true)`, moves the
wait to the consumer via an exported `VK_KHR_external_semaphore_fd` timeline
semaphore. The producer creates one exportable timeline semaphore per stream and
signals the next value on each frame's copy submit
(`wgpu_hal::vulkan::Queue::add_signal_semaphore`, no `poll(Wait)`), attaching
the semaphore fd and value to the emitted dma-buf in `OwnedDmaBuf`'s optional
`SyncFd` plus value slot. `DmaBufSink` ships the semaphore fd once as a
`TAG_SYNC` record over `SCM_RIGHTS`, ahead of the first synced frame, and tags
each frame with its timeline value. `DmaBufSrc` re-shares the one semaphore
across the frames it reconstructs. `DmaBufToWgpu` imports it once and, before
reading, waits for each frame's value by polling the timeline counter with
`vkGetSemaphoreCounterValue` and yielding cooperatively between polls rather
than blocking the runtime on `vkWaitSemaphores`. The common case, where the copy
is already done at arrival, passes on the first poll.

A timeline semaphore rather than a per-frame binary one keeps the fd single and
the wait a plain host wait, so no multi-fd ancillary passing or per-frame
semaphore churn is needed. The producer reclaims its exportable copy buffers
lazily once the timeline counter passes their value, using a non-blocking
`vkGetSemaphoreCounterValue`, so it never stalls yet never frees a buffer whose
copy is still in flight. wgpu-hal 29 exposes signal-semaphore injection but not
wait injection, so the consumer wait is a CPU-side cooperative poll, a counter
poll plus `yield_now` off the runtime's hot path, rather than a GPU-queue wait.
That still removes the producer stall and decouples the two pipelines. Validated
cross-device and cross-process on the RTX 3060: `dmabuf_timeline_probe` for the
bare primitive, `m562_dmabuf_semaphore_sync` for the element handoff, and
`gpu_dmabuf_ipc` with `G2G_DMABUF_SEM=1` for the full cross-process chain.

## SMPTE ST 2110

ST 2110 media transport rides on the shared PTP clock
([DESIGN-timing.md](DESIGN-timing.md)).
`MediaClock` (`g2g-core`, ST 2110-10) maps a PTP/TAI time to a 32-bit wrapping RTP
timestamp and back, a media clock counting at 90 kHz for video or the sample rate
for audio from the PTP epoch, so two receivers on the same grandmaster compute the
same timestamp for the same sampling instant.

### Audio and ancillary data

`st2110audio` (`g2g-plugins`, ST 2110-30) is the sans-IO PCM payloader and
depayloader, L16 and L24 big-endian in the RTP payload with timestamps off the
media clock. `st2110anc` (ST 2110-40, RFC 8331) carries SMPTE ST 291 ancillary data,
closed captions and timecode, as bit-packed 10-bit words with parity and checksum
validation, so the caption stack can ride 2110.

The sans-IO cores get network element wrappers. `st2110audiortp`, giving
`St2110AudioSink` and `St2110AudioSrc` behind the `st2110` feature, puts -30 audio
on the wire over UDP, with the sink mapping each frame's PTS through the elected PTP
clock to the media-clock timestamp and the source reconstructing PTS from it, so a
receiver on the same grandmaster stays in sync. `AudioFormat::PcmS16Le` rides as L16
and `PcmF32Le` as L24, the float scaled to the 24-bit wire, and
`AudioFormat::PcmS24Le`, integer 24-bit, rides the L24 wire directly.

`st2110ancrtp` does the same for -40 captions. `St2110AncSink` taps a compressed
H.264 or H.265 stream, a teed branch leaf like `CcExtract`, mines each access unit's
caption triples, wraps them in a Caption Distribution Packet (CDP, CEA-708 and SMPTE
ST 334-2) carried in a DID 0x61 ANC packet, and sends the RFC 8331 RTP timestamped
at the frame's PTP time. `St2110AncSrc` depacketizes -40 back into triples and,
through the shared `CaptionDecoder`, the decode core factored out of `CcExtract` that
drives the same CEA-608 and 708 state machines from triples mined from SEI or carried
in a CDP, emits timed `Caps::Text{Utf8}` cues. So captions travel end to end over
2110 and stay frame-aligned on a common grandmaster.

### Uncompressed video

`st2110video` (ST 2110-20, RFC 4175) carries uncompressed active video. The
packetizer slices a packed frame into Sample Row Data line runs, an Extended
Sequence Number then per-run headers giving scan line, pixel offset and octet
length, sized to the MTU, and the depacketizer writes each run back into the frame,
completing it on the RTP marker bit. `st2110videortp`, giving `St2110VideoSink` and
`St2110VideoSrc`, puts it on UDP with the 90 kHz media-clock timestamp shared by
every packet of a frame.

Each sampling is a `Layout` reading or writing one pgroup at a time, so the
packetizer and depacketizer stay layout-agnostic across three mappings: RGBA 8-bit,
packed and byte-identical, YCbCr-4:2:2 8-bit, packed `Yuyv` with luma and chroma
bytes swapped to the wire, and YCbCr-4:2:2 10-bit, the broadcast norm, from the
planar `I422p10` buffer where the four 10-bit samples Cb0 Y0 Cr0 Y1 are MSB-first
bit-packed into a 5-octet pgroup, crossing both a planar-to-packed and a
byte-to-bit boundary.

### SDP

The source's geometry comes from properties, or from the stream's SDP. `st2110sdp`
(RFC 4566 plus SMPTE ST 2110-10, -20, -30 and -40) is the sans-IO generator and
parser for the out-of-band description a receiver configures from, carrying the
essence (video sampling, size and rate, audio depth, rate, channels and ptime, or
ancillary), the payload type, the multicast group and port, and the `a=ts-refclk`
PTP grandmaster all the streams share. Every sink has an `sdp()` that publishes its
stream and every source an `apply_sdp()` that auto-configures from a parsed one, so
a stream self-describes end to end across video, audio and ancillary.

SDP covers all essences, including -22 (`jxsv`), and `St2110Session` bundles video,
audio and ancillary into one multi-section session document, each media tagged with
`a=mid` and a shared `a=ts-refclk`, so a whole program self-describes.

### JPEG XS

`st2110jxs` (ST 2110-22, RFC 9134) carries the compressed mezzanine essence, JPEG
XS. The packetizer slices an opaque codestream into codestream-mode packets, the
4-octet RFC 9134 payload header carrying transmode, packetmode, last-packet, frame
counter and packet counter, with the marker bit ending the frame and every packet on
the same 90 kHz media clock. `st2110jxsrtp`, giving `St2110JxsSink` and
`St2110JxsSrc`, puts it on UDP, taking and emitting `Caps::CompressedVideo{JpegXs}`
frames.

The JPEG XS codec itself is `SvtJpegXsEnc` and `SvtJpegXsDec` (`jpegxs` feature),
hand-rolled FFI to Intel SVT-JPEG-XS (ISO/IEC 21122, with no libavcodec), covering
planar 4:2:0 and 4:2:2 8-bit and 4:2:2 10-bit, with the encoder targeting a
bits-per-pixel budget and the decoder discovering geometry from the first
codestream. So a plant can move visually lossless video at a fraction of -20's
bandwidth with sub-frame latency, end to end from raw through encode and -22 to
decode.

### Redundancy and pacing

`st2110dup` implements ST 2110-7 seamless protection: a receive-side sequence-number
merge of two identical redundant streams, first arrival winning so a loss on one path
is filled by the other, with `a=group:DUP` in the session SDP. `SeamlessDedup` is the
sans-IO core, and `RedundantRtpReceiver`, behind the `st2110` feature, is the
socket-bound sibling that binds several receive paths, polls them round-robin so two
in-order streams merge back into sequence order, and yields deduplicated packets.
`St2110VideoSrc` adopts it behind a `redundant` property, a second blue path, and
being essence-agnostic it can serve the other essences the same way.

`st2110pacing` implements ST 2110-21 sender pacing: a schedule spreading a frame's
packets across the frame period, linear or gapped, which both `St2110VideoSink` and
the -22 `St2110JxsSink` realize over the tokio timer through a shared `pace_send`, so
the network sees a smooth flow instead of a burst. `VrxValidator` is the full
per-format -21 compliance check, the leaky-bucket virtual-receive-buffer model where
a receiver drains one packet every `TRS` after a `TR_OFFSET` head start, which over a
run of actual emission offsets reports the peak buffer occupancy, whether a packet
arrived late and starved the receiver, and whether it stays within the profile's
`Cmax`.

What is built spans -10, -20, -21, -22, -30, -40 and -7 plus SDP, written from the
RFCs and loopback-tested, not yet interop-validated against reference gear.
Multicast interop remains.
