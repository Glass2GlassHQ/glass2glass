//! Sans-IO RTMP protocol, both directions. Pure `no_std + alloc`, no sockets.
//!
//! - [`RtmpSession`] is the ingest (server) side, the transport half of
//!   [`RtmpSrc`](crate::rtmpsrc): feed received bytes with [`RtmpSession::push`],
//!   drain the peer responses ([`take_outbound`](RtmpSession::take_outbound)) and
//!   the demuxed FLV byte stream ([`take_flv`](RtmpSession::take_flv)).
//! - [`RtmpPublisher`] is the egress (client) side, the transport half of
//!   [`RtmpSink`](crate::rtmpsink): it connects out and *publishes*. Drain
//!   [`take_outbound`](RtmpPublisher::take_outbound) to the socket, feed the
//!   socket's bytes to [`push`](RtmpPublisher::push), and once
//!   [`is_publishing`](RtmpPublisher::is_publishing) feed an FLV byte stream to
//!   [`push_flv`](RtmpPublisher::push_flv); its audio/video tags are reframed
//!   into RTMP messages on the outbound buffer.
//!
//! An RTMP audio/video message payload is exactly an FLV tag *body*, so the two
//! halves are inverses: the session demuxes RTMP messages into an FLV stream,
//! the publisher reframes an FLV stream back into RTMP messages. Both share the
//! [`ChunkReader`] reassembly, the AMF0 codec, and the simple (non-digest)
//! handshake ffmpeg/OBS use. Scope: one stream, H.264 + AAC, AMF0 only. Verified
//! against the Adobe RTMP 1.0 spec.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// RTMP protocol version (the C0/S0 byte).
const RTMP_VERSION: u8 = 3;
/// C1/S1/C2/S2 are each 1536 bytes.
const HANDSHAKE_SIZE: usize = 1536;
/// Default chunk size before a `Set Chunk Size` changes it.
const DEFAULT_CHUNK_SIZE: usize = 128;

/// A simple-handshake C1/S1: a deterministic non-zero pattern in the random
/// region (bytes [0..8] left zero). The simple handshake does not validate it.
fn simple_sig() -> [u8; HANDSHAKE_SIZE] {
    let mut sig = [0u8; HANDSHAKE_SIZE];
    for (i, b) in sig.iter_mut().enumerate().skip(8) {
        *b = (i & 0xFF) as u8;
    }
    sig
}

/// A simple-handshake echo block (S2 echoing C1, or C2 echoing S1): the peer's
/// time, a zero time2, then the peer's 1528 random bytes.
fn simple_echo(peer_sig: &[u8]) -> [u8; HANDSHAKE_SIZE] {
    let mut echo = [0u8; HANDSHAKE_SIZE];
    echo[0..4].copy_from_slice(&peer_sig[0..4]);
    echo[8..].copy_from_slice(&peer_sig[8..]);
    echo
}

/// Build the initial client C1: a digest ("genuine FP") block when `complex`
/// (the `rtmp` feature supplies the crypto), else a simple block.
#[cfg(feature = "rtmp")]
fn client_c1(complex: bool) -> [u8; HANDSHAKE_SIZE] {
    if complex {
        crate::rtmphandshake::build_c1(0)
    } else {
        simple_sig()
    }
}
#[cfg(not(feature = "rtmp"))]
fn client_c1(_complex: bool) -> [u8; HANDSHAKE_SIZE] {
    simple_sig()
}

/// Build the client C2 replying to the server's `s1`: a digest response keyed
/// off S1's digest when `complex` and S1 actually carries one, else an echo of
/// S1 (the simple handshake / fallback when the server is not genuine-FMS).
#[cfg(feature = "rtmp")]
fn client_c2(complex: bool, s1: &[u8]) -> [u8; HANDSHAKE_SIZE] {
    if complex {
        if let Some(c2) = crate::rtmphandshake::build_c2(s1) {
            return c2;
        }
    }
    simple_echo(s1)
}
#[cfg(not(feature = "rtmp"))]
fn client_c2(_complex: bool, s1: &[u8]) -> [u8; HANDSHAKE_SIZE] {
    simple_echo(s1)
}

/// Build the server's S1 + S2 replying to the client's `c1`: a genuine-FMS
/// digest reply when `c1` requests the complex handshake (and the crypto is
/// available), else the simple S1 pattern + S2 echo.
#[cfg(feature = "rtmp")]
fn server_s1_s2(c1: &[u8]) -> ([u8; HANDSHAKE_SIZE], [u8; HANDSHAKE_SIZE]) {
    if crate::rtmphandshake::c1_has_digest(c1) {
        if let Some(s2) = crate::rtmphandshake::build_s2(c1) {
            return (crate::rtmphandshake::build_s1(0), s2);
        }
    }
    (simple_sig(), simple_echo(c1))
}
#[cfg(not(feature = "rtmp"))]
fn server_s1_s2(c1: &[u8]) -> ([u8; HANDSHAKE_SIZE], [u8; HANDSHAKE_SIZE]) {
    (simple_sig(), simple_echo(c1))
}

/// Default Window Acknowledgement Size (RFC §5.4.4): the peer acks after this
/// many bytes. The value FMS advertises and OBS / ffmpeg expect.
pub const DEFAULT_WINDOW_ACK_SIZE: u32 = 2_500_000;

// RTMP message type ids.
const MSG_SET_CHUNK_SIZE: u8 = 1;
const MSG_ACK: u8 = 3;
const MSG_WINDOW_ACK_SIZE: u8 = 5;
const MSG_SET_PEER_BW: u8 = 6;
const MSG_AUDIO: u8 = 8;
const MSG_VIDEO: u8 = 9;
const MSG_AMF0_COMMAND: u8 = 20;

// AMF0 type markers.
const AMF0_NUMBER: u8 = 0x00;
const AMF0_BOOLEAN: u8 = 0x01;
const AMF0_STRING: u8 = 0x02;
const AMF0_OBJECT: u8 = 0x03;
const AMF0_NULL: u8 = 0x05;
const AMF0_OBJECT_END: u8 = 0x09;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Awaiting the client's C0 (version) + C1 (1536 bytes).
    WaitC0C1,
    /// Sent S0/S1/S2; awaiting the client's C2 echo (consumed, not validated).
    WaitC2,
    /// Handshake done; parsing the chunk stream.
    Streaming,
}

/// Per-chunk-stream reassembly state (RTMP multiplexes messages by chunk stream
/// id, each fragment after the first carrying an `fmt 3` header).
#[derive(Debug, Default)]
struct ChunkStream {
    timestamp: u32,
    msg_length: usize,
    msg_type: u8,
    msg_stream_id: u32,
    ext_timestamp: bool,
    /// Payload accumulated for the in-progress message (empty = none in progress).
    payload: Vec<u8>,
}

/// Outcome of consuming one chunk from a [`ChunkReader`].
enum ChunkStep {
    /// More bytes are needed before another chunk can be parsed.
    Need,
    /// A chunk was consumed but did not complete a message (a fragment, or an
    /// internally-handled `Set Chunk Size`).
    Progress,
    /// A complete message: `(type, timestamp, payload)`.
    Message(u8, u32, Vec<u8>),
}

/// Reassembles the RTMP chunk stream into complete messages. Shared by the
/// ingest session ([`RtmpSession`]) and the publisher ([`RtmpPublisher`]). It
/// owns the inbound chunk size because `Set Chunk Size` changes how the
/// *following* chunks are read; that control message is applied internally and
/// never surfaced as a message.
#[derive(Debug)]
struct ChunkReader {
    inbound: Vec<u8>,
    chunk_size: usize,
    streams: BTreeMap<u32, ChunkStream>,
}

impl ChunkReader {
    fn new() -> Self {
        Self { inbound: Vec::new(), chunk_size: DEFAULT_CHUNK_SIZE, streams: BTreeMap::new() }
    }

    fn push(&mut self, data: &[u8]) {
        self.inbound.extend_from_slice(data);
    }

    /// Pop the next complete message, or `None` when more bytes are needed.
    fn next_message(&mut self) -> Option<(u8, u32, Vec<u8>)> {
        loop {
            match self.try_chunk() {
                ChunkStep::Need => return None,
                ChunkStep::Progress => continue,
                ChunkStep::Message(MSG_SET_CHUNK_SIZE, _, payload) => {
                    if payload.len() >= 4 {
                        let size =
                            u32::from_be_bytes(payload[0..4].try_into().expect("4")) & 0x7FFF_FFFF;
                        self.chunk_size = (size as usize).max(1);
                    }
                }
                ChunkStep::Message(msg_type, ts, payload) => {
                    return Some((msg_type, ts, payload));
                }
            }
        }
    }

    fn ensure_stream(&mut self, csid: u32) {
        self.streams.entry(csid).or_default();
    }

    /// Parse one chunk from `inbound`, returning what it produced.
    fn try_chunk(&mut self) -> ChunkStep {
        let len = self.inbound.len();
        if len == 0 {
            return ChunkStep::Need;
        }
        let fmt = self.inbound[0] >> 6;
        let marker = self.inbound[0] & 0x3F;
        let (csid, basic_len) = match marker {
            0 => {
                if len < 2 {
                    return ChunkStep::Need;
                }
                (self.inbound[1] as u32 + 64, 2)
            }
            1 => {
                if len < 3 {
                    return ChunkStep::Need;
                }
                (self.inbound[2] as u32 * 256 + self.inbound[1] as u32 + 64, 3)
            }
            _ => (marker as u32, 1),
        };

        let mh_len = match fmt {
            0 => 11,
            1 => 7,
            2 => 3,
            _ => 0,
        };
        if len < basic_len + mh_len {
            return ChunkStep::Need;
        }

        // Snapshot the per-stream inheritance (Copy fields + payload length) so no
        // borrow of `self` is held while reading the header and chunk size.
        self.ensure_stream(csid);
        let prev = {
            let s = &self.streams[&csid];
            (s.timestamp, s.msg_length, s.msg_type, s.msg_stream_id, s.ext_timestamp, s.payload.len())
        };
        let (prev_ts, prev_len, prev_type, prev_msid, prev_ext, prev_payload) = prev;

        // Resolve the message header fields against the inheritance.
        let mh = &self.inbound[basic_len..basic_len + mh_len];
        let (mut ts_field, msg_length, msg_type, msg_stream_id, is_delta) = match fmt {
            0 => (be24(mh, 0), be24(mh, 3) as usize, mh[6], le32(mh, 7), false),
            1 => (be24(mh, 0), be24(mh, 3) as usize, mh[6], prev_msid, true),
            2 => (be24(mh, 0), prev_len, prev_type, prev_msid, true),
            _ => (0, prev_len, prev_type, prev_msid, true),
        };

        // Extended timestamp follows the message header when the 24-bit field is
        // saturated (fmt 0/1/2), or when continuing a stream that used one (fmt 3).
        let needs_ext = if fmt == 3 { prev_ext } else { ts_field == 0xFF_FFFF };
        let ext_len = if needs_ext { 4 } else { 0 };
        if len < basic_len + mh_len + ext_len {
            return ChunkStep::Need;
        }
        if needs_ext {
            ts_field = u32::from_be_bytes(
                self.inbound[basic_len + mh_len..basic_len + mh_len + 4]
                    .try_into()
                    .expect("4 bytes"),
            );
        }

        let header_total = basic_len + mh_len + ext_len;
        let in_progress = prev_payload > 0;
        let target = if in_progress { prev_len } else { msg_length };
        let remaining = target.saturating_sub(prev_payload);
        let fragment = remaining.min(self.chunk_size);
        if len < header_total + fragment {
            return ChunkStep::Need;
        }

        let timestamp = if in_progress {
            prev_ts
        } else if fmt == 0 {
            ts_field
        } else if is_delta {
            prev_ts.wrapping_add(ts_field)
        } else {
            ts_field
        };

        let payload_bytes = self.inbound[header_total..header_total + fragment].to_vec();
        self.inbound.drain(..header_total + fragment);

        let stream = self.streams.get_mut(&csid).expect("inserted");
        if !in_progress {
            stream.timestamp = timestamp;
            stream.msg_length = msg_length;
            stream.msg_type = msg_type;
            stream.msg_stream_id = msg_stream_id;
            stream.ext_timestamp = needs_ext;
        }
        stream.payload.extend_from_slice(&payload_bytes);
        if stream.payload.len() >= stream.msg_length {
            ChunkStep::Message(stream.msg_type, stream.timestamp, core::mem::take(&mut stream.payload))
        } else {
            ChunkStep::Progress
        }
    }
}

/// Sans-IO RTMP receive session: a publisher-side state machine producing the
/// peer responses and the demuxed FLV byte stream.
#[derive(Debug)]
pub struct RtmpSession {
    phase: Phase,
    hs_inbound: Vec<u8>,
    outbound: Vec<u8>,
    reader: ChunkReader,
    flv: Vec<u8>,
    flv_header_written: bool,
    flv_prev_tag_size: u32,
    publishing: bool,
    /// Window Acknowledgement Size (RFC §5.4.4): the receiver sends an
    /// Acknowledgement after receiving this many bytes. Advertised to the peer at
    /// `connect` and used as our own ack cadence.
    window_ack_size: u32,
    /// Total chunk-stream bytes received (post-handshake), and the byte count at
    /// the last Acknowledgement we sent (RFC §5.4.3 flow control).
    bytes_received: u64,
    last_ack_bytes: u64,
}

impl Default for RtmpSession {
    fn default() -> Self {
        Self::new()
    }
}

impl RtmpSession {
    pub fn new() -> Self {
        Self {
            phase: Phase::WaitC0C1,
            hs_inbound: Vec::new(),
            outbound: Vec::new(),
            reader: ChunkReader::new(),
            flv: Vec::new(),
            flv_header_written: false,
            flv_prev_tag_size: 0,
            publishing: false,
            window_ack_size: DEFAULT_WINDOW_ACK_SIZE,
            bytes_received: 0,
            last_ack_bytes: 0,
        }
    }

    /// Set the Window Acknowledgement Size advertised to the publisher at
    /// `connect` (and used as our own ack cadence). Smaller values ack more often
    /// (tighter flow control); the default is [`DEFAULT_WINDOW_ACK_SIZE`].
    pub fn with_window_ack_size(mut self, bytes: u32) -> Self {
        self.window_ack_size = bytes.max(1);
        self
    }

    /// Whether the publisher has reached `NetStream.Publish.Start` (media flows).
    pub fn is_publishing(&self) -> bool {
        self.publishing
    }

    /// Total chunk-stream bytes received from the publisher (post-handshake).
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
    }

    /// Feed received bytes and advance the state machine.
    pub fn push(&mut self, data: &[u8]) {
        match self.phase {
            Phase::WaitC0C1 | Phase::WaitC2 => {
                self.hs_inbound.extend_from_slice(data);
                loop {
                    let progressed = match self.phase {
                        Phase::WaitC0C1 => self.try_handshake_c0c1(),
                        Phase::WaitC2 => self.try_handshake_c2(),
                        Phase::Streaming => false,
                    };
                    if !progressed {
                        break;
                    }
                }
                // Handshake completed inside this push: the leftover bytes are the
                // start of the chunk stream.
                if self.phase == Phase::Streaming {
                    let leftover = core::mem::take(&mut self.hs_inbound);
                    self.bytes_received += leftover.len() as u64;
                    self.reader.push(&leftover);
                }
            }
            Phase::Streaming => {
                self.bytes_received += data.len() as u64;
                self.reader.push(data);
            }
        }
        while let Some((msg_type, ts, payload)) = self.reader.next_message() {
            self.dispatch(msg_type, ts, &payload);
        }
        // Flow control (RFC §5.4.3): once a window's worth of bytes has arrived
        // since the last one, acknowledge the running byte count so the publisher
        // can advance its send window (back-pressure release).
        if self.bytes_received.saturating_sub(self.last_ack_bytes) >= self.window_ack_size as u64 {
            self.send_control(MSG_ACK, &(self.bytes_received as u32).to_be_bytes());
            self.last_ack_bytes = self.bytes_received;
        }
    }

    /// Take the bytes queued to send back to the peer (handshake + control +
    /// command replies).
    pub fn take_outbound(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.outbound)
    }

    /// Take the FLV byte stream demuxed so far (the init header is emitted once,
    /// before the first media tag).
    pub fn take_flv(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.flv)
    }

    fn try_handshake_c0c1(&mut self) -> bool {
        if self.hs_inbound.len() < 1 + HANDSHAKE_SIZE {
            return false;
        }
        let c1 = self.hs_inbound[1..1 + HANDSHAKE_SIZE].to_vec();
        self.hs_inbound.drain(..1 + HANDSHAKE_SIZE);

        // S0 + S1 + S2. A genuine-FMS digest reply if the client's C1 requests
        // the complex handshake (and its digest validates), else the simple
        // deterministic S1 + S2-echoes-C1 that OBS / ffmpeg's simple path expect.
        self.outbound.push(RTMP_VERSION);
        let (s1, s2) = server_s1_s2(&c1);
        self.outbound.extend_from_slice(&s1);
        self.outbound.extend_from_slice(&s2);

        self.phase = Phase::WaitC2;
        true
    }

    fn try_handshake_c2(&mut self) -> bool {
        if self.hs_inbound.len() < HANDSHAKE_SIZE {
            return false;
        }
        self.hs_inbound.drain(..HANDSHAKE_SIZE); // C2 not validated (simple handshake)
        self.phase = Phase::Streaming;
        true
    }

    /// Act on a complete message: answer AMF0 commands and reframe audio/video
    /// into the FLV byte stream. (`Set Chunk Size` is handled by [`ChunkReader`].)
    fn dispatch(&mut self, msg_type: u8, timestamp: u32, payload: &[u8]) {
        match msg_type {
            MSG_AMF0_COMMAND => self.handle_command(payload),
            MSG_AUDIO | MSG_VIDEO => {
                let tag_type = if msg_type == MSG_AUDIO { 8 } else { 9 };
                self.write_flv_tag(tag_type, timestamp, payload);
            }
            _ => {} // window-ack / set-peer-bw / acknowledgement / data: ignored
        }
    }

    /// Drive the publish handshake: reply to `connect` / `createStream` /
    /// `publish`. Only the command name + transaction id are needed to respond.
    fn handle_command(&mut self, payload: &[u8]) {
        let mut at = 0;
        let Some(name) = amf0_read_string(payload, &mut at) else { return };
        let txn = amf0_read_number(payload, &mut at).unwrap_or(0.0);
        match name.as_str() {
            "connect" => {
                self.send_control(MSG_WINDOW_ACK_SIZE, &self.window_ack_size.to_be_bytes());
                let mut bw = self.window_ack_size.to_be_bytes().to_vec();
                bw.push(2); // dynamic limit
                self.send_control(MSG_SET_PEER_BW, &bw);
                self.send_control(MSG_SET_CHUNK_SIZE, &(DEFAULT_CHUNK_SIZE as u32).to_be_bytes());
                let mut body = Vec::new();
                amf0_string(&mut body, "_result");
                amf0_number(&mut body, txn);
                amf0_object(&mut body, &[("fmsVer", AmfVal::Str("FMS/3,0,1,123")), ("capabilities", AmfVal::Num(31.0))]);
                amf0_object(
                    &mut body,
                    &[
                        ("level", AmfVal::Str("status")),
                        ("code", AmfVal::Str("NetConnection.Connect.Success")),
                        ("description", AmfVal::Str("Connection succeeded.")),
                        ("objectEncoding", AmfVal::Num(0.0)),
                    ],
                );
                self.send_command(0, &body);
            }
            "createStream" => {
                let mut body = Vec::new();
                amf0_string(&mut body, "_result");
                amf0_number(&mut body, txn);
                body.push(AMF0_NULL);
                amf0_number(&mut body, 1.0); // stream id
                self.send_command(0, &body);
            }
            "publish" => {
                let mut body = Vec::new();
                amf0_string(&mut body, "onStatus");
                amf0_number(&mut body, 0.0);
                body.push(AMF0_NULL);
                amf0_object(
                    &mut body,
                    &[
                        ("level", AmfVal::Str("status")),
                        ("code", AmfVal::Str("NetStream.Publish.Start")),
                        ("description", AmfVal::Str("Start publishing.")),
                    ],
                );
                self.send_command(1, &body);
                self.publishing = true;
            }
            _ => {} // releaseStream / FCPublish / etc.: safely ignored
        }
    }

    /// Send a protocol control message (chunk stream id 2, message stream id 0).
    fn send_control(&mut self, msg_type: u8, payload: &[u8]) {
        write_message(&mut self.outbound, 2, msg_type, 0, 0, payload, DEFAULT_CHUNK_SIZE);
    }

    /// Send an AMF0 command reply (chunk stream id 3) on `msg_stream_id`. Larger
    /// replies (the `connect` result) fragment at the advertised chunk size.
    fn send_command(&mut self, msg_stream_id: u32, payload: &[u8]) {
        write_message(&mut self.outbound, 3, MSG_AMF0_COMMAND, msg_stream_id, 0, payload, DEFAULT_CHUNK_SIZE);
    }

    /// Append one FLV tag (the RTMP message payload is the tag body) to the FLV
    /// output, writing the FLV header before the first tag.
    fn write_flv_tag(&mut self, tag_type: u8, timestamp: u32, body: &[u8]) {
        if !self.flv_header_written {
            self.flv.extend_from_slice(b"FLV");
            self.flv.push(1); // version
            self.flv.push(0x05); // flags: audio + video present
            self.flv.extend_from_slice(&9u32.to_be_bytes()); // data offset
            self.flv_header_written = true;
        }
        self.flv.extend_from_slice(&self.flv_prev_tag_size.to_be_bytes());
        let start = self.flv.len();
        self.flv.push(tag_type);
        write_u24(&mut self.flv, body.len() as u32);
        write_u24(&mut self.flv, timestamp & 0x00FF_FFFF);
        self.flv.push((timestamp >> 24) as u8);
        write_u24(&mut self.flv, 0); // stream id
        self.flv.extend_from_slice(body);
        self.flv_prev_tag_size = (self.flv.len() - start) as u32;
    }
}

/// The outbound chunk size the publisher negotiates up front (the 128-byte
/// default would fragment every video keyframe into dozens of chunks).
const PUB_CHUNK_SIZE: usize = 4096;
/// A timestamp at/above this triggers the extended-timestamp field; the
/// publisher clamps below it (it does not emit extended timestamps), good for
/// streams under ~4.6 hours.
const TS_MAX: u32 = 0xFF_FFFE;

/// Publisher state machine: handshake, then the `connect` -> `createStream` ->
/// `publish` command ladder driven by the server's replies, then `Publishing`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PubPhase {
    /// Sent C0/C1; awaiting S0 + S1 + S2 (then send C2 + the `connect` command).
    Handshake,
    /// Sent `connect`; awaiting its `_result`.
    WaitConnect,
    /// Sent `createStream`; awaiting its `_result` (carries the stream id).
    WaitCreate,
    /// Sent `publish`; awaiting `onStatus` `NetStream.Publish.Start`.
    WaitPublish,
    /// `Publish.Start` seen; audio/video messages flow.
    Publishing,
}

/// Sans-IO RTMP publish client, the transport half of [`RtmpSink`](crate::rtmpsink).
/// Construct with the target `app` / `tcUrl` / stream key; the C0/C1 handshake is
/// queued immediately. Drain [`take_outbound`](Self::take_outbound) to the socket
/// and feed the socket's bytes to [`push`](Self::push); once
/// [`is_publishing`](Self::is_publishing), feed an FLV byte stream to
/// [`push_flv`](Self::push_flv). The inverse of [`RtmpSession`].
#[derive(Debug)]
pub struct RtmpPublisher {
    app: String,
    tc_url: String,
    stream_key: String,
    phase: PubPhase,
    s_inbound: Vec<u8>,
    outbound: Vec<u8>,
    reader: ChunkReader,
    stream_id: u32,
    flv: FlvTagReader,
    /// Tags parsed before `Publish.Start`, replayed once publishing begins.
    pending_media: Vec<(u8, u32, Vec<u8>)>,
    /// Use the "genuine FP" digest (complex) handshake, required by strict CDNs;
    /// C2 falls back to a plain S1 echo when the server is not genuine-FMS, so it
    /// stays compatible with simple servers. Ignored without the `rtmp` feature.
    complex: bool,
    /// The server's Window Acknowledgement Size (RFC §5.4.4), learned from its
    /// control message: after this many unacknowledged bytes the publisher must
    /// pause until the server sends an Acknowledgement. 0 = no window yet (unbounded).
    peer_window: u32,
    /// Total bytes handed to the socket (via `take_outbound`), and the sequence
    /// number the server last acknowledged (RFC §5.4.3). The unacknowledged
    /// backlog is `bytes_sent - ack_seq` in 32-bit sequence space.
    bytes_sent: u64,
    ack_seq: u32,
}

impl RtmpPublisher {
    /// `app` is the RTMP application (first URL path segment), `stream_key` the
    /// rest, and `tc_url` the `rtmp://host[:port]/app` the server echoes back.
    pub fn new(app: impl Into<String>, tc_url: impl Into<String>, stream_key: impl Into<String>) -> Self {
        let mut me = Self {
            app: app.into(),
            tc_url: tc_url.into(),
            stream_key: stream_key.into(),
            phase: PubPhase::Handshake,
            s_inbound: Vec::new(),
            outbound: Vec::new(),
            reader: ChunkReader::new(),
            stream_id: 1,
            flv: FlvTagReader::default(),
            pending_media: Vec::new(),
            complex: true,
            peer_window: 0,
            bytes_sent: 0,
            ack_seq: 0,
        };
        // C0 (version) + C1: a "genuine FP" digest block by default (strict-CDN
        // ready), or a simple deterministic block if the digest is disabled.
        me.outbound.push(RTMP_VERSION);
        me.outbound.extend_from_slice(&client_c1(me.complex));
        me
    }

    /// Choose the handshake: the digest (complex) handshake (default, required by
    /// strict CDNs like Facebook Live / Wowza), or the plain simple handshake.
    /// The complex C2 auto-falls-back to an echo against a non-genuine-FMS
    /// server, so the default is safe against simple servers too.
    pub fn with_complex_handshake(mut self, complex: bool) -> Self {
        self.complex = complex;
        // Rebuild the queued C0/C1 for the chosen mode (nothing has been sent).
        self.outbound.clear();
        self.outbound.push(RTMP_VERSION);
        self.outbound.extend_from_slice(&client_c1(self.complex));
        self
    }

    /// Whether the server has acknowledged `publish` (media may be sent).
    pub fn is_publishing(&self) -> bool {
        self.phase == PubPhase::Publishing
    }

    /// Take the bytes queued to send to the server (handshake + commands + media),
    /// counting them toward the send window for flow control.
    pub fn take_outbound(&mut self) -> Vec<u8> {
        let out = core::mem::take(&mut self.outbound);
        self.bytes_sent += out.len() as u64;
        out
    }

    /// Unacknowledged bytes in flight: `bytes_sent - ack_seq` in the RTMP 32-bit
    /// sequence space (RFC §5.4.3).
    pub fn unacked_bytes(&self) -> u32 {
        (self.bytes_sent as u32).wrapping_sub(self.ack_seq)
    }

    /// Whether the publisher has hit the server's acknowledgement window and must
    /// pause sending media until an Acknowledgement advances the window (back-
    /// pressure). Always false until the server has advertised a window.
    pub fn throttled(&self) -> bool {
        self.peer_window != 0 && self.unacked_bytes() >= self.peer_window
    }

    /// The server's advertised Window Acknowledgement Size, once learned (0 = not yet).
    pub fn peer_window(&self) -> u32 {
        self.peer_window
    }

    /// Feed bytes received from the server and advance the publish ladder.
    pub fn push(&mut self, data: &[u8]) {
        if self.phase == PubPhase::Handshake {
            self.s_inbound.extend_from_slice(data);
            // Wait for the full S0 + S1 + S2 before replying (the simple handshake
            // servers send all three together after C0/C1).
            if self.s_inbound.len() < 1 + 2 * HANDSHAKE_SIZE {
                return;
            }
            // C2: a digest response keyed off the server's S1 digest (complex
            // handshake), or an echo of S1 (simple / fallback). S2 is not
            // validated. Server bytes are S0 at [0], S1 at [1 .. 1+1536].
            let s1 = self.s_inbound[1..1 + HANDSHAKE_SIZE].to_vec();
            self.outbound.extend_from_slice(&client_c2(self.complex, &s1));
            let leftover = self.s_inbound.split_off(1 + 2 * HANDSHAKE_SIZE);
            self.s_inbound.clear();
            // Raise the chunk size, then open the publish ladder.
            self.send_control(MSG_SET_CHUNK_SIZE, &(PUB_CHUNK_SIZE as u32).to_be_bytes());
            self.send_connect();
            self.phase = PubPhase::WaitConnect;
            self.reader.push(&leftover);
        } else {
            self.reader.push(data);
        }
        while let Some((msg_type, _ts, payload)) = self.reader.next_message() {
            self.handle_message(msg_type, &payload);
        }
    }

    /// Feed an FLV byte stream (from `flvmux`). Its audio/video/script tags are
    /// reframed into RTMP messages once publishing has started; earlier tags are
    /// buffered and replayed.
    pub fn push_flv(&mut self, data: &[u8]) {
        self.flv.push(data);
        while let Some((tag_type, ts, body)) = self.flv.next_tag() {
            if self.phase == PubPhase::Publishing {
                self.send_media(tag_type, ts, &body);
            } else {
                self.pending_media.push((tag_type, ts, body));
            }
        }
    }

    /// Advance the command ladder on the server's `_result` / `onStatus`. The
    /// command name and (for `createStream`) the assigned stream id are all that
    /// is needed; the ladder proceeds even on `_error` so a strict server cannot
    /// wedge it.
    fn handle_message(&mut self, msg_type: u8, payload: &[u8]) {
        // Flow-control control messages (RFC §5.4.3-5.4.4): the server's Window
        // Acknowledgement Size bounds the publisher's in-flight bytes, and each
        // Acknowledgement advances the acknowledged sequence, releasing the window.
        match msg_type {
            MSG_WINDOW_ACK_SIZE => {
                if let Some(w) = payload.get(..4) {
                    self.peer_window = u32::from_be_bytes([w[0], w[1], w[2], w[3]]);
                }
                return;
            }
            MSG_ACK => {
                if let Some(w) = payload.get(..4) {
                    self.ack_seq = u32::from_be_bytes([w[0], w[1], w[2], w[3]]);
                }
                return;
            }
            MSG_AMF0_COMMAND => {}
            _ => return, // set-peer-bw / user-control: ignored
        }
        let mut at = 0;
        let Some(name) = amf0_read_string(payload, &mut at) else { return };
        match self.phase {
            PubPhase::WaitConnect if name == "_result" || name == "_error" => {
                self.send_create_stream();
                self.phase = PubPhase::WaitCreate;
            }
            PubPhase::WaitCreate if name == "_result" => {
                // body: "_result", txn(number), NULL, streamId(number)
                let _txn = amf0_read_number(payload, &mut at);
                if payload.get(at) == Some(&AMF0_NULL) {
                    at += 1;
                }
                if let Some(sid) = amf0_read_number(payload, &mut at) {
                    self.stream_id = sid as u32;
                }
                self.send_publish();
                self.phase = PubPhase::WaitPublish;
            }
            PubPhase::WaitPublish if name == "onStatus" => {
                self.phase = PubPhase::Publishing;
                let pending = core::mem::take(&mut self.pending_media);
                for (tag_type, ts, body) in pending {
                    self.send_media(tag_type, ts, &body);
                }
            }
            _ => {}
        }
    }

    fn send_connect(&mut self) {
        let mut body = Vec::new();
        amf0_string(&mut body, "connect");
        amf0_number(&mut body, 1.0);
        amf0_object(
            &mut body,
            &[
                ("app", AmfVal::Str(&self.app)),
                ("type", AmfVal::Str("nonprivate")),
                ("flashVer", AmfVal::Str("FMLE/3.0 (compatible; g2g)")),
                ("tcUrl", AmfVal::Str(&self.tc_url)),
            ],
        );
        self.send_command(0, &body);
    }

    fn send_create_stream(&mut self) {
        let mut body = Vec::new();
        amf0_string(&mut body, "createStream");
        amf0_number(&mut body, 2.0);
        body.push(AMF0_NULL);
        self.send_command(0, &body);
    }

    fn send_publish(&mut self) {
        let mut body = Vec::new();
        amf0_string(&mut body, "publish");
        amf0_number(&mut body, 3.0);
        body.push(AMF0_NULL);
        amf0_string(&mut body, &self.stream_key);
        amf0_string(&mut body, "live");
        self.send_command(self.stream_id, &body);
    }

    /// Send a protocol control message (chunk stream id 2, message stream id 0).
    fn send_control(&mut self, msg_type: u8, payload: &[u8]) {
        write_message(&mut self.outbound, 2, msg_type, 0, 0, payload, PUB_CHUNK_SIZE);
    }

    /// Send an AMF0 command (chunk stream id 3) on `msg_stream_id`.
    fn send_command(&mut self, msg_stream_id: u32, payload: &[u8]) {
        write_message(&mut self.outbound, 3, MSG_AMF0_COMMAND, msg_stream_id, 0, payload, PUB_CHUNK_SIZE);
    }

    /// Reframe one FLV tag as an RTMP message (the tag body is the message
    /// payload). FLV tag type == RTMP message type for audio (8) / video (9) /
    /// data (18); audio and video ride distinct chunk streams.
    fn send_media(&mut self, tag_type: u8, timestamp: u32, body: &[u8]) {
        let csid = match tag_type {
            MSG_AUDIO => 4,
            MSG_VIDEO => 6,
            _ => 5, // script / data
        };
        write_message(
            &mut self.outbound,
            csid,
            tag_type,
            self.stream_id,
            timestamp.min(TS_MAX),
            body,
            PUB_CHUNK_SIZE,
        );
    }
}

/// Splits an FLV byte stream into raw tags `(tag_type, timestamp, body)`. The
/// 9-byte FLV header (and its data offset) is consumed once; each tag is
/// preceded by a 4-byte previous-tag-size field. The inverse of
/// [`RtmpSession::write_flv_tag`].
#[derive(Debug, Default)]
struct FlvTagReader {
    buf: Vec<u8>,
    header_consumed: bool,
}

impl FlvTagReader {
    fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    fn next_tag(&mut self) -> Option<(u8, u32, Vec<u8>)> {
        if !self.header_consumed {
            if self.buf.len() < 9 {
                return None;
            }
            // The header's data-offset field gives the first tag's start.
            let data_offset = u32::from_be_bytes(self.buf[5..9].try_into().expect("4 bytes")) as usize;
            let data_offset = data_offset.max(9);
            if self.buf.len() < data_offset {
                return None;
            }
            self.buf.drain(..data_offset);
            self.header_consumed = true;
        }
        // 4-byte previous-tag-size, then an 11-byte tag header, then the body.
        if self.buf.len() < 4 + 11 {
            return None;
        }
        let tag_type = self.buf[4];
        let data_size = be24(&self.buf, 4 + 1) as usize;
        let total = 4 + 11 + data_size;
        if self.buf.len() < total {
            return None;
        }
        let ts_lo = be24(&self.buf, 4 + 4);
        let ts_hi = self.buf[4 + 7] as u32;
        let timestamp = (ts_hi << 24) | ts_lo;
        let body = self.buf[4 + 11..total].to_vec();
        self.buf.drain(..total);
        Some((tag_type, timestamp, body))
    }
}

/// Write an outbound message as RTMP chunks, fragmenting the payload at
/// `chunk_size`: the first chunk carries an `fmt 0` header (timestamp, length,
/// type, little-endian message stream id), continuations carry a 1-byte `fmt 3`
/// header. `csid` is assumed to be in `2..=63` (1-byte basic header).
fn write_message(
    out: &mut Vec<u8>,
    csid: u32,
    msg_type: u8,
    msg_stream_id: u32,
    timestamp: u32,
    payload: &[u8],
    chunk_size: usize,
) {
    let basic = (csid as u8) & 0x3F;
    let mut off = 0;
    let mut first = true;
    loop {
        if first {
            out.push(basic); // fmt 0
            write_u24(out, timestamp);
            write_u24(out, payload.len() as u32);
            out.push(msg_type);
            out.extend_from_slice(&msg_stream_id.to_le_bytes());
            first = false;
        } else {
            out.push((3 << 6) | basic); // fmt 3 continuation
        }
        let take = (payload.len() - off).min(chunk_size.max(1));
        out.extend_from_slice(&payload[off..off + take]);
        off += take;
        if off >= payload.len() {
            break;
        }
    }
}

/// Write a 24-bit big-endian integer.
fn write_u24(out: &mut Vec<u8>, v: u32) {
    out.push((v >> 16) as u8);
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

fn be24(b: &[u8], at: usize) -> u32 {
    ((b[at] as u32) << 16) | ((b[at + 1] as u32) << 8) | b[at + 2] as u32
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

/// An AMF0 value the encoder writes (the subset the command replies need).
enum AmfVal<'a> {
    Num(f64),
    Str(&'a str),
}

fn amf0_number(out: &mut Vec<u8>, v: f64) {
    out.push(AMF0_NUMBER);
    out.extend_from_slice(&v.to_be_bytes());
}

fn amf0_string(out: &mut Vec<u8>, s: &str) {
    out.push(AMF0_STRING);
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Write an AMF0 object: bare (unmarked) length-prefixed keys + values, then the
/// empty-key object-end terminator.
fn amf0_object(out: &mut Vec<u8>, pairs: &[(&str, AmfVal)]) {
    out.push(AMF0_OBJECT);
    for (key, val) in pairs {
        out.extend_from_slice(&(key.len() as u16).to_be_bytes());
        out.extend_from_slice(key.as_bytes());
        match val {
            AmfVal::Num(n) => amf0_number(out, *n),
            AmfVal::Str(s) => amf0_string(out, s),
        }
    }
    out.extend_from_slice(&[0, 0, AMF0_OBJECT_END]);
}

/// Read a marker-prefixed AMF0 string at `*at`, advancing the cursor.
fn amf0_read_string(buf: &[u8], at: &mut usize) -> Option<alloc::string::String> {
    if *buf.get(*at)? != AMF0_STRING {
        return None;
    }
    let len = u16::from_be_bytes(buf.get(*at + 1..*at + 3)?.try_into().ok()?) as usize;
    let s = core::str::from_utf8(buf.get(*at + 3..*at + 3 + len)?).ok()?;
    *at += 3 + len;
    Some(alloc::string::String::from(s))
}

/// Read a marker-prefixed AMF0 number at `*at`, advancing the cursor.
fn amf0_read_number(buf: &[u8], at: &mut usize) -> Option<f64> {
    if *buf.get(*at)? != AMF0_NUMBER {
        return None;
    }
    let v = f64::from_be_bytes(buf.get(*at + 1..*at + 9)?.try_into().ok()?);
    *at += 9;
    Some(v)
}

// Keep the boolean marker referenced (publishers may send it in command objects,
// which the reader skips); silences an unused-const warning without hiding it.
const _: u8 = AMF0_BOOLEAN;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flv::{FlvDemuxer, FlvTrack};
    use alloc::vec;

    fn push_u24(out: &mut Vec<u8>, v: u32) {
        out.push((v >> 16) as u8);
        out.push((v >> 8) as u8);
        out.push(v as u8);
    }

    fn amf_string(out: &mut Vec<u8>, s: &str) {
        out.push(AMF0_STRING);
        out.extend_from_slice(&(s.len() as u16).to_be_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn amf_number(out: &mut Vec<u8>, v: f64) {
        out.push(AMF0_NUMBER);
        out.extend_from_slice(&v.to_be_bytes());
    }

    /// A single `fmt 0` chunk (the whole message fits one chunk).
    fn chunk(csid: u8, msg_type: u8, msid: u32, ts: u32, payload: &[u8]) -> Vec<u8> {
        let mut c = vec![csid & 0x3F];
        push_u24(&mut c, ts & 0x00FF_FFFF);
        push_u24(&mut c, payload.len() as u32);
        c.push(msg_type);
        c.extend_from_slice(&msid.to_le_bytes());
        c.extend_from_slice(payload);
        c
    }

    /// Drive C0/C1/C2 then connect / createStream / publish.
    fn handshake_and_publish(s: &mut RtmpSession) {
        let mut hs = vec![RTMP_VERSION];
        hs.extend((0..HANDSHAKE_SIZE as u32).map(|i| (i % 256) as u8)); // C1
        hs.extend(vec![0u8; HANDSHAKE_SIZE]); // C2
        s.push(&hs);
        let _ = s.take_outbound();

        let mut connect = Vec::new();
        amf_string(&mut connect, "connect");
        amf_number(&mut connect, 1.0);
        connect.push(AMF0_NULL);
        s.push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &connect));

        let mut create = Vec::new();
        amf_string(&mut create, "createStream");
        amf_number(&mut create, 2.0);
        create.push(AMF0_NULL);
        s.push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &create));

        let mut publish = Vec::new();
        amf_string(&mut publish, "publish");
        amf_number(&mut publish, 0.0);
        publish.push(AMF0_NULL);
        amf_string(&mut publish, "key");
        amf_string(&mut publish, "live");
        s.push(&chunk(3, MSG_AMF0_COMMAND, 1, 0, &publish));
    }

    #[test]
    fn handshake_replies_s0_s1_s2_echoing_c1() {
        let mut s = RtmpSession::new();
        let mut c0c1 = vec![RTMP_VERSION];
        c0c1.extend((0..HANDSHAKE_SIZE as u32).map(|i| (i % 256) as u8));
        s.push(&c0c1);
        let out = s.take_outbound();
        assert_eq!(out.len(), 1 + 2 * HANDSHAKE_SIZE, "S0 + S1 + S2");
        assert_eq!(out[0], RTMP_VERSION);
        let s2 = &out[1 + HANDSHAKE_SIZE..];
        assert_eq!(&s2[8..], &c0c1[1 + 8..1 + HANDSHAKE_SIZE], "S2 echoes C1's random bytes");
    }

    #[test]
    fn publish_flow_reaches_publishing_and_replies() {
        let mut s = RtmpSession::new();
        let mut hs = vec![RTMP_VERSION];
        hs.extend(vec![0u8; HANDSHAKE_SIZE]);
        hs.extend(vec![0u8; HANDSHAKE_SIZE]);
        s.push(&hs);
        assert!(!s.is_publishing());
        let _ = s.take_outbound();

        let mut connect = Vec::new();
        amf_string(&mut connect, "connect");
        amf_number(&mut connect, 1.0);
        connect.push(AMF0_NULL);
        s.push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &connect));
        // connect reply carries the success code.
        let reply = s.take_outbound();
        assert!(
            reply.windows("NetConnection.Connect.Success".len()).any(|w| w == b"NetConnection.Connect.Success"),
            "connect _result advertises success",
        );

        let mut create = Vec::new();
        amf_string(&mut create, "createStream");
        amf_number(&mut create, 4.0);
        create.push(AMF0_NULL);
        s.push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &create));

        let mut publish = Vec::new();
        amf_string(&mut publish, "publish");
        amf_number(&mut publish, 0.0);
        publish.push(AMF0_NULL);
        s.push(&chunk(3, MSG_AMF0_COMMAND, 1, 0, &publish));
        let reply = s.take_outbound();
        assert!(
            reply.windows("NetStream.Publish.Start".len()).any(|w| w == b"NetStream.Publish.Start"),
            "publish onStatus starts the stream",
        );
        assert!(s.is_publishing());
    }

    #[test]
    fn audio_and_video_messages_become_an_flv_stream() {
        let mut s = RtmpSession::new();
        handshake_and_publish(&mut s);

        // Video tag body: keyframe|AVC, NALU packet, then one AVCC access unit.
        let au = [0u8, 0, 0, 3, 0x65, 0x11, 0x22]; // 4-byte length=3 + NAL
        let mut vbody = vec![0x17u8, 0x01, 0x00, 0x00, 0x00];
        vbody.extend_from_slice(&au);
        s.push(&chunk(6, MSG_VIDEO, 1, 33, &vbody));

        // Audio tag body: AAC raw frame.
        let frame = [0xDEu8, 0xAD];
        let mut abody = vec![0xAFu8, 0x01];
        abody.extend_from_slice(&frame);
        s.push(&chunk(4, MSG_AUDIO, 1, 33, &abody));

        let mut demux = FlvDemuxer::new();
        demux.push_data(&s.take_flv());
        let units = demux.take_units();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], FlvUnitView(FlvTrack::Video, au.to_vec(), 33));
        assert_eq!(units[1], FlvUnitView(FlvTrack::Audio, frame.to_vec(), 33));
    }

    #[test]
    fn reassembles_a_message_split_across_chunks() {
        let mut s = RtmpSession::new();
        handshake_and_publish(&mut s);

        // A 200-byte video message exceeds the 128-byte default chunk size, so it
        // arrives as a fmt-0 chunk + an fmt-3 continuation.
        let nal: Vec<u8> = (0..191u32).map(|i| (i as u8).wrapping_mul(7)).collect();
        let mut vbody = vec![0x17u8, 0x01, 0x00, 0x00, 0x00];
        vbody.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        vbody.extend_from_slice(&nal);
        assert_eq!(vbody.len(), 200);

        let csid = 6u8;
        let mut bytes = vec![csid & 0x3F]; // fmt 0
        push_u24(&mut bytes, 0);
        push_u24(&mut bytes, vbody.len() as u32);
        bytes.push(MSG_VIDEO);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&vbody[..DEFAULT_CHUNK_SIZE]); // first fragment
        bytes.push((3 << 6) | (csid & 0x3F)); // fmt 3 continuation
        bytes.extend_from_slice(&vbody[DEFAULT_CHUNK_SIZE..]);
        s.push(&bytes);

        let mut demux = FlvDemuxer::new();
        demux.push_data(&s.take_flv());
        let units = demux.take_units();
        assert_eq!(units.len(), 1);
        // flvdemux forwards the AVCC body after the 5-byte AVC header.
        let mut expected = (nal.len() as u32).to_be_bytes().to_vec();
        expected.extend_from_slice(&nal);
        assert_eq!(units[0].data, expected, "the split message reassembled byte-exact");
    }

    /// The publisher (RtmpSink's transport) and the session (RtmpSrc's
    /// transport) are inverses, so pitting them against each other with no
    /// socket proves the egress path: drive the handshake + publish ladder to
    /// completion, then confirm a published access unit survives the RTMP round
    /// trip back into an FLV stream. (The sandbox blocks live RTMP anyway.)
    #[test]
    fn publisher_round_trips_media_to_a_server_session() {
        use crate::flv::{FlvDemuxer, FlvMuxer};

        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "secret");
        let mut server = RtmpSession::new();

        // Exchange bytes both ways until the publish ladder completes.
        for _ in 0..8 {
            let to_server = publisher.take_outbound();
            if !to_server.is_empty() {
                server.push(&to_server);
            }
            let to_pub = server.take_outbound();
            if !to_pub.is_empty() {
                publisher.push(&to_pub);
            }
            if publisher.is_publishing() {
                break;
            }
        }
        assert!(publisher.is_publishing(), "publisher reached Publish.Start");
        assert!(server.is_publishing(), "server accepted the publish");

        // Publish one keyframe access unit as an FLV stream; recover it server-side.
        let au = [0u8, 0, 0, 3, 0x65, 0x11, 0x22]; // 4-byte length=3 + NAL
        let mut mux = FlvMuxer::new(FlvTrack::Video);
        let flv = mux.push_au(&au, 40);
        publisher.push_flv(&flv);
        server.push(&publisher.take_outbound());

        let mut demux = FlvDemuxer::new();
        demux.push_data(&server.take_flv());
        let units = demux.take_units();
        assert_eq!(units.len(), 1, "one access unit survived the RTMP round trip");
        assert_eq!(units[0], FlvUnitView(FlvTrack::Video, au.to_vec(), 40));
    }

    /// The publisher and server negotiate the digest (complex) handshake, not
    /// the simple one: C1 carries a valid FP digest, the server answers with a
    /// valid FMS S1 digest, and each side's response proves it validated the
    /// other's digest, exactly as a genuine-FMS server + Flash client check.
    #[cfg(feature = "rtmp")]
    #[test]
    fn publisher_and_server_negotiate_the_complex_handshake() {
        use crate::rtmphandshake::{
            c1_has_digest, genuine_fms_key, genuine_fp_key, own_digest_scheme1, verify_response,
            SIG_SIZE,
        };

        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        // C0 (1) + C1 (1536): the publisher requests the complex handshake.
        let c0c1 = publisher.take_outbound();
        assert_eq!(c0c1.len(), 1 + SIG_SIZE, "C0 + C1 queued up front");
        let c1 = &c0c1[1..];
        assert!(c1_has_digest(c1), "publisher C1 carries a valid FP digest");

        let mut server = RtmpSession::new();
        server.push(&c0c1);
        let s = server.take_outbound();
        assert!(s.len() > 2 * SIG_SIZE, "server replied S0 + S1 + S2");
        let s1 = &s[1..1 + SIG_SIZE];
        let s2 = &s[1 + SIG_SIZE..1 + 2 * SIG_SIZE];
        // The server's S2 proves it validated our C1 digest (using the full FMS key).
        let client_c1_digest = own_digest_scheme1(c1);
        assert!(
            verify_response(s2, &client_c1_digest, &genuine_fms_key()),
            "server S2 proves it validated the publisher's C1 digest"
        );

        // Feed S0S1S2 back; the client's C2 proves it validated the server's S1.
        publisher.push(&s);
        let c2 = publisher.take_outbound();
        assert!(c2.len() >= SIG_SIZE, "publisher emitted C2");
        let server_s1_digest = own_digest_scheme1(s1);
        assert!(
            verify_response(&c2[..SIG_SIZE], &server_s1_digest, &genuine_fp_key()),
            "publisher C2 proves it validated the server's S1 digest"
        );
    }

    /// A payload larger than the negotiated chunk size must fragment on the
    /// publisher side and reassemble on the server side.
    #[test]
    fn publisher_fragments_large_media_across_chunks() {
        use crate::flv::{FlvDemuxer, FlvMuxer};

        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        let mut server = RtmpSession::new();
        for _ in 0..8 {
            let to_server = publisher.take_outbound();
            if !to_server.is_empty() {
                server.push(&to_server);
            }
            let to_pub = server.take_outbound();
            if !to_pub.is_empty() {
                publisher.push(&to_pub);
            }
            if publisher.is_publishing() {
                break;
            }
        }
        assert!(publisher.is_publishing());

        // A NAL well past PUB_CHUNK_SIZE forces fmt-0 + fmt-3 fragmentation.
        let nal: Vec<u8> = (0..9000u32).map(|i| (i as u8).wrapping_mul(31)).collect();
        let mut au = (nal.len() as u32).to_be_bytes().to_vec();
        au.extend_from_slice(&nal);
        let mut mux = FlvMuxer::new(FlvTrack::Video);
        let flv = mux.push_au(&au, 100);
        publisher.push_flv(&flv);
        server.push(&publisher.take_outbound());

        let mut demux = FlvDemuxer::new();
        demux.push_data(&server.take_flv());
        let units = demux.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, au, "the fragmented access unit reassembled byte-exact");
    }

    /// Drive a coupled publisher + server through the publish ladder, returning
    /// both once media may flow.
    fn coupled_to_publishing(server: &mut RtmpSession) -> RtmpPublisher {
        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        for _ in 0..8 {
            let to_server = publisher.take_outbound();
            if !to_server.is_empty() {
                server.push(&to_server);
            }
            let to_pub = server.take_outbound();
            if !to_pub.is_empty() {
                publisher.push(&to_pub);
            }
            if publisher.is_publishing() {
                break;
            }
        }
        assert!(publisher.is_publishing());
        publisher
    }

    /// Window-acknowledgement flow control (RFC §5.4.3): the session advertises a
    /// small window at connect, the publisher learns it, and after a window's
    /// worth of unacknowledged media the publisher reports `throttled`. Feeding it
    /// the session's Acknowledgement clears the throttle (back-pressure release).
    #[test]
    fn publisher_throttles_until_the_server_acknowledges() {
        use crate::flv::{FlvMuxer, FlvTrack};

        // A 4 KB window so a couple of small tags trip it.
        let mut server = RtmpSession::new().with_window_ack_size(4096);
        let mut publisher = coupled_to_publishing(&mut server);
        // The publisher learned the server's advertised window at connect.
        assert_eq!(publisher.peer_window(), 4096, "publisher learned the ack window");
        assert!(!publisher.throttled(), "not throttled before sending media");

        // Push media well past the window without delivering the bytes to the
        // server (so no acknowledgement comes back): the publisher must throttle.
        let mut mux = FlvMuxer::new(FlvTrack::Video);
        let big: Vec<u8> = (0..6000u32).map(|i| i as u8).collect();
        let mut au = (big.len() as u32).to_be_bytes().to_vec();
        au.extend_from_slice(&big);
        let flv = mux.push_au(&au, 10);
        publisher.push_flv(&flv);
        let media = publisher.take_outbound();
        assert!(media.len() as u32 >= 4096, "queued more than one window of media");
        assert!(publisher.throttled(), "unacked media past the window throttles the publisher");

        // Deliver it to the server; it acknowledges after a window of bytes.
        server.push(&media);
        let ack = server.take_outbound();
        assert!(!ack.is_empty(), "server sent an Acknowledgement after a window of bytes");
        // The control message is type 3 (Acknowledgement) on chunk stream 2.
        assert_eq!(ack[0] & 0x3F, 2, "ack rides the protocol-control chunk stream");
        assert_eq!(ack[7], MSG_ACK, "message type is Acknowledgement (3)");

        // Feeding the ack back to the publisher advances its window and clears it.
        publisher.push(&ack);
        assert!(!publisher.throttled(), "the server's acknowledgement released the window");
        assert!(publisher.unacked_bytes() < 4096, "unacked backlog dropped below the window");
    }

    /// Compact comparator for `FlvUnit` in assertions.
    #[derive(Debug, PartialEq, Eq)]
    struct FlvUnitView(FlvTrack, Vec<u8>, u32);
    impl PartialEq<FlvUnitView> for crate::flv::FlvUnit {
        fn eq(&self, o: &FlvUnitView) -> bool {
            self.track == o.0 && self.data == o.1 && self.pts_ms == o.2
        }
    }
}
