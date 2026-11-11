// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Sans-IO RTMP protocol, both directions. Pure `no_std + alloc`, no sockets.
//!
//! - [`RtmpSession`] is the ingest (server) side, the transport half of
//!   `RtmpSrc`: feed received bytes with [`RtmpSession::push`],
//!   drain the peer responses ([`take_outbound`](RtmpSession::take_outbound)) and
//!   the demuxed FLV byte stream ([`take_flv`](RtmpSession::take_flv)).
//! - [`RtmpPublisher`] is the egress (client) side, the transport half of
//!   `RtmpSink`: it connects out and *publishes*. Drain
//!   [`take_outbound`](RtmpPublisher::take_outbound) to the socket, feed the
//!   socket's bytes to [`push`](RtmpPublisher::push), and once
//!   [`is_publishing`](RtmpPublisher::is_publishing) feed an FLV byte stream to
//!   [`push_flv`](RtmpPublisher::push_flv); its audio/video tags are reframed
//!   into RTMP messages on the outbound buffer.
//!
//! An RTMP audio/video message payload is exactly an FLV tag *body*, so the two
//! halves are inverses: the session demuxes RTMP messages into an FLV stream,
//! the publisher reframes an FLV stream back into RTMP messages. Both share the
//! `ChunkReader` reassembly, the AMF0 codec, and the simple (non-digest)
//! handshake ffmpeg/OBS use. Scope: one stream, H.264 + AAC, AMF0 only. Verified
//! against the Adobe RTMP 1.0 spec.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::flv::{
    emit_tag, file_header_len, put_file_header, tag_at, write_u24, FLV_FLAG_AUDIO, FLV_FLAG_VIDEO,
};

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
const AMF0_UNDEFINED: u8 = 0x06;
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
        Self {
            inbound: Vec::new(),
            chunk_size: DEFAULT_CHUNK_SIZE,
            streams: BTreeMap::new(),
        }
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
                (
                    self.inbound[2] as u32 * 256 + self.inbound[1] as u32 + 64,
                    3,
                )
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
            (
                s.timestamp,
                s.msg_length,
                s.msg_type,
                s.msg_stream_id,
                s.ext_timestamp,
                s.payload.len(),
            )
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
        let needs_ext = if fmt == 3 {
            prev_ext
        } else {
            ts_field == EXTENDED_TIMESTAMP_MARKER
        };
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
            ChunkStep::Message(
                stream.msg_type,
                stream.timestamp,
                core::mem::take(&mut stream.payload),
            )
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

    /// Fuzzing entry: feed an attacker-controlled chunk stream straight into a
    /// post-handshake server session (chunk reassembly + AMF0 command parsing,
    /// the remote surface a malicious publisher reaches). The server handshake
    /// accepts any C1/C2, so forcing `Streaming` faithfully models that state.
    /// Exposed only under `--cfg fuzzing` so the public API is unchanged.
    #[cfg(fuzzing)]
    pub fn fuzz_feed_chunk_stream(data: &[u8]) {
        let mut session = Self::new();
        session.phase = Phase::Streaming;
        session.push(data);
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
        let Some(name) = amf0_read_string(payload, &mut at) else {
            return;
        };
        let txn = amf0_read_number(payload, &mut at).unwrap_or(0.0);
        match name.as_str() {
            "connect" => {
                self.send_control(MSG_WINDOW_ACK_SIZE, &self.window_ack_size.to_be_bytes());
                let mut bw = self.window_ack_size.to_be_bytes().to_vec();
                bw.push(2); // dynamic limit
                self.send_control(MSG_SET_PEER_BW, &bw);
                self.send_control(
                    MSG_SET_CHUNK_SIZE,
                    &(DEFAULT_CHUNK_SIZE as u32).to_be_bytes(),
                );
                let mut body = Vec::new();
                amf0_string(&mut body, "_result");
                amf0_number(&mut body, txn);
                amf0_object(
                    &mut body,
                    &[
                        ("fmsVer", AmfVal::Str("FMS/3,0,1,123")),
                        ("capabilities", AmfVal::Num(31.0)),
                    ],
                );
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
        write_message(
            &mut self.outbound,
            2,
            msg_type,
            0,
            0,
            payload,
            DEFAULT_CHUNK_SIZE,
        );
    }

    /// Send an AMF0 command reply (chunk stream id 3) on `msg_stream_id`. Larger
    /// replies (the `connect` result) fragment at the advertised chunk size.
    fn send_command(&mut self, msg_stream_id: u32, payload: &[u8]) {
        write_message(
            &mut self.outbound,
            3,
            MSG_AMF0_COMMAND,
            msg_stream_id,
            0,
            payload,
            DEFAULT_CHUNK_SIZE,
        );
    }

    /// Append one FLV tag (the RTMP message payload is the tag body) to the FLV
    /// output, writing the FLV header before the first tag.
    fn write_flv_tag(&mut self, tag_type: u8, timestamp: u32, body: &[u8]) {
        if !self.flv_header_written {
            put_file_header(&mut self.flv, FLV_FLAG_AUDIO | FLV_FLAG_VIDEO);
            self.flv_header_written = true;
        }
        emit_tag(&mut self.flv, tag_type, timestamp, &[body]);
    }
}

/// The outbound chunk size the publisher negotiates up front (the 128-byte
/// default would fragment every video keyframe into dozens of chunks).
const PUB_CHUNK_SIZE: usize = 4096;

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

/// Sans-IO RTMP publish client, the transport half of `RtmpSink`.
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
    // AMF0 values appended to the `connect` command, from `encode_extra_connect_args`
    extra_connect_args: Vec<u8>,
}

impl RtmpPublisher {
    /// `app` is the RTMP application (first URL path segment), `stream_key` the
    /// rest, and `tc_url` the `rtmp://host[:port]/app` the server echoes back.
    pub fn new(
        app: impl Into<String>,
        tc_url: impl Into<String>,
        stream_key: impl Into<String>,
    ) -> Self {
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
            extra_connect_args: Vec::new(),
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

    pub fn with_extra_connect_args(mut self, encoded: Vec<u8>) -> Self {
        self.extra_connect_args = encoded;
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
    pub fn push(&mut self, data: &[u8]) -> Result<(), PublishRefusal> {
        if self.phase == PubPhase::Handshake {
            self.s_inbound.extend_from_slice(data);
            // Wait for the full S0 + S1 + S2 before replying (the simple handshake
            // servers send all three together after C0/C1).
            if self.s_inbound.len() < 1 + 2 * HANDSHAKE_SIZE {
                return Ok(());
            }
            // C2: a digest response keyed off the server's S1 digest (complex
            // handshake), or an echo of S1 (simple / fallback). S2 is not
            // validated. Server bytes are S0 at [0], S1 at [1 .. 1+1536].
            let s1 = self.s_inbound[1..1 + HANDSHAKE_SIZE].to_vec();
            self.outbound
                .extend_from_slice(&client_c2(self.complex, &s1));
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
            self.handle_message(msg_type, &payload)?;
        }
        Ok(())
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
    /// is needed to go on. An `_error` or an error `onStatus` ends the publish.
    fn handle_message(&mut self, msg_type: u8, payload: &[u8]) -> Result<(), PublishRefusal> {
        // Flow-control control messages (RFC §5.4.3-5.4.4): the server's Window
        // Acknowledgement Size bounds the publisher's in-flight bytes, and each
        // Acknowledgement advances the acknowledged sequence, releasing the window.
        match msg_type {
            MSG_WINDOW_ACK_SIZE => {
                if let Some(w) = payload.get(..4) {
                    self.peer_window = u32::from_be_bytes([w[0], w[1], w[2], w[3]]);
                    // mediamtx acks only a publisher that sends its own window, half keeps acks ahead of the throttle
                    let own_window = (self.peer_window / 2).max(1);
                    self.send_control(MSG_WINDOW_ACK_SIZE, &own_window.to_be_bytes());
                }
                return Ok(());
            }
            MSG_ACK => {
                if let Some(w) = payload.get(..4) {
                    self.ack_seq = u32::from_be_bytes([w[0], w[1], w[2], w[3]]);
                }
                return Ok(());
            }
            MSG_AMF0_COMMAND => {}
            _ => return Ok(()), // set-peer-bw / user-control: ignored
        }
        let mut at = 0;
        let Some(name) = amf0_read_string(payload, &mut at) else {
            return Ok(());
        };
        match self.phase {
            PubPhase::WaitConnect if name == "_result" => {
                self.send_create_stream();
                self.phase = PubPhase::WaitCreate;
            }
            PubPhase::WaitConnect | PubPhase::WaitCreate if name == "_error" => {
                return Err(PublishRefusal::from_info(command_info(payload, at)));
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
                let info = command_info(payload, at);
                if info.get("level").is_some_and(|level| level == "error") {
                    return Err(PublishRefusal::from_info(info));
                }
                if info
                    .get("code")
                    .is_none_or(|code| code != "NetStream.Publish.Start")
                {
                    return Ok(());
                }
                self.phase = PubPhase::Publishing;
                let pending = core::mem::take(&mut self.pending_media);
                for (tag_type, ts, body) in pending {
                    self.send_media(tag_type, ts, &body);
                }
            }
            _ => {}
        }
        Ok(())
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
        body.extend_from_slice(&self.extra_connect_args);
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

    pub fn unpublish(&mut self) {
        // gst rtmp2sink and ffmpeg both send these two commands before closing
        let mut fc_unpublish = Vec::new();
        amf0_string(&mut fc_unpublish, "FCUnpublish");
        amf0_number(&mut fc_unpublish, 4.0);
        fc_unpublish.push(AMF0_NULL);
        amf0_string(&mut fc_unpublish, &self.stream_key);
        self.send_command(0, &fc_unpublish);

        let mut delete_stream = Vec::new();
        amf0_string(&mut delete_stream, "deleteStream");
        amf0_number(&mut delete_stream, 5.0);
        delete_stream.push(AMF0_NULL);
        amf0_number(&mut delete_stream, f64::from(self.stream_id));
        self.send_command(0, &delete_stream);
    }

    /// Send a protocol control message (chunk stream id 2, message stream id 0).
    fn send_control(&mut self, msg_type: u8, payload: &[u8]) {
        write_message(
            &mut self.outbound,
            2,
            msg_type,
            0,
            0,
            payload,
            PUB_CHUNK_SIZE,
        );
    }

    /// Send an AMF0 command (chunk stream id 3) on `msg_stream_id`.
    fn send_command(&mut self, msg_stream_id: u32, payload: &[u8]) {
        write_message(
            &mut self.outbound,
            3,
            MSG_AMF0_COMMAND,
            msg_stream_id,
            0,
            payload,
            PUB_CHUNK_SIZE,
        );
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
            timestamp,
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
            let header_len = file_header_len(&self.buf).filter(|len| self.buf.len() >= *len)?;
            self.buf.drain(..header_len);
            self.header_consumed = true;
        }
        let (tag, body) = tag_at(&self.buf, 0)?;
        let tag_body = self.buf[body.clone()].to_vec();
        self.buf.drain(..body.end);
        Some((tag.tag_type, tag.timestamp_ms, tag_body))
    }
}

/// A 24-bit timestamp field holding this value means a 4-byte extended
/// timestamp follows the message header.
const EXTENDED_TIMESTAMP_MARKER: u32 = 0xFF_FFFF;

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
    let extended = timestamp >= EXTENDED_TIMESTAMP_MARKER;
    let mut off = 0;
    let mut first = true;
    loop {
        if first {
            out.push(basic); // fmt 0
            write_u24(out, timestamp.min(EXTENDED_TIMESTAMP_MARKER));
            write_u24(out, payload.len() as u32);
            out.push(msg_type);
            out.extend_from_slice(&msg_stream_id.to_le_bytes());
            first = false;
        } else {
            out.push((3 << 6) | basic); // fmt 3 continuation
        }
        // rtmp repeats the extended timestamp on fmt 3 chunks of a message that has one
        if extended {
            out.extend_from_slice(&timestamp.to_be_bytes());
        }
        let take = (payload.len() - off).min(chunk_size.max(1));
        out.extend_from_slice(&payload[off..off + take]);
        off += take;
        if off >= payload.len() {
            break;
        }
    }
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

// the string entries of an AMF0 object, servers put only scalars in command reply objects
fn amf0_read_scalar_object(buf: &[u8], at: &mut usize) -> Option<BTreeMap<String, String>> {
    if *buf.get(*at)? != AMF0_OBJECT {
        return None;
    }
    *at += 1;
    let mut strings = BTreeMap::new();
    loop {
        let key_len = u16::from_be_bytes(buf.get(*at..*at + 2)?.try_into().ok()?) as usize;
        let key = core::str::from_utf8(buf.get(*at + 2..*at + 2 + key_len)?).ok()?;
        *at += 2 + key_len;
        match *buf.get(*at)? {
            AMF0_OBJECT_END if key_len == 0 => {
                *at += 1;
                return Some(strings);
            }
            AMF0_STRING => {
                let value = amf0_read_string(buf, at)?;
                strings.insert(String::from(key), value);
            }
            AMF0_NUMBER => {
                amf0_read_number(buf, at)?;
            }
            AMF0_BOOLEAN => {
                buf.get(*at + 1)?;
                *at += 2;
            }
            AMF0_NULL | AMF0_UNDEFINED => *at += 1,
            _ => return None,
        }
    }
}

// `_error` and `onStatus` carry the transaction id, null or a command object, then the info object
fn command_info(payload: &[u8], mut at: usize) -> BTreeMap<String, String> {
    let mut read = || {
        amf0_read_number(payload, &mut at)?;
        if payload.get(at) == Some(&AMF0_NULL) {
            at += 1;
        } else {
            amf0_read_scalar_object(payload, &mut at)?;
        }
        amf0_read_scalar_object(payload, &mut at)
    };
    read().unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishRefusal {
    pub code: String,
    pub description: String,
}

impl PublishRefusal {
    fn from_info(mut info: BTreeMap<String, String>) -> Self {
        Self {
            code: info.remove("code").unwrap_or_default(),
            description: info.remove("description").unwrap_or_default(),
        }
    }
}

// gst rtmp2sink's `extra-connect-args`, split on `conn=` as gst does so a string keeps its spaces, plus librtmp's objects and named values
pub fn encode_extra_connect_args(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut object_depth = 0usize;
    for item in text
        .split("conn=")
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let (kind, rest) = item.split_once(':')?;
        let (kind, data) = match kind.strip_prefix('N').filter(|kind| kind.len() == 1) {
            Some(kind) => {
                let (name, data) = rest.split_once(':')?;
                // AMF0 objects hold only named entries, and only objects name them
                if object_depth == 0 || kind == "O" && data == "0" {
                    return None;
                }
                out.extend_from_slice(&u16::try_from(name.len()).ok()?.to_be_bytes());
                out.extend_from_slice(name.as_bytes());
                (kind, data)
            }
            None => {
                let ends_object = kind == "O" && rest == "0";
                if (object_depth == 0) == ends_object {
                    return None;
                }
                (kind, rest)
            }
        };
        match (kind, data) {
            ("B", "0" | "1") => out.extend_from_slice(&[AMF0_BOOLEAN, u8::from(data == "1")]),
            ("N", _) => amf0_number(&mut out, data.parse().ok()?),
            ("S", _) => {
                u16::try_from(data.len()).ok()?;
                amf0_string(&mut out, data);
            }
            ("Z", _) => out.push(AMF0_NULL),
            ("O", "1") => {
                out.push(AMF0_OBJECT);
                object_depth += 1;
            }
            ("O", "0") => {
                out.extend_from_slice(&[0, 0, AMF0_OBJECT_END]);
                object_depth -= 1;
            }
            _ => return None,
        }
    }
    (object_depth == 0).then_some(out)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flv::{FlvCodec, FlvDemuxer, FlvTrack};
    use alloc::vec;

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
        write_u24(&mut c, ts & 0x00FF_FFFF);
        write_u24(&mut c, payload.len() as u32);
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
        assert_eq!(
            &s2[8..],
            &c0c1[1 + 8..1 + HANDSHAKE_SIZE],
            "S2 echoes C1's random bytes"
        );
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
            reply
                .windows("NetConnection.Connect.Success".len())
                .any(|w| w == b"NetConnection.Connect.Success"),
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
            reply
                .windows("NetStream.Publish.Start".len())
                .any(|w| w == b"NetStream.Publish.Start"),
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

        let flv = s.take_flv();
        let mut demux = FlvDemuxer::new();
        demux.push_data(&flv);
        let units = demux.take_units();
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], FlvUnitView(FlvTrack::Video, au.to_vec(), 33));
        assert_eq!(units[1], FlvUnitView(FlvTrack::Audio, frame.to_vec(), 33));
        // gst's flvdemux drops a last tag that no PreviousTagSize follows
        let last_tag_size = (crate::flv::TAG_HEADER_LEN + abody.len()) as u32;
        assert_eq!(flv[flv.len() - 4..], last_tag_size.to_be_bytes());
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
        write_u24(&mut bytes, 0);
        write_u24(&mut bytes, vbody.len() as u32);
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
        assert_eq!(
            units[0].data, expected,
            "the split message reassembled byte-exact"
        );
    }

    /// The publisher (RtmpSink's transport) and the session (RtmpSrc's
    /// transport) are inverses, so pitting them against each other with no
    /// socket proves the egress path: drive the handshake + publish ladder to
    /// completion, then confirm a published access unit survives the RTMP round
    /// trip back into an FLV stream. (The sandbox blocks live RTMP anyway.)
    fn publishing_pair() -> (RtmpPublisher, RtmpSession) {
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
                publisher.push(&to_pub).unwrap();
            }
            if publisher.is_publishing() {
                break;
            }
        }
        assert!(publisher.is_publishing(), "publisher reached Publish.Start");
        assert!(server.is_publishing(), "server accepted the publish");
        (publisher, server)
    }

    #[test]
    fn publisher_round_trips_media_to_a_server_session() {
        use crate::flv::{FlvDemuxer, FlvMuxer};

        let (mut publisher, mut server) = publishing_pair();

        // Publish one keyframe access unit as an FLV stream; recover it server-side.
        let au = [0u8, 0, 0, 3, 0x65, 0x11, 0x22]; // 4-byte length=3 + NAL
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let flv = mux.push_au(&au, 40);
        publisher.push_flv(&flv);
        server.push(&publisher.take_outbound());

        let mut demux = FlvDemuxer::new();
        demux.push_data(&server.take_flv());
        let units = demux.take_units();
        assert_eq!(
            units.len(),
            1,
            "one access unit survived the RTMP round trip"
        );
        assert_eq!(units[0], FlvUnitView(FlvTrack::Video, au.to_vec(), 40));
    }

    /// A publish running past the 24-bit timestamp range (about 4 h 40 min) keeps
    /// its real timestamps through the extended-timestamp field, including on the
    /// continuation chunks of a message larger than the chunk size.
    #[test]
    fn publisher_sends_extended_timestamps() {
        use crate::flv::{FlvDemuxer, FlvMuxer};

        let (mut publisher, mut server) = publishing_pair();
        let nal: Vec<u8> = (0..2 * PUB_CHUNK_SIZE as u32).map(|i| i as u8).collect();
        let mut au = (nal.len() as u32).to_be_bytes().to_vec();
        au.extend_from_slice(&nal);
        let timestamps_ms = [
            EXTENDED_TIMESTAMP_MARKER - 1,
            EXTENDED_TIMESTAMP_MARKER,
            20_000_000,
        ];
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        for timestamp_ms in timestamps_ms {
            publisher.push_flv(&mux.push_au(&au, timestamp_ms));
        }
        server.push(&publisher.take_outbound());

        let mut demux = FlvDemuxer::new();
        demux.push_data(&server.take_flv());
        let units = demux.take_units();
        let received: Vec<u32> = units.iter().map(|u| u.pts_ms).collect();
        assert_eq!(received, timestamps_ms);
        assert!(
            units.iter().all(|u| u.data == au),
            "payloads survived the chunking"
        );
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
        publisher.push(&s).unwrap();
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
                publisher.push(&to_pub).unwrap();
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
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let flv = mux.push_au(&au, 100);
        publisher.push_flv(&flv);
        server.push(&publisher.take_outbound());

        let mut demux = FlvDemuxer::new();
        demux.push_data(&server.take_flv());
        let units = demux.take_units();
        assert_eq!(units.len(), 1);
        assert_eq!(
            units[0].data, au,
            "the fragmented access unit reassembled byte-exact"
        );
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
                publisher.push(&to_pub).unwrap();
            }
            if publisher.is_publishing() {
                break;
            }
        }
        assert!(publisher.is_publishing());
        publisher
    }

    // a publisher that has sent `connect` and waits for its reply
    fn publisher_awaiting_connect(publisher: &mut RtmpPublisher) -> Vec<u8> {
        let mut server = RtmpSession::new();
        server.push(&publisher.take_outbound());
        publisher.push(&server.take_outbound()).unwrap();
        publisher.take_outbound()
    }

    fn status_reply(name: &str, level: &str, code: &str) -> Vec<u8> {
        let mut body = Vec::new();
        amf_string(&mut body, name);
        amf_number(&mut body, 0.0);
        body.push(AMF0_NULL);
        amf0_object(
            &mut body,
            &[("level", AmfVal::Str(level)), ("code", AmfVal::Str(code))],
        );
        chunk(3, MSG_AMF0_COMMAND, 0, 0, &body)
    }

    #[test]
    fn a_connect_error_fails_the_publish() {
        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        publisher_awaiting_connect(&mut publisher);
        let rejected = status_reply("_error", "error", "NetConnection.Connect.Rejected");
        assert_eq!(
            publisher.push(&rejected),
            Err(PublishRefusal {
                code: "NetConnection.Connect.Rejected".into(),
                description: String::new(),
            })
        );
        assert!(
            publisher.take_outbound().is_empty(),
            "no createStream follows"
        );
    }

    #[test]
    fn an_error_status_for_publish_fails_the_publish() {
        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        publisher_awaiting_connect(&mut publisher);
        let mut connected = Vec::new();
        amf_string(&mut connected, "_result");
        amf_number(&mut connected, 1.0);
        publisher
            .push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &connected))
            .unwrap();
        let mut created = Vec::new();
        amf_string(&mut created, "_result");
        amf_number(&mut created, 2.0);
        created.push(AMF0_NULL);
        amf_number(&mut created, 1.0);
        publisher
            .push(&chunk(3, MSG_AMF0_COMMAND, 0, 0, &created))
            .unwrap();
        let bad_name = status_reply("onStatus", "error", "NetStream.Publish.BadName");
        assert_eq!(
            publisher.push(&bad_name),
            Err(PublishRefusal {
                code: "NetStream.Publish.BadName".into(),
                description: String::new(),
            })
        );
        assert!(!publisher.is_publishing());
    }

    #[test]
    fn extra_connect_args_follow_the_connect_object() {
        const ARGS: &str =
            "conn=B:1 conn=S:auth me conn=O:1 conn=NN:code:1.23 conn=NS:flag:ok conn=O:0 conn=Z:";
        let encoded = encode_extra_connect_args(ARGS).expect("well formed");
        let mut expected = vec![AMF0_BOOLEAN, 1];
        amf_string(&mut expected, "auth me");
        amf0_object(
            &mut expected,
            &[("code", AmfVal::Num(1.23)), ("flag", AmfVal::Str("ok"))],
        );
        expected.push(AMF0_NULL);
        assert_eq!(encoded, expected);

        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k")
            .with_extra_connect_args(encoded.clone());
        let sent = publisher_awaiting_connect(&mut publisher);
        let mut reader = ChunkReader::new();
        reader.push(&sent[HANDSHAKE_SIZE..]);
        let (_, _, connect) = core::iter::from_fn(|| reader.next_message())
            .find(|(msg_type, _, _)| *msg_type == MSG_AMF0_COMMAND)
            .expect("connect was sent");
        assert!(connect.ends_with(&encoded));
    }

    #[test]
    fn malformed_extra_connect_args_are_refused() {
        for args in [
            "conn=B:2",
            "conn=N:x",
            "conn=N:",
            "conn=NS:name:v",
            "conn=O:1 conn=S:unnamed conn=O:0",
            "conn=O:1",
            "conn=O:0",
            "conn=X:1",
            "B:1 S:x",
        ] {
            assert_eq!(encode_extra_connect_args(args), None, "{args}");
        }
        // gst reads what comes before the first `conn=` as an item too
        assert_eq!(encode_extra_connect_args("S:a"), {
            let mut out = Vec::new();
            amf_string(&mut out, "a");
            Some(out)
        });
    }

    #[test]
    fn unpublish_sends_fcunpublish_and_deletestream() {
        let mut server = RtmpSession::new();
        let mut publisher = coupled_to_publishing(&mut server);
        publisher.unpublish();
        let mut reader = ChunkReader::new();
        reader.push(&publisher.take_outbound());
        let commands: Vec<(String, Vec<u8>)> = core::iter::from_fn(|| reader.next_message())
            .map(|(msg_type, _, payload)| {
                assert_eq!(msg_type, MSG_AMF0_COMMAND);
                let mut at = 0;
                let name = amf0_read_string(&payload, &mut at).unwrap();
                amf0_read_number(&payload, &mut at).unwrap();
                assert_eq!(payload[at], AMF0_NULL);
                (name, payload[at + 1..].to_vec())
            })
            .collect();
        let mut stream_key = Vec::new();
        amf_string(&mut stream_key, "k");
        let mut stream_id = Vec::new();
        amf_number(&mut stream_id, f64::from(publisher.stream_id));
        assert_eq!(
            commands,
            vec![
                (String::from("FCUnpublish"), stream_key),
                (String::from("deleteStream"), stream_id),
            ]
        );
    }

    /// Window-acknowledgement flow control (RFC §5.4.3): the session advertises a
    /// small window at connect, the publisher learns it, and after a window's
    /// worth of unacknowledged media the publisher reports `throttled`. Feeding it
    /// the session's Acknowledgement clears the throttle (back-pressure release).
    #[test]
    fn publisher_throttles_until_the_server_acknowledges() {
        use crate::flv::{FlvCodec, FlvMuxer};

        // A 4 KB window so a couple of small tags trip it.
        let mut server = RtmpSession::new().with_window_ack_size(4096);
        let mut publisher = coupled_to_publishing(&mut server);
        // The publisher learned the server's advertised window at connect.
        assert_eq!(
            publisher.peer_window(),
            4096,
            "publisher learned the ack window"
        );
        assert!(!publisher.throttled(), "not throttled before sending media");

        // Push media well past the window without delivering the bytes to the
        // server (so no acknowledgement comes back): the publisher must throttle.
        let mut mux = FlvMuxer::new(FlvCodec::H264);
        let big: Vec<u8> = (0..6000u32).map(|i| i as u8).collect();
        let mut au = (big.len() as u32).to_be_bytes().to_vec();
        au.extend_from_slice(&big);
        let flv = mux.push_au(&au, 10);
        publisher.push_flv(&flv);
        let media = publisher.take_outbound();
        assert!(
            media.len() as u32 >= 4096,
            "queued more than one window of media"
        );
        assert!(
            publisher.throttled(),
            "unacked media past the window throttles the publisher"
        );

        // Deliver it to the server; it acknowledges after a window of bytes.
        server.push(&media);
        let ack = server.take_outbound();
        assert!(
            !ack.is_empty(),
            "server sent an Acknowledgement after a window of bytes"
        );
        // The control message is type 3 (Acknowledgement) on chunk stream 2.
        assert_eq!(
            ack[0] & 0x3F,
            2,
            "ack rides the protocol-control chunk stream"
        );
        assert_eq!(ack[7], MSG_ACK, "message type is Acknowledgement (3)");

        // Feeding the ack back to the publisher advances its window and clears it.
        publisher.push(&ack).unwrap();
        assert!(
            !publisher.throttled(),
            "the server's acknowledgement released the window"
        );
        assert!(
            publisher.unacked_bytes() < 4096,
            "unacked backlog dropped below the window"
        );
    }

    // gortmplib, mediamtx's RTMP stack, acknowledges only after the peer's own Window Acknowledgement Size
    #[test]
    fn publisher_announces_half_the_server_window_so_the_server_acknowledges() {
        const SERVER_WINDOW: u32 = 4096;
        let mut server = RtmpSession::new().with_window_ack_size(SERVER_WINDOW);
        let mut publisher = RtmpPublisher::new("live", "rtmp://localhost/live", "k");
        let mut sent = Vec::new();
        for _ in 0..8 {
            let to_server = publisher.take_outbound();
            sent.extend_from_slice(&to_server);
            server.push(&to_server);
            publisher.push(&server.take_outbound()).unwrap();
        }
        let mut reader = ChunkReader::new();
        reader.push(&sent[1 + 2 * HANDSHAKE_SIZE..]);
        let announced = core::iter::from_fn(|| reader.next_message())
            .find(|(msg_type, _, _)| *msg_type == MSG_WINDOW_ACK_SIZE)
            .map(|(_, _, payload)| payload);
        assert_eq!(announced, Some((SERVER_WINDOW / 2).to_be_bytes().to_vec()));
    }

    /// Compact comparator for `FlvUnit` in assertions.
    #[derive(Debug, PartialEq, Eq)]
    struct FlvUnitView(FlvTrack, Vec<u8>, u32);
    impl PartialEq<FlvUnitView> for crate::flv::FlvUnit {
        fn eq(&self, o: &FlvUnitView) -> bool {
            self.track() == o.0 && self.data == o.1 && self.pts_ms == o.2
        }
    }
}

#[cfg(all(test, feature = "rtmp"))]
mod auth_tests {
    use super::*;
    use crate::rtmpauth::{adobe_response, llnw_response, query_value};
    use crate::rtmpsink::RtmpSink;
    use crate::testutil::{frame, CollectSink};
    use alloc::format;
    use alloc::string::ToString;
    use g2g_core::{
        AsyncElement, ByteStreamEncoding, Caps, G2gError, NetworkFailure, PipelinePacket, PropValue,
    };
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::process::Command;
    use std::sync::mpsc;
    use std::time::Duration;

    const USERNAME: &str = "publisher";
    const PASSWORD: &str = "s3cret";
    const SALT: &str = "c2FsdA==";
    const SERVER_CHALLENGE: &str = "Y2hhbA==";
    const OPAQUE: &str = "b3BhcXVl";
    const NONCE: &str = "abc123";
    const PEER_TIMEOUT: Duration = Duration::from_secs(30);

    fn reject(authmod: &str, reason_query: &str) -> String {
        format!("[ AccessManager.Reject ] : [ authmod={authmod} ] : ?{reason_query}")
    }

    // the refusal description for one connect, None accepts it
    fn verdict(authmod: &str, app_with_query: &str) -> Option<String> {
        let Some((app, query)) = app_with_query.split_once('?') else {
            return Some(format!(
                "[ AccessManager.Reject ] : [ code=403 need auth; authmod={authmod} ] : "
            ));
        };
        let field = |key| query_value(query, key);
        if field("authmod") != Some(authmod) || field("user") != Some(USERNAME) {
            return Some(reject(authmod, "reason=authfailed"));
        }
        let Some(response) = field("response") else {
            let challenge = match authmod {
                "adobe" => format!(
                    "reason=needauth&user={USERNAME}&salt={SALT}&challenge={SERVER_CHALLENGE}&opaque={OPAQUE}"
                ),
                _ => format!("reason=needauth&user={USERNAME}&nonce={NONCE}"),
            };
            return Some(reject(authmod, &challenge));
        };
        let expected = match authmod {
            "adobe" => adobe_response(
                USERNAME,
                PASSWORD,
                SALT,
                Some(OPAQUE),
                field("challenge").unwrap_or_default(),
            ),
            _ => llnw_response(
                USERNAME,
                PASSWORD,
                app,
                field("nonce").unwrap_or_default(),
                field("cnonce").unwrap_or_default(),
            ),
        };
        (response != expected).then(|| reject(authmod, "reason=authfailed"))
    }

    fn connect_app(msg_type: u8, payload: &[u8]) -> Option<String> {
        let mut at = 0;
        if msg_type != MSG_AMF0_COMMAND || amf0_read_string(payload, &mut at)? != "connect" {
            return None;
        }
        amf0_read_number(payload, &mut at)?;
        amf0_read_scalar_object(payload, &mut at)?.remove("app")
    }

    fn serve_connection(mut stream: TcpStream, authmod: &str) -> bool {
        let mut session = RtmpSession::new();
        let mut commands = ChunkReader::new();
        let mut handshake_left = 1 + 2 * HANDSHAKE_SIZE;
        let mut buf = [0u8; 4096];
        loop {
            let n = stream.read(&mut buf).unwrap_or(0);
            if n == 0 {
                return session.is_publishing();
            }
            let handshake = handshake_left.min(n);
            handshake_left -= handshake;
            commands.push(&buf[handshake..n]);
            session.push(&buf[..n]);
            let mut reply = session.take_outbound();
            let rejection = core::iter::from_fn(|| commands.next_message())
                .find_map(|(msg_type, _, payload)| connect_app(msg_type, &payload))
                .and_then(|app| verdict(authmod, &app));
            if let Some(description) = &rejection {
                let mut body = Vec::new();
                amf0_string(&mut body, "_error");
                amf0_number(&mut body, 1.0);
                body.push(AMF0_NULL);
                amf0_object(
                    &mut body,
                    &[
                        ("level", AmfVal::Str("error")),
                        ("code", AmfVal::Str("NetConnection.Connect.Rejected")),
                        ("description", AmfVal::Str(description)),
                    ],
                );
                reply.clear();
                write_message(
                    &mut reply,
                    3,
                    MSG_AMF0_COMMAND,
                    0,
                    0,
                    &body,
                    DEFAULT_CHUNK_SIZE,
                );
            }
            stream.write_all(&reply).unwrap();
            if rejection.is_some() {
                // linux resets a socket closed with unread input, which can drop the refusal
                while stream.read(&mut buf).unwrap_or(0) > 0 {}
                return false;
            }
        }
    }

    // serves `connects` connections, true when the last one published
    fn spawn_server(authmod: &'static str, connects: usize) -> (u16, mpsc::Receiver<bool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (published, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut last_published = false;
            for stream in listener.incoming().take(connects) {
                last_published = serve_connection(stream.unwrap(), authmod);
            }
            published.send(last_published).unwrap();
        });
        (port, receiver)
    }

    fn publish_with_sink(sink: &mut RtmpSink) -> Result<(), G2gError> {
        let mut flv = Vec::new();
        put_file_header(&mut flv, FLV_FLAG_VIDEO);
        emit_tag(&mut flv, MSG_VIDEO, 0, &[&[0x17, 1, 0, 0, 0]]);
        sink.configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Flv,
        })?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let mut out = CollectSink::default();
        runtime.block_on(async {
            sink.process(PipelinePacket::DataFrame(frame(flv)), &mut out)
                .await?;
            sink.process(PipelinePacket::Eos, &mut out).await
        })
    }

    fn set(sink: &mut RtmpSink, name: &str, value: &str) {
        sink.set_property(name, PropValue::Str(value.to_string()))
            .unwrap();
    }

    #[test]
    fn sink_answers_an_adobe_challenge_in_auto_mode() {
        let (port, published) = spawn_server("adobe", 3);
        let mut sink = RtmpSink::new(format!("rtmp://127.0.0.1:{port}/live/key"));
        set(&mut sink, "username", USERNAME);
        set(&mut sink, "password", PASSWORD);
        publish_with_sink(&mut sink).unwrap();
        assert!(published.recv_timeout(PEER_TIMEOUT).unwrap());
    }

    #[test]
    fn sink_answers_a_limelight_challenge_with_url_credentials() {
        let (port, published) = spawn_server("llnw", 2);
        let mut sink = RtmpSink::new(format!(
            "rtmp://{USERNAME}:{PASSWORD}@127.0.0.1:{port}/live/key"
        ));
        set(&mut sink, "authmod", "llnw");
        publish_with_sink(&mut sink).unwrap();
        assert!(published.recv_timeout(PEER_TIMEOUT).unwrap());
    }

    #[test]
    fn sink_with_a_wrong_password_fails_the_publish() {
        let (port, published) = spawn_server("adobe", 3);
        let mut sink = RtmpSink::new(format!("rtmp://127.0.0.1:{port}/live/key"));
        set(&mut sink, "username", USERNAME);
        set(&mut sink, "password", "wrong");
        assert_eq!(
            publish_with_sink(&mut sink),
            Err(G2gError::Network(NetworkFailure::Connect))
        );
        assert!(!published.recv_timeout(PEER_TIMEOUT).unwrap());
    }

    fn reference_client_publishes(authmod: &'static str, client: impl Fn(u16) -> Command) {
        let (port, published) = spawn_server(authmod, 3);
        let mut child = client(port).spawn().expect("reference client runs");
        let result = published.recv_timeout(PEER_TIMEOUT);
        child.kill().ok();
        child.wait().unwrap();
        assert_eq!(result, Ok(true), "{authmod}");
    }

    #[test]
    #[ignore = "needs gst-launch-1.0 with rtmp2sink and x264enc, opens a local TCP port"]
    fn gst_rtmp2sink_passes_the_auth_checks() {
        for authmod in ["adobe", "llnw"] {
            reference_client_publishes(authmod, |port| {
                let mut gst = Command::new("gst-launch-1.0");
                gst.args([
                    "-q",
                    "videotestsrc",
                    "num-buffers=10",
                    "!",
                    "x264enc",
                    "!",
                    "flvmux",
                    "!",
                    "rtmp2sink",
                    &format!("location=rtmp://127.0.0.1:{port}/live/key"),
                    &format!("username={USERNAME}"),
                    &format!("password={PASSWORD}"),
                ]);
                gst
            });
        }
    }

    #[test]
    #[ignore = "needs ffmpeg with libx264, opens a local TCP port"]
    fn ffmpeg_passes_the_auth_checks() {
        for authmod in ["adobe", "llnw"] {
            reference_client_publishes(authmod, |port| {
                let mut ffmpeg = Command::new("ffmpeg");
                ffmpeg.args([
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=1:size=64x64:rate=10",
                    "-c:v",
                    "libx264",
                    "-f",
                    "flv",
                    &format!("rtmp://{USERNAME}:{PASSWORD}@127.0.0.1:{port}/live/key"),
                ]);
                ffmpeg
            });
        }
    }
}
