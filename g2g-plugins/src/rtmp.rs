//! Sans-IO RTMP ingest protocol (server side), the transport half of
//! [`RtmpSrc`](crate::rtmpsrc). Pure `no_std + alloc`, no sockets: feed received
//! bytes with [`RtmpSession::push`], then drain the bytes to send back to the
//! peer ([`take_outbound`](RtmpSession::take_outbound)) and the FLV byte stream
//! to forward downstream ([`take_flv`](RtmpSession::take_flv)).
//!
//! Scope: the simple (non-digest) handshake ffmpeg/OBS publishers use, the chunk
//! stream protocol, the `connect` / `createStream` / `publish` command flow, and
//! audio/video messages. An RTMP audio/video message payload is exactly an FLV
//! tag *body*, so the session reframes the messages into an FLV byte stream
//! (`flvdemux` then recovers the H.264 / AAC access units). One publisher, one
//! stream; AMF0 only. Verified against the Adobe RTMP 1.0 spec.

use alloc::vec::Vec;

/// RTMP protocol version (the C0/S0 byte).
const RTMP_VERSION: u8 = 3;
/// C1/S1/C2/S2 are each 1536 bytes.
const HANDSHAKE_SIZE: usize = 1536;
/// Default chunk size before a `Set Chunk Size` changes it.
const DEFAULT_CHUNK_SIZE: usize = 128;

// RTMP message type ids.
const MSG_SET_CHUNK_SIZE: u8 = 1;
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

/// Sans-IO RTMP receive session: a publisher-side state machine producing the
/// peer responses and the demuxed FLV byte stream.
#[derive(Debug)]
pub struct RtmpSession {
    phase: Phase,
    inbound: Vec<u8>,
    outbound: Vec<u8>,
    flv: Vec<u8>,
    flv_header_written: bool,
    flv_prev_tag_size: u32,
    chunk_size: usize,
    streams: Vec<(u32, ChunkStream)>,
    publishing: bool,
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
            inbound: Vec::new(),
            outbound: Vec::new(),
            flv: Vec::new(),
            flv_header_written: false,
            flv_prev_tag_size: 0,
            chunk_size: DEFAULT_CHUNK_SIZE,
            streams: Vec::new(),
            publishing: false,
        }
    }

    /// Whether the publisher has reached `NetStream.Publish.Start` (media flows).
    pub fn is_publishing(&self) -> bool {
        self.publishing
    }

    /// Feed received bytes and advance the state machine.
    pub fn push(&mut self, data: &[u8]) {
        self.inbound.extend_from_slice(data);
        loop {
            let progressed = match self.phase {
                Phase::WaitC0C1 => self.try_handshake_c0c1(),
                Phase::WaitC2 => self.try_handshake_c2(),
                Phase::Streaming => self.try_chunk(),
            };
            if !progressed {
                break;
            }
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
        if self.inbound.len() < 1 + HANDSHAKE_SIZE {
            return false;
        }
        let c1 = self.inbound[1..1 + HANDSHAKE_SIZE].to_vec();
        self.inbound.drain(..1 + HANDSHAKE_SIZE);

        // S0 + S1 (time 0, zero u32, zeroed random) + S2 (echo C1).
        self.outbound.push(RTMP_VERSION);
        let mut s1 = [0u8; HANDSHAKE_SIZE];
        // A deterministic non-zero pattern in S1's random region; the simple
        // handshake does not validate it.
        for (i, b) in s1.iter_mut().enumerate().skip(8) {
            *b = (i & 0xFF) as u8;
        }
        self.outbound.extend_from_slice(&s1);
        // S2 echoes C1: its time, then time2 (0), then its 1528 random bytes.
        let mut s2 = [0u8; HANDSHAKE_SIZE];
        s2[0..4].copy_from_slice(&c1[0..4]);
        s2[8..].copy_from_slice(&c1[8..]);
        self.outbound.extend_from_slice(&s2);

        self.phase = Phase::WaitC2;
        true
    }

    fn try_handshake_c2(&mut self) -> bool {
        if self.inbound.len() < HANDSHAKE_SIZE {
            return false;
        }
        self.inbound.drain(..HANDSHAKE_SIZE); // C2 not validated (simple handshake)
        self.phase = Phase::Streaming;
        true
    }

    /// Parse one chunk from `inbound`, returning whether one was consumed. A
    /// complete message dispatches to [`Self::dispatch`].
    fn try_chunk(&mut self) -> bool {
        let len = self.inbound.len();
        if len == 0 {
            return false;
        }
        let fmt = self.inbound[0] >> 6;
        let marker = self.inbound[0] & 0x3F;
        let (csid, basic_len) = match marker {
            0 => {
                if len < 2 {
                    return false;
                }
                (self.inbound[1] as u32 + 64, 2)
            }
            1 => {
                if len < 3 {
                    return false;
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
            return false;
        }

        // Snapshot the per-stream inheritance (Copy fields + payload length) so no
        // borrow of `self` is held while reading the header and chunk size.
        self.ensure_stream(csid);
        let prev = {
            let s = &self.streams.iter().find(|(id, _)| *id == csid).expect("inserted").1;
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
            return false;
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
            return false;
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

        let dispatch = {
            let stream = &mut self
                .streams
                .iter_mut()
                .find(|(id, _)| *id == csid)
                .expect("inserted")
                .1;
            if !in_progress {
                stream.timestamp = timestamp;
                stream.msg_length = msg_length;
                stream.msg_type = msg_type;
                stream.msg_stream_id = msg_stream_id;
                stream.ext_timestamp = needs_ext;
            }
            stream.payload.extend_from_slice(&payload_bytes);
            if stream.payload.len() >= stream.msg_length {
                Some((stream.msg_type, stream.timestamp, core::mem::take(&mut stream.payload)))
            } else {
                None
            }
        };
        if let Some((mtype, mts, message)) = dispatch {
            self.dispatch(mtype, mts, &message);
        }
        true
    }

    fn ensure_stream(&mut self, csid: u32) {
        if !self.streams.iter().any(|(id, _)| *id == csid) {
            self.streams.push((csid, ChunkStream::default()));
        }
    }

    /// Act on a complete message: honor `Set Chunk Size`, answer AMF0 commands,
    /// and reframe audio/video into the FLV byte stream.
    fn dispatch(&mut self, msg_type: u8, timestamp: u32, payload: &[u8]) {
        match msg_type {
            MSG_SET_CHUNK_SIZE => {
                if payload.len() >= 4 {
                    let size = u32::from_be_bytes(payload[0..4].try_into().expect("4")) & 0x7FFF_FFFF;
                    self.chunk_size = (size as usize).max(1);
                }
            }
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
                self.send_control(MSG_WINDOW_ACK_SIZE, &2_500_000u32.to_be_bytes());
                let mut bw = 2_500_000u32.to_be_bytes().to_vec();
                bw.push(2); // dynamic limit
                self.send_control(MSG_SET_PEER_BW, &bw);
                self.send_control(MSG_SET_CHUNK_SIZE, &(self.chunk_size as u32).to_be_bytes());
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
        write_chunk(&mut self.outbound, 2, msg_type, 0, payload);
    }

    /// Send an AMF0 command reply (chunk stream id 3) on `msg_stream_id`.
    fn send_command(&mut self, msg_stream_id: u32, payload: &[u8]) {
        write_chunk(&mut self.outbound, 3, MSG_AMF0_COMMAND, msg_stream_id, payload);
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

/// Write an outbound message as a single `fmt 0` chunk (our replies are smaller
/// than the chunk size, so no fragmentation is needed). `msg_stream_id` is
/// little-endian per the chunk spec.
fn write_chunk(out: &mut Vec<u8>, csid: u8, msg_type: u8, msg_stream_id: u32, payload: &[u8]) {
    out.push(csid & 0x3F); // fmt 0, 1-byte basic header
    write_u24(out, 0); // timestamp
    write_u24(out, payload.len() as u32);
    out.push(msg_type);
    out.extend_from_slice(&msg_stream_id.to_le_bytes());
    out.extend_from_slice(payload);
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

    /// Compact comparator for `FlvUnit` in assertions.
    #[derive(Debug, PartialEq, Eq)]
    struct FlvUnitView(FlvTrack, Vec<u8>, u32);
    impl PartialEq<FlvUnitView> for crate::flv::FlvUnit {
        fn eq(&self, o: &FlvUnitView) -> bool {
            self.track == o.0 && self.data == o.1 && self.pts_ms == o.2
        }
    }
}
