//! The wire of GStreamer's `unixfd` plugin (`gst/unixfd` in gst-plugins-bad,
//! 1.24+), which the `unixfdsink` / `unixfdsrc` pair speaks.
//!
//! Every message is an 8-byte header, `{u32 type, u32 payload_size}` in host
//! byte order, then the payload. A buffer's file descriptors ride as one
//! `SCM_RIGHTS` item on the send that carries the header, one fd per memory.
//! Metas follow the memories in GStreamer's `gst_meta_serialize` framing. The
//! only one read or written here is `GstVideoMeta`, the per-plane offsets and
//! strides.
//!
//! Everything a peer sends is untrusted: [`read_command`] bounds the payload
//! before allocating it, and [`NewBuffer::decode`] bounds the memory and meta
//! counts and checks every length against the payload.

use alloc::vec::Vec;

use std::io::{Error as IoError, ErrorKind};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

use tokio::io::unix::AsyncFd;

use g2g_core::log::io_err;
use g2g_core::{G2gError, RawVideoFormat};

use crate::scmfd;

/// Sink to source: a buffer, its memories' fds attached.
pub const COMMAND_NEW_BUFFER: u32 = 0;
/// Source to sink: done with every memory of the buffer with this id.
pub const COMMAND_RELEASE_BUFFER: u32 = 1;
/// Sink to source: the caps string, NUL-terminated.
pub const COMMAND_CAPS: u32 = 2;
/// Sink to source: end of stream, no payload.
pub const COMMAND_EOS: u32 = 3;

pub const HEADER_BYTES: usize = 8;
/// The fixed part of a NEW_BUFFER payload, before its memories.
pub const NEW_BUFFER_FIXED_BYTES: usize = 56;
/// One `{u64 size, u64 offset}` memory entry.
pub const MEMORY_BYTES: usize = 16;
pub const RELEASE_BUFFER_BYTES: usize = 8;

/// Plain fd memory (memfd, shm).
pub const MEMORY_TYPE_DEFAULT: u8 = 0;
/// Every memory of the buffer is a dma-buf.
pub const MEMORY_TYPE_DMABUF: u8 = 1;

/// `GST_CLOCK_TIME_NONE`, also `GST_BUFFER_OFFSET_NONE`.
pub const CLOCK_TIME_NONE: u64 = u64::MAX;
/// `GST_BUFFER_FLAG_DELTA_UNIT`: the buffer is not a keyframe.
pub const BUFFER_FLAG_DELTA_UNIT: u32 = 1 << 13;

/// Memories one buffer may carry, GStreamer's own per-buffer limit.
pub const MAX_MEMORIES: usize = scmfd::MAX_FDS;
/// Serialized metas one buffer may carry.
pub const MAX_METAS: usize = 64;
/// Largest payload read off the socket, bounding the allocation a peer can ask
/// for. A caps string with codec headers stays far below it.
pub const MAX_PAYLOAD_BYTES: usize = 1 << 20;

const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// `GST_VIDEO_MAX_PLANES`.
pub const MAX_VIDEO_PLANES: usize = 4;
pub const VIDEO_META_NAME: &str = "GstVideoMeta";
const VIDEO_META_VERSION: u8 = 0;
/// flags, format, width, height and n_planes.
const VIDEO_META_FIXED_BYTES: usize = 20;
/// padding top, bottom, left and right.
const VIDEO_META_PADDING_BYTES: usize = 16;
/// One plane's u64 offset, i32 stride and u32 stride alignment.
const VIDEO_META_PLANE_BYTES: usize = 16;
/// `[u32 total_size][u32 name_len]` in front of a meta's name.
const META_LENGTHS_BYTES: usize = 8;
/// The name's NUL and the version byte.
const META_NAME_TRAILER_BYTES: usize = 2;

/// `GstVideoFormat` for each raw format this carries a `GstVideoMeta` for.
const GST_VIDEO_FORMATS: &[(RawVideoFormat, i32)] = &[
    (RawVideoFormat::I420, 2),
    (RawVideoFormat::Yuyv, 4),
    (RawVideoFormat::Rgba8, 11),
    (RawVideoFormat::Bgra8, 12),
    (RawVideoFormat::Rgb8, 15),
    (RawVideoFormat::I422, 18),
    (RawVideoFormat::I444, 20),
    (RawVideoFormat::Nv12, 23),
    (RawVideoFormat::I420p10, 43),
    (RawVideoFormat::I422p10, 45),
    (RawVideoFormat::I444p10, 47),
    (RawVideoFormat::P010, 62),
    (RawVideoFormat::I420p12, 73),
    (RawVideoFormat::I422p12, 75),
    (RawVideoFormat::I444p12, 77),
];

/// The `GstVideoFormat` number of `format`.
pub fn gst_video_format(format: RawVideoFormat) -> Option<i32> {
    GST_VIDEO_FORMATS
        .iter()
        .find(|(known, _)| *known == format)
        .map(|(_, number)| *number)
}

pub fn protocol_error(reason: &'static str) -> G2gError {
    io_err(IoError::new(ErrorKind::InvalidData, reason))
}

/// Where the socket lives: gst's `socket-type`, `path` or `abstract`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SocketType {
    #[default]
    Path,
    Abstract,
}

pub const SOCKET_TYPE_NAMES: &str = "path | abstract";

const SOCKET_TYPES: [SocketType; 2] = [SocketType::Path, SocketType::Abstract];

impl SocketType {
    pub fn from_name(name: &str) -> Option<Self> {
        SOCKET_TYPES
            .into_iter()
            .find(|socket_type| socket_type.name() == name)
    }

    pub const fn name(self) -> &'static str {
        match self {
            SocketType::Path => "path",
            SocketType::Abstract => "abstract",
        }
    }

    fn address(self, socket_path: &str) -> std::io::Result<SocketAddr> {
        match self {
            SocketType::Path => SocketAddr::from_pathname(socket_path),
            SocketType::Abstract => SocketAddr::from_abstract_name(socket_path.as_bytes()),
        }
    }

    /// Bind and listen, taking over a path socket file left behind by a
    /// process that is gone.
    pub fn bind(self, socket_path: &str) -> std::io::Result<UnixListener> {
        let address = self.address(socket_path)?;
        match UnixListener::bind_addr(&address) {
            Err(error) if error.kind() == ErrorKind::AddrInUse && self == SocketType::Path => {
                if UnixStream::connect_addr(&address).is_ok() {
                    return Err(error);
                }
                std::fs::remove_file(socket_path)?;
                UnixListener::bind_addr(&address)
            }
            bound => bound,
        }
    }

    pub fn connect(self, socket_path: &str) -> std::io::Result<UnixStream> {
        UnixStream::connect_addr(&self.address(socket_path)?)
    }
}

/// A connected socket the runtime can wait on. Reads that must not wait go to
/// the inner stream directly, so they see bytes the runtime has not noticed yet.
pub type Socket = AsyncFd<UnixStream>;

/// Register `stream` with the runtime. Only callable from inside it.
pub fn socket(stream: UnixStream) -> std::io::Result<Socket> {
    stream.set_nonblocking(true)?;
    AsyncFd::new(stream)
}

/// One `{u64 size, u64 offset}` entry: the bytes of a memory's fd it covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    pub size: u64,
    pub offset: u64,
}

/// One plane of a `GstVideoMeta`, offsets counted from the buffer start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoPlane {
    pub offset: u64,
    pub stride: i32,
}

/// The `GstVideoMeta` fields this reads and writes. Alignment is written as
/// none and not read back: the planes say where the rows are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoMeta {
    pub flags: i32,
    pub format: i32,
    pub width: u32,
    pub height: u32,
    pub planes: Vec<VideoPlane>,
}

impl VideoMeta {
    fn payload_bytes(planes: usize) -> usize {
        VIDEO_META_FIXED_BYTES + VIDEO_META_PADDING_BYTES + planes * VIDEO_META_PLANE_BYTES
    }

    /// Append in `gst_meta_serialize` framing.
    fn serialize(&self, out: &mut Vec<u8>) {
        let name = VIDEO_META_NAME.as_bytes();
        let header = META_LENGTHS_BYTES + name.len() + META_NAME_TRAILER_BYTES;
        let total = header + Self::payload_bytes(self.planes.len());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name);
        out.push(0);
        out.push(VIDEO_META_VERSION);
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.format.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&(self.planes.len() as u32).to_le_bytes());
        for plane in &self.planes {
            out.extend_from_slice(&plane.offset.to_le_bytes());
        }
        for plane in &self.planes {
            out.extend_from_slice(&plane.stride.to_le_bytes());
        }
        out.extend_from_slice(&[0; VIDEO_META_PADDING_BYTES]);
        for _ in &self.planes {
            out.extend_from_slice(&0u32.to_le_bytes());
        }
    }

    fn deserialize(payload: &[u8], version: u8) -> Option<Self> {
        if version != VIDEO_META_VERSION {
            return None;
        }
        let mut reader = Reader::new(payload);
        let flags = reader.i32_le()?;
        let format = reader.i32_le()?;
        let width = reader.u32_le()?;
        let height = reader.u32_le()?;
        let count = reader.u32_le()? as usize;
        if count > MAX_VIDEO_PLANES {
            return None;
        }
        let mut offsets = [0u64; MAX_VIDEO_PLANES];
        for offset in offsets.iter_mut().take(count) {
            *offset = reader.u64_le()?;
        }
        let mut planes = Vec::with_capacity(count);
        for offset in offsets.iter().take(count) {
            planes.push(VideoPlane {
                offset: *offset,
                stride: reader.i32_le()?,
            });
        }
        reader.skip(VIDEO_META_PADDING_BYTES + count * core::mem::size_of::<u32>())?;
        Some(VideoMeta {
            flags,
            format,
            width,
            height,
            planes,
        })
    }
}

/// A NEW_BUFFER payload. Times are absolute `CLOCK_MONOTONIC` nanoseconds or
/// [`CLOCK_TIME_NONE`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewBuffer {
    pub id: u64,
    pub pts: u64,
    pub dts: u64,
    pub duration: u64,
    pub offset: u64,
    pub offset_end: u64,
    pub flags: u32,
    pub memory_type: u8,
    pub memories: Vec<Memory>,
    /// The first `GstVideoMeta` on the buffer. Every other meta is skipped.
    pub video_meta: Option<VideoMeta>,
}

impl NewBuffer {
    pub fn encode(&self) -> Vec<u8> {
        let mut out =
            Vec::with_capacity(NEW_BUFFER_FIXED_BYTES + self.memories.len() * MEMORY_BYTES);
        for field in [
            self.id,
            self.pts,
            self.dts,
            self.duration,
            self.offset,
            self.offset_end,
        ] {
            out.extend_from_slice(&field.to_ne_bytes());
        }
        out.extend_from_slice(&self.flags.to_ne_bytes());
        out.push(self.memory_type);
        out.push(self.memories.len() as u8);
        let meta_count = u16::from(self.video_meta.is_some());
        out.extend_from_slice(&meta_count.to_ne_bytes());
        for memory in &self.memories {
            out.extend_from_slice(&memory.size.to_ne_bytes());
            out.extend_from_slice(&memory.offset.to_ne_bytes());
        }
        if let Some(meta) = &self.video_meta {
            meta.serialize(&mut out);
        }
        out
    }

    /// `None` on anything malformed: short, too many memories or metas, an
    /// unknown memory type, a meta whose framing does not fit the payload, or a
    /// `GstVideoMeta` that does not parse.
    pub fn decode(payload: &[u8]) -> Option<Self> {
        let mut reader = Reader::new(payload);
        let id = reader.u64_ne()?;
        let pts = reader.u64_ne()?;
        let dts = reader.u64_ne()?;
        let duration = reader.u64_ne()?;
        let offset = reader.u64_ne()?;
        let offset_end = reader.u64_ne()?;
        let flags = reader.u32_ne()?;
        let memory_type = reader.u8()?;
        let memory_count = reader.u8()? as usize;
        let meta_count = reader.u16_ne()? as usize;
        if memory_count > MAX_MEMORIES || meta_count > MAX_METAS || memory_type > MEMORY_TYPE_DMABUF
        {
            return None;
        }
        let mut memories = Vec::with_capacity(memory_count);
        for _ in 0..memory_count {
            memories.push(Memory {
                size: reader.u64_ne()?,
                offset: reader.u64_ne()?,
            });
        }
        let mut video_meta = None;
        for _ in 0..meta_count {
            let total = reader.peek_u32_le()? as usize;
            let framed = reader.take(total)?;
            let (name, version, body) = split_meta(framed)?;
            if name == VIDEO_META_NAME.as_bytes() && video_meta.is_none() {
                video_meta = Some(VideoMeta::deserialize(body, version)?);
            }
        }
        Some(NewBuffer {
            id,
            pts,
            dts,
            duration,
            offset,
            offset_end,
            flags,
            memory_type,
            memories,
            video_meta,
        })
    }
}

/// Split one `gst_meta_serialize` record into its name, version and payload.
fn split_meta(framed: &[u8]) -> Option<(&[u8], u8, &[u8])> {
    let mut reader = Reader::new(framed);
    reader.u32_le()?;
    let name_len = reader.u32_le()? as usize;
    let header = META_LENGTHS_BYTES
        .checked_add(name_len)?
        .checked_add(META_NAME_TRAILER_BYTES)?;
    if framed.len() < header {
        return None;
    }
    let name = reader.take(name_len)?;
    if reader.u8()? != 0 {
        return None;
    }
    let version = reader.u8()?;
    Some((name, version, &framed[header..]))
}

pub fn release_payload(id: u64) -> [u8; RELEASE_BUFFER_BYTES] {
    id.to_ne_bytes()
}

/// The id a RELEASE_BUFFER names. Longer payloads are read the way gst reads
/// them, by their first eight bytes.
pub fn decode_release(payload: &[u8]) -> Option<u64> {
    Reader::new(payload).u64_ne()
}

pub fn caps_payload(caps: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(caps.len() + 1);
    out.extend_from_slice(caps.as_bytes());
    out.push(0);
    out
}

/// The caps string of a CAPS payload, which must end in its NUL.
pub fn decode_caps(payload: &[u8]) -> Option<&str> {
    let (last, text) = payload.split_last()?;
    if *last != 0 {
        return None;
    }
    let end = text
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(text.len());
    core::str::from_utf8(&text[..end]).ok()
}

/// One message read off the socket, its fds already owned.
#[derive(Debug)]
pub struct Command {
    pub command: u32,
    pub payload: Vec<u8>,
    pub fds: Vec<OwnedFd>,
}

/// Send one message, `fds` riding the header. The header and payload go out
/// as one byte stream, which a gst peer reads as header then payload.
pub async fn send_command(
    socket: &Socket,
    command: u32,
    payload: &[u8],
    fds: &[RawFd],
) -> std::io::Result<()> {
    let payload_size = u32::try_from(payload.len())
        .map_err(|_| IoError::new(ErrorKind::InvalidInput, "payload exceeds a u32"))?;
    let mut message = Vec::with_capacity(HEADER_BYTES + payload.len());
    message.extend_from_slice(&command.to_ne_bytes());
    message.extend_from_slice(&payload_size.to_ne_bytes());
    message.extend_from_slice(payload);
    let mut sent = 0;
    let mut attached = fds;
    while sent < message.len() {
        let mut ready = socket.writable().await?;
        match ready
            .try_io(|stream| scmfd::send_with_fds(stream.as_raw_fd(), &message[sent..], attached))
        {
            Ok(Ok(count)) => {
                sent += count;
                attached = &[];
            }
            Ok(Err(error)) => return Err(error),
            Err(_would_block) => continue,
        }
    }
    Ok(())
}

/// Fill `buffer`, collecting any fds that arrive. `Ok(false)` on a clean end of
/// stream before the first byte.
async fn read_exact_with_fds(
    socket: &Socket,
    buffer: &mut [u8],
    fds: &mut Vec<OwnedFd>,
) -> Result<bool, G2gError> {
    let mut filled = 0;
    while filled < buffer.len() {
        let mut ready = socket.readable().await.map_err(io_err)?;
        let count = match ready
            .try_io(|stream| scmfd::recv_with_fds(stream.as_raw_fd(), &mut buffer[filled..], fds))
        {
            Ok(Ok(count)) => count,
            Ok(Err(error)) => return Err(io_err(error)),
            Err(_would_block) => continue,
        };
        if count == 0 {
            if filled == 0 {
                return Ok(false);
            }
            return Err(protocol_error("peer closed the socket inside a message"));
        }
        filled += count;
    }
    Ok(true)
}

/// Read one message. `Ok(None)` when the peer closed the socket between
/// messages. Fds may only ride the header, and any received fd is closed on
/// every error.
pub async fn read_command(socket: &Socket) -> Result<Option<Command>, G2gError> {
    let mut header = [0u8; HEADER_BYTES];
    let mut fds = Vec::new();
    if !read_exact_with_fds(socket, &mut header, &mut fds).await? {
        return Ok(None);
    }
    let mut reader = Reader::new(&header);
    let command = reader
        .u32_ne()
        .ok_or_else(|| protocol_error("short header"))?;
    let payload_size = reader
        .u32_ne()
        .ok_or_else(|| protocol_error("short header"))? as usize;
    if payload_size > MAX_PAYLOAD_BYTES {
        return Err(protocol_error("payload size exceeds the limit"));
    }
    let mut payload = alloc::vec![0u8; payload_size];
    let mut late_fds = Vec::new();
    if !read_exact_with_fds(socket, &mut payload, &mut late_fds).await? {
        return Err(protocol_error("peer closed the socket inside a message"));
    }
    if !late_fds.is_empty() {
        return Err(protocol_error("descriptors arrived inside a payload"));
    }
    Ok(Some(Command {
        command,
        payload,
        fds,
    }))
}

/// Pull complete messages out of bytes read from a peer that sends no fds. A
/// payload over [`MAX_PAYLOAD_BYTES`] is an error, a partial message stays put.
pub fn take_command(buffered: &mut Vec<u8>) -> Result<Option<(u32, Vec<u8>)>, G2gError> {
    let mut reader = Reader::new(buffered);
    let (Some(command), Some(payload_size)) = (reader.u32_ne(), reader.u32_ne()) else {
        return Ok(None);
    };
    let payload_size = payload_size as usize;
    if payload_size > MAX_PAYLOAD_BYTES {
        return Err(protocol_error("payload size exceeds the limit"));
    }
    if buffered.len() < HEADER_BYTES + payload_size {
        return Ok(None);
    }
    let payload = buffered[HEADER_BYTES..HEADER_BYTES + payload_size].to_vec();
    buffered.drain(..HEADER_BYTES + payload_size);
    Ok(Some((command, payload)))
}

/// `CLOCK_MONOTONIC` now, the timeline the wire timestamps are on.
pub fn monotonic_now_ns() -> u64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a live timespec the call writes into.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    (now.tv_sec as u64)
        .saturating_mul(NANOS_PER_SECOND)
        .saturating_add(now.tv_nsec as u64)
}

/// `CLOCK_MONOTONIC` minus `pipeline_now`, sampled now: gst's `clock_diff`.
pub fn monotonic_offset(pipeline_now: u64) -> i64 {
    let offset = i128::from(monotonic_now_ns()) - i128::from(pipeline_now);
    offset.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

/// A running time on the wire: `running + base_time + latency`, moved onto
/// `CLOCK_MONOTONIC`. [`CLOCK_TIME_NONE`] stays as it is.
pub fn to_wire_time(running: u64, base_time: u64, latency: u64, offset: i64) -> u64 {
    if running == CLOCK_TIME_NONE {
        return CLOCK_TIME_NONE;
    }
    let pipeline = running.saturating_add(base_time).saturating_add(latency);
    pipeline
        .saturating_add_signed(offset)
        .min(CLOCK_TIME_NONE - 1)
}

/// The running time of a wire timestamp against this end's `base_time`,
/// clamped at zero the way gst clamps it.
pub fn from_wire_time(wire: u64, base_time: u64, offset: i64) -> u64 {
    if wire == CLOCK_TIME_NONE {
        return CLOCK_TIME_NONE;
    }
    let negated = offset.checked_neg().unwrap_or(i64::MAX);
    wire.saturating_add_signed(negated)
        .saturating_sub(base_time)
}

/// Bounds-checked reads off untrusted bytes.
#[derive(Debug)]
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        if self.bytes.len() < count {
            return None;
        }
        let (taken, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Some(taken)
    }

    fn skip(&mut self, count: usize) -> Option<()> {
        self.take(count).map(|_| ())
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N)?.try_into().ok()
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.array::<1>()?[0])
    }

    fn u16_ne(&mut self) -> Option<u16> {
        Some(u16::from_ne_bytes(self.array()?))
    }

    fn u32_ne(&mut self) -> Option<u32> {
        Some(u32::from_ne_bytes(self.array()?))
    }

    fn u64_ne(&mut self) -> Option<u64> {
        Some(u64::from_ne_bytes(self.array()?))
    }

    fn u32_le(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.array()?))
    }

    fn i32_le(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.array()?))
    }

    fn u64_le(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.array()?))
    }

    fn peek_u32_le(&self) -> Option<u32> {
        Reader::new(self.bytes).u32_le()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_buffer() -> NewBuffer {
        NewBuffer {
            id: 7,
            pts: 1_000,
            dts: CLOCK_TIME_NONE,
            duration: 33_333_333,
            offset: CLOCK_TIME_NONE,
            offset_end: CLOCK_TIME_NONE,
            flags: 0,
            memory_type: MEMORY_TYPE_DEFAULT,
            memories: alloc::vec![Memory {
                size: 64,
                offset: 0
            }],
            video_meta: Some(VideoMeta {
                flags: 0,
                format: 11,
                width: 4,
                height: 4,
                planes: alloc::vec![VideoPlane {
                    offset: 0,
                    stride: 16
                }],
            }),
        }
    }

    #[test]
    fn a_new_buffer_round_trips() {
        let buffer = sample_buffer();
        let encoded = buffer.encode();
        let meta_bytes = META_LENGTHS_BYTES
            + VIDEO_META_NAME.len()
            + META_NAME_TRAILER_BYTES
            + VideoMeta::payload_bytes(1);
        assert_eq!(
            encoded.len(),
            NEW_BUFFER_FIXED_BYTES + MEMORY_BYTES + meta_bytes
        );
        assert_eq!(NewBuffer::decode(&encoded), Some(buffer));
    }

    #[test]
    fn every_truncation_of_a_new_buffer_is_rejected() {
        let encoded = sample_buffer().encode();
        for length in 0..encoded.len() {
            assert_eq!(NewBuffer::decode(&encoded[..length]), None, "{length}");
        }
    }

    #[test]
    fn an_unknown_meta_is_skipped_by_its_size() {
        let mut buffer = sample_buffer();
        buffer.video_meta = None;
        let mut encoded = buffer.encode();
        let meta_count_at = NEW_BUFFER_FIXED_BYTES - core::mem::size_of::<u16>();
        encoded[meta_count_at..NEW_BUFFER_FIXED_BYTES].copy_from_slice(&1u16.to_ne_bytes());
        let name = b"GstSomeOtherMeta";
        let body = [9u8; 5];
        let total = META_LENGTHS_BYTES + name.len() + META_NAME_TRAILER_BYTES + body.len();
        encoded.extend_from_slice(&(total as u32).to_le_bytes());
        encoded.extend_from_slice(&(name.len() as u32).to_le_bytes());
        encoded.extend_from_slice(name);
        encoded.extend_from_slice(&[0, 0]);
        encoded.extend_from_slice(&body);
        assert_eq!(NewBuffer::decode(&encoded), Some(buffer));
    }

    #[test]
    fn too_many_memories_are_rejected() {
        let mut buffer = sample_buffer();
        buffer.memories = alloc::vec![
            Memory {
                size: 1,
                offset: 0
            };
            MAX_MEMORIES + 1
        ];
        assert_eq!(NewBuffer::decode(&buffer.encode()), None);
    }

    #[test]
    fn caps_need_their_nul() {
        assert_eq!(
            decode_caps(&caps_payload("video/x-raw")),
            Some("video/x-raw")
        );
        assert_eq!(decode_caps(b"video/x-raw"), None);
        assert_eq!(decode_caps(b""), None);
    }

    #[test]
    fn a_partial_command_waits_and_an_oversized_one_fails() {
        let mut buffered = Vec::new();
        buffered.extend_from_slice(&COMMAND_RELEASE_BUFFER.to_ne_bytes());
        buffered.extend_from_slice(&(RELEASE_BUFFER_BYTES as u32).to_ne_bytes());
        buffered.extend_from_slice(&release_payload(5)[..3]);
        assert_eq!(take_command(&mut buffered), Ok(None));
        buffered.extend_from_slice(&release_payload(5)[3..]);
        let (command, payload) = take_command(&mut buffered).unwrap().unwrap();
        assert_eq!(command, COMMAND_RELEASE_BUFFER);
        assert_eq!(decode_release(&payload), Some(5));
        assert!(buffered.is_empty());

        buffered.extend_from_slice(&COMMAND_RELEASE_BUFFER.to_ne_bytes());
        buffered.extend_from_slice(&(MAX_PAYLOAD_BYTES as u32 + 1).to_ne_bytes());
        assert!(take_command(&mut buffered).is_err());
    }

    #[test]
    fn wire_time_round_trips_through_both_ends() {
        let (running, sink_base, latency, offset) = (40_000_000, 5_000, 1_000, -2_000);
        let wire = to_wire_time(running, sink_base, latency, offset);
        assert_eq!(wire, running + sink_base + latency - 2_000);
        assert_eq!(from_wire_time(wire, sink_base + latency, offset), running);
        assert_eq!(from_wire_time(wire, u64::MAX - 1, offset), 0);
        assert_eq!(to_wire_time(CLOCK_TIME_NONE, 1, 1, 1), CLOCK_TIME_NONE);
        assert_eq!(from_wire_time(CLOCK_TIME_NONE, 1, 1), CLOCK_TIME_NONE);
    }
}
