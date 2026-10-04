//! M1216: the fd-passing IPC pair. The loopbacks run a g2g `unixfdsink` into a
//! g2g `unixfdsrc` over a real socket, a raw client and a raw server stand in
//! for a peer that holds buffers or sends garbage, and the `#[ignore]`d interop
//! legs pair each element with its GStreamer counterpart. Run those with:
//!
//! ```sh
//! cargo test -p g2g-plugins --features unixfd --test m1216_unixfd -- --ignored --nocapture
//! ```
#![cfg(all(target_os = "linux", feature = "unixfd"))]

use std::io::Write;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, RunStats, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ClockSync, Colorimetry, Dim, FrameTiming, G2gError, Interlace,
    MonotonicClock, OutputSink, PipelineClock, PipelinePacket, PropError, PropValue, PropertySpec,
    PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::dmabufmap::DmaBufReadMap;
use g2g_plugins::registry::default_registry;
use g2g_plugins::scmfd;
use g2g_plugins::unixfd::{UnixFdSink, UnixFdSrc};
use g2g_plugins::unixfdwire::{
    caps_payload, gst_video_format, release_payload, Memory, NewBuffer, SocketType, VideoMeta,
    VideoPlane, CLOCK_TIME_NONE, COMMAND_CAPS, COMMAND_EOS, COMMAND_NEW_BUFFER,
    COMMAND_RELEASE_BUFFER, HEADER_BYTES, MAX_PAYLOAD_BYTES, MEMORY_TYPE_DEFAULT,
};

const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;
const FRAMERATE: u32 = 30;
const RGBA_BYTES_PER_PIXEL: usize = 4;
const ROW_BYTES: usize = WIDTH as usize * RGBA_BYTES_PER_PIXEL;
const FRAME_BYTES: usize = ROW_BYTES * HEIGHT as usize;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const FRAME_PERIOD_NS: u64 = NANOS_PER_SECOND / FRAMERATE as u64;
const FRAMES: u64 = 12;
/// Both ends sample the clock offset per frame, a few reads apart.
const PTS_TOLERANCE_NS: u64 = 1_000_000;
/// A receiver that starts after the sender sees its first frames clamped to
/// zero, so only a bound on the first timestamp holds across graphs.
const FIRST_PTS_BOUND_NS: u64 = NANOS_PER_SECOND;
const RUN_DEADLINE: Duration = Duration::from_secs(20);
const LINK_CAPACITY: usize = 4;
/// Rows padded past their pixels, the stride a GPU export would use.
const PADDED_STRIDE: u32 = 64;
/// Where plane 0 starts inside the dma-buf.
const PLANE_OFFSET: u32 = 128;
const GST_DEADLINE: Duration = Duration::from_secs(60);
const GST_FRAMES: u64 = 30;
const GST_WIDTH: u32 = 64;
const GST_HEIGHT: u32 = 48;
const GST_FRAME_BYTES: usize = GST_WIDTH as usize * GST_HEIGHT as usize * RGBA_BYTES_PER_PIXEL;
const MEMFD_NAME: &std::ffi::CStr = c"m1216";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn unique_name(name: &str) -> String {
    static SERIAL: AtomicU32 = AtomicU32::new(0);
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    format!("g2g_m1216_{name}_{}_{serial}", std::process::id())
}

fn unique_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(unique_name(name));
    let _ = std::fs::remove_file(&path);
    path
}

fn socket_name(socket_type: SocketType, name: &str) -> String {
    match socket_type {
        SocketType::Path => unique_path(name).display().to_string(),
        SocketType::Abstract => unique_name(name),
    }
}

fn rgba_caps(width: u32, height: u32) -> Caps {
    rgba_caps_interlaced(width, height, Interlace::Any)
}

fn rgba_caps_interlaced(width: u32, height: u32, interlace: Interlace) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(FRAMERATE << 16),
        interlace,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn rgba_caps_text() -> String {
    format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate={FRAMERATE}/1")
}

fn frame_fill(index: u64) -> u8 {
    (index % u64::from(u8::MAX)) as u8 + 1
}

fn frame_bytes(index: u64) -> Vec<u8> {
    (0..FRAME_BYTES)
        .map(|byte| frame_fill(index).wrapping_add(byte as u8))
        .collect()
}

fn timing(index: u64) -> FrameTiming {
    FrameTiming {
        pts_ns: index * FRAME_PERIOD_NS,
        dts_ns: index * FRAME_PERIOD_NS,
        duration_ns: FRAME_PERIOD_NS,
        ..FrameTiming::default()
    }
}

fn system_frame(index: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            frame_bytes(index).into_boxed_slice(),
        )),
        timing(index),
        index,
    ))
}

fn memfd_holding(bytes: &[u8]) -> OwnedFd {
    // SAFETY: the name is a NUL-terminated literal and the flag a documented bit.
    let raw = unsafe { libc::memfd_create(MEMFD_NAME.as_ptr(), libc::MFD_CLOEXEC) };
    assert!(raw >= 0, "memfd_create");
    // SAFETY: memfd_create returned a fresh descriptor nothing else owns.
    let mut file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(raw) });
    file.write_all(bytes).expect("fill the memfd");
    file.into()
}

/// `lead` bytes, then the rows of frame `index` `PADDED_STRIDE` apart.
fn padded_bytes(index: u64, lead: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; lead + PADDED_STRIDE as usize * HEIGHT as usize];
    for (row, pixels) in frame_bytes(index).chunks(ROW_BYTES).enumerate() {
        let start = lead + row * PADDED_STRIDE as usize;
        bytes[start..start + ROW_BYTES].copy_from_slice(pixels);
    }
    bytes
}

/// A memfd standing in for a dma-buf: `PLANE_OFFSET` bytes of lead-in, then
/// rows `PADDED_STRIDE` apart holding frame `index`.
fn padded_dmabuf(index: u64) -> OwnedDmaBuf {
    let fd = memfd_holding(&padded_bytes(index, PLANE_OFFSET as usize));
    // SAFETY: the memfd was just created and is handed over whole.
    unsafe {
        OwnedDmaBuf::from_raw(
            std::os::fd::IntoRawFd::into_raw_fd(fd),
            PADDED_STRIDE,
            PLANE_OFFSET,
        )
    }
}

fn dmabuf_frame(dmabuf: OwnedDmaBuf, index: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::DmaBuf(dmabuf),
        timing(index),
        index,
    ))
}

/// One clock for both ends, so a received timestamp is the sent one.
fn shared_clock() -> ClockSync {
    ClockSync::new(Arc::new(MonotonicClock), MonotonicClock.now_ns())
}

#[derive(Debug, Default)]
struct CollectingOutput {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectingOutput {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        self.packets.extend(packet.take());
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl CollectingOutput {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }
}

async fn feed(sink: &mut UnixFdSink, caps: &Caps, packets: Vec<PipelinePacket>) {
    let mut out = CollectingOutput::default();
    sink.process(PipelinePacket::CapsChanged(caps.clone()), &mut out)
        .await
        .expect("caps");
    for packet in packets {
        sink.process(packet, &mut out).await.expect("frame");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos");
}

/// Drops every frame as it arrives, as a sink that is done with it would.
#[derive(Debug, Default)]
struct DroppingOutput;

impl OutputSink for DroppingOutput {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        drop(packet.take());
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn receive_into(
    source: &mut UnixFdSrc,
    out: &mut impl OutputSink,
) -> Result<(Caps, u64), G2gError> {
    let caps = source.intercept_caps().await?;
    source.configure_pipeline(&caps)?;
    let count = source.run(out).await?;
    Ok((caps, count))
}

async fn receive(source: &mut UnixFdSrc) -> Result<(Caps, u64, CollectingOutput), G2gError> {
    let mut out = CollectingOutput::default();
    let (caps, count) = receive_into(source, &mut out).await?;
    Ok((caps, count, out))
}

/// A sink and a source joined over one socket, both on `clock`.
fn connected_pair(socket_type: SocketType, name: &str) -> (UnixFdSink, UnixFdSrc) {
    let socket = socket_name(socket_type, name);
    let clock = shared_clock();
    let mut sink = UnixFdSink::new(socket.clone())
        .with_socket_type(socket_type)
        .with_wait_for_connection(true);
    AsyncElement::set_clock_sync(&mut sink, clock.clone());
    let mut source = UnixFdSrc::new(socket).with_socket_type(socket_type);
    SourceLoop::set_clock_sync(&mut source, clock);
    (sink, source)
}

async fn system_loopback(socket_type: SocketType) {
    let caps = rgba_caps(WIDTH, HEIGHT);
    let (mut sink, mut source) = connected_pair(socket_type, "system");
    sink.configure_pipeline(&caps).expect("the sink binds");
    let packets = (0..FRAMES).map(system_frame).collect();
    let (_, received) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(feed(&mut sink, &caps, packets), receive(&mut source))
    })
    .await
    .expect("the loopback finishes");
    let (received_caps, count, out) = received.expect("the source runs");

    assert_eq!(received_caps, caps);
    assert_eq!(count, FRAMES);
    assert!(matches!(out.packets.last(), Some(PipelinePacket::Eos)));
    for (index, frame) in out.frames().into_iter().enumerate() {
        let index = index as u64;
        let bytes = frame.domain.as_system_slice().expect("system memory");
        assert_eq!(bytes, frame_bytes(index).as_slice(), "frame {index}");
        let sent = timing(index);
        assert!(
            frame.timing.pts_ns.abs_diff(sent.pts_ns) <= PTS_TOLERANCE_NS,
            "frame {index}: pts {} for {}",
            frame.timing.pts_ns,
            sent.pts_ns
        );
        assert_eq!(frame.timing.duration_ns, sent.duration_ns);
    }
}

#[tokio::test]
async fn system_frames_cross_a_path_socket() {
    system_loopback(SocketType::Path).await;
}

#[tokio::test]
async fn system_frames_cross_an_abstract_socket() {
    system_loopback(SocketType::Abstract).await;
}

async fn run_launch(line: String) -> Result<RunStats, G2gError> {
    let registry = default_registry();
    let graph = parse_launch(&registry, &line).expect("parses");
    run_graph(graph, &ZeroClock, LINK_CAPACITY).await
}

/// Two launch lines in two graphs: the registry, the negotiation and the
/// runner's clocks all in the path.
#[tokio::test]
async fn launch_lines_link_two_graphs() {
    let socket = unique_path("launch");
    let feed_channel = unique_name("launch_in");
    let pull_channel = unique_name("launch_out");
    let feed = register_appsrc(&feed_channel);
    let pull = register_appsink_pull(&pull_channel);
    let sender = format!(
        "appsrc channel={feed_channel} caps={} ! unixfdsink socket-path={} wait-for-connection=true",
        rgba_caps_text(),
        socket.display()
    );
    let receiver = format!(
        "unixfdsrc socket-path={} ! appsink channel={pull_channel} caps={}",
        socket.display(),
        rgba_caps_text()
    );
    let push = async {
        for index in 0..FRAMES {
            while !feed.push(&frame_bytes(index), timing(index).pts_ns) {
                tokio::task::yield_now().await;
            }
        }
        while !feed.end_of_stream() {
            tokio::task::yield_now().await;
        }
    };
    let collect = async {
        let mut frames = Vec::new();
        while let Some(frame) = pull.pull().await {
            frames.push(frame);
        }
        frames
    };
    let (sent, received, (), frames) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(run_launch(sender), run_launch(receiver), push, collect)
    })
    .await
    .expect("both graphs end");
    sent.expect("the sender graph runs");
    received.expect("the receiver graph runs");
    assert!(matches!(pull.try_pull(), Pull::Ended));

    assert_eq!(frames.len() as u64, FRAMES);
    for (index, frame) in frames.iter().enumerate() {
        let bytes = frame.domain.as_system_slice().expect("system memory");
        assert_eq!(bytes, frame_bytes(index as u64).as_slice(), "frame {index}");
    }
    let pts: Vec<u64> = frames.iter().map(|frame| frame.timing.pts_ns).collect();
    assert!(pts[0] <= FIRST_PTS_BOUND_NS, "first pts {}", pts[0]);
    let last_step = pts[pts.len() - 1] - pts[pts.len() - 2];
    assert!(
        last_step.abs_diff(FRAME_PERIOD_NS) <= PTS_TOLERANCE_NS,
        "pts advance by the frame period: {pts:?}"
    );
}

#[tokio::test]
async fn a_dmabuf_crosses_as_a_dmabuf_with_its_layout() {
    let caps = rgba_caps(WIDTH, HEIGHT);
    let (mut sink, mut source) = connected_pair(SocketType::Path, "dmabuf");
    sink.configure_pipeline(&caps).expect("the sink binds");
    let packets = (0..FRAMES)
        .map(|index| dmabuf_frame(padded_dmabuf(index), index))
        .collect();
    let (_, received) = tokio::time::timeout(RUN_DEADLINE, async {
        tokio::join!(feed(&mut sink, &caps, packets), receive(&mut source))
    })
    .await
    .expect("the loopback finishes");
    let (received_caps, count, out) = received.expect("the source runs");

    assert_eq!(received_caps, caps);
    assert_eq!(source.output_memory(), MemoryDomainKind::DmaBuf);
    assert_eq!(count, FRAMES);
    for (index, frame) in out.frames().into_iter().enumerate() {
        let MemoryDomain::DmaBuf(dmabuf) = &frame.domain else {
            panic!("frame {index} is not a dma-buf");
        };
        assert_eq!(dmabuf.stride, PADDED_STRIDE);
        assert_eq!(dmabuf.offset, PLANE_OFFSET);
        let map = DmaBufReadMap::read(dmabuf).expect("map the received fd");
        let rows: Vec<u8> = map
            .as_slice()
            .chunks(PADDED_STRIDE as usize)
            .take(HEIGHT as usize)
            .flat_map(|row| row[..ROW_BYTES].to_vec())
            .collect();
        assert_eq!(rows, frame_bytes(index as u64), "frame {index}");
    }
}

/// Read one message the way a gst client does, fds off the header.
fn read_message(stream: &UnixStream) -> (u32, Vec<u8>, Vec<OwnedFd>) {
    let mut fds = Vec::new();
    let mut header = [0u8; HEADER_BYTES];
    read_exact(stream, &mut header, &mut fds);
    let command = u32::from_ne_bytes(header[..4].try_into().unwrap());
    let size = u32::from_ne_bytes(header[4..].try_into().unwrap()) as usize;
    let mut payload = vec![0u8; size];
    read_exact(stream, &mut payload, &mut fds);
    (command, payload, fds)
}

fn read_exact(stream: &UnixStream, buffer: &mut [u8], fds: &mut Vec<OwnedFd>) {
    let mut filled = 0;
    while filled < buffer.len() {
        let count = scmfd::recv_with_fds(stream.as_raw_fd(), &mut buffer[filled..], fds)
            .expect("read the socket");
        assert!(count > 0, "the peer closed the socket");
        filled += count;
    }
}

fn write_message(stream: &UnixStream, command: u32, payload: &[u8], fds: &[i32]) {
    let mut message = Vec::new();
    message.extend_from_slice(&command.to_ne_bytes());
    message.extend_from_slice(&(payload.len() as u32).to_ne_bytes());
    message.extend_from_slice(payload);
    let sent = scmfd::send_with_fds(stream.as_raw_fd(), &message, fds).expect("write the socket");
    assert_eq!(sent, message.len());
}

/// A client that has not released a buffer keeps it alive in the sink, until
/// its release or its disconnect. A release naming a buffer it was never sent
/// drops the client without failing the sink.
#[tokio::test]
async fn the_sink_holds_a_buffer_until_the_client_releases_it() {
    let socket = unique_path("hold");
    let caps = rgba_caps(WIDTH, HEIGHT);
    let mut sink = UnixFdSink::new(socket.display().to_string());
    sink.configure_pipeline(&caps).expect("the sink binds");
    let client = UnixStream::connect(&socket).expect("connect");
    let mut out = CollectingOutput::default();

    let first = padded_dmabuf(0);
    let first_watch = first.clone();
    sink.process(dmabuf_frame(first, 0), &mut out)
        .await
        .expect("send the first frame");
    assert_eq!(sink.get_property("num-clients"), Some(PropValue::Uint(1)));
    assert_eq!(first_watch.share_count(), 2, "the sink holds the frame");

    let (command, _, _) = read_message(&client);
    assert_eq!(command, COMMAND_CAPS);
    let (command, payload, fds) = read_message(&client);
    assert_eq!(command, COMMAND_NEW_BUFFER);
    assert_eq!(fds.len(), 1, "one fd per memory");
    let buffer = NewBuffer::decode(&payload).expect("a well-formed buffer");
    write_message(
        &client,
        COMMAND_RELEASE_BUFFER,
        &release_payload(buffer.id),
        &[],
    );

    let second = padded_dmabuf(1);
    let second_watch = second.clone();
    sink.process(dmabuf_frame(second, 1), &mut out)
        .await
        .expect("send the second frame");
    assert_eq!(first_watch.share_count(), 1, "the release freed the frame");
    assert_eq!(second_watch.share_count(), 2, "the second is still held");

    let unknown_id = buffer.id + FRAMES;
    write_message(
        &client,
        COMMAND_RELEASE_BUFFER,
        &release_payload(unknown_id),
        &[],
    );
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("a bad release does not fail the sink");
    assert_eq!(sink.get_property("num-clients"), Some(PropValue::Uint(0)));
    assert_eq!(
        second_watch.share_count(),
        1,
        "a dropped client releases what it held"
    );
}

#[tokio::test]
async fn a_sink_that_may_not_copy_refuses_system_frames() {
    let socket = unique_path("no_copy");
    let caps = rgba_caps(WIDTH, HEIGHT);
    let mut sink = UnixFdSink::new(socket.display().to_string());
    sink.set_property("min-memory-size", PropValue::Int(-1))
        .expect("-1 disables copying");
    assert!(!sink.input_domains().contains(MemoryDomainKind::System));
    sink.configure_pipeline(&caps).expect("the sink binds");
    let _client = UnixStream::connect(&socket).expect("connect");
    let result = sink
        .process(system_frame(0), &mut CollectingOutput::default())
        .await;
    assert_eq!(result, Err(G2gError::UnsupportedDomain));
}

/// A raw server plays the sink: `caps`, then `sink_script`, while a source
/// reads into `out`.
async fn source_against_with<O: OutputSink>(
    caps: String,
    out: &mut O,
    sink_script: impl FnOnce(&UnixStream) + Send + 'static,
) -> Result<u64, G2gError> {
    let socket = unique_path("raw_sink");
    let listener = UnixListener::bind(&socket).expect("bind");
    let peer = std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("accept");
        write_message(&stream, COMMAND_CAPS, &caps_payload(&caps), &[]);
        sink_script(&stream);
        stream
    });
    let mut source = UnixFdSrc::new(socket.display().to_string());
    let result = tokio::time::timeout(RUN_DEADLINE, receive_into(&mut source, out))
        .await
        .expect("the source gives up");
    drop(peer.join().expect("the peer thread"));
    result.map(|(_, count)| count)
}

async fn source_against(
    sink_script: impl FnOnce(&UnixStream) + Send + 'static,
) -> Result<CollectingOutput, G2gError> {
    let mut out = CollectingOutput::default();
    source_against_with(rgba_caps_text(), &mut out, sink_script).await?;
    Ok(out)
}

/// The raw sink sends one buffer and waits for its release before EOS, so the
/// source has to send it once downstream drops the frame.
async fn release_on_drop(caps: String) {
    let mut out = DroppingOutput;
    let count = source_against_with(caps, &mut out, |stream| {
        let mut buffer = one_memory_buffer();
        buffer.id = FRAMES;
        let fd = memfd_holding(&frame_bytes(0));
        write_message(
            stream,
            COMMAND_NEW_BUFFER,
            &buffer.encode(),
            &[fd.as_raw_fd()],
        );
        let (command, payload, _) = read_message(stream);
        assert_eq!(command, COMMAND_RELEASE_BUFFER);
        assert_eq!(payload, release_payload(FRAMES));
        write_message(stream, COMMAND_EOS, &[], &[]);
    })
    .await
    .expect("the source runs");
    assert_eq!(count, 1);
}

#[tokio::test]
async fn a_dropped_memfd_frame_is_released() {
    release_on_drop(rgba_caps_text()).await;
}

#[tokio::test]
async fn a_dropped_dmabuf_frame_is_released() {
    let caps = rgba_caps(WIDTH, HEIGHT);
    let dmabuf_caps = g2g_plugins::capsfilter::dmabuf_gst_caps(&caps).expect("RGBA has a fourcc");
    release_on_drop(dmabuf_caps).await;
}

#[tokio::test]
async fn several_memories_are_joined_into_one_frame() {
    let out = source_against(|stream| {
        let bytes = frame_bytes(0);
        let (first, second) = bytes.split_at(FRAME_BYTES / 2);
        let mut buffer = one_memory_buffer();
        buffer.memories = [first, second]
            .iter()
            .map(|half| Memory {
                size: half.len() as u64,
                offset: 0,
            })
            .collect();
        let fds = [memfd_holding(first), memfd_holding(second)];
        let raw: Vec<i32> = fds.iter().map(AsRawFd::as_raw_fd).collect();
        write_message(stream, COMMAND_NEW_BUFFER, &buffer.encode(), &raw);
        write_message(stream, COMMAND_EOS, &[], &[]);
    })
    .await
    .expect("the source runs");
    let frames = out.frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].domain.as_system_slice(),
        Some(frame_bytes(0).as_slice())
    );
}

/// Rows a `GstVideoMeta` says are padded reach a downstream that did not ask
/// for a `PlaneLayout` packed tight.
#[tokio::test]
async fn padded_rows_are_packed_for_downstream() {
    let out = source_against(|stream| {
        let bytes = padded_bytes(0, 0);
        let mut buffer = one_memory_buffer();
        buffer.memories[0].size = bytes.len() as u64;
        buffer.video_meta = Some(VideoMeta {
            flags: 0,
            format: gst_video_format(RawVideoFormat::Rgba8).expect("RGBA has a gst number"),
            width: WIDTH,
            height: HEIGHT,
            planes: vec![VideoPlane {
                offset: 0,
                stride: PADDED_STRIDE as i32,
            }],
        });
        let fd = memfd_holding(&bytes);
        write_message(
            stream,
            COMMAND_NEW_BUFFER,
            &buffer.encode(),
            &[fd.as_raw_fd()],
        );
        write_message(stream, COMMAND_EOS, &[], &[]);
    })
    .await
    .expect("the source runs");
    let frames = out.frames();
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].domain.as_system_slice(),
        Some(frame_bytes(0).as_slice())
    );
}

fn one_memory_buffer() -> NewBuffer {
    NewBuffer {
        id: 1,
        pts: CLOCK_TIME_NONE,
        dts: CLOCK_TIME_NONE,
        duration: CLOCK_TIME_NONE,
        offset: CLOCK_TIME_NONE,
        offset_end: CLOCK_TIME_NONE,
        flags: 0,
        memory_type: MEMORY_TYPE_DEFAULT,
        memories: vec![Memory {
            size: FRAME_BYTES as u64,
            offset: 0,
        }],
        video_meta: None,
    }
}

#[tokio::test]
async fn an_oversized_payload_fails_the_source() {
    let result = source_against(|stream| {
        let mut header = Vec::new();
        header.extend_from_slice(&COMMAND_NEW_BUFFER.to_ne_bytes());
        header.extend_from_slice(&(MAX_PAYLOAD_BYTES as u32 + 1).to_ne_bytes());
        (&*stream).write_all(&header).expect("write the header");
    })
    .await;
    assert!(result.is_err(), "{result:?}");
}

#[tokio::test]
async fn an_fd_count_unlike_the_memory_count_fails_the_source() {
    let result = source_against(|stream| {
        let first = memfd_holding(&frame_bytes(0));
        let second = memfd_holding(&frame_bytes(1));
        write_message(
            stream,
            COMMAND_NEW_BUFFER,
            &one_memory_buffer().encode(),
            &[first.as_raw_fd(), second.as_raw_fd()],
        );
    })
    .await;
    assert!(result.is_err(), "{result:?}");

    let result = source_against(|stream| {
        write_message(
            stream,
            COMMAND_NEW_BUFFER,
            &one_memory_buffer().encode(),
            &[],
        );
    })
    .await;
    assert!(result.is_err(), "{result:?}");
}

#[tokio::test]
async fn a_truncated_payload_fails_the_source() {
    let result = source_against(|stream| {
        let payload = one_memory_buffer().encode();
        let fd = memfd_holding(&frame_bytes(0));
        let mut message = Vec::new();
        message.extend_from_slice(&COMMAND_NEW_BUFFER.to_ne_bytes());
        message.extend_from_slice(&(payload.len() as u32).to_ne_bytes());
        message.extend_from_slice(&payload[..payload.len() / 2]);
        scmfd::send_with_fds(stream.as_raw_fd(), &message, &[fd.as_raw_fd()]).expect("write");
        stream
            .shutdown(std::net::Shutdown::Write)
            .expect("close the write side");
    })
    .await;
    assert!(result.is_err(), "{result:?}");
}

#[tokio::test]
async fn a_memory_past_the_end_of_its_fd_fails_the_source() {
    let result = source_against(|stream| {
        let fd = memfd_holding(&frame_bytes(0)[..FRAME_BYTES / 2]);
        write_message(
            stream,
            COMMAND_NEW_BUFFER,
            &one_memory_buffer().encode(),
            &[fd.as_raw_fd()],
        );
    })
    .await;
    assert!(result.is_err(), "{result:?}");
}

fn declares(specs: &[PropertySpec], name: &str) -> bool {
    specs.iter().any(|spec| spec.name == name)
}

fn declared_default(specs: &[PropertySpec], name: &str) -> PropValue {
    let spec = specs
        .iter()
        .find(|spec| spec.name == name)
        .unwrap_or_else(|| panic!("`{name}` is declared"));
    let text = spec
        .default
        .unwrap_or_else(|| panic!("`{name}` declares a default"));
    spec.parse_value(text).expect("the default parses")
}

#[test]
fn sink_properties_round_trip() {
    let mut sink = UnixFdSink::default();
    let specs = sink.properties();
    for name in ["socket-type", "wait-for-connection", "min-memory-size"] {
        assert_eq!(
            sink.get_property(name),
            Some(declared_default(specs, name)),
            "{name}"
        );
    }
    let values = [
        ("socket-path", PropValue::Str("/tmp/m1216-props".into())),
        ("socket-type", PropValue::Str("abstract".into())),
        ("wait-for-connection", PropValue::Bool(true)),
        ("min-memory-size", PropValue::Int(4096)),
    ];
    for (name, value) in values {
        assert!(declares(specs, name), "{name}");
        sink.set_property(name, value.clone()).expect(name);
        assert_eq!(sink.get_property(name), Some(value), "{name}");
    }
    assert!(declares(specs, "num-clients"));
    assert_eq!(sink.get_property("num-clients"), Some(PropValue::Uint(0)));
    assert_eq!(
        sink.set_property("num-clients", PropValue::Uint(1)),
        Err(PropError::ReadOnly)
    );
    assert_eq!(
        sink.set_property("socket-type", PropValue::Str("anonymous".into())),
        Err(PropError::Value)
    );
    assert_eq!(
        sink.set_property("min-memory-size", PropValue::Int(-2)),
        Err(PropError::Value)
    );
}

#[test]
fn source_properties_round_trip() {
    let mut source = UnixFdSrc::default();
    let specs = source.properties();
    assert_eq!(
        source.get_property("socket-type"),
        Some(declared_default(specs, "socket-type"))
    );
    let values = [
        ("socket-path", PropValue::Str("/tmp/m1216-props".into())),
        ("socket-type", PropValue::Str("abstract".into())),
    ];
    for (name, value) in values {
        assert!(declares(specs, name), "{name}");
        source.set_property(name, value.clone()).expect(name);
        assert_eq!(source.get_property(name), Some(value), "{name}");
    }
}

#[test]
fn both_elements_build_from_a_launch_line() {
    let registry = default_registry();
    for line in [
        "videotestsrc num-buffers=1 ! unixfdsink socket-path=/tmp/m1216-launch socket-type=abstract min-memory-size=-1",
        "unixfdsrc socket-path=/tmp/m1216-launch socket-type=abstract ! fakesink",
    ] {
        assert!(parse_launch(&registry, line).is_ok(), "{line}");
    }
}

// ---- real-peer interop against gst-launch-1.0 (ignored: needs GStreamer) ----

fn gst(description: &str) -> Command {
    let mut command = Command::new("gst-launch-1.0");
    command.arg("-q").args(description.split_whitespace());
    command
}

fn wait_for_gst(mut child: Child) {
    let deadline = std::time::Instant::now() + GST_DEADLINE;
    loop {
        match child.try_wait().expect("poll the gst peer") {
            Some(status) => {
                assert!(status.success(), "the gst peer failed: {status}");
                return;
            }
            None if std::time::Instant::now() >= deadline => {
                let _ = child.kill();
                panic!("the gst peer did not finish within {GST_DEADLINE:?}");
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

fn gst_test_pattern() -> String {
    format!(
        "videotestsrc num-buffers={GST_FRAMES} pattern=ball ! \
         video/x-raw,format=RGBA,width={GST_WIDTH},height={GST_HEIGHT},framerate={FRAMERATE}/1"
    )
}

/// gst sends, g2g receives. gst 1.26's unixfdsink refuses buffers that are not
/// fd memory, which a tee in front of it hands over, so the reference file
/// comes from a second run of the same deterministic pattern.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with gst-plugins-bad"]
async fn unixfdsrc_reads_gst_unixfdsink() {
    let reference = unique_path("gst_reference.raw");
    let status = gst(&format!(
        "{} ! filesink location={}",
        gst_test_pattern(),
        reference.display()
    ))
    .status()
    .expect("gst-launch-1.0 is on PATH");
    assert!(status.success());

    let socket = unique_path("gst_sink.sock");
    let peer = gst(&format!(
        "{} ! unixfdsink socket-path={} wait-for-connection=true",
        gst_test_pattern(),
        socket.display()
    ))
    .spawn()
    .expect("gst-launch-1.0 is on PATH");
    let mut source = UnixFdSrc::new(socket.display().to_string());
    let (caps, count, out) = tokio::time::timeout(GST_DEADLINE, receive(&mut source))
        .await
        .expect("the receiver finishes")
        .expect("the source runs");
    wait_for_gst(peer);

    let progressive = rgba_caps_interlaced(GST_WIDTH, GST_HEIGHT, Interlace::Progressive);
    assert_eq!(caps, progressive, "gst names its interlace mode");
    assert_eq!(count, GST_FRAMES);
    let received: Vec<u8> = out
        .frames()
        .iter()
        .flat_map(|frame| frame.domain.as_system_slice().expect("system").to_vec())
        .collect();
    assert_eq!(received.len(), GST_FRAME_BYTES * GST_FRAMES as usize);
    assert!(received == std::fs::read(&reference).expect("read the reference"));
    let pts: Vec<u64> = out
        .frames()
        .iter()
        .map(|frame| frame.timing.pts_ns)
        .collect();
    assert!(pts[0] <= FIRST_PTS_BOUND_NS, "first pts {}", pts[0]);
    let last_step = pts[pts.len() - 1] - pts[pts.len() - 2];
    assert!(
        last_step.abs_diff(FRAME_PERIOD_NS) <= PTS_TOLERANCE_NS,
        "pts advance by the frame period: {pts:?}"
    );
}

/// g2g sends, gst receives and writes the bytes out.
#[tokio::test]
#[ignore = "needs gst-launch-1.0 with gst-plugins-bad"]
async fn unixfdsink_feeds_gst_unixfdsrc() {
    let socket = unique_path("gst_src.sock");
    let output = unique_path("gst_output.raw");
    let caps = rgba_caps(WIDTH, HEIGHT);
    let mut sink = UnixFdSink::new(socket.display().to_string()).with_wait_for_connection(true);
    sink.configure_pipeline(&caps).expect("the sink binds");
    let peer = gst(&format!(
        "unixfdsrc socket-path={} ! filesink location={}",
        socket.display(),
        output.display()
    ))
    .spawn()
    .expect("gst-launch-1.0 is on PATH");
    let packets = (0..FRAMES).map(system_frame).collect();
    tokio::time::timeout(GST_DEADLINE, feed(&mut sink, &caps, packets))
        .await
        .expect("the sender finishes");
    let waiter = tokio::task::spawn_blocking(move || wait_for_gst(peer));
    waiter.await.expect("the gst peer ends");
    drop(sink);

    let expected: Vec<u8> = (0..FRAMES).flat_map(frame_bytes).collect();
    assert!(std::fs::read(&output).expect("read the gst output") == expected);
}
