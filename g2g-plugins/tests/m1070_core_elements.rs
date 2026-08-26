//! M1070: the gst `coreelements` g2g was missing, driven the way a pipeline
//! drives them.
//!
//! `valve` has to swallow data without ending the stream (a closed valve is not
//! an EOS), `fakesrc` has to produce the bytes its `filltype` names, and the
//! `fdsrc` / `fdsink` pair has to carry a byte stream through a descriptor
//! unchanged.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::ffi::c_void;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::set_appsink_callback;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Link capacity for these runs: enough that a source never stalls on it, small
/// enough to keep the in-flight set trivial.
const LINK_CAPACITY: usize = 4;

/// Buffers the valve runs push at the sink.
const VALVE_BUFFERS: u64 = 20;

/// Bytes per buffer in the valve runs, `fakesrc`'s own `sizemax` default.
const VALVE_BUFFER_SIZE: usize = 4096;

/// Buffers the fill-pattern run pushes.
const PATTERN_BUFFERS: u64 = 5;

/// Bytes per buffer in the fill-pattern run.
const PATTERN_BUFFER_SIZE: usize = 100;

/// What one appsink channel received.
#[derive(Default)]
struct Received {
    frames: Vec<Vec<u8>>,
    eos: bool,
}

extern "C" fn collect(data: *const u8, len: usize, _pts_ns: u64, user: *mut c_void) {
    // SAFETY: `user` is the &Mutex<Received> registered below, alive for the run.
    let received = unsafe { &*(user as *const Mutex<Received>) };
    let mut guard = received.lock().unwrap();
    if data.is_null() && len == 0 {
        guard.eos = true;
        return;
    }
    // SAFETY: appsink passes `len` readable bytes for the call.
    guard
        .frames
        .push(unsafe { std::slice::from_raw_parts(data, len) }.to_vec());
}

/// Run `line`, whose `appsink channel=` is `channel`, and return what arrived
/// there alongside the run's frame counts.
async fn run_into_appsink(channel: &str, line: &str) -> (Received, u64, u64) {
    let received = Box::new(Mutex::new(Received::default()));
    let user = (&*received as *const Mutex<Received>) as *mut c_void;
    set_appsink_callback(channel, collect, user);

    let registry = default_registry();
    let graph = parse_launch(&registry, line).expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .expect("pipeline runs");

    let collected = core::mem::take(&mut *received.lock().unwrap());
    (collected, stats.frames_emitted, stats.frames_consumed)
}

#[tokio::test]
async fn a_closed_valve_drops_every_frame_and_the_stream_still_ends() {
    let (received, emitted, consumed) = run_into_appsink(
        "m1070valveclosed",
        &format!(
            "fakesrc num-buffers={VALVE_BUFFERS} sizemax={VALVE_BUFFER_SIZE} \
             ! valve drop=true ! appsink channel=m1070valveclosed"
        ),
    )
    .await;

    assert_eq!(emitted, VALVE_BUFFERS, "the source produced every buffer");
    assert_eq!(consumed, 0, "the closed valve swallowed all of them");
    assert!(received.frames.is_empty());
    assert!(received.eos, "EOS still reached the sink");
}

#[tokio::test]
async fn an_open_valve_forwards_every_frame() {
    let (received, emitted, consumed) = run_into_appsink(
        "m1070valveopen",
        &format!(
            "fakesrc num-buffers={VALVE_BUFFERS} sizemax={VALVE_BUFFER_SIZE} \
             ! valve drop=false ! appsink channel=m1070valveopen"
        ),
    )
    .await;

    assert_eq!(emitted, VALVE_BUFFERS);
    assert_eq!(consumed, VALVE_BUFFERS, "an open valve is a pass-through");
    assert_eq!(received.frames.len(), VALVE_BUFFERS as usize);
    assert!(received.frames.iter().all(|f| f.len() == VALVE_BUFFER_SIZE));
    assert!(received.eos);
}

#[tokio::test]
async fn fakesrc_fills_each_buffer_with_the_pattern() {
    let (received, emitted, consumed) = run_into_appsink(
        "m1070pattern",
        &format!(
            "fakesrc num-buffers={PATTERN_BUFFERS} sizemax={PATTERN_BUFFER_SIZE} \
             filltype=pattern ! appsink channel=m1070pattern"
        ),
    )
    .await;

    assert_eq!(emitted, PATTERN_BUFFERS);
    assert_eq!(consumed, PATTERN_BUFFERS);
    // The pattern counts 0x00 -> 0xff and restarts at each buffer, so every
    // buffer is the same run of bytes.
    let expected: Vec<u8> = (0..PATTERN_BUFFER_SIZE).map(|i| i as u8).collect();
    assert_eq!(received.frames.len(), PATTERN_BUFFERS as usize);
    for frame in &received.frames {
        assert_eq!(frame, &expected, "buffer carries the pattern");
    }
}

/// Direct `process` drive: what a closed valve does with the ordered control
/// packets. `forward-sticky-events` lets caps and segment through while the
/// data is dropped; `drop-all` holds them back until the valve reopens.
mod control_packets {
    use g2g_core::memory::SystemSlice;
    use g2g_core::{
        AsyncElement, ByteStreamEncoding, Caps, Frame, FrameTiming, G2gError, MemoryDomain,
        OutputSink, PipelinePacket, PropValue, PushOutcome, Segment,
    };
    use g2g_plugins::valve::Valve;

    #[derive(Default)]
    struct CollectSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            self.packets
                .push(packet_slot.take().expect("poll_push without a packet"));
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }

    fn data_frame() -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(vec![1u8, 2, 3].into_boxed_slice())),
            FrameTiming::default(),
            0,
        ))
    }

    fn closed_valve(drop_mode: &str) -> Valve {
        let mut valve = Valve::new();
        valve.configure_pipeline(&caps()).unwrap();
        valve.set_property("drop", PropValue::Bool(true)).unwrap();
        valve
            .set_property("drop-mode", PropValue::Str(drop_mode.into()))
            .unwrap();
        valve
    }

    #[tokio::test]
    async fn forward_sticky_events_passes_caps_and_segment_while_dropping() {
        let mut valve = closed_valve("forward-sticky-events");
        let mut out = CollectSink::default();
        valve
            .process(PipelinePacket::CapsChanged(caps()), &mut out)
            .await
            .unwrap();
        valve
            .process(PipelinePacket::Segment(Segment::new()), &mut out)
            .await
            .unwrap();
        valve.process(data_frame(), &mut out).await.unwrap();
        valve
            .process(PipelinePacket::Flush, &mut out)
            .await
            .unwrap();

        assert!(matches!(out.packets[0], PipelinePacket::CapsChanged(_)));
        assert!(matches!(out.packets[1], PipelinePacket::Segment(_)));
        assert!(matches!(out.packets[2], PipelinePacket::Flush));
        assert_eq!(out.packets.len(), 3, "the frame was the only thing dropped");
        assert_eq!(valve.dropped(), 1);
    }

    #[tokio::test]
    async fn drop_all_holds_caps_and_segment_until_the_valve_reopens() {
        let mut valve = closed_valve("drop-all");
        let mut out = CollectSink::default();
        valve
            .process(PipelinePacket::CapsChanged(caps()), &mut out)
            .await
            .unwrap();
        valve
            .process(PipelinePacket::Segment(Segment::new()), &mut out)
            .await
            .unwrap();
        valve.process(data_frame(), &mut out).await.unwrap();
        assert!(out.packets.is_empty(), "nothing leaves a drop-all valve");

        valve.set_property("drop", PropValue::Bool(false)).unwrap();
        valve.process(data_frame(), &mut out).await.unwrap();
        assert!(matches!(out.packets[0], PipelinePacket::CapsChanged(_)));
        assert!(matches!(out.packets[1], PipelinePacket::Segment(_)));
        assert!(matches!(out.packets[2], PipelinePacket::DataFrame(_)));
        assert_eq!(out.packets.len(), 3, "the held state precedes the frame");
    }
}

/// `fdsrc` / `fdsink` over real descriptors. Unix only, like the elements.
#[cfg(unix)]
mod file_descriptors {
    use std::fs::{File, OpenOptions};
    use std::os::fd::AsRawFd;
    use std::path::PathBuf;

    use super::{run_into_appsink, ZeroClock, LINK_CAPACITY};
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_plugins::registry::default_registry;

    /// Buffers the descriptor round-trip writes.
    const ROUNDTRIP_BUFFERS: u64 = 3;

    /// Bytes per written buffer.
    const ROUNDTRIP_BUFFER_SIZE: usize = 1000;

    /// Read size for the descriptor round-trip, deliberately not a divisor of
    /// the written total, so the last chunk is a short read.
    const ROUNDTRIP_BLOCKSIZE: usize = 512;

    /// A checked-in byte stream `fdsrc` has to reproduce exactly. Its length is
    /// not a multiple of [`ROUNDTRIP_BLOCKSIZE`] either.
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_64x48_bframes.h264"
    );

    fn scratch_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    #[tokio::test]
    async fn a_byte_stream_survives_the_descriptor_round_trip() {
        let path = scratch_path("g2g_m1070_fd_roundtrip.bin");
        let _ = std::fs::remove_file(&path);
        // The element borrows the descriptor and never closes it, so the file
        // has to outlive the run that writes through its fd.
        let write_file = File::create(&path).expect("scratch file");
        let registry = default_registry();
        let graph = parse_launch(
            &registry,
            &format!(
                "fakesrc num-buffers={ROUNDTRIP_BUFFERS} sizemax={ROUNDTRIP_BUFFER_SIZE} \
                 filltype=pattern ! fdsink fd={}",
                write_file.as_raw_fd()
            ),
        )
        .expect("fdsink pipeline parses");
        run_graph(graph, &ZeroClock, LINK_CAPACITY)
            .await
            .expect("fdsink pipeline runs");
        drop(write_file);

        let one_buffer: Vec<u8> = (0..ROUNDTRIP_BUFFER_SIZE).map(|i| i as u8).collect();
        let expected: Vec<u8> = one_buffer.repeat(ROUNDTRIP_BUFFERS as usize);
        assert_eq!(
            std::fs::read(&path).unwrap(),
            expected,
            "fdsink wrote every buffer's bytes in order"
        );

        let read_file = OpenOptions::new().read(true).open(&path).expect("reopen");
        let (received, _, consumed) = run_into_appsink(
            "m1070fdroundtrip",
            &format!(
                "fdsrc fd={} blocksize={ROUNDTRIP_BLOCKSIZE} ! appsink channel=m1070fdroundtrip",
                read_file.as_raw_fd()
            ),
        )
        .await;
        drop(read_file);
        let _ = std::fs::remove_file(&path);

        assert_eq!(received.frames.concat(), expected, "fdsrc read them back");
        assert_eq!(
            consumed as usize,
            expected.len().div_ceil(ROUNDTRIP_BLOCKSIZE),
            "one frame per blocksize chunk, the last one short"
        );
    }

    #[tokio::test]
    async fn fdsrc_reproduces_a_file_byte_exact() {
        let file = File::open(FIXTURE).expect("fixture opens");
        let (received, _, _) = run_into_appsink(
            "m1070fdfixture",
            &format!(
                "fdsrc fd={} blocksize={ROUNDTRIP_BLOCKSIZE} ! appsink channel=m1070fdfixture",
                file.as_raw_fd()
            ),
        )
        .await;
        drop(file);

        assert_eq!(
            received.frames.concat(),
            std::fs::read(FIXTURE).unwrap(),
            "the descriptor read reproduces the file"
        );
        assert!(received.eos);
    }
}
