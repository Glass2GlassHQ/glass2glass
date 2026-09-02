//! M931 step 0: a `Segment` reaching a fan-in element.
//!
//! A paced sink maps a frame's PTS to running time through the `Segment` in
//! effect. A fan-in sat in the middle of that path and swallowed the segment, so
//! a graph like `demux ! decode ! convert ! compositor ! sink` left the sink with
//! no mapping: a DVD title whose PTS starts at 2267 s stalled for 37 minutes at
//! zero CPU. The compositor's output frames carry input 0's PTS, so input 0's
//! segment is the one that has to reach the sink.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    Caps, Dim, G2gError, MultiInputElement, OutputSink, PushOutcome, Rate, RawVideoFormat, Seek,
    Segment,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};

const W: u32 = 64;
const H: u32 = 32;
/// A DVD title's PTS base: far enough out that a sink pacing against it without
/// a segment waits over half an hour.
const BASE_NS: u64 = 2_267_767_267_000;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");

        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn segments(&self) -> Vec<Segment> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::Segment(s) => Some(*s),
                _ => None,
            })
            .collect()
    }

    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(pts_ns: u64) -> PipelinePacket {
    let bytes = alloc_canvas();
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes)),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn alloc_canvas() -> Box<[u8]> {
    vec![0u8; (W * H * 4) as usize].into_boxed_slice()
}

fn compositor() -> Compositor {
    Compositor::new(W, H, vec![CompositorPad::at(0, 0), CompositorPad::at(0, 0)])
}

/// The step-0 contract: input 0's segment reaches the output, once, carrying the
/// mapping that puts the first frame at running time 0.
#[tokio::test]
async fn a_fan_in_forwards_the_timing_inputs_segment() {
    let mut c = compositor();
    for pad in 0..2 {
        c.configure_pipeline(pad, &rgba_caps()).expect("configure");
    }
    let mut sink = Collect::default();

    let seg = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
    c.process(0, PipelinePacket::Segment(seg), &mut sink)
        .await
        .expect("segment on the timing input");
    // The overlay is primed first: the compositor holds input-0 frames at startup
    // until each overlay has delivered, so a lone input-0 frame emits nothing.
    c.process(1, frame(BASE_NS), &mut sink)
        .await
        .expect("overlay frame");
    c.process(0, frame(BASE_NS), &mut sink)
        .await
        .expect("first frame");

    let segments = sink.segments();
    assert_eq!(segments.len(), 1, "input 0's segment reaches the output");
    assert_eq!(segments[0].start, BASE_NS, "carrying its own mapping");
    assert_eq!(
        segments[0].to_running_time(BASE_NS),
        Some(0),
        "so the first frame presents immediately rather than 37 minutes late"
    );

    // And it leads the frame it describes: a sink that saw the frame first would
    // already have computed a deadline from the raw PTS.
    let seg_at = sink
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::Segment(_)))
        .expect("a segment");
    let frame_at = sink
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .expect("a frame");
    assert!(seg_at < frame_at, "the segment leads the first frame");
}

/// An overlay input's segment is not the output's: the output frames are stamped
/// from input 0, so forwarding a subtitle track's segment would remap the video.
#[tokio::test]
async fn an_overlay_inputs_segment_is_not_forwarded() {
    let mut c = compositor();
    for pad in 0..2 {
        c.configure_pipeline(pad, &rgba_caps()).expect("configure");
    }
    let mut sink = Collect::default();

    let overlay_seg = Segment::for_flush_seek(&Seek::flush_to(999_000_000_000), None);
    c.process(1, PipelinePacket::Segment(overlay_seg), &mut sink)
        .await
        .expect("segment on the overlay input");
    assert!(
        sink.segments().is_empty(),
        "an overlay's segment does not remap the output"
    );

    // Input 0's still does, and is the one that goes out.
    let seg = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
    c.process(0, PipelinePacket::Segment(seg), &mut sink)
        .await
        .expect("segment on the timing input");
    assert_eq!(sink.segments(), [seg], "only the timing input's");
}

/// The regression guard for the freeze this milestone chased. A link opens with
/// the runner's default segment (`start = 0`) and the demuxer's real one arrives
/// after it, so a fan-in that forwards only the FIRST segment pins the sink to
/// the wrong mapping: frames stamped 2267 s against a `start = 0` segment are
/// 2267 s of running time away, and a paced sink holds every one of them at zero
/// CPU. A later segment supersedes an earlier one.
#[tokio::test]
async fn a_later_segment_supersedes_the_runners_default() {
    let mut c = compositor();
    for pad in 0..2 {
        c.configure_pipeline(pad, &rgba_caps()).expect("configure");
    }
    let mut sink = Collect::default();

    // Exactly the order a real graph delivers: the open segment, then the
    // demuxer's stream-start one.
    let opening = Segment::new();
    let real = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
    c.process(0, PipelinePacket::Segment(opening), &mut sink)
        .await
        .unwrap();
    c.process(0, PipelinePacket::Segment(real), &mut sink)
        .await
        .unwrap();

    let segments = sink.segments();
    assert_eq!(
        segments.last(),
        Some(&real),
        "the demuxer's segment is the one left in force, not the opening default"
    );
    assert_eq!(
        segments.last().unwrap().to_running_time(BASE_NS),
        Some(0),
        "so the first frame presents immediately"
    );

    // An unchanged repeat is not re-emitted: a segment per frame would reset the
    // sink's mapping continuously.
    let before = sink.segments().len();
    c.process(0, PipelinePacket::Segment(real), &mut sink)
        .await
        .unwrap();
    assert_eq!(
        sink.segments().len(),
        before,
        "re-sending the same segment changes nothing"
    );
}

/// A flush re-arms it: the stream after a seek carries its own mapping.
#[tokio::test]
async fn a_flush_re_arms_the_segment() {
    let mut c = compositor();
    for pad in 0..2 {
        c.configure_pipeline(pad, &rgba_caps()).expect("configure");
    }
    let mut sink = Collect::default();

    let first = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
    c.process(0, PipelinePacket::Segment(first), &mut sink)
        .await
        .unwrap();
    c.process(0, PipelinePacket::Flush, &mut sink)
        .await
        .unwrap();
    let after = Segment::for_flush_seek(&Seek::flush_to(BASE_NS + 60_000_000_000), None);
    c.process(0, PipelinePacket::Segment(after), &mut sink)
        .await
        .unwrap();

    assert_eq!(
        sink.segments(),
        [first, after],
        "the post-seek segment goes out too"
    );
}

/// The compositing itself is unchanged by a segment passing through: it is
/// control, not pixels.
#[tokio::test]
async fn a_segment_does_not_disturb_compositing() {
    let mut with = compositor();
    let mut without = compositor();
    for c in [&mut with, &mut without] {
        for pad in 0..2 {
            c.configure_pipeline(pad, &rgba_caps()).expect("configure");
        }
    }
    let (mut a, mut b) = (Collect::default(), Collect::default());

    let seg = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
    with.process(0, PipelinePacket::Segment(seg), &mut a)
        .await
        .unwrap();
    for pts in [BASE_NS, BASE_NS + 40_000_000] {
        with.process(0, frame(pts), &mut a).await.unwrap();
        without.process(0, frame(pts), &mut b).await.unwrap();
    }

    let (fa, fb) = (a.frames(), b.frames());
    assert_eq!(fa.len(), fb.len(), "the same number of composited frames");
    for (x, y) in fa.iter().zip(fb.iter()) {
        assert_eq!(x.timing.pts_ns, y.timing.pts_ns, "same timestamps");
        assert_eq!(
            x.domain.as_system_slice(),
            y.domain.as_system_slice(),
            "same pixels"
        );
    }
}

// ---- blast radius: a segment reaching a muxer must change nothing ----

/// The step-0 contract now delivers `Segment` to every fan-in element, muxers
/// included. A container's timestamps are already mapped by its own headers, so
/// a segment arriving must be consumed, not written: the muxed bytes have to be
/// identical to a run where none arrived.
mod muxers {
    use super::*;
    use g2g_core::AudioFormat;

    /// A byte sink recording exactly what a muxer wrote.
    #[derive(Default)]
    struct Bytes {
        out: Vec<u8>,
        segments: usize,
    }

    impl OutputSink for Bytes {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");

            match &packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.out.extend_from_slice(s);
                    }
                }
                PipelinePacket::Segment(_) => self.segments += 1,
                _ => {}
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn aac_caps() -> Caps {
        Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    /// One ADTS AAC frame, enough for a muxer to write a track.
    fn adts(pts_ns: u64, seq: u64) -> PipelinePacket {
        let mut au = vec![0xFF, 0xF1, 0x50, 0x80, 0x03, 0x9F, 0xFC];
        au.extend_from_slice(&[0x21, 0x1A, 0x8F, 0xE0]);
        let len = au.len() as u16;
        au[3] = (au[3] & 0xFC) | ((len >> 11) & 0x03) as u8;
        au[4] = ((len >> 3) & 0xFF) as u8;
        au[5] = (au[5] & 0x1F) | (((len & 0x07) << 5) as u8);
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                keyframe: true,
                ..FrameTiming::default()
            },
            seq,
        ))
    }

    /// Drive `mux` over one audio input, optionally handing it a segment first,
    /// and return the bytes it wrote plus how many segments it forwarded.
    async fn run(mux: &mut dyn MuxUnderTest, with_segment: bool) -> (Vec<u8>, usize) {
        let mut sink = Bytes::default();
        mux.configure(0, &aac_caps());
        if with_segment {
            let seg = Segment::for_flush_seek(&Seek::flush_to(BASE_NS), None);
            mux.feed(0, PipelinePacket::Segment(seg), &mut sink).await;
        }
        for i in 0..4u64 {
            mux.feed(0, adts(BASE_NS + i * 21_333_333, i), &mut sink)
                .await;
        }
        mux.feed(0, PipelinePacket::Eos, &mut sink).await;
        (sink.out, sink.segments)
    }

    /// Object-safe shim so one runner drives each muxer type.
    trait MuxUnderTest {
        fn configure(&mut self, pad: usize, caps: &Caps);
        fn feed<'a>(
            &'a mut self,
            pad: usize,
            packet: PipelinePacket,
            out: &'a mut Bytes,
        ) -> Pin<Box<dyn Future<Output = ()> + 'a>>;
    }

    impl<T: MultiInputElement> MuxUnderTest for T {
        fn configure(&mut self, pad: usize, caps: &Caps) {
            let _ = self.configure_pipeline(pad, caps);
        }
        fn feed<'a>(
            &'a mut self,
            pad: usize,
            packet: PipelinePacket,
            out: &'a mut Bytes,
        ) -> Pin<Box<dyn Future<Output = ()> + 'a>> {
            Box::pin(async move {
                let _ = self.process(pad, packet, out).await;
            })
        }
    }

    macro_rules! byte_identical {
        ($name:ident, $ctor:expr, $label:literal) => {
            #[tokio::test]
            async fn $name() {
                let (plain, plain_segs) = run(&mut $ctor, false).await;
                let (with, with_segs) = run(&mut $ctor, true).await;
                assert!(!plain.is_empty(), "{} wrote a container", $label);
                assert_eq!(
                    plain, with,
                    "{} output is byte-identical with a segment delivered",
                    $label
                );
                assert_eq!(
                    (plain_segs, with_segs),
                    (0, 0),
                    "{} forwards no segment into its byte stream",
                    $label
                );
            }
        };
    }

    byte_identical!(
        mkvmux_is_byte_identical,
        g2g_plugins::mkvmuxn::MkvMuxN::new(1),
        "matroskamux"
    );
    byte_identical!(
        tsmux_is_byte_identical,
        g2g_plugins::tsmuxn::TsMux::new(1),
        "mpegtsmux"
    );
    // `oggmux` and `flvmux` take no AAC track, so this ADTS fixture cannot drive
    // them; their ignore arm is the same code, and their own muxer suites (which
    // compare against ffmpeg's read of the output) cover the bytes.
    byte_identical!(
        mp4mux_is_byte_identical,
        g2g_plugins::mp4muxn::Mp4MuxN::new(1),
        "mp4mux"
    );
}

// ---- overlay inputs advance by timestamp, not arrival ----

/// A compositor's overlay inputs must apply by *timestamp*: the canvas in force
/// for an input-0 frame at pts T is the newest overlay canvas whose pts is at or
/// before T, held until a successor comes due.
///
/// Taking the latest-arrived instead only looks right when nothing paces. Live,
/// the display sink paces input 0 to real time while the subtitle branch runs
/// flat out, so every cue and its clearing canvas arrive within the first
/// moments and the last to land (a clear) is what every visible frame
/// composites with: video and audio play, no subtitles ever appear. Dumped
/// headless to a file, nothing paces, arrival order matches the file interleave
/// and the cues land correctly by accident.
mod overlay_timing {
    use super::*;

    const MS: u64 = 1_000_000;

    /// A canvas filled with one value, so "which overlay is in force" is a
    /// single byte to read back.
    fn canvas(fill: u8, pts_ms: u64) -> PipelinePacket {
        let px = [fill, fill, fill, 255];
        let mut buf = Vec::with_capacity((W * H * 4) as usize);
        for _ in 0..W * H {
            buf.extend_from_slice(&px);
        }
        let pts = pts_ms * MS;
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
            FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    /// A fully transparent canvas: the "clear" a cue ends with. Alpha 0, so
    /// compositing it leaves the video underneath untouched (an opaque black
    /// canvas would blank the picture instead).
    fn clear(pts_ms: u64) -> PipelinePacket {
        let pts = pts_ms * MS;
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                vec![0u8; (W * H * 4) as usize].into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                ..FrameTiming::default()
            },
            0,
        ))
    }

    /// Top-left pixel of each composited output, paired with its pts in ms.
    fn composited(sink: &Collect) -> Vec<(u64, [u8; 4])> {
        sink.frames()
            .iter()
            .map(|f| {
                let px = f.domain.as_system_slice().expect("system frame");
                (f.timing.pts_ns / MS, [px[0], px[1], px[2], px[3]])
            })
            .collect()
    }

    async fn configured() -> Compositor {
        let mut c = Compositor::new(W, H, vec![CompositorPad::at(0, 0), CompositorPad::at(0, 0)]);
        for pad in 0..2 {
            c.configure_pipeline(pad, &rgba_caps()).expect("configure");
        }
        c
    }

    /// The screen race, exactly: every overlay canvas is delivered before the
    /// second input-0 frame. Arrival order says "the clear is in force for all
    /// of them"; timestamps say the cue covers pts 2..5.
    #[tokio::test]
    async fn overlay_canvases_delivered_early_still_apply_at_their_own_pts() {
        let mut c = configured().await;
        let mut sink = Collect::default();

        // Input 0 opens the stream.
        c.process(0, canvas(10, 0), &mut sink).await.unwrap();
        // The whole subtitle branch lands at once, ahead of the video.
        c.process(1, canvas(200, 2), &mut sink).await.unwrap();
        c.process(1, clear(5), &mut sink).await.unwrap();
        // Then the video frames arrive, paced.
        for pts in 1..=7u64 {
            c.process(0, canvas(10, pts), &mut sink).await.unwrap();
        }

        let out = composited(&sink);
        let at = |ms: u64| {
            out.iter()
                .find(|(p, _)| *p == ms)
                .unwrap_or_else(|| panic!("an output at {ms} ms: {out:?}"))
                .1
        };
        // Before the cue's pts: no cue.
        assert_eq!(at(1)[0], 10, "pts 1 is before the cue");
        // Inside the cue's window: the cue is composited (it is opaque white
        // over the video, so the pixel is the cue's).
        assert_eq!(at(3)[0], 200, "pts 3 carries the cue");
        assert_eq!(at(4)[0], 200, "and it is held until the clear is due");
        // After the clear's pts: the cue is gone.
        assert_eq!(at(6)[0], 10, "pts 6 is after the clear");
    }

    /// The same rule with the overlays interleaved in timestamp order, which is
    /// how a headless (unpaced) run happens to deliver them.
    #[tokio::test]
    async fn overlay_canvases_delivered_interleaved_apply_at_their_own_pts() {
        let mut c = configured().await;
        let mut sink = Collect::default();

        c.process(0, canvas(10, 0), &mut sink).await.unwrap();
        c.process(1, canvas(200, 2), &mut sink).await.unwrap();
        for pts in 1..=4u64 {
            c.process(0, canvas(10, pts), &mut sink).await.unwrap();
        }
        c.process(1, clear(5), &mut sink).await.unwrap();
        for pts in 5..=7u64 {
            c.process(0, canvas(10, pts), &mut sink).await.unwrap();
        }

        let out = composited(&sink);
        let at = |ms: u64| {
            out.iter()
                .find(|(p, _)| *p == ms)
                .unwrap_or_else(|| panic!("an output at {ms} ms: {out:?}"))
                .1
        };
        assert_eq!(at(1)[0], 10, "before the cue");
        assert_eq!(at(3)[0], 200, "inside the cue window");
        assert_eq!(at(6)[0], 10, "after the clear");
    }

    /// A cue whose pts is still ahead of every video frame never composites: it
    /// is queued, not applied early.
    #[tokio::test]
    async fn a_future_cue_does_not_apply_early() {
        let mut c = configured().await;
        let mut sink = Collect::default();
        c.process(0, canvas(10, 0), &mut sink).await.unwrap();
        c.process(1, canvas(200, 900), &mut sink).await.unwrap();
        for pts in 1..=4u64 {
            c.process(0, canvas(10, pts), &mut sink).await.unwrap();
        }
        for (pts, px) in composited(&sink) {
            assert_eq!(px[0], 10, "no cue is due yet at {pts} ms");
        }
    }
}
