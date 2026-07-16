//! M28: `Mp4Src` reads back what `Mp4Mux` muxes. Round trip is
//! byte-exact (Annex-B in, fMP4, Annex-B out), the caps probe recovers the
//! recorded geometry during negotiation, and on Windows the full circle
//! runs encode -> container -> demux -> decode through both real MFTs.

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, Rate, TagList, VideoCodec};
use g2g_plugins::mp4mux::Mp4Mux;
use g2g_plugins::mp4src::Mp4Src;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m28_{}_{}.mp4", std::process::id(), name))
}

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
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

fn frame_bytes(f: &Frame) -> &[u8] {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("System frames expected");
    };
    slice.as_slice()
}

fn au_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// A sink that concatenates the ISO-BMFF byte-stream frames `Mp4Mux` forwards.
#[derive(Default)]
struct Capture {
    bytes: Vec<u8>,
}

impl OutputSink for Capture {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Mux `aus` through `Mp4Mux`, returning the forwarded fMP4 byte stream. Tags,
/// when present, go in the init segment's moov.
async fn mux_to_bytes(aus: &[Vec<u8>], w: u32, h: u32, tags: Option<TagList>) -> Vec<u8> {
    let mut mux = match tags {
        Some(t) => Mp4Mux::new().with_tags(t),
        None => Mp4Mux::new(),
    };
    mux.configure_pipeline(&h264_caps(w, h)).expect("configure mux");
    let mut cap = Capture::default();
    for (i, au) in aus.iter().enumerate() {
        mux.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut cap,
        )
        .await
        .expect("mux AU");
    }
    mux.process(PipelinePacket::Eos, &mut cap).await.expect("eos");
    cap.bytes
}

/// Record `aus` to an fMP4 file at `path` (muxed via `Mp4Mux`, then written).
async fn record(path: &PathBuf, aus: &[Vec<u8>], w: u32, h: u32) {
    let bytes = mux_to_bytes(aus, w, h, None).await;
    std::fs::write(path, &bytes).unwrap();
}

#[tokio::test]
async fn round_trip_recovers_access_units_and_timing() {
    let path = temp_path("roundtrip");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr_au: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, 0xBB],
    ]
    .concat();
    let p_au = |fill: u8| [&[0, 0, 0, 1][..], &[0x41, fill, fill, fill]].concat();
    let aus = vec![idr_au, p_au(1), p_au(2)];

    record(&path, &aus, 64, 48).await;

    // probe before negotiation: dims recovered from the moov.
    let mut src = Mp4Src::new(&path);
    let caps = src.intercept_caps().await.expect("probe header");
    assert_eq!(
        caps,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            // advisory range, not `Any`: per-frame PTS carries the real timing
            // and `Any` would abort fixate when nothing downstream pins the rate.
            framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
        }
    );

    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    let produced = src.run(&mut out).await.expect("demux to EOS");
    assert_eq!(produced, 3);

    let frames = out.frames();
    assert_eq!(frames.len(), 3);
    for (i, original) in aus.iter().enumerate() {
        assert_eq!(
            frame_bytes(frames[i]),
            &original[..],
            "AU {i} must round trip byte-exactly"
        );
    }
    // timing recovered from tfdt/trun at 90 kHz granularity.
    assert_eq!(frames[0].timing.pts_ns, 0);
    let pts1 = frames[1].timing.pts_ns;
    assert!(
        (pts1 as i64 - 33_333_333).abs() < 20_000,
        "second AU pts {pts1} should be ~33.33 ms (90 kHz rounding)"
    );
    assert!(frames[0].timing.duration_ns > 33_000_000);
    assert!(
        matches!(out.packets.last(), Some(PipelinePacket::Eos)),
        "EOS terminates the stream"
    );
    let _ = std::fs::remove_file(&path);
}

/// A size-prefixed MP4 box `[u32 size][4cc][payload]`.
fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut b = (payload.len() as u32 + 8).to_be_bytes().to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(payload);
    b
}

/// An iTunes UTF-8 text item (`©nam`, ...) wrapping a `data` box.
fn text_item(kind: &[u8; 4], value: &str) -> Vec<u8> {
    let mut data = 1u32.to_be_bytes().to_vec(); // type 1 = UTF-8
    data.extend_from_slice(&0u32.to_be_bytes()); // locale
    data.extend_from_slice(value.as_bytes());
    mp4_box(kind, &mp4_box(b"data", &data))
}

/// Insert `udta` at the end of the top-level `moov`'s children, patching the
/// moov box size so the file stays well-formed.
fn splice_into_moov(mp4: &[u8], udta: &[u8]) -> Vec<u8> {
    let mut at = 0usize;
    while at + 8 <= mp4.len() {
        let size = u32::from_be_bytes(mp4[at..at + 4].try_into().unwrap()) as usize;
        if &mp4[at + 4..at + 8] == b"moov" {
            let new_size = (size + udta.len()) as u32;
            let mut out = mp4[..at].to_vec();
            out.extend_from_slice(&new_size.to_be_bytes());
            out.extend_from_slice(&mp4[at + 4..at + size]); // moov 4cc + existing children
            out.extend_from_slice(udta); // appended child
            out.extend_from_slice(&mp4[at + size..]); // the rest (moof/mdat)
            return out;
        }
        at += size;
    }
    panic!("no moov box found");
}

#[tokio::test]
async fn sink_written_tags_round_trip_to_the_source_bus() {
    use g2g_core::{Bus, BusMessage, Tag};

    let path = temp_path("tag_roundtrip");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr_au: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xAA]].concat();

    // Record with tags attached to the mux's init-segment moov.
    let tags: TagList =
        [Tag::Title("Recorded".into()), Tag::Encoder("g2g".into())].into_iter().collect();
    let bytes = mux_to_bytes(&[idr_au], 64, 48, Some(tags.clone())).await;
    std::fs::write(&path, &bytes).unwrap();

    // Read it back; the source surfaces the same tags on the bus.
    let (bus, handle) = Bus::new(8);
    let mut src = Mp4Src::new(&path).with_bus(handle);
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    src.run(&mut out).await.expect("demux");

    let mut posted = None;
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Tag(t) = m {
            posted = Some(t);
        }
    }
    assert_eq!(posted.expect("a Tag message").tags(), tags.tags());
    assert_eq!(out.frames().len(), 1, "the AU still demuxes alongside the tags");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn surfaces_ilst_tags_on_the_bus() {
    use g2g_core::{Bus, BusMessage, Tag};

    let path = temp_path("tags");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr_au: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xAA]].concat();
    record(&path, &[idr_au], 64, 48).await;

    // splice a udta/meta/ilst (title + encoder) into the recorded moov.
    let ilst = [text_item(b"\xA9nam", "Spliced Clip"), text_item(b"\xA9too", "g2g")].concat();
    let mut meta = vec![0u8, 0, 0, 0]; // meta full box version/flags
    meta.extend_from_slice(&mp4_box(b"ilst", &ilst));
    let udta = mp4_box(b"udta", &mp4_box(b"meta", &meta));
    let original = std::fs::read(&path).unwrap();
    std::fs::write(&path, splice_into_moov(&original, &udta)).unwrap();

    let (bus, handle) = Bus::new(8);
    let mut src = Mp4Src::new(&path).with_bus(handle);
    let caps = src.intercept_caps().await.expect("probe still works with udta");
    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    src.run(&mut out).await.expect("demux");
    assert_eq!(out.frames().len(), 1, "the sample still demuxes alongside the tags");

    let mut posted = None;
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Tag(t) = m {
            posted = Some(t);
        }
    }
    let tags = posted.expect("a Tag message was posted");
    assert_eq!(tags.tags(), &[Tag::Title("Spliced Clip".into()), Tag::Encoder("g2g".into())]);
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn flush_seek_repositions_to_keyframe() {
    use g2g_core::runtime::SeekController;
    use g2g_core::Seek;

    let path = temp_path("seek");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    // The first AU carries the parameter sets in-band (so the moov picks them up);
    // a later IDR does not, exercising the post-seek param-set prepend.
    let idr0: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xA0]].concat();
    let p = |tag: u8| [&[0, 0, 0, 1][..], &[0x41, tag][..]].concat();
    let idr2: Vec<u8> = [&[0, 0, 0, 1][..], &[0x65, 0xA2][..]].concat();
    let aus = vec![idr0, p(0xA1), idr2, p(0xA3)]; // keyframes at index 0 and 2
    record(&path, &aus, 64, 48).await;

    let ctl = SeekController::new();
    // Target ~70 ms snaps back to the keyframe at ~66.6 ms (the 3rd AU).
    ctl.seek(Seek::flush_to(70_000_000));

    let mut src = Mp4Src::new(&path).with_seek(ctl.clone());
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    let produced = src.run(&mut out).await.expect("run");

    assert!(
        out.packets.iter().any(|p| matches!(p, PipelinePacket::Flush)),
        "the flushing seek flushed downstream"
    );
    let seg = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::Segment(s) => Some(s),
            _ => None,
        })
        .expect("a post-seek segment");
    assert_eq!(seg.start, 70_000_000, "segment starts at the requested target");

    // Resumed from the keyframe (index 2), not from the file start: two frames.
    let frames = out.frames();
    assert_eq!(frames.len(), 2, "keyframe + following P-frame");
    assert_eq!(produced, 2);
    let first = frame_bytes(frames[0]);
    assert!(first.windows(2).any(|w| w == [0x65, 0xA2]), "snapped to the 3rd AU's IDR");
    assert!(first.windows(2).any(|w| w == [0x67, 0x42]), "parameter sets prepended for resume");
    assert!(
        (frames[0].timing.pts_ns as i64 - 66_666_666).abs() < 50_000,
        "keyframe pts ~66.6 ms, got {}",
        frames[0].timing.pts_ns
    );
}

/// A sink that records like `Collect` and fires a one-shot seek the moment it
/// receives its first frame, so the source repositions mid-stream (the scrub
/// case): the source awaits this push, so the seek is pending before the next
/// frame is produced.
struct ScrubSink {
    inner: Collect,
    ctl: g2g_core::runtime::SeekController,
    target: u64,
    armed: bool,
}

impl OutputSink for ScrubSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if self.armed {
                if let PipelinePacket::DataFrame(_) = &packet {
                    self.ctl.seek(g2g_core::Seek::flush_to(self.target));
                    self.armed = false;
                }
            }
            self.inner.push(packet).await
        })
    }
}

#[tokio::test]
async fn mid_stream_scrub_repositions_after_frames_flow() {
    use g2g_core::runtime::SeekController;

    let path = temp_path("scrub");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr0: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xA0]].concat();
    let p = |tag: u8| [&[0, 0, 0, 1][..], &[0x41, tag][..]].concat();
    let idr2: Vec<u8> = [&[0, 0, 0, 1][..], &[0x65, 0xA2][..]].concat();
    record(&path, &[idr0, p(0xA1), idr2, p(0xA3)], 64, 48).await;

    let ctl = SeekController::new();
    let mut src = Mp4Src::new(&path).with_seek(ctl.clone());
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure");
    let mut sink = ScrubSink { inner: Collect::default(), ctl, target: 70_000_000, armed: true };
    src.run(&mut sink).await.expect("run");

    // First frame (idr0 at 0 ms) flows, then the scrub jumps to the keyframe near
    // 70 ms: idr0, Flush, Segment, idr2, p3.
    let kinds: Vec<&str> = sink
        .inner
        .packets
        .iter()
        .map(|p| match p {
            PipelinePacket::DataFrame(_) => "frame",
            PipelinePacket::Flush => "flush",
            PipelinePacket::Segment(_) => "segment",
            PipelinePacket::Eos => "eos",
            _ => "other",
        })
        .collect();
    assert_eq!(kinds, ["frame", "flush", "segment", "frame", "frame", "eos"]);
    let frames = sink.inner.frames();
    assert!(frame_bytes(frames[0]).windows(2).any(|w| w == [0x65, 0xA0]), "played idr0 first");
    assert!(frame_bytes(frames[1]).windows(2).any(|w| w == [0x65, 0xA2]), "scrubbed to idr2");
    let _ = std::fs::remove_file(&path);
}

/// M359: `Mp4Src` loops a clip gaplessly on a `SeekFlags::SEGMENT` seek. The
/// source plays `[0, stop]`, reports segment-done at the boundary instead of
/// Eos, and the app re-arms a non-flushing SEGMENT seek to loop (accumulating
/// running time, no flush); after N loops the app shuts the source down.
#[tokio::test]
async fn segment_seek_loops_clip_gaplessly_then_shuts_down() {
    use g2g_core::runtime::SeekController;
    use g2g_core::{Seek, SeekFlags, SeekType};

    fn segment_seek(flush: bool, stop: u64) -> Seek {
        let flags = if flush {
            SeekFlags::FLUSH | SeekFlags::SEGMENT
        } else {
            SeekFlags::SEGMENT
        };
        Seek { rate: 1.0, flags, start_type: SeekType::Set, start: 0, stop_type: SeekType::Set, stop }
    }

    let path = temp_path("segloop");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr0: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xA0]].concat();
    let p = |tag: u8| [&[0, 0, 0, 1][..], &[0x41, tag][..]].concat();
    // 4 AUs at pts 0, 33.3, 66.6, 100 ms; keyframe only at index 0.
    record(&path, &[idr0, p(0xA1), p(0xA2), p(0xA3)], 64, 48).await;

    // stop = 50 ms: the segment [0, 50ms] holds AUs 0 (0ms) and 1 (33.3ms); AU 2
    // (66.6ms) is past stop and triggers the boundary, so 2 frames per loop.
    let stop = 50_000_000u64;
    let n_loops = 3u64;

    let ctl = SeekController::new();
    ctl.seek(segment_seek(true, stop)); // arm the initial flushing segment seek

    let mut src = Mp4Src::new(&path).with_seek(ctl.clone());
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();

    let driver_ctl = ctl.clone();
    let driver = async move {
        // Track how many segment-dones we've handled, so a done that lands before
        // we first look (the source runs a whole segment in one poll, no yield) is
        // still seen rather than waited-on forever.
        let mut handled = 0u64;
        loop {
            while driver_ctl.segment_done_count() <= handled {
                tokio::task::yield_now().await;
            }
            handled = driver_ctl.segment_done_count();
            assert_eq!(driver_ctl.take_segment_done(), Some(stop));
            if handled >= n_loops {
                driver_ctl.shutdown();
                break;
            }
            driver_ctl.seek(segment_seek(false, stop)); // non-flushing loop
        }
    };

    let (run_res, ()) = tokio::join!(src.run(&mut out), driver);
    run_res.expect("run");

    assert_eq!(ctl.segment_done_count(), n_loops, "one segment-done per loop");
    // 2 frames per loop, N loops.
    assert_eq!(out.frames().len(), (n_loops * 2) as usize);
    // Each loop replays the same clip from its keyframe: every other frame is the
    // IDR (idr0), so the loop genuinely restarted at the segment start.
    let frames = out.frames();
    for k in 0..n_loops as usize {
        assert!(
            frame_bytes(frames[k * 2]).windows(2).any(|w| w == [0x65, 0xA0]),
            "loop {k} restarted at the keyframe idr0"
        );
    }

    // Only the initial seek flushed; the loop seeks are gapless (no flush).
    let flushes = out.packets.iter().filter(|p| matches!(p, PipelinePacket::Flush)).count();
    assert_eq!(flushes, 1, "only the initial segment seek flushes");
    // The running-time base accumulates one span (stop) per loop, so the last
    // loop's segment starts at (n_loops - 1) * stop: gapless across the loops.
    let segments: Vec<_> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::Segment(s) => Some(s),
            _ => None,
        })
        .collect();
    assert_eq!(segments.len() as u64, n_loops, "one segment per loop iteration");
    let last = segments.last().expect("a loop segment");
    assert_eq!(last.base, (n_loops - 1) * stop, "base accumulates one span per loop");
    assert_eq!(last.to_running_time(0), Some((n_loops - 1) * stop), "gapless across the loop");

    assert!(out.packets.iter().any(|p| matches!(p, PipelinePacket::Eos)), "shutdown ends with Eos");
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn missing_or_invalid_file_fails_loud() {
    let mut missing = Mp4Src::new(temp_path("missing"));
    assert!(missing.intercept_caps().await.is_err());

    // a non-MP4 file is rejected at probe, not silently emitted.
    let path = temp_path("garbage");
    std::fs::write(&path, b"not an mp4 at all").expect("write");
    let mut garbage = Mp4Src::new(&path);
    assert_eq!(
        garbage.intercept_caps().await.err(),
        Some(G2gError::CapsMismatch)
    );
    let _ = std::fs::remove_file(&path);
}

/// Full circle on Windows: real encode -> container -> demux -> real decode.
#[cfg(all(target_os = "windows", feature = "mf-encode", feature = "mf-decode"))]
#[tokio::test(flavor = "current_thread")]
async fn encode_mux_demux_decode_full_circle() {
    use g2g_plugins::mfdecode::MfDecode;
    use g2g_plugins::mfencode::MfEncode;

    const W: u32 = 320;
    const H: u32 = 240;
    const FRAMES: usize = 10;

    // encode synthetic NV12.
    let mut enc = MfEncode::new();
    let nv12 = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    enc.configure_pipeline(&nv12).expect("encoder init");
    let mut encoded = Collect::default();
    for i in 0..FRAMES {
        let mut data = vec![128u8; (W * H * 3 / 2) as usize];
        for (j, b) in data.iter_mut().take((W * H) as usize).enumerate() {
            *b = ((j + i * 16) % 256) as u8;
        }
        enc.process(
            PipelinePacket::DataFrame(au_frame(data, i as u64 * 33_333_333, i as u64)),
            &mut encoded,
        )
        .await
        .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded).await.expect("drain");

    // mux to fMP4.
    let path = temp_path("full_circle");
    let mut mux = Mp4Mux::new();
    mux.configure_pipeline(&h264_caps(W, H)).expect("configure mux");
    let mut cap = Capture::default();
    for f in encoded.frames() {
        mux.process(
            PipelinePacket::DataFrame(au_frame(
                frame_bytes(f).to_vec(),
                f.timing.pts_ns,
                f.sequence,
            )),
            &mut cap,
        )
        .await
        .expect("mux");
    }
    mux.process(PipelinePacket::Eos, &mut cap).await.expect("eos");
    std::fs::write(&path, &cap.bytes).unwrap();

    // demux and decode.
    let mut src = Mp4Src::new(&path);
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure src");
    let mut demuxed = Collect::default();
    src.run(&mut demuxed).await.expect("demux");

    let mut dec = MfDecode::new();
    dec.configure_pipeline(&caps).expect("decoder init");
    let mut decoded = Collect::default();
    for f in demuxed.frames() {
        dec.process(
            PipelinePacket::DataFrame(au_frame(
                frame_bytes(f).to_vec(),
                f.timing.pts_ns,
                f.sequence,
            )),
            &mut decoded,
        )
        .await
        .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded).await.expect("drain");

    let frames = decoded.frames();
    assert_eq!(frames.len(), FRAMES, "every frame survives the full circle");
    let expected_len = (W * H * 3 / 2) as usize;
    for f in frames {
        assert_eq!(frame_bytes(f).len(), expected_len, "packed NV12 out");
    }
    let _ = std::fs::remove_file(&path);
}

/// M203: `Mp4Src::query_duration` reads the `mdhd` movie duration. The fragmented
/// writer leaves it `0` (unknown until fragments), so a recorded file reports
/// `None`; patching a known `mdhd` duration in is read back as nanoseconds.
#[tokio::test]
async fn query_duration_reads_mdhd_duration() {
    let path = temp_path("duration");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr_au: Vec<u8> =
        [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps, &[0, 0, 0, 1], &[0x65, 0xAA]].concat();
    let p_au = |fill: u8| [&[0, 0, 0, 1][..], &[0x41, fill]].concat();
    let aus = vec![idr_au, p_au(1), p_au(2)];
    record(&path, &aus, 64, 48).await;

    // As recorded: fragmented init segment, mdhd duration 0 -> unknown.
    let mut src0 = Mp4Src::new(&path);
    let _ = src0.intercept_caps().await.expect("probe");
    assert_eq!(src0.query_duration(), None, "fragmented file: duration unknown");

    // Patch a known mdhd duration (2 s at the file's own timescale) and read back.
    let mut bytes = std::fs::read(&path).unwrap();
    let m = bytes.windows(4).position(|w| w == b"mdhd").expect("mdhd present");
    let ts_off = m + 4 + 12; // payload starts after the 4cc; timescale at +12
    let dur_off = m + 4 + 16; // duration at +16 (mdhd v0)
    let timescale = u32::from_be_bytes(bytes[ts_off..ts_off + 4].try_into().unwrap());
    let units = timescale * 2; // two seconds
    bytes[dur_off..dur_off + 4].copy_from_slice(&units.to_be_bytes());
    let patched = temp_path("duration_patched");
    std::fs::write(&patched, &bytes).unwrap();

    let mut src = Mp4Src::new(&patched);
    let _ = src.intercept_caps().await.expect("probe patched");
    assert_eq!(src.query_duration(), Some(2_000_000_000), "2 s read back as ns");

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&patched);
}
