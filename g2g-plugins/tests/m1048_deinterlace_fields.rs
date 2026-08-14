//! M1048: the `deinterlace` element's `fields` and `tff` properties, and the
//! planar formats at 10 and 12 bits.
//!
//! Every kernel claim here is checked against ffmpeg's own `yadif` on the same
//! raw frames, so a mismatch is a porting bug and not a decoder difference:
//! `fields=all` against `yadif=1` (send_field), `tff=bff` against
//! `setfield=bff,yadif`, and the deep planar formats against the depth ffmpeg
//! filters them at. The structural claims the oracle cannot make (output caps,
//! field timestamps, the semi-planar chroma pair) are asserted directly.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::{AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PushOutcome, Rate};
use g2g_core::{Interlace, PropValue, RawVideoFormat};

use g2g_plugins::deinterlace::{
    Deinterlace, DeinterlaceFields, DeinterlaceMethod, DeinterlaceMode, FieldOrder,
};

const W: usize = 320;
const H: usize = 240;
const FPS_Q16: u32 = 25 << 16;
const FRAME_NS: u64 = 40_000_000;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1048-{}-{name}", std::process::id()))
}

fn ffmpeg(args: &[&str]) {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

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
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn frames(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }

    fn timings(&self) -> Vec<FrameTiming> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing),
                _ => None,
            })
            .collect()
    }

    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
}

fn caps(format: RawVideoFormat, w: usize, h: usize) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w as u32),
        height: Dim::Fixed(h as u32),
        framerate: Rate::Fixed(FPS_Q16),
        interlace: Interlace::Interleaved,
    }
}

/// Push every frame of `input` through a `deinterlace` configured as given, then
/// EOS, and return what came out.
async fn run(
    input: &[Vec<u8>],
    format: RawVideoFormat,
    w: usize,
    h: usize,
    method: DeinterlaceMethod,
    fields: DeinterlaceFields,
    field_order: FieldOrder,
) -> Collect {
    let mut element = Deinterlace::new()
        .with_method(method)
        .with_fields(fields)
        .with_field_order(field_order);
    element.configure_pipeline(&caps(format, w, h)).unwrap();
    let mut out = Collect::default();
    for (i, bytes) in input.iter().enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming {
                pts_ns: i as u64 * FRAME_NS,
                dts_ns: i as u64 * FRAME_NS,
                duration_ns: FRAME_NS,
                ..Default::default()
            },
            i as u64,
        );
        element
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .unwrap();
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .unwrap();
    out
}

fn split_frames(bytes: &[u8], size: usize) -> Vec<Vec<u8>> {
    bytes.chunks_exact(size).map(|c| c.to_vec()).collect()
}

fn frame_bytes(pixel_format: &str) -> usize {
    match pixel_format {
        "yuv420p" => W * H * 3 / 2,
        "yuv420p10le" | "yuv420p12le" => W * H * 3,
        "yuv422p10le" => W * H * 4,
        "yuv444p12le" => W * H * 6,
        other => unreachable!("unmodeled fixture format {other}"),
    }
}

/// Interlace `testsrc2` at twice the target rate into single frames whose two
/// fields are a field period apart: every frame has real motion between its
/// fields, which is what yadif's temporal branch keys on.
fn interlaced_fixture(name: &str, pixel_format: &str) -> Vec<Vec<u8>> {
    let path = temp_path(name);
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        &format!("testsrc2=size={W}x{H}:rate=50:duration=1"),
        "-vf",
        &format!("tinterlace=mode=interleave_top,format={pixel_format}"),
        "-f",
        "rawvideo",
        path.to_str().unwrap(),
    ]);
    let frames = split_frames(&std::fs::read(&path).unwrap(), frame_bytes(pixel_format));
    let _ = std::fs::remove_file(&path);
    assert!(frames.len() >= 10, "fixture is {} frames", frames.len());
    frames
}

/// The same fixture back through ffmpeg's `yadif` under `filter`.
///
/// `-cpuflags 0` picks ffmpeg's C kernel, which is what yadif's behavior is
/// defined by. Its 16-bit SIMD path does not agree with it: on this fixture the
/// two ffmpeg builds of the same filter differ on 73 samples by up to 264, so
/// only the C output is a reference worth being bit-exact against.
fn ffmpeg_yadif(name: &str, pixel_format: &str, filter: &str, input: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let raw = temp_path(&format!("{name}-in"));
    let reference = temp_path(&format!("{name}-out"));
    std::fs::write(&raw, input.concat()).unwrap();
    ffmpeg(&[
        "-cpuflags",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-s",
        &format!("{W}x{H}"),
        "-r",
        "25",
        "-i",
        raw.to_str().unwrap(),
        "-vf",
        filter,
        "-pix_fmt",
        pixel_format,
        "-f",
        "rawvideo",
        reference.to_str().unwrap(),
    ]);
    let frames = split_frames(
        &std::fs::read(&reference).unwrap(),
        frame_bytes(pixel_format),
    );
    let _ = std::fs::remove_file(&raw);
    let _ = std::fs::remove_file(&reference);
    frames
}

fn assert_bit_exact(got: &[Vec<u8>], want: &[Vec<u8>], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: output frame count");
    for (i, (g, r)) in got.iter().zip(want).enumerate() {
        let differing = g.iter().zip(r.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{what}: frame {i} differs from ffmpeg");
    }
}

// ---- differential: g2g vs ffmpeg yadif ----

/// `fields=all` is ffmpeg's `yadif=1` (send_field): one output per field, both
/// fields of every input frame, in presentation order.
#[tokio::test]
async fn send_field_matches_ffmpeg_bit_exactly() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let input = interlaced_fixture("sendfield", "yuv420p");
    let want = ffmpeg_yadif("sendfield", "yuv420p", "yadif=1:-1:0", &input);
    assert_eq!(want.len(), 2 * input.len(), "send_field doubles the rate");

    let got = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::All,
        FieldOrder::Auto,
    )
    .await;
    assert_bit_exact(&got.frames(), &want, "fields=all");

    // Each field lands half a frame period apart and covers half the span.
    let timings = got.timings();
    let pts: Vec<u64> = timings.iter().take(4).map(|t| t.pts_ns).collect();
    assert_eq!(pts, [0, 20_000_000, 40_000_000, 60_000_000]);
    assert!(timings.iter().all(|t| t.duration_ns == 20_000_000));
}

/// `tff=bff` is ffmpeg's bottom-field-first parity: the bottom field survives
/// and the top field's lines are the rebuilt ones.
#[tokio::test]
async fn bottom_field_first_matches_ffmpeg_bit_exactly() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let input = interlaced_fixture("bff", "yuv420p");
    let want = ffmpeg_yadif("bff", "yuv420p", "setfield=bff,yadif=0:-1:0", &input);

    let got = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::Auto,
        FieldOrder::BottomFirst,
    )
    .await;
    assert_bit_exact(&got.frames(), &want, "tff=bff");

    // The flipped parity has to change the pixels: a bff pass keeps the rows a
    // tff pass rebuilds.
    let tff = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::Auto,
        FieldOrder::TopFirst,
    )
    .await;
    assert_ne!(got.frames(), tff.frames(), "bff must not equal tff");
}

/// Both properties at once: the two passes swap which field they keep, and each
/// one swaps which temporal pair it reads.
#[tokio::test]
async fn bottom_field_first_send_field_matches_ffmpeg_bit_exactly() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let input = interlaced_fixture("bff-sendfield", "yuv420p");
    let want = ffmpeg_yadif(
        "bff-sendfield",
        "yuv420p",
        "setfield=bff,yadif=1:-1:0",
        &input,
    );

    let got = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::All,
        FieldOrder::BottomFirst,
    )
    .await;
    assert_bit_exact(&got.frames(), &want, "fields=all tff=bff");
}

/// The deep planar formats: ffmpeg filters each at its own depth, so a wrong
/// sample width, plane stride or endianness shows up as a mismatch.
#[tokio::test]
async fn deep_planar_formats_match_ffmpeg_bit_exactly() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for (format, pixel_format) in [
        (RawVideoFormat::I420p10, "yuv420p10le"),
        (RawVideoFormat::I422p10, "yuv422p10le"),
        (RawVideoFormat::I444p12, "yuv444p12le"),
    ] {
        let input = interlaced_fixture(pixel_format, pixel_format);
        let want = ffmpeg_yadif(pixel_format, pixel_format, "yadif=0:-1:0", &input);
        let got = run(
            &input,
            format,
            W,
            H,
            DeinterlaceMethod::Yadif,
            DeinterlaceFields::Auto,
            FieldOrder::Auto,
        )
        .await;
        assert_bit_exact(&got.frames(), &want, pixel_format);
        assert_ne!(
            got.frames()[2],
            input[2],
            "{pixel_format}: the fixture was actually filtered"
        );
    }
}

// ---- caps and element surface ----

#[tokio::test]
async fn all_fields_doubles_the_declared_framerate() {
    let input = vec![vec![0x40u8; W * H * 3 / 2]; 3];
    let doubled = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::All,
        FieldOrder::Auto,
    )
    .await;
    assert_eq!(
        doubled.caps_changes(),
        vec![Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(W as u32),
            height: Dim::Fixed(H as u32),
            framerate: Rate::Fixed(50 << 16),
            interlace: Interlace::Progressive,
        }]
    );

    let single = run(
        &input,
        RawVideoFormat::I420,
        W,
        H,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::Auto,
        FieldOrder::Auto,
    )
    .await;
    assert_eq!(
        single.caps_changes(),
        vec![Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(W as u32),
            height: Dim::Fixed(H as u32),
            framerate: Rate::Fixed(FPS_Q16),
            interlace: Interlace::Progressive,
        }],
        "the default rate is untouched"
    );
    assert_eq!(single.frames().len(), input.len());
}

/// The negotiated output caps have to carry the doubled rate too, or the runner
/// would hand downstream a rate the element then contradicts.
#[test]
fn the_derived_output_caps_carry_the_doubled_rate() {
    let element = Deinterlace::new()
        .with_mode(DeinterlaceMode::Auto)
        .with_fields(DeinterlaceFields::All);
    let g2g_core::CapsConstraint::DerivedOutput(derive) = element.caps_constraint_as_transform()
    else {
        panic!("deinterlace derives its output caps");
    };
    let solved = derive(&caps(RawVideoFormat::I420, W, H));
    assert_eq!(
        solved.alternatives(),
        &[Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(W as u32),
            height: Dim::Fixed(H as u32),
            framerate: Rate::Fixed(50 << 16),
            interlace: Interlace::Progressive,
        }]
    );

    // A progressive stream never reaches the kernels under `auto`, so it keeps
    // its rate.
    let progressive = Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(W as u32),
        height: Dim::Fixed(H as u32),
        framerate: Rate::Fixed(FPS_Q16),
        interlace: Interlace::Progressive,
    };
    assert_eq!(derive(&progressive).alternatives(), &[progressive]);
}

/// The P010 chroma plane interleaves 16-bit Cb and Cr: a layout that treated the
/// pair as one component would average them together.
#[tokio::test]
async fn p010_chroma_pairs_stay_separate() {
    let (w, h) = (16usize, 8usize);
    let (low, high) = (0x0100u16, 0xFF00u16);
    let mut frame = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let luma = if y % 2 == 0 { low } else { high };
            frame[(y * w + x) * 2..][..2].copy_from_slice(&luma.to_le_bytes());
        }
    }
    let chroma = w * h * 2;
    for i in 0..(w / 2 * h / 2) {
        frame[chroma + i * 4..][..2].copy_from_slice(&low.to_le_bytes());
        frame[chroma + i * 4 + 2..][..2].copy_from_slice(&high.to_le_bytes());
    }

    let out = run(
        &vec![frame.clone(); 3],
        RawVideoFormat::P010,
        w,
        h,
        DeinterlaceMethod::Yadif,
        DeinterlaceFields::Auto,
        FieldOrder::Auto,
    )
    .await;
    let got = &out.frames()[1];
    for i in 0..(w / 2 * h / 2) {
        let cb = u16::from_le_bytes([got[chroma + i * 4], got[chroma + i * 4 + 1]]);
        let cr = u16::from_le_bytes([got[chroma + i * 4 + 2], got[chroma + i * 4 + 3]]);
        assert_eq!((cb, cr), (low, high), "chroma sample {i} mixed the pair");
    }
}

/// `linear` has no temporal window, so `tff` shows up purely as which rows it
/// rebuilds.
#[tokio::test]
async fn linear_rebuilds_the_other_field_under_bff() {
    let (w, h) = (8usize, 8usize);
    let mut frame = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            frame[y * w + x] = if y % 2 == 0 { 0 } else { 240 };
        }
    }
    let luma = |out: &Collect, y: usize| out.frames()[0][y * w];

    let tff = run(
        &[frame.clone()],
        RawVideoFormat::I420,
        w,
        h,
        DeinterlaceMethod::Linear,
        DeinterlaceFields::Auto,
        FieldOrder::TopFirst,
    )
    .await;
    assert_eq!(
        (luma(&tff, 2), luma(&tff, 3)),
        (0, 0),
        "tff rebuilds odd rows"
    );

    let bff = run(
        &[frame.clone()],
        RawVideoFormat::I420,
        w,
        h,
        DeinterlaceMethod::Linear,
        DeinterlaceFields::Auto,
        FieldOrder::BottomFirst,
    )
    .await;
    assert_eq!(
        (luma(&bff, 2), luma(&bff, 3)),
        (240, 240),
        "bff rebuilds even rows"
    );
}

/// `blend` mixes both fields uniformly, so it has no parity to flip, but it
/// still owes one output frame per field under `fields=all`.
#[tokio::test]
async fn blend_ignores_field_order_and_still_doubles() {
    let input = vec![vec![0x30u8; W * H * 3 / 2]; 3];
    let mut produced = Vec::new();
    for field_order in [FieldOrder::TopFirst, FieldOrder::BottomFirst] {
        let out = run(
            &input,
            RawVideoFormat::I420,
            W,
            H,
            DeinterlaceMethod::Blend,
            DeinterlaceFields::All,
            field_order,
        )
        .await;
        assert_eq!(out.frames().len(), 2 * input.len());
        produced.push(out.frames());
    }
    assert_eq!(produced[0], produced[1], "blend is field-order agnostic");
}

/// End to end through the runner: the solver has to accept the doubled rate the
/// element derives, and the sink has to see two frames per source buffer.
#[tokio::test]
async fn a_launched_graph_doubles_the_frames_reaching_the_sink() {
    use g2g_core::runtime::{parse_launch, run_graph};
    use g2g_core::PipelineClock;
    use g2g_plugins::registry::default_registry;

    struct ZeroClock;
    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    let buffers = 6;
    for (fields, expected) in [("auto", buffers), ("all", 2 * buffers)] {
        let graph = parse_launch(
            &default_registry(),
            &format!("videotestsrc num-buffers={buffers} ! deinterlace fields={fields} ! fakesink"),
        )
        .expect("the pipeline parses");
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .expect("the pipeline runs");
        assert_eq!(stats.frames_consumed, expected, "fields={fields}");
    }
}

#[test]
fn fields_and_field_order_are_launch_properties() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    for fields in ["all", "top", "bottom", "auto"] {
        parse_launch(
            &default_registry(),
            &format!("videotestsrc num-buffers=1 ! deinterlace fields={fields} ! fakesink"),
        )
        .unwrap_or_else(|e| panic!("deinterlace fields={fields}: {e}"));
    }
    for order in ["auto", "tff", "bff"] {
        parse_launch(
            &default_registry(),
            &format!("videotestsrc num-buffers=1 ! deinterlace tff={order} ! fakesink"),
        )
        .unwrap_or_else(|e| panic!("deinterlace tff={order}: {e}"));
    }

    let mut element = Deinterlace::new();
    assert_eq!(
        element.get_property("fields"),
        Some(PropValue::Str("auto".into())),
        "the default stays one output frame per input frame"
    );
    assert_eq!(
        element.get_property("tff"),
        Some(PropValue::Str("auto".into()))
    );
    element
        .set_property("fields", PropValue::Str("all".into()))
        .unwrap();
    element
        .set_property("tff", PropValue::Str("bff".into()))
        .unwrap();
    assert_eq!(
        element.get_property("fields"),
        Some(PropValue::Str("all".into()))
    );
    assert_eq!(
        element.get_property("tff"),
        Some(PropValue::Str("bff".into()))
    );
    assert!(element
        .set_property("tff", PropValue::Str("middle".into()))
        .is_err());
}
