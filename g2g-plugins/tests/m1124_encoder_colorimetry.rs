//! M1124: `FfmpegH264Enc` writes the caps colorimetry into the SPS VUI colour
//! description and carries it on its output caps.
//!
//! The oracle is ffprobe reading the encoded elementary stream, not a g2g
//! loopback: what ffmpeg reports out of the VUI is what any other decoder sees.
//! No colour name is spelled here, every expected value comes from probing a
//! reference stream ffmpeg itself encoded with the same colour request, and the
//! two references (tagged and untagged) have to disagree, so a probe that
//! reported nothing could not pass the test.
#![cfg(feature = "ffmpeg")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, G2gError, Interlace, OutputSink, PushOutcome, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::ffmpegenc::{Backend, FfmpegH264Enc};
use g2g_plugins::h264parse::H264Parse;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMES: u64 = 10;
const FPS: u32 = 30;
const FRAME_PERIOD_NS: u64 = 1_000_000_000 / FPS as u64;

/// Flat chroma in every source frame; only the luma moves, so successive frames
/// differ and the encoder emits real inter frames.
const U_VAL: u8 = 110;
const V_VAL: u8 = 150;

fn i420_frame(seq: u64) -> Vec<u8> {
    let (w, h) = (WIDTH as usize, HEIGHT as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut v = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            v.push(((x + y + seq as usize * 7) & 0xff) as u8);
        }
    }
    v.extend(std::iter::repeat_n(U_VAL, cw * ch));
    v.extend(std::iter::repeat_n(V_VAL, cw * ch));
    v
}

fn raw_caps(colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FPS << 16),
        interlace: Interlace::Any,
        colorimetry,
    }
}

#[derive(Default)]
struct CaptureSink {
    caps: Vec<Caps>,
    access_units: Vec<Vec<u8>>,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::CapsChanged(caps) => self.caps.push(caps),
            PipelinePacket::DataFrame(frame) => {
                if let Some(slice) = frame.domain.as_system_slice() {
                    self.access_units.push(slice.to_vec());
                }
            }
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Encode the test pattern under `colorimetry`. `None` when `backend` is missing
/// on this host, which skips the test rather than failing it.
async fn encode_with(backend: Backend, colorimetry: Colorimetry) -> Option<CaptureSink> {
    let mut encoder = FfmpegH264Enc::new().with_backend(backend);
    encoder.configure_pipeline(&raw_caps(colorimetry)).ok()?;
    let mut sink = CaptureSink::default();
    for seq in 0..FRAMES {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(i420_frame(seq).into_boxed_slice())),
            FrameTiming {
                pts_ns: seq * FRAME_PERIOD_NS,
                ..FrameTiming::default()
            },
            seq,
        );
        encoder
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("encode frame");
    }
    encoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("flush encoder");
    Some(sink)
}

/// The libx264 encode every test but the NVENC one runs.
async fn encode(colorimetry: Colorimetry) -> Option<CaptureSink> {
    encode_with(Backend::Software, colorimetry).await
}

/// The four colour fields ffprobe reports for a stream, as it spells them.
#[derive(Debug, PartialEq, Eq)]
struct ProbedColour {
    range: String,
    space: String,
    transfer: String,
    primaries: String,
}

/// Tests run concurrently and reuse tags, so every file gets its own number.
static FILE_NO: AtomicU64 = AtomicU64::new(0);

fn temp_path(tag: &str) -> PathBuf {
    let file_no = FILE_NO.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "g2g-m1124-{tag}-{}-{file_no}.h264",
        std::process::id()
    ))
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
        && Command::new("ffprobe").arg("-version").output().is_ok()
}

/// What ffprobe reads out of an Annex-B stream's VUI colour description.
fn probe_colour(annex_b: &[u8], tag: &str) -> ProbedColour {
    let path = temp_path(tag);
    std::fs::write(&path, annex_b).expect("write the stream to probe");
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=color_range,color_space,color_transfer,color_primaries",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(out.status.success(), "ffprobe read {tag}: {text}");
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{name}=")))
            .unwrap_or_else(|| panic!("ffprobe reported {name} for {tag}, got:\n{text}"))
            .to_string()
    };
    let probed = ProbedColour {
        range: field("color_range"),
        space: field("color_space"),
        transfer: field("color_transfer"),
        primaries: field("color_primaries"),
    };
    let _ = std::fs::remove_file(&path);
    probed
}

/// Encode the same geometry with the ffmpeg CLI, asking libx264 for `colour`
/// through x264's own parameters, and probe the result. This is the expected
/// value the element's stream is compared against.
fn ffmpeg_reference(colour_args: &[&str], tag: &str) -> ProbedColour {
    let path = temp_path(&format!("ref-{tag}"));
    let status = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size={WIDTH}x{HEIGHT}:rate={FPS}"),
            "-frames:v",
            &FRAMES.to_string(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ])
        .args(colour_args)
        .arg(&path)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg authored the {tag} reference");
    let bytes = std::fs::read(&path).expect("read the reference stream");
    let _ = std::fs::remove_file(&path);
    probe_colour(&bytes, &format!("ref-{tag}"))
}

/// A bt709-tagged input encodes to a stream whose VUI says bt709, matching what
/// ffmpeg's own encoder writes when asked for the same description.
#[tokio::test]
async fn bt709_caps_reach_the_vui_colour_description() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let Some(sink) = encode(Colorimetry::BT709).await else {
        eprintln!("skipping: libx264 not available");
        return;
    };
    let tagged = ffmpeg_reference(
        &[
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-color_range",
            "tv",
        ],
        "bt709",
    );
    let untagged = ffmpeg_reference(&[], "untagged");
    assert_ne!(
        tagged, untagged,
        "the reference encodes have to differ, or the probe sees nothing"
    );
    assert_eq!(
        probe_colour(&sink.access_units.concat(), "bt709"),
        tagged,
        "bt709 caps must encode to the colour description ffmpeg writes for bt709"
    );
}

/// NVENC writes the description from the same context fields, so the hardware
/// backend tags its stream too. Skips on a host without an NVENC encoder.
#[tokio::test]
async fn nvenc_writes_the_vui_colour_description() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let Some(sink) = encode_with(Backend::Nvenc, Colorimetry::BT709).await else {
        eprintln!("skipping: h264_nvenc not available");
        return;
    };
    let tagged = ffmpeg_reference(
        &[
            "-x264-params",
            "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-color_range",
            "tv",
        ],
        "bt709",
    );
    assert_eq!(
        probe_colour(&sink.access_units.concat(), "nvenc-bt709"),
        tagged,
        "NVENC must write the same colour description libx264 does"
    );
}

/// An untagged input still encodes to an untagged stream: the encoder writes
/// what the caps say and invents nothing.
#[tokio::test]
async fn untagged_caps_write_no_colour_description() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let Some(sink) = encode(Colorimetry::UNKNOWN).await else {
        eprintln!("skipping: libx264 not available");
        return;
    };
    assert_eq!(
        probe_colour(&sink.access_units.concat(), "untagged"),
        ffmpeg_reference(&[], "untagged"),
        "untagged caps must leave the stream as untagged as ffmpeg's own"
    );
}

/// The colorimetry survives the whole hop: it rides the encoder's output caps,
/// and `h264parse` recovers it from the bitstream the encoder produced.
#[tokio::test]
async fn encoded_colorimetry_comes_back_through_h264parse() {
    let Some(sink) = encode(Colorimetry::BT709).await else {
        eprintln!("skipping: libx264 not available");
        return;
    };
    assert_eq!(
        sink.caps,
        vec![Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(FPS << 16),
            colorimetry: Colorimetry::BT709,
        }],
        "the encoder announces the input colorimetry on its output caps"
    );

    let mut parse = H264Parse::new();
    parse
        .configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
        })
        .expect("configure h264parse");
    let mut parsed = CaptureSink::default();
    for (seq, access_unit) in sink.access_units.iter().enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(
                access_unit.clone().into_boxed_slice(),
            )),
            FrameTiming {
                pts_ns: seq as u64 * FRAME_PERIOD_NS,
                ..FrameTiming::default()
            },
            seq as u64,
        );
        parse
            .process(PipelinePacket::DataFrame(frame), &mut parsed)
            .await
            .expect("parse access unit");
    }
    let refined = parsed
        .caps
        .iter()
        .find_map(|caps| match caps {
            Caps::CompressedVideo { colorimetry, .. } => Some(*colorimetry),
            _ => None,
        })
        .expect("h264parse refined the caps");
    assert_eq!(
        refined,
        Colorimetry::BT709,
        "the VUI the encoder wrote parses back as the input colorimetry"
    );
}
