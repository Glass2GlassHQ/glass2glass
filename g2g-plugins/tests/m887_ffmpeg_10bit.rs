//! M887: 10-bit output from `FfmpegVideoDec`.
//!
//! A High 10 H.264 stream decodes to the planar 10-bit format (`I420p10`) under
//! `OutputFormat::Auto` and to the semi-planar `P010` when that layout is
//! requested, both bit-exact against ffmpeg's own raw decode of the same clip
//! (the oracle is dumped at test time; the test skips if the ffmpeg CLI is
//! absent). A fixed 8-bit request from the same stream stays a loud error rather
//! than truncating the samples.
//!
//! Fixture: `tests/fixtures/h264_high10_320x240.h264`, 12 frames of 320x240
//! High 10 (`profile=High 10`, `pix_fmt=yuv420p10le`), authored with
//! `ffmpeg -f lavfi -i testsrc2=size=320x240:rate=25 -frames:v 12 -c:v libx264
//! -pix_fmt yuv420p10le -profile:v high10 -bf 2 -g 12 -f h264`.

#![cfg(all(target_os = "linux", feature = "ffmpeg"))]

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::ffmpegdec::{FfmpegVideoDec, OutputFormat};
use g2g_plugins::h264parse::H264Parse;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_high10_320x240.h264");
const W: usize = 320;
const H: usize = 240;
const FRAMES: usize = 12;

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
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
    fn frame_bytes(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn data_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

/// Split the fixture into access units with the real `H264Parse`, the framing the
/// decoder sees in a pipeline.
async fn access_units() -> Vec<Vec<u8>> {
    let mut parse = H264Parse::reframing();
    parse
        .configure_pipeline(&h264_caps())
        .expect("h264parse accepts the stream");
    let mut sink = Collect::default();
    parse
        .process(data_frame(FIXTURE.to_vec()), &mut sink)
        .await
        .expect("parse the fixture");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at EOS");
    let aus = sink.frame_bytes();
    assert!(
        aus.len() >= FRAMES,
        "expected one AU per picture, got {}",
        aus.len()
    );
    aus
}

/// Decode the fixture with the given output layout. Returns the emitted caps and
/// frame payloads, or the first decode error.
async fn decode(output: OutputFormat) -> Result<(Vec<Caps>, Vec<Vec<u8>>), G2gError> {
    let aus = access_units().await;
    let mut dec = FfmpegVideoDec::new().with_output_format(output);
    dec.configure_pipeline(&h264_caps())
        .expect("libavcodec opens the H.264 decoder");
    let mut sink = Collect::default();
    for au in aus {
        dec.process(data_frame(au), &mut sink).await?;
    }
    dec.process(PipelinePacket::Eos, &mut sink).await?;
    Ok((sink.caps_changes(), sink.frame_bytes()))
}

/// ffmpeg's own decode of the fixture into `pix_fmt`, as one buffer of
/// display-order frames. `None` when the ffmpeg CLI is not on PATH.
fn ffmpeg_reference(pix_fmt: &str) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join("g2g_m887");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let clip = dir.join("high10.h264");
    let out = dir.join(format!("ref_{pix_fmt}.raw"));
    std::fs::write(&clip, FIXTURE).expect("write clip");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&clip)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt])
        .arg(&out)
        .status();
    match status {
        Ok(s) if s.success() => Some(std::fs::read(&out).expect("read reference")),
        Ok(s) => panic!("ffmpeg reference decode failed: {s}"),
        Err(_) => {
            eprintln!("skipping m887 oracle: ffmpeg not on PATH");
            None
        }
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Compare each decoded frame against the reference buffer's frame of
/// `frame_bytes` bytes, byte for byte.
fn assert_matches_reference(frames: &[Vec<u8>], reference: &[u8], frame_bytes: usize, what: &str) {
    assert_eq!(
        reference.len(),
        frame_bytes * FRAMES,
        "{what}: reference frame size / count mismatch"
    );
    assert_eq!(frames.len(), FRAMES, "{what}: decoded frame count");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.len(), frame_bytes, "{what}: frame {i} plane bytes");
        let want = &reference[i * frame_bytes..(i + 1) * frame_bytes];
        assert_eq!(
            fnv64(f),
            fnv64(want),
            "{what}: frame {i} checksum differs from ffmpeg"
        );
        assert!(f.as_slice() == want, "{what}: frame {i} bytes differ");
    }
    eprintln!(
        "m887 {what}: {FRAMES} frames x {frame_bytes} B bit-exact vs ffmpeg (checksum {:#018x})",
        fnv64(reference)
    );
}

#[tokio::test]
async fn high10_decodes_to_planar_10bit_matching_ffmpeg() {
    let (caps, frames) = decode(OutputFormat::Auto).await.expect("decode High 10");
    let cw = W / 2;
    let ch = H / 2;
    let frame_bytes = (W * H + 2 * cw * ch) * 2;
    assert!(
        caps.iter().any(|c| matches!(
            c,
            Caps::RawVideo {
                format: RawVideoFormat::I420p10,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if *w == W as u32 && *h == H as u32
        )),
        "Auto must emit I420_10LE caps for a High 10 stream, got {caps:?}"
    );
    let Some(reference) = ffmpeg_reference("yuv420p10le") else {
        return;
    };
    assert_matches_reference(&frames, &reference, frame_bytes, "I420p10");
}

#[tokio::test]
async fn high10_decodes_to_p010_matching_ffmpeg() {
    let (caps, frames) = decode(OutputFormat::P010).await.expect("decode High 10");
    // P010 = Y plus interleaved UV, all 16-bit samples: w*h*3 bytes at even dims.
    let frame_bytes = W * H * 3;
    assert!(
        caps.iter().any(|c| matches!(
            c,
            Caps::RawVideo {
                format: RawVideoFormat::P010,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if *w == W as u32 && *h == H as u32
        )),
        "a P010 request must emit P010 caps, got {caps:?}"
    );
    let Some(reference) = ffmpeg_reference("p010le") else {
        return;
    };
    assert_matches_reference(&frames, &reference, frame_bytes, "P010");
}

/// No silent truncation: asking an 8-bit layout of a 10-bit stream fails loud
/// (depth conversion belongs in `videoconvert`, which takes the planar 10-bit
/// family as input).
#[tokio::test]
async fn eight_bit_request_from_high10_fails_loud() {
    let err = decode(OutputFormat::I420)
        .await
        .expect_err("an I420 request cannot serve a 10-bit stream");
    assert_eq!(err, G2gError::CapsMismatch);
}

/// Negotiation reaches the new formats: a downstream pinning P010 auto-plugs the
/// ffmpeg decoder and the built element accepts the runner's pre-fixed P010 output
/// caps (an `Auto`- or NV12-built decoder rejects them).
#[tokio::test]
async fn autoplug_builds_a_p010_decoder_for_a_p010_sink() {
    use g2g_core::element::DynAsyncElement;
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();
    let h264 = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(W as u32),
        height: Dim::Fixed(H as u32),
        framerate: Rate::Fixed(25 << 16),
    };
    let strict_p010 = |c: &Caps| {
        matches!(
            c,
            Caps::RawVideo {
                format: RawVideoFormat::P010,
                ..
            }
        )
    };
    let mut chain = reg
        .autoplug(&h264, &strict_p010, 4)
        .expect("a decode chain reaches P010");
    let dec = chain.last_mut().expect("non-empty chain").as_mut();
    DynAsyncElement::configure_pipeline(dec, &h264).expect("libavcodec opens the decoder");

    let p010 = Caps::RawVideo {
        format: RawVideoFormat::P010,
        width: Dim::Fixed(W as u32),
        height: Dim::Fixed(H as u32),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
    };
    let mut sink = Collect::default();
    DynAsyncElement::process(dec, PipelinePacket::CapsChanged(p010), &mut sink)
        .await
        .expect("the auto-plugged decoder must accept P010 output caps");
}
