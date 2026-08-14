//! M935: interlace signaling on `Caps::RawVideo` and the universal `auto`
//! deinterlace.
//!
//! Covers the caps algebra (`Interlace` intersect / fixate / gst-string
//! round-trip), the `deinterlace mode=auto` element behavior (weave only on
//! caps-declared interleaved input, byte-exact passthrough otherwise, including
//! formats the kernels cannot process), the mid-stream flip a decoder's
//! `CapsChanged` triggers, and (under the `ffmpeg` feature) the decoder latch:
//! `FfmpegVideoDec` reports libavcodec's per-picture interlaced flag as
//! `Interlace::Interleaved` output caps, sticky for the rest of the stream.
#![cfg(feature = "std")]

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, Interlace, MemoryDomain, OutputSink, PropValue, PushOutcome,
    Rate, RawVideoFormat,
};
use g2g_plugins::capsfilter::parse_caps;
use g2g_plugins::deinterlace::{Deinterlace, DeinterlaceMode};

const W: usize = 64;
const H: usize = 32;
const I420_BYTES: usize = W * H * 3 / 2;

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
    fn caps_changes(&self) -> Vec<Caps> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(c) => Some(c.clone()),
                _ => None,
            })
            .collect()
    }
    fn frames(&self) -> Vec<Vec<u8>> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }
}

fn raw_caps(format: RawVideoFormat, interlace: Interlace) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(W as u32),
        height: Dim::Fixed(H as u32),
        framerate: Rate::Fixed(25 << 16),
        interlace,
    }
}

fn frame(bytes: &[u8], i: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns: i * 40_000_000,
            ..Default::default()
        },
        i,
    ))
}

/// A combed I420 frame: fields are horizontally shifted copies of a
/// vertical-edge pattern.
fn combed() -> Vec<u8> {
    let mut f = vec![128u8; I420_BYTES];
    for y in 0..H {
        for x in 0..W {
            let sx = if y % 2 == 0 { x } else { x + 4 };
            f[y * W + x] = if (sx / 8) % 2 == 0 { 20 } else { 220 };
        }
    }
    f
}

async fn run_mode(mode: DeinterlaceMode, caps: &Caps, frames: &[Vec<u8>]) -> Collect {
    let mut el = Deinterlace::new().with_mode(mode);
    el.configure_pipeline(caps).unwrap();
    let mut out = Collect::default();
    for (i, b) in frames.iter().enumerate() {
        el.process(frame(b, i as u64), &mut out).await.unwrap();
    }
    el.process(PipelinePacket::Eos, &mut out).await.unwrap();
    out
}

// ---- caps algebra ----

#[test]
fn interlace_any_is_the_intersect_identity_and_survives_fixate() {
    assert_eq!(
        Interlace::Any.intersect(&Interlace::Interleaved),
        Some(Interlace::Interleaved)
    );
    assert_eq!(
        Interlace::Progressive.intersect(&Interlace::Any),
        Some(Interlace::Progressive)
    );
    assert_eq!(
        Interlace::Progressive.intersect(&Interlace::Interleaved),
        None
    );

    // Field-wise through Caps: a sink advertising Any accepts an interleaved
    // decoder, and fixation keeps whatever the field carries (the wildcard
    // reads as "progressive unless declared", so it is already concrete).
    let narrowed = raw_caps(RawVideoFormat::I420, Interlace::Any)
        .intersect(&raw_caps(RawVideoFormat::I420, Interlace::Interleaved))
        .unwrap();
    assert_eq!(
        narrowed.fixate().unwrap(),
        raw_caps(RawVideoFormat::I420, Interlace::Interleaved)
    );
    assert_eq!(
        raw_caps(RawVideoFormat::I420, Interlace::Any)
            .fixate()
            .unwrap(),
        raw_caps(RawVideoFormat::I420, Interlace::Any)
    );
}

#[test]
fn gst_string_prints_only_interleaved_and_parses_back() {
    let interleaved = raw_caps(RawVideoFormat::I420, Interlace::Interleaved);
    let s = interleaved.to_gst_string();
    assert!(
        s.contains("interlace-mode=interleaved"),
        "missing field in {s}"
    );
    assert_eq!(parse_caps(&s), Some(interleaved));

    // Progressive and Any both print nothing; an absent field parses as Any
    // (a filter should not constrain what it does not name).
    let s = raw_caps(RawVideoFormat::I420, Interlace::Progressive).to_gst_string();
    assert!(!s.contains("interlace-mode"), "unexpected field in {s}");
    assert_eq!(
        parse_caps(&s),
        Some(raw_caps(RawVideoFormat::I420, Interlace::Any))
    );
    assert_eq!(
        parse_caps(
            "video/x-raw,format=I420,width=64,height=32,framerate=25/1,interlace-mode=progressive"
        ),
        Some(raw_caps(RawVideoFormat::I420, Interlace::Progressive))
    );
}

// ---- deinterlace mode=auto ----

#[tokio::test]
async fn auto_passes_progressive_through_byte_exact() {
    let frames = vec![combed(); 3];
    let out = run_mode(
        DeinterlaceMode::Auto,
        &raw_caps(RawVideoFormat::I420, Interlace::Progressive),
        &frames,
    )
    .await;
    assert_eq!(out.frames(), frames, "passthrough must not touch the bytes");
    assert_eq!(
        out.caps_changes(),
        vec![raw_caps(RawVideoFormat::I420, Interlace::Progressive)],
        "passthrough forwards the upstream caps verbatim, once"
    );
}

#[tokio::test]
async fn auto_weaves_interleaved_input_like_the_forced_mode() {
    let frames = vec![combed(); 4];
    let auto = run_mode(
        DeinterlaceMode::Auto,
        &raw_caps(RawVideoFormat::I420, Interlace::Interleaved),
        &frames,
    )
    .await;
    let forced = run_mode(
        DeinterlaceMode::Interlaced,
        &raw_caps(RawVideoFormat::I420, Interlace::Interleaved),
        &frames,
    )
    .await;
    assert_eq!(
        auto.frames(),
        forced.frames(),
        "auto == forced when declared"
    );
    assert_ne!(auto.frames(), frames, "the comb was actually processed");
    assert_eq!(
        auto.caps_changes(),
        vec![raw_caps(RawVideoFormat::I420, Interlace::Progressive)],
        "woven output declares itself progressive"
    );
}

#[tokio::test]
async fn auto_passes_unweavable_formats_through_even_when_interleaved() {
    // Packed YUYV is outside the kernel formats: an interleaved declaration must
    // not fail the branch, it stays a byte-exact passthrough.
    let bytes = vec![vec![0x5Au8; W * H * 2]; 2];
    let out = run_mode(
        DeinterlaceMode::Auto,
        &raw_caps(RawVideoFormat::Yuyv, Interlace::Interleaved),
        &bytes,
    )
    .await;
    assert_eq!(out.frames(), bytes);
    assert_eq!(
        out.caps_changes(),
        vec![raw_caps(RawVideoFormat::Yuyv, Interlace::Interleaved)]
    );
}

#[tokio::test]
async fn disabled_never_weaves() {
    let frames = vec![combed(); 2];
    let out = run_mode(
        DeinterlaceMode::Disabled,
        &raw_caps(RawVideoFormat::I420, Interlace::Interleaved),
        &frames,
    )
    .await;
    assert_eq!(out.frames(), frames);
}

#[tokio::test]
async fn a_mid_stream_caps_flip_switches_auto_on() {
    let mut el = Deinterlace::new().with_mode(DeinterlaceMode::Auto);
    el.configure_pipeline(&raw_caps(RawVideoFormat::I420, Interlace::Progressive))
        .unwrap();
    let mut out = Collect::default();
    let comb = combed();
    el.process(frame(&comb, 0), &mut out).await.unwrap();
    assert_eq!(out.frames(), vec![comb.clone()], "progressive: passthrough");

    // The decoder latches interlaced mid-stream: the runner re-calls
    // `configure_pipeline` with the new input caps, then pushes the pre-fixed
    // forward *output* caps as a CapsChanged packet (which must only be
    // forwarded, never adopted as input; the `VideoConvert` contract).
    el.configure_pipeline(&raw_caps(RawVideoFormat::I420, Interlace::Interleaved))
        .unwrap();
    el.process(
        PipelinePacket::CapsChanged(raw_caps(RawVideoFormat::I420, Interlace::Progressive)),
        &mut out,
    )
    .await
    .unwrap();
    for i in 1..4 {
        el.process(frame(&comb, i), &mut out).await.unwrap();
    }
    el.process(PipelinePacket::Eos, &mut out).await.unwrap();

    let frames = out.frames();
    assert_eq!(frames.len(), 4, "single rate across the switch");
    assert_ne!(frames[2], comb, "post-flip frames are woven");
    let caps = out.caps_changes();
    assert_eq!(
        caps.last(),
        Some(&raw_caps(RawVideoFormat::I420, Interlace::Progressive)),
        "woven output declares progressive after the flip"
    );
}

#[test]
fn mode_is_a_launch_property() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;
    for mode in ["auto", "interlaced", "disabled"] {
        parse_launch(
            &default_registry(),
            &format!("videotestsrc num-buffers=1 ! deinterlace mode={mode} ! fakesink"),
        )
        .unwrap_or_else(|e| panic!("deinterlace mode={mode}: {e}"));
    }
    let mut el = Deinterlace::new();
    assert_eq!(
        el.get_property("mode"),
        Some(PropValue::Str("interlaced".into())),
        "default stays the always-on pre-M935 behavior"
    );
    el.set_property("mode", PropValue::Str("auto".into()))
        .unwrap();
    assert_eq!(el.get_property("mode"), Some(PropValue::Str("auto".into())));
    assert!(el
        .set_property("mode", PropValue::Str("nope".into()))
        .is_err());
}

// ---- the decoder latch (libavcodec per-picture flag -> output caps) ----

#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod ffmpeg_latch {
    use super::*;
    use g2g_core::VideoCodec;
    use g2g_plugins::ffmpegdec::FfmpegVideoDec;
    use std::path::PathBuf;
    use std::process::Command;

    fn have_ffmpeg() -> bool {
        Command::new("ffmpeg").arg("-version").output().is_ok()
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("g2g-m935-{}-{name}", std::process::id()))
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

    /// Split an MPEG-2 elementary stream on picture start codes, headers glued
    /// to the picture that follows them, one access unit per chunk.
    fn split_access_units(es: &[u8]) -> Vec<Vec<u8>> {
        let mut starts = Vec::new();
        for i in 0..es.len().saturating_sub(3) {
            if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 && es[i + 3] == 0x00 {
                starts.push(i);
            }
        }
        let mut units = Vec::new();
        let mut prev = 0usize;
        for (n, &s) in starts.iter().enumerate() {
            // Headers (sequence / GOP) between pictures ride with the next one,
            // so `prev` only advances at the following picture's start.
            if n + 1 < starts.len() {
                units.push(es[prev..starts[n + 1]].to_vec());
                prev = starts[n + 1];
            } else {
                units.push(es[prev..].to_vec());
            }
            let _ = s;
        }
        units
    }

    async fn decode_caps(path: &std::path::Path) -> Vec<Caps> {
        let es = std::fs::read(path).unwrap();
        let units = split_access_units(&es);
        assert!(units.len() >= 5, "fixture has {} access units", units.len());
        let mut dec = FfmpegVideoDec::new();
        dec.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Fixed(160),
            height: Dim::Fixed(128),
            framerate: Rate::Fixed(25 << 16),
        })
        .unwrap();
        let mut out = Collect::default();
        for (i, au) in units.iter().enumerate() {
            dec.process(frame(au, i as u64), &mut out).await.unwrap();
        }
        dec.process(PipelinePacket::Eos, &mut out).await.unwrap();
        assert!(
            !out.frames().is_empty(),
            "decoder produced no frames from {path:?}"
        );
        out.caps_changes()
    }

    fn interlace_of(caps: &Caps) -> Interlace {
        match caps {
            Caps::RawVideo { interlace, .. } => *interlace,
            other => panic!("decoder emitted non-raw caps {other:?}"),
        }
    }

    #[tokio::test]
    async fn decoder_latches_interleaved_caps_from_the_picture_flags() {
        if !have_ffmpeg() {
            eprintln!("ffmpeg not present: skipping");
            return;
        }
        let interlaced = temp_path("interlaced.m2v");
        let progressive = temp_path("progressive.m2v");
        ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x128:rate=50:duration=1",
            "-vf",
            "tinterlace=mode=interleave_top",
            "-c:v",
            "mpeg2video",
            "-flags",
            "+ilme+ildct",
            "-top",
            "1",
            "-f",
            "mpeg2video",
            interlaced.to_str().unwrap(),
        ]);
        ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x128:rate=25:duration=1",
            "-c:v",
            "mpeg2video",
            "-f",
            "mpeg2video",
            progressive.to_str().unwrap(),
        ]);

        let caps = decode_caps(&interlaced).await;
        assert_eq!(
            interlace_of(caps.last().unwrap()),
            Interlace::Interleaved,
            "interlaced stream must end declared interleaved: {caps:?}"
        );
        // Sticky: once interleaved, no later change reverts to progressive.
        let first_il = caps
            .iter()
            .position(|c| interlace_of(c) == Interlace::Interleaved)
            .unwrap();
        assert!(
            caps[first_il..]
                .iter()
                .all(|c| interlace_of(c) == Interlace::Interleaved),
            "latch flapped: {caps:?}"
        );

        let caps = decode_caps(&progressive).await;
        assert!(
            caps.iter()
                .all(|c| interlace_of(c) != Interlace::Interleaved),
            "progressive stream must never declare interleaved: {caps:?}"
        );

        let _ = std::fs::remove_file(&interlaced);
        let _ = std::fs::remove_file(&progressive);
    }
}
