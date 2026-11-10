//! M2185: writers keep the caps pixel aspect ratio. A y4m stream declaring
//! `A16:15` is encoded and muxed by a launch line, then the written file is read
//! back by the g2g demuxer and, when it is installed, by ffprobe.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AspectRatio, AsyncElement, ByteStreamEncoding, Caps, G2gError, OutputSink, PipelineClock,
    PixelAspect, PushOutcome,
};
use g2g_plugins::mkvdemux::MkvDemux;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::registry::default_registry;

// 60 columns of 16:15 pixels show as 64, so WebM's whole-pixel display width is exact
const WIDTH: u32 = 60;
const HEIGHT: u32 = 48;
const PIXEL_ASPECT: (u32, u32) = (16, 15);
const FRAMES: usize = 5;
const LINK_CAPACITY: usize = 4;
const CHUNK_LEN: usize = 4096;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn expected_pixel_aspect() -> PixelAspect {
    let (numerator, denominator) = PIXEL_ASPECT;
    PixelAspect::Exact(AspectRatio::new(numerator, denominator).expect("nonzero"))
}

fn temp_path(tag: &str, extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "g2g-m2185-{tag}-{}.{extension}",
        std::process::id()
    ))
}

// a grey I420 y4m stream whose header declares PIXEL_ASPECT
fn write_source(path: &Path) {
    let (numerator, denominator) = PIXEL_ASPECT;
    let mut stream =
        format!("YUV4MPEG2 W{WIDTH} H{HEIGHT} F25:1 Ip A{numerator}:{denominator} C420jpeg\n")
            .into_bytes();
    let frame_bytes = (WIDTH * HEIGHT * 3 / 2) as usize;
    for _ in 0..FRAMES {
        stream.extend_from_slice(b"FRAME\n");
        stream.resize(stream.len() + frame_bytes, 128);
    }
    std::fs::write(path, stream).expect("write the y4m source");
}

// runs `y4m source ! {chain} ! filesink` and returns the written file
async fn write_through(tag: &str, chain: &str, extension: &str) -> Vec<u8> {
    let source = temp_path(tag, "y4m");
    let out = temp_path(tag, extension);
    write_source(&source);
    // y4mdec negotiates on a 1x1 placeholder, which vpxenc, x264enc and av1enc refuse to open at
    let line = format!(
        "filesrc location={} ! y4mdec ! video/x-raw,width={WIDTH},height={HEIGHT} ! {chain} ! filesink location={}",
        source.display(),
        out.display()
    );
    let graph = parse_launch(&default_registry(), &line)
        .unwrap_or_else(|e| panic!("`{line}` negotiates: {e:?}"));
    run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
    let written = std::fs::read(&out).expect("the pipeline wrote a file");
    check_ffprobe(&out, chain);
    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&out);
    written
}

// ffprobe as a reader g2g did not write, skipped where it is not installed
fn check_ffprobe(path: &Path, chain: &str) {
    let Ok(output) = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=sample_aspect_ratio",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
    else {
        eprintln!("ffprobe not present; skipping the reference read of `{chain}`");
        return;
    };
    let (numerator, denominator) = PIXEL_ASPECT;
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        format!("{numerator}:{denominator}"),
        "ffprobe reads the pixel aspect `{chain}` wrote"
    );
}

#[derive(Default)]
struct CapsSink {
    caps: Vec<Caps>,
}

impl OutputSink for CapsSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::CapsChanged(caps)) = packet_slot.take() {
            self.caps.push(caps);
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

// the pixel aspect of the last video caps the demuxer announces for `file`
async fn demuxed_pixel_aspect<E: AsyncElement>(
    demux: &mut E,
    encoding: ByteStreamEncoding,
    file: &[u8],
) -> PixelAspect {
    demux
        .configure_pipeline(&Caps::ByteStream { encoding })
        .expect("the demuxer takes the container");
    let mut sink = CapsSink::default();
    for (sequence, chunk) in file.chunks(CHUNK_LEN).enumerate() {
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
            FrameTiming::default(),
            sequence as u64,
        );
        demux
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .expect("demux");
    }
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("demux to the end");
    let pixel_aspect = sink.caps.iter().rev().find_map(|caps| match caps {
        Caps::CompressedVideo { pixel_aspect, .. } => Some(*pixel_aspect),
        _ => None,
    });
    pixel_aspect.expect("the demuxer announced video caps")
}

async fn mp4_pixel_aspect(file: &[u8]) -> PixelAspect {
    demuxed_pixel_aspect(&mut Mp4Demux::new(), ByteStreamEncoding::Mp4, file).await
}

async fn mkv_pixel_aspect(stream: &str, file: &[u8]) -> PixelAspect {
    let mut demux = MkvDemux::new();
    demux
        .set_property("stream", g2g_core::PropValue::Str(stream.into()))
        .expect("a known stream name");
    demuxed_pixel_aspect(&mut demux, ByteStreamEncoding::Matroska, file).await
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn x264enc_writes_the_vui_sample_aspect() {
    let stream = write_through("x264", "x264enc", "h264").await;
    let mut parse = g2g_plugins::h264parse::H264Parse::reframing();
    // h264parse prefers an upstream ratio, so open caps leave the SPS as the only source
    let h264 = Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::H264,
        width: g2g_core::Dim::Any,
        height: g2g_core::Dim::Any,
        framerate: g2g_core::Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    };
    parse
        .configure_pipeline(&h264)
        .expect("h264parse takes H.264");
    let mut sink = CapsSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(stream.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    parse
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .expect("parse");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("parse");
    let parsed = sink.caps.iter().rev().find_map(|caps| match caps {
        Caps::CompressedVideo { pixel_aspect, .. } => Some(*pixel_aspect),
        _ => None,
    });
    assert_eq!(parsed, Some(expected_pixel_aspect()));
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn matroskamux_writes_a_display_aspect() {
    let file = write_through("mkv-h264", "x264enc ! h264parse ! matroskamux", "mkv").await;
    assert_eq!(
        mkv_pixel_aspect("h264", &file).await,
        expected_pixel_aspect()
    );
}

#[cfg(feature = "vpx")]
#[tokio::test]
async fn webm_writes_a_display_size_in_pixels() {
    let file = write_through("webm-vp9", "vpxenc ! matroskamux", "webm").await;
    assert_eq!(
        mkv_pixel_aspect("vp9", &file).await,
        expected_pixel_aspect()
    );
}

#[cfg(feature = "vpx")]
#[tokio::test]
async fn mp4mux_writes_pasp_for_vp9() {
    let file = write_through("mp4-vp9", "vpxenc ! mp4mux", "mp4").await;
    assert_eq!(mp4_pixel_aspect(&file).await, expected_pixel_aspect());
}

#[cfg(feature = "openh264")]
#[tokio::test]
async fn openh264enc_carries_the_ratio_to_mp4mux() {
    let file = write_through("mp4-openh264", "openh264enc ! h264parse ! mp4mux", "mp4").await;
    assert_eq!(mp4_pixel_aspect(&file).await, expected_pixel_aspect());
}

#[cfg(feature = "av1-encode")]
#[tokio::test]
async fn av1enc_carries_the_ratio_to_mp4mux() {
    let file = write_through("mp4-av1", "av1enc ! mp4mux", "mp4").await;
    assert_eq!(mp4_pixel_aspect(&file).await, expected_pixel_aspect());
}
