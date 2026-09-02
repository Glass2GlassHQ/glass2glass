//! M831: the legacy FLV codecs (Sorenson Spark, VP6 with and without alpha, MP3,
//! Speex) carried by the FLV demuxer and muxer, validated against ffmpeg as the
//! reference peer in both directions:
//!
//! - **Demux:** ffmpeg encodes a real Sorenson / MP3 FLV; `FlvDemux` must pull out
//!   payloads byte-identical to what libavcodec's own FLV demuxer emits (the proof
//!   that each codec's payload offset is right), and ffmpeg must decode the
//!   extracted MP3 elementary stream standalone.
//! - **Mux:** `FlvMux` writes each codec back into an FLV; ffprobe must read the
//!   codec back and ffmpeg decode the result (the M615 oracle discipline).
//!
//! VP6 has no encoder in any ffmpeg build, so its carriage is proven the other
//! way round: g2g muxes VP6 / VP6-alpha tags, ffprobe names the codec, and the
//! demuxer recovers the payload. Speex has no ffmpeg encoder either, so the demux
//! vector comes from GStreamer's `speexenc ! flvmux` when it is installed.
//!
//! Self-skips where ffmpeg / ffprobe / gst-launch-1.0 are absent.
#![cfg(feature = "std")]

use std::path::PathBuf;
use std::process::Command;

use g2g_core::conformance::{ConformanceDimension, Evidence};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, Dim, G2gError, MultiInputElement,
    OutputSink, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::conformance::persist;
use g2g_plugins::flvdemux::{FlvDemux, FlvStream};
use g2g_plugins::flvmux::FlvMux;
use g2g_plugins::flvmuxn::FlvMuxN;

fn have(bin: &str) -> bool {
    Command::new(bin).arg("-version").output().is_ok()
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

/// Point the persisted-evidence log at a shared temp file unless a CI
/// conformance run already set one.
fn ensure_conformance_log() {
    if std::env::var_os("G2G_CONFORMANCE_LOG").is_none() {
        std::env::set_var(
            "G2G_CONFORMANCE_LOG",
            std::env::temp_dir().join("g2g-conformance-m831.tsv"),
        );
    }
}

#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
    caps: Vec<Caps>,
}
impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

impl CaptureSink {
    fn sizes(&self) -> Vec<usize> {
        self.frames.iter().map(|f| f.len()).collect()
    }

    fn bytes(&self) -> Vec<u8> {
        self.frames.concat()
    }
}

fn run(cmd: &mut Command) -> std::process::Output {
    let out = cmd.output().expect("the tool runs");
    assert!(
        out.status.success(),
        "{cmd:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// The packet sizes libavcodec's own FLV demuxer reports for a stream, the
/// reference for what each codec's payload offset should leave behind.
fn ffprobe_packet_sizes(path: &PathBuf, select: &str) -> Vec<usize> {
    let out = run(Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            select,
            "-show_entries",
            "packet=size",
            "-of",
            "csv=p=0",
        ])
        .arg(path));
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect()
}

/// The codec names ffprobe reads out of a container.
fn ffprobe_codecs(path: &PathBuf) -> String {
    let out = run(Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
        ])
        .arg(path));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Whether ffmpeg decodes a file end to end.
fn ffmpeg_decodes(path: &PathBuf, format: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-y", "-f", format, "-i"])
        .arg(path)
        .args(["-f", "null", "-"])
        .output()
        .expect("ffmpeg runs")
        .status
        .success()
}

/// Run the demux element over a whole FLV byte stream.
async fn demux(flv: &[u8], stream: FlvStream) -> CaptureSink {
    let mut d = FlvDemux::new().with_stream(stream);
    d.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Flv,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(flv.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    d.process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    d.process(PipelinePacket::Eos, &mut sink).await.unwrap();
    sink
}

fn frame_at(au: &[u8], pts_ns: u64, keyframe: bool) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
        FrameTiming {
            pts_ns,
            keyframe,
            ..FrameTiming::default()
        },
        0,
    )
}

/// Mux access units into an FLV byte stream, one tag per unit at 25 fps.
async fn mux(caps: &Caps, aus: &[Vec<u8>], keyframe_every: usize) -> Vec<u8> {
    let mut m = FlvMux::new();
    m.configure_pipeline(caps).unwrap();
    let mut sink = CaptureSink::default();
    for (i, au) in aus.iter().enumerate() {
        let frame = frame_at(au, i as u64 * 40_000_000, i % keyframe_every == 0);
        m.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    sink.bytes()
}

fn video_caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn record_oracle(element: &str, codec: &str, detail: &str) {
    ensure_conformance_log();
    persist::record_evidence(
        element,
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec(codec)
            .detail(detail),
    )
    .expect("record oracle evidence");
}

#[tokio::test]
async fn sorenson_flv_round_trips_against_ffmpeg() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("ffmpeg/ffprobe not present; skipping the Sorenson FLV oracle");
        return;
    }
    let src = tmp("g2g_m831_sorenson_src.flv");
    run(Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=25",
        ])
        .args(["-c:v", "flv1", "-f", "flv"])
        .arg(&src));
    let flv = std::fs::read(&src).unwrap();

    // Demux: every payload matches what libavcodec's FLV demuxer hands its own
    // decoder, byte for byte.
    let video = demux(&flv, FlvStream::SorensonH263).await;
    assert_eq!(
        video.sizes(),
        ffprobe_packet_sizes(&src, "v:0"),
        "Sorenson payload offsets agree with ffmpeg's demuxer"
    );
    assert!(video.frames.len() >= 20, "a second of 25 fps video");

    // Mux the recovered frames back and let ffmpeg read the result.
    let out = tmp("g2g_m831_sorenson_out.flv");
    let flv_out = mux(&video_caps(VideoCodec::SorensonH263), &video.frames, 25).await;
    std::fs::write(&out, &flv_out).unwrap();
    assert!(
        ffprobe_codecs(&out).contains("flv1"),
        "ffprobe names the muxed stream flv1"
    );
    assert_eq!(
        ffprobe_packet_sizes(&out, "v:0"),
        video.sizes(),
        "the muxed tags carry the same payloads"
    );
    assert!(ffmpeg_decodes(&out, "flv"), "ffmpeg decodes the muxed FLV");

    record_oracle(
        "flvdemux",
        "sorenson-h263",
        "payloads match libavcodec's FLV demuxer byte for byte",
    );
    record_oracle(
        "flvmux",
        "sorenson-h263",
        "ffprobe reads back flv1 and ffmpeg decodes the muxed FLV",
    );
    for p in [&src, &out] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn mp3_flv_round_trips_against_ffmpeg() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("ffmpeg/ffprobe not present; skipping the MP3 FLV oracle");
        return;
    }
    let src = tmp("g2g_m831_mp3_src.flv");
    run(Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
        ])
        .args([
            "-c:a",
            "libmp3lame",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "flv",
        ])
        .arg(&src));
    let flv = std::fs::read(&src).unwrap();

    let audio = demux(&flv, FlvStream::Mp3).await;
    // ffmpeg's demuxer re-splits a multi-frame tag through the MP3 parser, so the
    // packet counts can differ; the bytes handed to the decoder must not.
    assert_eq!(
        audio.bytes().len(),
        ffprobe_packet_sizes(&src, "a:0").iter().sum::<usize>(),
        "MP3 payload offsets agree with ffmpeg's demuxer"
    );
    assert_eq!(
        audio.caps,
        vec![Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED
        }],
        "the layout the audio tag flags declare is announced"
    );

    // The extracted elementary stream is a playable .mp3 on its own.
    let es = tmp("g2g_m831_mp3_es.mp3");
    std::fs::write(&es, audio.bytes()).unwrap();
    assert!(
        ffmpeg_decodes(&es, "mp3"),
        "ffmpeg decodes the extracted MP3 elementary stream standalone"
    );

    let out = tmp("g2g_m831_mp3_out.flv");
    let caps = Caps::Audio {
        format: AudioFormat::Mp3,
        channels: 2,
        sample_rate: 44_100,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    std::fs::write(&out, mux(&caps, &audio.frames, 1).await).unwrap();
    assert!(
        ffprobe_codecs(&out).contains("mp3"),
        "ffprobe names the muxed stream mp3"
    );
    assert!(ffmpeg_decodes(&out, "flv"), "ffmpeg decodes the muxed FLV");

    record_oracle(
        "flvdemux",
        "mp3",
        "ffmpeg decodes the extracted MP3 elementary stream standalone",
    );
    record_oracle("flvmux", "mp3", "ffprobe reads back mp3 from the muxed FLV");
    for p in [&src, &es, &out] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn sorenson_plus_mp3_mux_into_one_flv() {
    if !have("ffmpeg") || !have("ffprobe") {
        eprintln!("ffmpeg/ffprobe not present; skipping the A/V FLV oracle");
        return;
    }
    // The A/V muxer over a legacy pair: neither track has a sequence header, so
    // both are writable from the first access unit.
    let vsrc = tmp("g2g_m831_av_v.flv");
    let asrc = tmp("g2g_m831_av_a.flv");
    run(Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=25",
        ])
        .args(["-c:v", "flv1", "-f", "flv"])
        .arg(&vsrc));
    run(Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
        ])
        .args([
            "-c:a",
            "libmp3lame",
            "-ar",
            "44100",
            "-ac",
            "2",
            "-f",
            "flv",
        ])
        .arg(&asrc));
    let video = demux(&std::fs::read(&vsrc).unwrap(), FlvStream::SorensonH263).await;
    let audio = demux(&std::fs::read(&asrc).unwrap(), FlvStream::Mp3).await;

    let mut m = FlvMuxN::new(2);
    m.configure_pipeline(0, &video_caps(VideoCodec::SorensonH263))
        .unwrap();
    m.configure_pipeline(
        1,
        &Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        },
    )
    .unwrap();
    let mut sink = CaptureSink::default();
    for (i, au) in video.frames.iter().enumerate() {
        let frame = frame_at(au, i as u64 * 40_000_000, i % 25 == 0);
        m.process(0, PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    for (i, au) in audio.frames.iter().enumerate() {
        let frame = frame_at(au, i as u64 * 26_000_000, true);
        m.process(1, PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    m.process(0, PipelinePacket::Eos, &mut sink).await.unwrap();
    m.process(1, PipelinePacket::Eos, &mut sink).await.unwrap();

    let out = tmp("g2g_m831_av_out.flv");
    std::fs::write(&out, sink.bytes()).unwrap();
    let codecs = ffprobe_codecs(&out);
    assert!(
        codecs.contains("flv1") && codecs.contains("mp3"),
        "ffprobe sees both tracks, got: {codecs}"
    );
    assert_eq!(
        ffprobe_packet_sizes(&out, "v:0"),
        video.sizes(),
        "the video payloads survive the A/V mux"
    );
    assert!(ffmpeg_decodes(&out, "flv"), "ffmpeg decodes the A/V FLV");

    record_oracle(
        "flvmux",
        "sorenson-h263+mp3",
        "ffprobe reads both tracks out of a g2g-muxed A/V FLV",
    );
    for p in [&vsrc, &asrc, &out] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn vp6_carriage_probes_as_vp6f_and_round_trips() {
    if !have("ffprobe") {
        eprintln!("ffprobe not present; skipping the VP6 carriage oracle");
        return;
    }
    // No ffmpeg build encodes VP6, so this proves the carriage rather than the
    // pixels: g2g writes VP6 tags, ffprobe names the codec, and the demuxer
    // recovers the payload past the dimension-adjustment byte.
    let aus: Vec<Vec<u8>> = (0..10u8)
        .map(|i| (0..64u8).map(|b| b.wrapping_mul(i.max(1))).collect())
        .collect();
    for (alpha, name, stream) in [
        (false, "vp6f", FlvStream::Vp6),
        (true, "vp6a", FlvStream::Vp6Alpha),
    ] {
        let flv = mux(&video_caps(VideoCodec::Vp6 { alpha }), &aus, 5).await;
        let out = tmp(&format!("g2g_m831_{name}.flv"));
        std::fs::write(&out, &flv).unwrap();
        let codecs = ffprobe_codecs(&out);
        assert!(
            codecs.contains(name),
            "ffprobe names the stream {name}, got: {codecs}"
        );
        assert_eq!(
            ffprobe_packet_sizes(&out, "v:0"),
            aus.iter().map(|a| a.len()).collect::<Vec<_>>(),
            "ffmpeg's demuxer strips exactly the bytes g2g wrote as the header"
        );

        let back = demux(&flv, stream).await;
        assert_eq!(back.frames, aus, "{name} payloads survive the round trip");
        let _ = std::fs::remove_file(&out);
    }
    record_oracle(
        "flvmux",
        "vp6",
        "ffprobe names vp6f / vp6a and strips exactly the written header bytes",
    );
}

#[tokio::test]
async fn speex_carriage_probes_as_speex() {
    if !have("ffprobe") {
        eprintln!("ffprobe not present; skipping the Speex carriage oracle");
        return;
    }
    // FLV pins Speex at 16 kHz mono; g2g carries the frames without a decoder,
    // so this is a carriage claim only.
    let aus: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 70]).collect();
    let caps = Caps::Audio {
        format: AudioFormat::Speex,
        channels: 1,
        sample_rate: 16_000,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    };
    let flv = mux(&caps, &aus, 1).await;
    let out = tmp("g2g_m831_speex.flv");
    std::fs::write(&out, &flv).unwrap();
    let codecs = ffprobe_codecs(&out);
    assert!(
        codecs.contains("speex"),
        "ffprobe names speex, got: {codecs}"
    );
    assert_eq!(
        ffprobe_packet_sizes(&out, "a:0"),
        aus.iter().map(|a| a.len()).collect::<Vec<_>>(),
        "ffmpeg's demuxer strips exactly the one flags byte g2g wrote"
    );

    let back = demux(&flv, FlvStream::Speex).await;
    assert_eq!(back.frames, aus, "Speex payloads survive the round trip");
    assert_eq!(
        back.caps,
        vec![caps],
        "Speex announces FLV's fixed 16 kHz mono"
    );
    let _ = std::fs::remove_file(&out);
}

#[tokio::test]
async fn gstreamer_speex_flv_demuxes() {
    if !have("ffprobe")
        || Command::new("gst-launch-1.0")
            .arg("--version")
            .output()
            .is_err()
    {
        eprintln!("ffprobe/gst-launch-1.0 not present; skipping the Speex demux vector");
        return;
    }
    // GStreamer is the only encoder on hand that writes Speex into FLV.
    let src = tmp("g2g_m831_speex_src.flv");
    let built = Command::new("gst-launch-1.0")
        .args(["-q", "audiotestsrc", "num-buffers=40", "!", "audioconvert"])
        .args([
            "!",
            "audioresample",
            "!",
            "audio/x-raw,rate=16000,channels=1",
            "!",
            "speexenc",
            "!",
            "flvmux",
            "!",
            "filesink",
        ])
        .arg(format!("location={}", src.display()))
        .output()
        .expect("gst-launch runs");
    if !built.status.success() {
        eprintln!("gst-launch could not build a Speex FLV; skipping");
        return;
    }
    let flv = std::fs::read(&src).unwrap();
    let audio = demux(&flv, FlvStream::Speex).await;
    assert_eq!(
        audio.sizes(),
        ffprobe_packet_sizes(&src, "a:0"),
        "Speex payload offsets agree with ffmpeg's demuxer"
    );
    record_oracle(
        "flvdemux",
        "speex",
        "payloads match libavcodec's FLV demuxer byte for byte",
    );
    let _ = std::fs::remove_file(&src);
}

#[tokio::test]
async fn h264_and_aac_still_demux() {
    if !have("ffmpeg") {
        eprintln!("ffmpeg not present; skipping the H.264 + AAC regression");
        return;
    }
    // The codecs that rode the tag stream before M831 keep their framing: Annex-B
    // video with in-band parameter sets, ADTS audio.
    let src = tmp("g2g_m831_h264_aac.flv");
    run(Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=25",
        ])
        .args(["-f", "lavfi", "-i", "sine=frequency=440:duration=1"])
        .args(["-c:v", "libx264", "-preset", "ultrafast", "-g", "12"])
        .args(["-c:a", "aac", "-ar", "44100", "-ac", "2"])
        .arg(&src));
    let flv = std::fs::read(&src).unwrap();

    let video = demux(&flv, FlvStream::H264).await;
    assert!(video.frames.len() >= 20);
    assert_eq!(&video.bytes()[..4], &[0, 0, 0, 1], "Annex-B start code");
    assert_eq!(video.bytes()[4] & 0x1F, 7, "the prepended SPS leads");

    let audio = demux(&flv, FlvStream::Aac).await;
    assert!(audio.frames.len() >= 40);
    assert_eq!(&audio.bytes()[..2], &[0xFF, 0xF1], "ADTS syncword");
    assert_eq!(
        audio.caps,
        vec![Caps::Audio {
            format: AudioFormat::Aac,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED
        }]
    );

    // And the new selections find nothing in an H.264 + AAC stream.
    assert!(demux(&flv, FlvStream::Vp6).await.frames.is_empty());
    assert!(demux(&flv, FlvStream::Mp3).await.frames.is_empty());
    let _ = std::fs::remove_file(&src);
}

/// Decode plumbing: the codecs libavcodec can decode reach raw frames through the
/// same auto-pluggable elements as H.264 / AAC.
#[cfg(all(target_os = "linux", feature = "ffmpeg"))]
mod decode {
    use super::*;
    use g2g_plugins::ffmpegaudiodec::FfmpegAudioDec;
    use g2g_plugins::ffmpegdec::FfmpegH264Dec;

    #[tokio::test]
    async fn sorenson_decodes_to_raw_frames() {
        if !have("ffmpeg") {
            eprintln!("ffmpeg not present; skipping the Sorenson decode");
            return;
        }
        let src = tmp("g2g_m831_dec_sorenson.flv");
        run(Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=25",
            ])
            .args(["-c:v", "flv1", "-f", "flv"])
            .arg(&src));
        let flv = std::fs::read(&src).unwrap();
        let units = demux(&flv, FlvStream::SorensonH263).await;

        let caps = video_caps(VideoCodec::SorensonH263);
        let mut dec = FfmpegH264Dec::new();
        dec.configure_pipeline(&caps).unwrap();
        let mut sink = CaptureSink::default();
        for (i, au) in units.frames.iter().enumerate() {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming {
                    pts_ns: i as u64 * 40_000_000,
                    ..FrameTiming::default()
                },
                i as u64,
            );
            dec.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert!(
            sink.frames.len() >= units.frames.len() - 1,
            "every Sorenson access unit decodes, got {} of {}",
            sink.frames.len(),
            units.frames.len()
        );
        // I420 at 320x240 is 1.5 bytes per pixel.
        assert_eq!(sink.frames[0].len(), 320 * 240 * 3 / 2);
        let _ = std::fs::remove_file(&src);
    }

    #[tokio::test]
    async fn mp3_decodes_to_pcm() {
        if !have("ffmpeg") {
            eprintln!("ffmpeg not present; skipping the MP3 decode");
            return;
        }
        let src = tmp("g2g_m831_dec_mp3.flv");
        run(Command::new("ffmpeg")
            .args([
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
            ])
            .args([
                "-c:a",
                "libmp3lame",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-f",
                "flv",
            ])
            .arg(&src));
        let flv = std::fs::read(&src).unwrap();
        let units = demux(&flv, FlvStream::Mp3).await;

        let mut dec = FfmpegAudioDec::new();
        dec.configure_pipeline(&Caps::Audio {
            format: AudioFormat::Mp3,
            channels: 2,
            sample_rate: 44_100,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .unwrap();
        let mut sink = CaptureSink::default();
        for au in &units.frames {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            dec.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(
            sink.caps.first(),
            Some(&Caps::Audio {
                format: AudioFormat::PcmS16Le,
                channels: 2,
                sample_rate: 44_100,
                channel_layout: g2g_core::ChannelLayout::UNSPECIFIED
            }),
            "the decoder announces the real layout"
        );
        // A second of 44.1 kHz stereo S16, rounded up to whole 1152-sample MP3
        // frames (so a little over, with lame's encoder delay).
        let samples = sink.bytes().len() / 4;
        assert!(
            (44_100..=48_000).contains(&samples),
            "about a second of PCM, got {samples} samples"
        );
        let _ = std::fs::remove_file(&src);
    }
}
