#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiInputElement, OutputSink, PropValue, PushOutcome, Rate,
    VideoCodec,
};
use g2g_plugins::tsmuxn::TsMux;

const VIDEO_FRAME_NS: u64 = 40_000_000;
const VIDEO_FRAMES: u64 = 75;
const KEYFRAME_SPACING: u64 = 25;
const AAC_SAMPLE_RATE: u64 = 48_000;
const AAC_SAMPLES_PER_FRAME: u64 = 1024;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const TARGET_BITRATE: u64 = 2_000_000;
const IDR_BYTES: usize = 3_000;
const INTER_BYTES: usize = 600;
const AAC_PAYLOAD_BYTES: usize = 200;
const VIDEO_INPUT: usize = 0;
const AUDIO_INPUT: usize = 1;

#[derive(Default)]
struct CaptureSink {
    bytes: Vec<u8>,
}
impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(f) = packet {
            if let Some(s) = f.domain.as_system_slice() {
                self.bytes.extend_from_slice(s);
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn frame(data: Vec<u8>, pts_ns: u64, keyframe: bool) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            keyframe,
            ..FrameTiming::default()
        },
        0,
    ))
}

fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

fn video_access_unit(keyframe: bool) -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    if keyframe {
        let mut idr = vec![0x65u8, 0x88, 0x84];
        idr.resize(IDR_BYTES, 0x5A);
        return annexb(&[&sps, &pps, &idr]);
    }
    let mut inter = vec![0x41u8, 0x9a];
    inter.resize(INTER_BYTES, 0x5A);
    annexb(&[&inter])
}

fn adts_frame() -> Vec<u8> {
    const ADTS_HEADER_BYTES: usize = 7;
    let length = ADTS_HEADER_BYTES + AAC_PAYLOAD_BYTES;
    let mut frame = vec![
        0xFF,
        0xF1,
        0x4C,
        0x80 | ((length >> 11) & 0x03) as u8,
        (length >> 3) as u8,
        (((length & 0x07) as u8) << 5) | 0x1F,
        0xFC,
    ];
    frame.resize(length, 0x00);
    frame
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(25 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: AAC_SAMPLE_RATE as u32,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

async fn mux_av(bitrate: u64) -> Vec<u8> {
    let mut mux = TsMux::new(2);
    mux.set_property("bitrate", PropValue::Uint(bitrate))
        .unwrap();
    mux.configure_pipeline(VIDEO_INPUT, &h264_caps()).unwrap();
    mux.configure_pipeline(AUDIO_INPUT, &aac_caps()).unwrap();
    let mut sink = CaptureSink::default();
    let audio_frame_ns = AAC_SAMPLES_PER_FRAME * NANOSECONDS_PER_SECOND / AAC_SAMPLE_RATE;
    let end_ns = VIDEO_FRAMES * VIDEO_FRAME_NS;
    let mut next_audio_ns = 0;
    for index in 0..VIDEO_FRAMES {
        let video_ns = index * VIDEO_FRAME_NS;
        let keyframe = index % KEYFRAME_SPACING == 0;
        mux.process(
            VIDEO_INPUT,
            frame(video_access_unit(keyframe), video_ns, keyframe),
            &mut sink,
        )
        .await
        .unwrap();
        while next_audio_ns < (video_ns + VIDEO_FRAME_NS).min(end_ns) {
            mux.process(
                AUDIO_INPUT,
                frame(adts_frame(), next_audio_ns, true),
                &mut sink,
            )
            .await
            .unwrap();
            next_audio_ns += audio_frame_ns;
        }
    }
    for input in [VIDEO_INPUT, AUDIO_INPUT] {
        mux.process(input, PipelinePacket::Eos, &mut sink)
            .await
            .unwrap();
    }
    sink.bytes
}

fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m1249-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).expect("write the muxed TS");
    path
}

fn tool_present(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn ffprobe_codecs(path: &Path) -> Output {
    Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe")
}

#[tokio::test]
async fn ffprobe_reads_both_streams_in_either_bitrate_mode() {
    if !tool_present("ffprobe") {
        eprintln!("ffprobe not present; skipping the ffprobe oracle");
        return;
    }
    for bitrate in [0, TARGET_BITRATE] {
        let path = write_temp(&format!("ffprobe-{bitrate}.ts"), &mux_av(bitrate).await);
        let out = ffprobe_codecs(&path);
        let codecs = String::from_utf8_lossy(&out.stdout);
        assert!(out.status.success(), "ffprobe ran on bitrate={bitrate}");
        assert!(codecs.contains("h264"), "bitrate={bitrate}: {codecs}");
        assert!(codecs.contains("aac"), "bitrate={bitrate}: {codecs}");
        let _ = std::fs::remove_file(&path);
    }
}

fn tsp_report(path: &Path) -> String {
    let out = Command::new("tsp")
        .args(["-I", "file"])
        .arg(path)
        .args(["-P", "continuity", "-P", "pcrverify", "-O", "drop"])
        .output()
        .expect("run tsp");
    assert!(out.status.success(), "tsp ran");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn pcr_estimated_bitrate(path: &Path) -> u64 {
    const FIELD: &str = "pcrbitrate=";
    let out = Command::new("tsanalyze")
        .arg("--normalized")
        .arg(path)
        .output()
        .expect("run tsanalyze");
    assert!(out.status.success(), "tsanalyze ran");
    let report = String::from_utf8_lossy(&out.stdout);
    let line = report
        .lines()
        .find(|l| l.starts_with("ts:"))
        .unwrap_or_else(|| panic!("no ts: line in {report}"));
    line.split(':')
        .find_map(|field| field.strip_prefix(FIELD))
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("no {FIELD} in {line}"))
}

#[tokio::test]
async fn tsduck_finds_no_continuity_error_and_the_cbr_pcr_rate_is_the_target() {
    if !tool_present("tsp") || !tool_present("tsanalyze") {
        eprintln!("TSDuck not present; skipping the TSDuck oracle");
        return;
    }
    for bitrate in [0, TARGET_BITRATE] {
        let path = write_temp(&format!("tsduck-{bitrate}.ts"), &mux_av(bitrate).await);
        let report = tsp_report(&path);
        assert!(
            report.contains("* pcrverify:"),
            "pcrverify ran on bitrate={bitrate}: {report}"
        );
        assert!(
            !report.contains("* continuity:"),
            "continuity errors at bitrate={bitrate}: {report}"
        );
        if bitrate > 0 {
            assert!(
                report.contains(" 0 with jitter"),
                "PCR jitter at bitrate={bitrate}: {report}"
            );
            assert_eq!(pcr_estimated_bitrate(&path), bitrate);
        }
        let _ = std::fs::remove_file(&path);
    }
}
