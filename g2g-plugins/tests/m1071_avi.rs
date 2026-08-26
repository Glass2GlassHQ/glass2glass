//! M1071 - AVI demux and mux against ffmpeg-generated fixtures.
//!
//! The fixtures under `tests/fixtures/` were generated with:
//!
//! ```text
//! ffmpeg -f lavfi -i "testsrc=size=160x120:rate=25:duration=1" \
//!        -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1" \
//!        -c:v mjpeg -q:v 20 -c:a pcm_s16le \
//!        -metadata title="g2g avi mjpeg" -metadata artist="glass2glass" \
//!        avi_mjpeg_pcm.avi
//!
//! ffmpeg -f lavfi -i "testsrc=size=160x120:rate=25:duration=1" \
//!        -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=1" \
//!        -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p \
//!        -c:a libmp3lame -b:a 64k \
//!        -metadata title="g2g avi h264" -metadata artist="glass2glass" \
//!        avi_h264_mp3.avi
//! ```
//!
//! No expected value is typed into this file: geometry, codec, framerate,
//! sample rate, channel count, per-stream frame counts, keyframe positions and
//! the container tags are all read back from `ffprobe` when the test runs.
//! `ffprobe -of flat` carries the same fields as `-of json` as `key=value`
//! lines, which this parses without a JSON dependency in the test crate.
//!
//! The malformed-input cases (a truncated file, an absurd `strl` count, a chunk
//! size past the file, an `idx1` offset past the end) are unit tests on the
//! parser itself, in `g2g_plugins::avi`.

#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{block_on, parse_launch, run_graph};
use g2g_core::{
    AudioFormat, Caps, Dim, G2gError, MultiOutputElement, MultiOutputSink, PipelineClock,
    PushOutcome, VideoCodec,
};
use g2g_plugins::avidemux::{forwardable_streams, AviDemuxN};
use g2g_plugins::registry::default_registry;

/// The fixtures this milestone reads, by file name.
const MJPEG_PCM: &str = "avi_mjpeg_pcm.avi";
const H264_MP3: &str = "avi_h264_mp3.avi";
/// Both fixtures hold the video stream first and the audio stream second, the
/// order ffmpeg wrote its `strl` lists in.
const VIDEO_INDEX: usize = 0;
const AUDIO_INDEX: usize = 1;

const NANOS_PER_SECOND: u64 = 1_000_000_000;
/// Per-edge queue depth for the launch runs; a file demux is not latency bound.
const LINK_CAPACITY: usize = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// The `key=value` lines of `ffprobe -of flat`, in probe order.
#[derive(Debug)]
struct Probe {
    entries: Vec<(String, String)>,
}

impl Probe {
    fn run(path: &Path, args: &[&str]) -> Probe {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-of", "flat"])
            .args(args)
            .arg(path)
            .output()
            .unwrap_or_else(|e| panic!("ffprobe must be installed to run this test: {e}"));
        assert!(
            output.status.success(),
            "ffprobe {args:?} on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        let entries = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.to_string(), v.trim_matches('"').to_string()))
            .collect();
        Probe { entries }
    }

    /// Everything `-show_streams -show_format` reports, with per-frame counts.
    fn of(path: &Path) -> Probe {
        Probe::run(
            path,
            &[
                "-show_streams",
                "-show_format",
                "-count_frames",
                "-count_packets",
            ],
        )
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn stream(&self, index: usize, field: &str) -> String {
        let key = format!("streams.stream.{index}.{field}");
        self.get(&key)
            .unwrap_or_else(|| panic!("ffprobe reported no {key}"))
            .to_string()
    }

    fn stream_u64(&self, index: usize, field: &str) -> u64 {
        let raw = self.stream(index, field);
        raw.parse()
            .unwrap_or_else(|e| panic!("{field} = {raw:?} is not a number: {e}"))
    }

    /// A stream's demuxed unit count: how many packets the container hands out,
    /// which is what a demuxer emits.
    fn packets(&self, index: usize) -> u64 {
        self.stream_u64(index, "nb_read_packets")
    }

    /// `r_frame_rate` as a nanosecond frame period.
    fn frame_period_ns(&self, index: usize) -> u64 {
        let rate = self.stream(index, "r_frame_rate");
        let (num, den) = rate
            .split_once('/')
            .unwrap_or_else(|| panic!("r_frame_rate {rate:?} is not a fraction"));
        let num: u64 = num.parse().expect("frame rate numerator");
        let den: u64 = den.parse().expect("frame rate denominator");
        assert!(num > 0, "a video stream has a frame rate");
        NANOS_PER_SECOND * den / num
    }

    /// How many packets of a stream ffprobe flags as keyframes: the `idx1`
    /// `AVIIF_KEYFRAME` bits, read through a reference demuxer.
    fn keyframe_packets(path: &Path, index: usize) -> usize {
        let output = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams"])
            .arg(index.to_string())
            .args([
                "-show_entries",
                "packet=flags",
                "-of",
                "csv=p=0",
                "-read_intervals",
                "%+#100000",
            ])
            .arg(path)
            .output()
            .expect("ffprobe runs");
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|line| line.starts_with('K'))
            .count()
    }
}

/// The g2g codec name a probed `codec_name` maps to.
fn probed_video_codec(name: &str) -> VideoCodec {
    match name {
        "mjpeg" => VideoCodec::Mjpeg,
        "h264" => VideoCodec::H264,
        "mpeg4" => VideoCodec::Mpeg4Part2,
        other => panic!("unexpected fixture video codec {other}"),
    }
}

/// The g2g audio format a probed `codec_name` + `bits_per_sample` maps to.
fn probed_audio_format(name: &str, bits: &str) -> AudioFormat {
    match (name, bits) {
        ("pcm_u8", _) => AudioFormat::PcmU8,
        ("pcm_s16le", _) => AudioFormat::PcmS16Le,
        ("pcm_s24le", _) => AudioFormat::PcmS24Le,
        ("pcm_s32le", _) => AudioFormat::PcmS32Le,
        ("mp3", _) => AudioFormat::Mp3,
        ("ac3", _) => AudioFormat::Ac3,
        ("aac", _) => AudioFormat::Aac,
        (other, _) => panic!("unexpected fixture audio codec {other}"),
    }
}

/// Records, per port, the caps and frames a demuxer pushed.
#[derive(Debug, Default)]
struct PortCapture {
    caps: Vec<Option<Caps>>,
    payloads: Vec<Vec<Vec<u8>>>,
    timings: Vec<Vec<FrameTiming>>,
}

impl PortCapture {
    fn new(ports: usize) -> Self {
        Self {
            caps: vec![None; ports],
            payloads: vec![Vec::new(); ports],
            timings: vec![Vec::new(); ports],
        }
    }
}

impl MultiOutputSink for PortCapture {
    fn port_count(&self) -> usize {
        self.payloads.len()
    }

    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet_slot.take().expect("poll_push without a packet") {
            PipelinePacket::CapsChanged(caps) => self.caps[port] = Some(caps),
            PipelinePacket::DataFrame(frame) => {
                self.timings[port].push(frame.timing);
                self.payloads[port]
                    .push(frame.domain.as_system_slice().unwrap_or_default().to_vec());
            }
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn byte_frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Run the real multi-output demuxer over a whole AVI and capture every port.
fn demux(bytes: &[u8]) -> PortCapture {
    let streams = forwardable_streams(bytes);
    assert!(!streams.is_empty(), "the file carries a forwardable stream");
    let mut demux = AviDemuxN::new(streams);
    let mut sink = PortCapture::new(demux.port_count());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Avi,
        })
        .expect("the byte stream is accepted");
    block_on(async {
        demux
            .process(byte_frame(bytes.to_vec()), &mut sink)
            .await
            .expect("the bytes buffer");
        demux
            .process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("the file demuxes");
    });
    sink
}

async fn run_line(line: &str) -> u64 {
    let registry = default_registry();
    let graph = parse_launch(&registry, line).unwrap_or_else(|e| panic!("{line:?} parse: {e}"));
    run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs {line:?}: {e:?}"))
        .frames_consumed
}

/// The `stream=` name for the video stream a probe describes.
fn video_stream_name(probe: &Probe) -> &'static str {
    match probed_video_codec(&probe.stream(VIDEO_INDEX, "codec_name")) {
        VideoCodec::Mjpeg => "mjpeg",
        VideoCodec::H264 => "h264",
        VideoCodec::Mpeg4Part2 => "mpeg4part2",
        other => panic!("no stream name for {other:?}"),
    }
}

/// The `stream=` name for the audio stream a probe describes.
fn audio_stream_name(probe: &Probe) -> &'static str {
    let format = probed_audio_format(
        &probe.stream(AUDIO_INDEX, "codec_name"),
        &probe.stream(AUDIO_INDEX, "bits_per_sample"),
    );
    match format {
        AudioFormat::PcmU8 => "pcm-u8",
        AudioFormat::PcmS16Le => "pcm-s16le",
        AudioFormat::PcmS24Le => "pcm-s24le",
        AudioFormat::PcmS32Le => "pcm-s32le",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Ac3 => "ac3",
        AudioFormat::Aac => "aac",
        other => panic!("no stream name for {other:?}"),
    }
}

#[test]
fn demuxes_each_fixture_to_the_streams_ffprobe_reports() {
    for name in [MJPEG_PCM, H264_MP3] {
        let path = fixture(name);
        let probe = Probe::of(&path);
        let bytes = std::fs::read(&path).expect("the fixture is checked in");
        let sink = demux(&bytes);

        assert_eq!(
            sink.port_count(),
            probe
                .get("format.nb_streams")
                .expect("nb_streams")
                .parse::<usize>()
                .expect("a count"),
            "{name}: one port per container stream"
        );

        // Per-stream counts: one frame per demuxed packet.
        assert_eq!(
            sink.payloads[VIDEO_INDEX].len() as u64,
            probe.packets(VIDEO_INDEX),
            "{name}: video frame count"
        );
        assert_eq!(
            sink.payloads[AUDIO_INDEX].len() as u64,
            probe.packets(AUDIO_INDEX),
            "{name}: audio frame count"
        );

        // The CapsChanged each branch sees carries the probed stream. The
        // framerate is not in it: AVI's rate travels per frame, so the caps keep
        // the span the branch negotiated against.
        let Some(Caps::CompressedVideo {
            codec,
            width,
            height,
            ..
        }) = &sink.caps[VIDEO_INDEX]
        else {
            panic!("{name}: the video port declared no video caps");
        };
        assert_eq!(
            *codec,
            probed_video_codec(&probe.stream(VIDEO_INDEX, "codec_name")),
            "{name}: video codec"
        );
        assert_eq!(
            *width,
            Dim::Fixed(probe.stream_u64(VIDEO_INDEX, "width") as u32),
            "{name}: video width"
        );
        assert_eq!(
            *height,
            Dim::Fixed(probe.stream_u64(VIDEO_INDEX, "height") as u32),
            "{name}: video height"
        );
        assert_eq!(
            sink.caps[AUDIO_INDEX],
            Some(Caps::Audio {
                format: probed_audio_format(
                    &probe.stream(AUDIO_INDEX, "codec_name"),
                    &probe.stream(AUDIO_INDEX, "bits_per_sample")
                ),
                channels: probe.stream_u64(AUDIO_INDEX, "channels") as u8,
                sample_rate: probe.stream_u64(AUDIO_INDEX, "sample_rate") as u32,
            }),
            "{name}: audio caps"
        );

        // Video pts advance by the probed frame period.
        let period = probe.frame_period_ns(VIDEO_INDEX);
        for (i, timing) in sink.timings[VIDEO_INDEX].iter().enumerate() {
            assert_eq!(
                timing.pts_ns,
                i as u64 * period,
                "{name}: video frame {i} sits one frame period past its predecessor"
            );
            assert_eq!(timing.duration_ns, period, "{name}: video frame duration");
        }

        // Keyframe flags come from `idx1`, so they must agree with what a
        // reference demuxer reads out of the same index.
        assert_eq!(
            sink.timings[VIDEO_INDEX]
                .iter()
                .filter(|t| t.keyframe)
                .count(),
            Probe::keyframe_packets(&path, VIDEO_INDEX),
            "{name}: keyframe count matches idx1"
        );
        assert!(
            sink.timings[VIDEO_INDEX][0].keyframe,
            "{name}: the first video frame opens the stream"
        );
    }
}

#[tokio::test]
async fn a_launch_line_demuxes_each_stream_and_the_fan_out() {
    for name in [MJPEG_PCM, H264_MP3] {
        let path = fixture(name);
        let probe = Probe::of(&path);
        let location = path.display();
        let video = probe.packets(VIDEO_INDEX);
        let audio = probe.packets(AUDIO_INDEX);

        let line = format!(
            "filesrc location={location} ! avidemux stream={} ! fakesink",
            video_stream_name(&probe)
        );
        assert_eq!(run_line(&line).await, video, "{name}: video-only line");

        let line = format!(
            "filesrc location={location} ! avidemux stream={} ! fakesink",
            audio_stream_name(&probe)
        );
        assert_eq!(run_line(&line).await, audio, "{name}: audio-only line");

        let line = format!(
            "filesrc location={location} ! avidemux name=d  \
             d.video_0 ! fakesink  d.audio_0 ! fakesink"
        );
        assert_eq!(
            run_line(&line).await,
            video + audio,
            "{name}: both branches of the fan-out"
        );
    }
}

#[tokio::test]
async fn types_an_avi_by_content_and_auto_plugs_the_demuxer() {
    let path = fixture(MJPEG_PCM);
    let probe = Probe::of(&path);
    let bytes = std::fs::read(&path).expect("the fixture is checked in");

    // The RIFF `AVI ` magic alone types the stream.
    assert_eq!(
        g2g_plugins::typefind::sniff_caps(&bytes[..g2g_plugins::typefind::SNIFF_LEN]),
        Some(Caps::ByteStream {
            encoding: g2g_core::ByteStreamEncoding::Avi
        })
    );

    // No `bytestream-format=`: `typefind` declares the type mid-graph.
    let line = format!(
        "filesrc location={} ! typefind ! avidemux stream={} ! fakesink",
        path.display(),
        video_stream_name(&probe)
    );
    assert_eq!(run_line(&line).await, probe.packets(VIDEO_INDEX));

    // What `decodebin` expands to over an AVI byte stream: the auto-plug search
    // reaches a compressed elementary stream through `avidemux`.
    let registry = default_registry();
    let chain = registry
        .autoplug_names(
            &Caps::ByteStream {
                encoding: g2g_core::ByteStreamEncoding::Avi,
            },
            &|caps: &Caps| matches!(caps, Caps::CompressedVideo { .. }),
            2,
        )
        .expect("an AVI byte stream reaches an elementary stream");
    assert_eq!(chain.first(), Some(&"avidemux"), "got {chain:?}");
}

#[tokio::test]
async fn a_mux_round_trip_preserves_every_chunk() {
    for name in [MJPEG_PCM, H264_MP3] {
        let path = fixture(name);
        let source = std::fs::read(&path).expect("the fixture is checked in");
        let out = std::env::temp_dir().join(format!("g2g-m1071-{}-{name}", std::process::id()));
        let line = format!(
            "filesrc location={} ! avidemux name=d  \
             d.video_0 ! m.video_0  d.audio_0 ! m.audio_0  \
             avimux name=m ! filesink location={}",
            path.display(),
            out.display()
        );
        run_line(&line).await;

        let remuxed = std::fs::read(&out).unwrap_or_else(|e| panic!("{name}: no output: {e}"));
        let before = demux(&source);
        let after = demux(&remuxed);
        assert_eq!(
            after.port_count(),
            before.port_count(),
            "{name}: stream count"
        );
        for port in 0..before.port_count() {
            assert_eq!(
                after.payloads[port].len(),
                before.payloads[port].len(),
                "{name}: port {port} frame count"
            );
            assert_eq!(
                after.payloads[port], before.payloads[port],
                "{name}: port {port} chunk payloads are byte-exact"
            );
            assert_eq!(
                after.caps[port], before.caps[port],
                "{name}: port {port} caps"
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

#[test]
fn posts_the_container_tags_ffprobe_reports() {
    let path = fixture(MJPEG_PCM);
    let probe = Probe::of(&path);
    let bytes = std::fs::read(&path).expect("the fixture is checked in");
    let tags = g2g_plugins::avidemux::probe_tags(&bytes).expect("the file parses");
    for (key, field) in [
        ("title", "format.tags.title"),
        ("artist", "format.tags.artist"),
    ] {
        let expected = probe
            .get(field)
            .unwrap_or_else(|| panic!("ffprobe reported no {field}"));
        let found = tags
            .tags()
            .iter()
            .find(|t| t.key() == key)
            .unwrap_or_else(|| panic!("no {key} tag on the bus"));
        assert_eq!(found.value_string(), expected);
    }
}

/// Reference-peer checks against the installed ffmpeg / GStreamer. Ignored by
/// default: they depend on tools CI does not carry.
mod interop {
    use super::*;

    /// `avimux`'s output must probe the same as the source it was remuxed from.
    #[tokio::test]
    #[ignore]
    async fn ffprobe_agrees_with_the_source_on_a_remuxed_file() {
        for name in [MJPEG_PCM, H264_MP3] {
            let path = fixture(name);
            let out = std::env::temp_dir().join(format!("g2g-m1071-interop-{name}"));
            let line = format!(
                "filesrc location={} ! avidemux name=d  \
                 d.video_0 ! m.video_0  d.audio_0 ! m.audio_0  \
                 avimux name=m ! filesink location={}",
                path.display(),
                out.display()
            );
            run_line(&line).await;
            let before = Probe::of(&path);
            let after = Probe::of(&out);
            for field in ["codec_name", "width", "height", "nb_read_packets"] {
                assert_eq!(
                    after.stream(VIDEO_INDEX, field),
                    before.stream(VIDEO_INDEX, field),
                    "{name}: video {field}"
                );
            }
            for field in ["codec_name", "sample_rate", "channels", "nb_read_packets"] {
                assert_eq!(
                    after.stream(AUDIO_INDEX, field),
                    before.stream(AUDIO_INDEX, field),
                    "{name}: audio {field}"
                );
            }
            let _ = std::fs::remove_file(&out);
        }
    }

    /// GStreamer's own `avidemux` must accept what g2g wrote, and g2g must
    /// accept what GStreamer's `avimux` wrote. GStreamer's `avidemux` needs a
    /// `queue` on each branch and its `avimux` needs a parser ahead of a
    /// compressed stream, so each fixture names the elements its codecs want.
    #[tokio::test]
    #[ignore]
    async fn gstreamer_reads_our_file_and_we_read_gstreamers() {
        for (name, video_parser, audio_parser) in [
            (MJPEG_PCM, "", ""),
            (H264_MP3, "h264parse !", "mpegaudioparse !"),
        ] {
            let path = fixture(name);
            let ours = std::env::temp_dir().join(format!("g2g-m1071-ours-{name}"));
            let line = format!(
                "filesrc location={} ! avidemux name=d  \
                 d.video_0 ! m.video_0  d.audio_0 ! m.audio_0  \
                 avimux name=m ! filesink location={}",
                path.display(),
                ours.display()
            );
            run_line(&line).await;
            gst_launch(&format!(
                "filesrc location={} ! avidemux name=d \
                 d.video_0 ! queue ! fakesink  d.audio_0 ! queue ! fakesink",
                ours.display()
            ));

            let theirs = std::env::temp_dir().join(format!("g2g-m1071-gst-{name}"));
            gst_launch(&format!(
                "filesrc location={} ! avidemux name=d \
                 d.video_0 ! queue ! {video_parser} m.video_0 \
                 d.audio_0 ! queue ! {audio_parser} m.audio_0 \
                 avimux name=m ! filesink location={}",
                path.display(),
                theirs.display()
            ));
            let gst_probe = Probe::of(&theirs);
            let bytes = std::fs::read(&theirs).expect("gstreamer wrote a file");
            let sink = demux(&bytes);
            assert_eq!(
                sink.payloads[VIDEO_INDEX].len() as u64,
                gst_probe.packets(VIDEO_INDEX),
                "{name}: we read every video chunk gstreamer wrote"
            );
            assert_eq!(
                sink.payloads[AUDIO_INDEX].len() as u64,
                gst_probe.packets(AUDIO_INDEX),
                "{name}: we read every audio chunk gstreamer wrote"
            );
            let _ = std::fs::remove_file(&ours);
            let _ = std::fs::remove_file(&theirs);
        }
    }

    /// Run one `gst-launch-1.0` pipeline to EOS, failing loud on its output.
    /// `gst-launch` parses one element per argument, so the description is split
    /// on whitespace (no path here carries a space).
    fn gst_launch(pipeline: &str) {
        let output = Command::new("gst-launch-1.0")
            .arg("-q")
            .args(pipeline.split_whitespace())
            .output()
            .expect("gst-launch-1.0 must be installed to run this test");
        assert!(
            output.status.success(),
            "gst-launch {pipeline:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
