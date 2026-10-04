//! M1215: `splitmuxsrc` plays the parts a `splitmuxsink` wrote as one stream.
//!
//! This repo's own `splitmuxsink` writes an H.264 fixture into mp4, matroska and
//! mpegts parts, and what comes back has to be the input on one unbroken
//! timeline. GStreamer's `splitmuxsink` wrote the checked-in
//! `splitmuxsink_gst_part*` fixtures (mp4 parts restart at zero, matroska and
//! mpegts parts keep counting), checked against the `ffprobe` output beside each
//! part since CI has neither tool. The parts were written by
//!
//! ```text
//! gst-launch-1.0 -q videotestsrc num-buffers=30 ! video/x-raw,width=64,height=48,framerate=30/1 ! openh264enc gop-size=10 ! h264parse ! splitmuxsink location=splitmuxsink_gst_part%02d.mp4 max-size-time=300000000
//! gst-launch-1.0 -q videotestsrc num-buffers=30 ! video/x-raw,width=64,height=48,framerate=30/1 ! openh264enc gop-size=10 ! h264parse ! splitmuxsink async-finalize=true muxer-factory=matroskamux location=splitmuxsink_gst_part%02d.mkv max-size-time=300000000
//! gst-launch-1.0 -q videotestsrc num-buffers=30 ! video/x-raw,width=64,height=48,framerate=30/1 ! openh264enc gop-size=10 ! h264parse ! splitmuxsink async-finalize=true muxer-factory=mpegtsmux location=splitmuxsink_gst_part%02d.ts max-size-time=300000000
//! ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate:packet=pts_time -of flat <part> > <part>.probe
//! ```
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std --test m1215_splitmuxsrc`, and add
//! `-- --ignored` to compare against GStreamer's own `splitmuxsrc`.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{Caps, Dim, G2gError, PipelineClock, PropError, PropValue, Rate, VideoCodec};
use g2g_plugins::gst_compat::{gst_equivalent, GstEquivalent};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::registry::default_registry;
use g2g_plugins::splitmuxsink::SplitMuxSink;
use g2g_plugins::splitmuxsrc::SplitMuxSrc;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
// two GOPs per pass, so four parts
const PASSES: usize = 2;
const FRAME_NS: u64 = 50_000_000;
// one part per GOP
const MAX_SIZE_TIME_NS: u64 = FRAME_NS;
// ffprobe prints pts_time to the microsecond
const PROBE_PRECISION_NS: u64 = 1_000;
const NS_PER_SECOND: f64 = 1e9;

const GST_PART_PREFIX: &str = "splitmuxsink_gst_part";
const GST_EXTENSIONS: [&str; 3] = ["mp4", "mkv", "ts"];

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

fn gst_parts_pattern(extension: &str) -> String {
    fixture(&format!("{GST_PART_PREFIX}*.{extension}"))
        .to_string_lossy()
        .into_owned()
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g-m1215-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the working directory is created");
    dir
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

#[derive(Default)]
struct Collect {
    frames: Vec<(FrameTiming, Vec<u8>)>,
    eos: usize,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        match packet.take().expect("poll_push without a packet") {
            PipelinePacket::DataFrame(f) => self.frames.push((
                f.timing,
                f.domain.as_system_slice().expect("system memory").to_vec(),
            )),
            PipelinePacket::Eos => self.eos += 1,
            _ => {}
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

async fn access_units() -> Vec<(bool, Vec<u8>)> {
    let mut parse = H264Parse::reframing();
    parse
        .configure_pipeline(&h264_caps())
        .expect("h264parse accepts the stream");
    let mut sink = Collect::default();
    let whole = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(FIXTURE.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    parse
        .process(PipelinePacket::DataFrame(whole), &mut sink)
        .await
        .expect("parse the fixture");
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain at Eos");
    let pass: Vec<(bool, Vec<u8>)> = sink
        .frames
        .into_iter()
        .map(|(timing, bytes)| (timing.keyframe, bytes))
        .collect();
    pass.iter()
        .cycle()
        .take(pass.len() * PASSES)
        .cloned()
        .collect()
}

// a container stores length prefixes, so a 3-byte start code comes back as 4 bytes
fn nal_units(access_unit: &[u8]) -> Vec<&[u8]> {
    let starts: Vec<usize> = access_unit
        .windows(3)
        .enumerate()
        .filter(|(_, window)| *window == [0, 0, 1])
        .map(|(at, _)| at + 3)
        .collect();
    starts
        .iter()
        .enumerate()
        .map(|(index, &start)| {
            let end = starts
                .get(index + 1)
                .map_or(access_unit.len(), |next| next - 3);
            let unit = &access_unit[start..end];
            let trailing_zeros = unit.iter().rev().take_while(|&&b| b == 0).count();
            &unit[..unit.len() - trailing_zeros]
        })
        .collect()
}

async fn play(location: &str) -> Collect {
    let mut src = SplitMuxSrc::new("");
    src.set_property("location", PropValue::Str(location.into()))
        .expect("location is a property");
    let caps = src.intercept_caps().await.expect("the first part types");
    src.configure_pipeline(&caps.fixate().expect("the caps fixate"))
        .expect("the source configures");
    let mut out = Collect::default();
    let frames = src.run(&mut out).await.expect("the parts play");
    assert_eq!(
        frames as usize,
        out.frames.len(),
        "run reports the frames sent"
    );
    out
}

async fn round_trip(muxer: &str, extension: &str) {
    let access_units = access_units().await;
    let dir = temp_dir(muxer);
    let mut sink = SplitMuxSink::new(
        dir.join(format!("part%03d.{extension}"))
            .to_string_lossy()
            .into_owned(),
    );
    sink.set_property("muxer", PropValue::Str(muxer.into()))
        .expect("muxer is a property");
    sink.set_property("max-size-time", PropValue::Uint(MAX_SIZE_TIME_NS))
        .expect("max-size-time is a property");
    sink.configure_pipeline(&h264_caps())
        .expect("splitmuxsink takes H.264");
    let mut discard = Collect::default();
    for (index, (keyframe, bytes)) in access_units.iter().enumerate() {
        let pts_ns = index as u64 * FRAME_NS;
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(bytes.clone().into_boxed_slice())),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: FRAME_NS,
                keyframe: *keyframe,
                ..FrameTiming::default()
            },
            index as u64,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut discard)
            .await
            .expect("splitmuxsink writes the frame");
    }
    sink.process(PipelinePacket::Eos, &mut discard)
        .await
        .expect("splitmuxsink finalizes");
    let keyframes = access_units
        .iter()
        .filter(|(keyframe, _)| *keyframe)
        .count();
    assert_eq!(
        sink.files_written() as usize,
        keyframes,
        "{muxer}: one part per GOP"
    );

    let out = play(&dir.join(format!("part*.{extension}")).to_string_lossy()).await;
    assert_eq!(out.eos, 1, "{muxer}: one Eos after the last part");
    assert_eq!(
        out.frames.len(),
        access_units.len(),
        "{muxer}: every access unit comes back"
    );
    for (index, ((_, played), (_, written))) in out.frames.iter().zip(&access_units).enumerate() {
        assert_eq!(
            nal_units(played),
            nal_units(written),
            "{muxer}: access unit {index} comes back in order and unchanged"
        );
    }
    for (index, pair) in out.frames.windows(2).enumerate() {
        assert_eq!(
            pair[1].0.pts_ns - pair[0].0.pts_ns,
            FRAME_NS,
            "{muxer}: frame {} follows frame {index} one frame period later, across part boundaries too",
            index + 1
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn mp4_parts_play_back_as_the_stream_written() {
    round_trip("mp4", "mp4").await;
}

#[tokio::test]
async fn matroska_parts_play_back_as_the_stream_written() {
    round_trip("matroska", "mkv").await;
}

#[tokio::test]
async fn mpegts_parts_play_back_as_the_stream_written() {
    round_trip("mpegts", "ts").await;
}

struct Probe {
    pts_ns: Vec<u64>,
    frame_period_ns: u64,
}

impl Probe {
    fn of_part(part: &Path) -> Probe {
        let path = PathBuf::from(format!("{}.probe", part.display()));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} is checked in beside the part: {e}", path.display()));
        let value = |line: &str| {
            line.split_once('=')
                .map(|(_, v)| v.trim_matches('"').to_string())
        };
        let pts_ns = text
            .lines()
            .filter(|line| line.starts_with("packets.packet.") && line.contains(".pts_time="))
            .map(|line| {
                let seconds: f64 = value(line)
                    .and_then(|v| v.parse().ok())
                    .expect("a pts_time");
                (seconds * NS_PER_SECOND).round() as u64
            })
            .collect();
        let rate = text
            .lines()
            .find(|line| line.starts_with("streams.stream.0.r_frame_rate="))
            .and_then(value)
            .expect("a frame rate");
        let (numerator, denominator) = rate.split_once('/').expect("a num/den rate");
        let numerator: u64 = numerator.parse().expect("a numerator");
        let denominator: u64 = denominator.parse().expect("a denominator");
        Probe {
            pts_ns,
            frame_period_ns: denominator * NS_PER_SECOND as u64 / numerator,
        }
    }
}

fn gst_parts(extension: &str) -> Vec<PathBuf> {
    let mut parts: Vec<PathBuf> = std::fs::read_dir(fixture(""))
        .expect("the fixture directory reads")
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            name.starts_with(GST_PART_PREFIX) && name.ends_with(&format!(".{extension}"))
        })
        .collect();
    parts.sort();
    parts
}

fn within_probe_precision(actual: u64, expected: u64) -> bool {
    actual.abs_diff(expected) <= PROBE_PRECISION_NS
}

#[tokio::test]
async fn gstreamer_parts_play_back_as_one_timeline() {
    for extension in GST_EXTENSIONS {
        let probes: Vec<Probe> = gst_parts(extension)
            .iter()
            .map(|p| Probe::of_part(p))
            .collect();
        assert!(
            probes.len() > 1,
            "{extension}: several parts are checked in"
        );
        let out = play(&gst_parts_pattern(extension)).await;
        assert_eq!(out.eos, 1, "{extension}: one Eos after the last part");
        let probed_frames: usize = probes.iter().map(|p| p.pts_ns.len()).sum();
        assert_eq!(
            out.frames.len(),
            probed_frames,
            "{extension}: every packet ffprobe sees in the parts plays"
        );
        assert!(
            within_probe_precision(out.frames[0].0.pts_ns, probes[0].pts_ns[0]),
            "{extension}: the first part plays at its own timestamps"
        );

        let mut played = out.frames.iter().map(|(timing, _)| timing.pts_ns);
        let mut previous: Option<u64> = None;
        for (part, probe) in probes.iter().enumerate() {
            let pts: Vec<u64> = played.by_ref().take(probe.pts_ns.len()).collect();
            if let Some(previous) = previous {
                let step = pts[0] - previous;
                assert!(
                    step > 0 && step <= probe.frame_period_ns + PROBE_PRECISION_NS,
                    "{extension}: part {part} starts {step} ns after the previous part's last frame, not within one frame period ({} ns)",
                    probe.frame_period_ns
                );
            }
            for (index, (played, probed)) in pts.windows(2).zip(probe.pts_ns.windows(2)).enumerate()
            {
                assert!(
                    within_probe_precision(played[1] - played[0], probed[1] - probed[0]),
                    "{extension}: part {part} frame {} keeps the spacing ffprobe reports",
                    index + 1
                );
            }
            previous = pts.last().copied();
        }
    }
}

// needs gst-launch-1.0, which CI does not have
#[tokio::test]
#[ignore]
async fn gstreamer_splitmuxsrc_reads_as_many_frames() {
    for extension in GST_EXTENSIONS {
        let pattern = gst_parts_pattern(extension);
        let run = Command::new("gst-launch-1.0")
            .arg("-v")
            .args(["splitmuxsrc", &format!("location={pattern}")])
            .args(["!", "fakesink", "silent=false"])
            .output()
            .expect("gst-launch-1.0 is installed");
        assert!(
            run.status.success(),
            "{extension}: gst-launch-1.0 read the parts"
        );
        // `fakesink silent=false` under `-v` prints one `chain` line per buffer
        let buffers = String::from_utf8_lossy(&run.stdout)
            .lines()
            .filter(|line| line.contains("chain"))
            .count();
        assert_eq!(play(&pattern).await.frames.len(), buffers, "{extension}");
    }
}

#[tokio::test]
async fn a_launch_line_plays_the_parts() {
    let pattern = gst_parts_pattern(GST_EXTENSIONS[0]);
    let expected: usize = gst_parts(GST_EXTENSIONS[0])
        .iter()
        .map(|p| Probe::of_part(p).pts_ns.len())
        .sum();
    let line = format!("splitmuxsrc location={pattern} ! fakesink");
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    assert_eq!(stats.frames_consumed as usize, expected);
}

#[tokio::test]
async fn a_pattern_matching_nothing_fails_to_start() {
    let dir = temp_dir("empty");
    let pattern = dir.join("nothing*.mp4").to_string_lossy().into_owned();
    let mut src = SplitMuxSrc::new(&pattern);
    assert_eq!(src.intercept_caps().await, Err(G2gError::CapsMismatch));
    assert_eq!(
        src.configure_pipeline(&h264_caps()).err(),
        Some(G2gError::CapsMismatch)
    );
    let line = format!("splitmuxsrc location={pattern} ! fakesink");
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    assert!(
        run_graph(graph, &ZeroClock, 4).await.is_err(),
        "no parts, no run"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn gstreamer_name_resolves_to_this_element() {
    assert_eq!(
        gst_equivalent(&default_registry(), "splitmuxsrc"),
        GstEquivalent::Available
    );
}

#[test]
fn location_round_trips_and_is_the_only_property() {
    let mut src = SplitMuxSrc::new("");
    let names: Vec<&str> = src.properties().iter().map(|spec| spec.name).collect();
    assert_eq!(names, ["location"]);
    let pattern = gst_parts_pattern(GST_EXTENSIONS[0]);
    src.set_property("location", PropValue::Str(pattern.clone()))
        .expect("location is a property");
    assert_eq!(src.get_property("location"), Some(PropValue::Str(pattern)));
    assert_eq!(
        src.set_property("num-open-fragments", PropValue::Uint(1)),
        Err(PropError::Unknown)
    );
}
