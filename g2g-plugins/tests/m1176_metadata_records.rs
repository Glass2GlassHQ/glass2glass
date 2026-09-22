#![cfg(feature = "analytics-json")]
//! M1176: the metadata record tools. Each element runs for real; the assertions
//! come from the fixture table below and from reading back what the elements
//! wrote, never from restating an element's own output.
//!
//! - `metasink` writes one line per frame and posts it on the bus.
//! - `metareplay` puts that same file back onto bare frames, and the detections
//!   land where they started (within the pixel the record rounded them to).
//! - `analyticsalert` fires its rule once per cooldown, in stream time, attaches
//!   the `alert` blob and paints the border.
//! - `alertrecorder` writes a clip that decodes back through `avidemux !
//!   mjpegdec`, covering the frames before and after the alert.

use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AnalyticsMeta, BBox, BlobMeta, Bus, BusMessage, Caps, Colorimetry, Dim, G2gError, Interlace,
    ObjectDetection, OutputSink, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::analyticsalert::AnalyticsAlert;
use g2g_plugins::metareplay::MetaReplay;
use g2g_plugins::metasink::MetaSink;

/// Fixed-point shift a `Rate` counts fps in.
const Q16_SHIFT: u32 = 16;
/// The rate the record frames are negotiated at. Nothing reads it back, but a
/// sink still needs a complete caps.
const RECORD_FPS: u32 = 30;

/// The geometry the records are written against.
const FRAME_WIDTH: u32 = 320;
const FRAME_HEIGHT: u32 = 240;
/// The class-name table the detections index.
const CLASS_NAMES: &[&str] = &["person", "car"];
/// The frames the record file is written from: `(pts_ns, label, x, y, w, h,
/// score)`, the box normalized. Every value is an exact binary fraction, so the
/// pixel conversion and the `f32` round trip through the file are exact.
const RECORDED: &[(u64, u32, f32, f32, f32, f32, f32)] = &[
    (1_000_000_000, 0, 0.25, 0.5, 0.125, 0.25, 0.875),
    (2_000_000_000, 1, 0.5, 0.25, 0.25, 0.5, 0.5),
];
/// A JSON blob riding on the first recorded frame, under its own header.
const BLOB_HEADER: &str = "note";
const BLOB_PAYLOAD: &[u8] = b"{\"camera\":\"north\"}";

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
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

impl RecordingSink {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }
}

fn rgba_caps(width: u32, height: u32, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate,
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

fn black_frame(width: u32, height: u32, pts_ns: u64) -> Frame {
    let pixels = vec![0u8; width as usize * height as usize * 4];
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence: pts_ns,
        meta: Default::default(),
    }
}

/// One recorded row as the detection it describes.
fn detection(row: &(u64, u32, f32, f32, f32, f32, f32)) -> ObjectDetection {
    let (_, label, x, y, w, h, score) = *row;
    ObjectDetection {
        bbox: BBox { x, y, w, h },
        label,
        confidence: score,
    }
}

/// A frame carrying one recorded row's detection, and the blob on the first.
fn recorded_frame(index: usize) -> Frame {
    let row = &RECORDED[index];
    let mut frame = black_frame(1, 1, row.0);
    let mut analytics = AnalyticsMeta::new();
    analytics.set_class_names(CLASS_NAMES.iter().copied());
    analytics.add_detection(detection(row));
    frame.meta.attach(analytics);
    if index == 0 {
        let mut blobs = BlobMeta::new();
        blobs.push(BLOB_HEADER, Vec::from(BLOB_PAYLOAD));
        frame.meta.attach(blobs);
    }
    frame
}

fn temp_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Run a `metasink` over the fixture frames, returning the file it wrote and the
/// records it posted on the bus.
async fn write_records(path: &PathBuf) -> (Vec<String>, Vec<BusMessage>) {
    let mut sink = MetaSink::new().with_location(path.to_string_lossy().into_owned());
    sink.set_instance_name(String::from("records"));
    let (bus, handle) = Bus::new(16);
    sink.set_bus(handle);
    sink.configure_pipeline(&rgba_caps(
        FRAME_WIDTH,
        FRAME_HEIGHT,
        Rate::Fixed(RECORD_FPS << Q16_SHIFT),
    ))
    .expect("metasink accepts the caps");
    let mut out = RecordingSink::default();
    for index in 0..RECORDED.len() {
        sink.process(PipelinePacket::DataFrame(recorded_frame(index)), &mut out)
            .await
            .expect("metasink writes the record");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("metasink flushes");
    assert_eq!(sink.records_written(), RECORDED.len() as u64);
    let lines = std::fs::read_to_string(path)
        .expect("the record file")
        .lines()
        .map(String::from)
        .collect();
    let mut posted = Vec::new();
    while let Some(message) = bus.try_recv() {
        posted.push(message);
    }
    (lines, posted)
}

#[tokio::test]
async fn metasink_writes_a_record_per_frame_and_posts_it() {
    let path = temp_path("m1176_metasink.jsonl");
    let (lines, posted) = write_records(&path).await;
    assert_eq!(lines.len(), RECORDED.len());
    for (index, line) in lines.iter().enumerate() {
        let (pts_ns, label, x, y, w, h, score) = RECORDED[index];
        assert!(
            line.contains(&format!("\"pts\":{}", pts_ns as f64 / 1e9)),
            "{line} carries the frame time in seconds"
        );
        assert!(
            line.contains(&format!("\"label\":\"{}\"", CLASS_NAMES[label as usize])),
            "{line} names the class"
        );
        assert!(line.contains(&format!("\"score\":{}", f64::from(score))));
        // The box in whole pixels of the negotiated geometry.
        for (key, normalized, span) in [
            ("x", x, FRAME_WIDTH),
            ("y", y, FRAME_HEIGHT),
            ("w", w, FRAME_WIDTH),
            ("h", h, FRAME_HEIGHT),
        ] {
            let pixels = (normalized * span as f32).round() as i64;
            assert!(
                line.contains(&format!("\"{key}\":{pixels}")),
                "{line} carries {key} in pixels"
            );
        }
    }
    // A JSON blob becomes a key of its own; an embedding's bytes would not.
    let blob_json = String::from_utf8(Vec::from(BLOB_PAYLOAD)).expect("the blob is text");
    let blob_body = blob_json.trim_start_matches('{').trim_end_matches('}');
    assert!(
        lines[0].contains(&format!("\"{BLOB_HEADER}\":{{")) && lines[0].contains(blob_body),
        "the blob is a key of the record"
    );
    let records: Vec<&String> = posted
        .iter()
        .filter_map(|message| match message {
            BusMessage::MetadataRecord { element, record } => {
                assert_eq!(element, "records", "the posting sink names itself");
                Some(record)
            }
            _ => None,
        })
        .collect();
    assert_eq!(records, lines.iter().collect::<Vec<_>>());
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn metareplay_reattaches_what_metasink_wrote() {
    let path = temp_path("m1176_metareplay.jsonl");
    write_records(&path).await;
    let mut replay = MetaReplay::new().with_location(path.to_string_lossy().into_owned());
    let caps = rgba_caps(
        FRAME_WIDTH,
        FRAME_HEIGHT,
        Rate::Fixed(RECORD_FPS << Q16_SHIFT),
    );
    replay
        .configure_pipeline(&caps)
        .expect("metareplay reads the file");
    assert_eq!(replay.records(), RECORDED.len());
    let mut out = RecordingSink::default();
    for row in RECORDED {
        replay
            .process(
                PipelinePacket::DataFrame(black_frame(1, 1, row.0)),
                &mut out,
            )
            .await
            .expect("metareplay forwards the frame");
    }
    let frames = out.frames();
    assert_eq!(frames.len(), RECORDED.len());
    for (frame, row) in frames.iter().zip(RECORDED) {
        let analytics = frame
            .meta
            .get::<AnalyticsMeta>()
            .expect("the record's detections are back");
        let replayed = *analytics.detections().next().expect("one detection");
        let started = detection(row);
        assert_eq!(
            analytics.class_name(replayed.label),
            Some(CLASS_NAMES[started.label as usize]),
            "the label name survives the file"
        );
        assert_eq!(replayed.confidence, started.confidence);
        // The record holds whole pixels, so a box comes back within one of them.
        for (replayed, started, span) in [
            (replayed.bbox.x, started.bbox.x, FRAME_WIDTH),
            (replayed.bbox.y, started.bbox.y, FRAME_HEIGHT),
            (replayed.bbox.w, started.bbox.w, FRAME_WIDTH),
            (replayed.bbox.h, started.bbox.h, FRAME_HEIGHT),
        ] {
            let off_by = (replayed - started).abs() * span as f32;
            assert!(off_by <= 1.0, "{replayed} is within a pixel of {started}");
        }
    }
    let blobs = frames[0]
        .meta
        .get::<BlobMeta>()
        .expect("the record's blob is back");
    let blob = blobs.get(BLOB_HEADER).expect("under its own header");
    assert_eq!(blob.payload, BLOB_PAYLOAD, "the blob's JSON comes back");
    let _ = std::fs::remove_file(&path);
}

/// The rule the alert test fires, and the stream times it is fed.
const ALERT_RULES: &str = "[{\"class\":\"person\",\"min_score\":0.5}]";
const ALERT_COOLDOWN_SECONDS: u64 = 1;
const ALERT_FRAME_SIZE: u32 = 32;

#[tokio::test]
async fn analyticsalert_fires_once_per_cooldown_and_paints_the_border() {
    let mut alert = AnalyticsAlert::new();
    alert
        .set_property("rules", PropValue::Str(String::from(ALERT_RULES)))
        .expect("the rules parse");
    alert
        .set_property("cooldown", PropValue::Uint(ALERT_COOLDOWN_SECONDS))
        .expect("cooldown is settable");
    alert
        .configure_pipeline(&rgba_caps(
            ALERT_FRAME_SIZE,
            ALERT_FRAME_SIZE,
            Rate::Fixed(RECORD_FPS << Q16_SHIFT),
        ))
        .expect("alert accepts RGBA");

    // A person over the threshold, then the same one a half second later (inside
    // the cooldown), then a second past the first (outside it).
    let cooldown_ns = ALERT_COOLDOWN_SECONDS * 1_000_000_000;
    let times = [0, cooldown_ns / 2, cooldown_ns];
    let mut out = RecordingSink::default();
    for pts_ns in times {
        let mut frame = black_frame(ALERT_FRAME_SIZE, ALERT_FRAME_SIZE, pts_ns);
        let mut analytics = AnalyticsMeta::new();
        analytics.set_class_names(CLASS_NAMES.iter().copied());
        analytics.add_detection(detection(&RECORDED[0]));
        frame.meta.attach(analytics);
        alert
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("the frame passes through");
    }
    let frames = out.frames();
    assert_eq!(frames.len(), times.len());
    let fired: Vec<bool> = frames
        .iter()
        .map(|frame| {
            frame
                .meta
                .get::<BlobMeta>()
                .and_then(|blobs| blobs.get("alert"))
                .is_some()
        })
        .collect();
    assert_eq!(
        fired,
        [true, false, true],
        "the cooldown is measured in stream time"
    );
    assert_eq!(alert.fired(), 2);

    let alerted = frames[0].domain.as_system_slice().expect("system pixels");
    let untouched = frames[1].domain.as_system_slice().expect("system pixels");
    let centre = (ALERT_FRAME_SIZE as usize / 2 * ALERT_FRAME_SIZE as usize
        + ALERT_FRAME_SIZE as usize / 2)
        * 4;
    assert_eq!(
        alerted[..4],
        [255u8, 0, 0, 255],
        "the border is painted red"
    );
    assert_eq!(
        alerted[centre..centre + 4],
        untouched[centre..centre + 4],
        "the picture inside the border is left alone"
    );
}

/// The clip the recorder is driven to write: 10 frames at this rate, with the
/// alert on one of them.
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_FRAME_SIZE: u32 = 16;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_FRAMES: u64 = 10;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_FPS: u32 = 10;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_FRAME_PERIOD_NS: u64 = 1_000_000_000 / CLIP_FPS as u64;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_ALERT_FRAME: u64 = 3;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_SECONDS_BEFORE: f64 = 0.2;
#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
const CLIP_SECONDS_AFTER: f64 = 0.2;

#[cfg(all(feature = "mjpeg-encode", feature = "mjpeg"))]
#[tokio::test]
async fn alertrecorder_writes_a_clip_that_decodes_back() {
    use g2g_plugins::alertrecorder::AlertRecorder;
    use g2g_plugins::appsink::{register_appsink_pull, Pull};

    let directory = std::env::temp_dir().join("m1176_alert_clips");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).expect("a directory for the clips");

    let mut recorder = AlertRecorder::new()
        .with_location(
            directory
                .join("alert-%s.avi")
                .to_string_lossy()
                .into_owned(),
        )
        .with_seconds_before(CLIP_SECONDS_BEFORE)
        .with_seconds_after(CLIP_SECONDS_AFTER);
    let caps = rgba_caps(
        CLIP_FRAME_SIZE,
        CLIP_FRAME_SIZE,
        Rate::Fixed(CLIP_FPS << Q16_SHIFT),
    );
    recorder
        .configure_pipeline(&caps)
        .expect("the recorder accepts the caps");
    let mut out = RecordingSink::default();
    for index in 0..CLIP_FRAMES {
        let mut frame = black_frame(
            CLIP_FRAME_SIZE,
            CLIP_FRAME_SIZE,
            index * CLIP_FRAME_PERIOD_NS,
        );
        if index == CLIP_ALERT_FRAME {
            let mut blobs = BlobMeta::new();
            blobs.push("alert", Vec::from(b"[]".as_slice()));
            frame.meta.attach(blobs);
        }
        recorder
            .process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("frames pass through the recorder");
    }
    recorder
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("the recorder closes its clip");
    assert_eq!(
        out.frames().len(),
        CLIP_FRAMES as usize,
        "identity on frames"
    );
    assert_eq!(recorder.clips_written(), 1);

    let written: Vec<PathBuf> = std::fs::read_dir(&directory)
        .expect("the clip directory")
        .map(|entry| entry.expect("a directory entry").path())
        .collect();
    assert_eq!(written.len(), 1, "one alert, one clip");
    let clip = &written[0];
    let name = clip.file_name().expect("a file name").to_string_lossy();
    assert!(
        name.starts_with("alert-") && name.ends_with(".avi"),
        "{name} is the location with its %s filled in"
    );

    // The window the ring and the tail cover: every frame from
    // `seconds-before` ahead of the alert to `seconds-after` behind it.
    let alert_ns = CLIP_ALERT_FRAME * CLIP_FRAME_PERIOD_NS;
    let first = alert_ns - (CLIP_SECONDS_BEFORE * 1e9) as u64;
    let last = alert_ns + (CLIP_SECONDS_AFTER * 1e9) as u64;
    let expected = (0..CLIP_FRAMES)
        .map(|index| index * CLIP_FRAME_PERIOD_NS)
        .filter(|pts| *pts >= first && *pts <= last)
        .count();

    let channel = "m1176-clip-readback";
    let pull = register_appsink_pull(channel);
    // The demuxer announces its stream before it has read the file, so the
    // clip's codec is named on the line.
    let line = format!(
        "filesrc location={} ! avidemux stream=mjpeg ! mjpegdec ! appsink channel={channel}",
        clip.to_string_lossy()
    );
    let graph = g2g_core::runtime::parse_launch(&g2g_plugins::registry::default_registry(), &line)
        .expect("the read-back pipeline builds");
    let clock = g2g_plugins::clock::WallClock::new();
    g2g_core::runtime::run_graph(graph, &clock, 4)
        .await
        .expect("the clip decodes");
    let mut decoded = 0;
    while let Pull::Frame(_) = pull.try_pull() {
        decoded += 1;
    }
    assert_eq!(decoded, expected, "the clip covers the alert's window");
    let _ = std::fs::remove_dir_all(&directory);
}
