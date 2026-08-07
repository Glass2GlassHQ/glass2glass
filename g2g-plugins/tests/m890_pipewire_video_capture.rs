//! Live interop tests for the PipeWire capture elements (M890, M894): capture
//! from a node published by GStreamer's `pipewiresink`, the reference PipeWire
//! peer.
//!
//! Needs a running user PipeWire daemon (plus its session manager, which makes
//! the link), `gst-launch-1.0` with the `pipewire` plugin, and `pw-dump`. Each
//! test publishes its own uniquely named node, waits for it to show up in the
//! registry, captures from it by name and checks the negotiated format, the
//! plane-exact frame size and the frame count. Without those host pieces the
//! tests report what is missing and pass vacuously.
//!
//! M894 adds the property paths: a pinned video `format` (honoured, or a loud
//! failure when the node cannot produce it) and an audio capture driven entirely
//! through `PipeWireSrc`'s runtime properties.
//!
//! ```sh
//! cargo test -p g2g-plugins --features pipewire \
//!     --test m890_pipewire_video_capture -- --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "pipewire"))]

use std::process::{Child, Command, Stdio};
use std::time::Duration;

use g2g_core::element::{BoxFuture, PushOutcome};
use g2g_core::frame::PipelinePacket;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::SourceLoop;
use g2g_core::{AudioFormat, Caps, Dim, G2gError, OutputSink, PropValue, Rate, RawVideoFormat};
use g2g_plugins::pipewiresrc::PipeWireSrc;
use g2g_plugins::pipewirevideosrc::PipeWireVideoSrc;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// A `gst-launch-1.0 ... ! pipewiresink` publishing one node, killed on every
/// exit path (including a panicking assertion).
struct Publisher {
    child: Child,
    node: String,
}

impl Drop for Publisher {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Publisher {
    /// Publish `format` at `width`x`height` / `fps` on a node named after `tag`.
    /// `mode=provide` makes the sink offer a node others capture from instead of
    /// connecting to one.
    fn spawn(tag: &str, format: &str, width: u32, height: u32, fps: u32) -> Option<Self> {
        Self::spawn_gst(
            tag,
            "Video/Source",
            &[
                "videotestsrc",
                "is-live=true",
                "!",
                &format!(
                    "video/x-raw,format={format},width={width},height={height},framerate={fps}/1"
                ),
            ],
        )
    }

    /// Publish an interleaved PCM audio node named after `tag`. Not live: a
    /// provide-mode audio node is pulled by the graph it drives, and a
    /// realtime-throttled source stalls that pull (even `pw-record` then
    /// captures nothing).
    fn spawn_audio(tag: &str, format: &str, rate: u32, channels: u8) -> Option<Self> {
        Self::spawn_gst(
            tag,
            "Audio/Source",
            &[
                "audiotestsrc",
                "!",
                &format!(
                    "audio/x-raw,format={format},rate={rate},channels={channels},layout=interleaved"
                ),
            ],
        )
    }

    fn spawn_gst(tag: &str, media_class: &str, head: &[&str]) -> Option<Self> {
        let node = format!("g2g-m890-{}-{}", std::process::id(), tag);
        let child = Command::new("gst-launch-1.0")
            .arg("-q")
            .args(head)
            .args([
                "!",
                "pipewiresink",
                "mode=provide",
                &format!("stream-properties=p,node.name={node},media.class={media_class}"),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        Some(Self { child, node })
    }

    /// Wait for the node to appear in the PipeWire registry. Polls rather than
    /// sleeping a fixed time, so startup order does not matter.
    async fn wait_until_registered(&self) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let needle = format!("\"node.name\": \"{}\"", self.node);
        while std::time::Instant::now() < deadline {
            if let Ok(out) = Command::new("pw-dump").stderr(Stdio::null()).output() {
                if String::from_utf8_lossy(&out.stdout).contains(&needle) {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        false
    }
}

/// `false` (with a reason on stderr) when the host cannot run a live capture.
fn host_can_run() -> bool {
    let socket = std::env::var("XDG_RUNTIME_DIR")
        .map(|dir| std::path::Path::new(&dir).join("pipewire-0"))
        .map(|p| p.exists())
        .unwrap_or(false);
    if !socket {
        eprintln!("skipping: no pipewire-0 socket in XDG_RUNTIME_DIR");
        return false;
    }
    for tool in ["gst-launch-1.0", "pw-dump"] {
        if Command::new(tool)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            eprintln!("skipping: {tool} is not installed");
            return false;
        }
    }
    if !Command::new("gst-inspect-1.0")
        .arg("pipewiresink")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        eprintln!("skipping: gstreamer has no pipewiresink element");
        return false;
    }
    true
}

/// Capture `limit` frames from `node` with the requested geometry / rate, with
/// the format left open (`None`) or pinned through the `format` property.
async fn capture(
    node: &str,
    req: (u32, u32, u32),
    limit: u64,
    pin: Option<&str>,
) -> (Vec<PipelinePacket>, Result<u64, G2gError>) {
    let (w, h, fps) = req;
    let mut src = PipeWireVideoSrc::new()
        .with_target(node)
        .with_size(w, h)
        .with_fps(fps)
        .with_frame_limit(limit);
    if let Some(format) = pin {
        src.set_property("format", PropValue::Str(format.to_string()))
            .expect("format is a known property");
    }
    let advertised = src.intercept_caps().await.expect("advertised caps");
    src.configure_pipeline(&advertised).expect("configure");
    let mut out = Collect::default();
    let result = tokio::time::timeout(Duration::from_secs(30), src.run(&mut out))
        .await
        .expect("capture finishes within 30s");
    (out.packets, result)
}

fn frame_sizes(packets: &[PipelinePacket]) -> Vec<usize> {
    packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => match &f.domain {
                MemoryDomain::System(s) => Some(s.as_slice().len()),
                other => panic!("captured frame is not in system memory: {other:?}"),
            },
            _ => None,
        })
        .collect()
}

fn first_caps_change(packets: &[PipelinePacket]) -> Option<(usize, Caps)> {
    packets.iter().enumerate().find_map(|(i, p)| match p {
        PipelinePacket::CapsChanged(c) => Some((i, c.clone())),
        _ => None,
    })
}

fn first_frame_index(packets: &[PipelinePacket]) -> Option<usize> {
    packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)))
}

fn raw_caps(format: RawVideoFormat, w: u32, h: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

/// The node produces exactly what was requested: I420 320x240 frames arrive
/// plane-exact (115200 bytes) and nothing renegotiates.
#[tokio::test]
async fn captures_i420_frames_from_a_gstreamer_node() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn("i420", "I420", 320, 240, 30) {
        Some(p) => p,
        None => {
            eprintln!("skipping: gst-launch-1.0 would not start");
            return;
        }
    };
    assert!(
        pub_.wait_until_registered().await,
        "publisher node never reached the registry"
    );

    let (packets, result) = capture(&pub_.node, (320, 240, 30), 12, None).await;
    let frames = frame_sizes(&packets);
    eprintln!(
        "captured {} frames, sizes {:?}, caps change {:?}",
        frames.len(),
        frames.first(),
        first_caps_change(&packets).map(|(_, c)| c)
    );
    assert_eq!(result, Ok(12), "source should emit the requested frames");
    assert_eq!(frames.len(), 12, "every emitted frame reaches the sink");
    assert!(
        frames.iter().all(|n| *n == 320 * 240 * 3 / 2),
        "I420 320x240 frames are 115200 bytes: {frames:?}"
    );
    assert_eq!(
        first_caps_change(&packets),
        None,
        "the node matched the request, so nothing renegotiates"
    );
    assert!(
        matches!(packets.last(), Some(PipelinePacket::Eos)),
        "a frame-limited capture ends with EOS"
    );
}

/// The node keeps its own geometry and rate: the element re-advertises them with
/// `CapsChanged` before the first frame, and the frames are the node's size.
#[tokio::test]
async fn node_geometry_arrives_as_caps_changed() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn("geom", "I420", 320, 240, 30) {
        Some(p) => p,
        None => return,
    };
    assert!(pub_.wait_until_registered().await, "node never registered");

    // ask for something the node will not produce
    let (packets, result) = capture(&pub_.node, (640, 480, 15), 8, None).await;
    let frames = frame_sizes(&packets);
    let change = first_caps_change(&packets);
    eprintln!("caps change {:?}, first frame {:?}", change, frames.first());
    assert_eq!(result, Ok(8));
    let (change_at, caps) = change.expect("the node's geometry is re-advertised");
    assert_eq!(caps, raw_caps(RawVideoFormat::I420, 320, 240, 30));
    assert!(
        change_at < first_frame_index(&packets).expect("frames arrived"),
        "CapsChanged precedes the first frame it describes"
    );
    assert!(frames.iter().all(|n| *n == 320 * 240 * 3 / 2));
}

/// A node that only offers packed YUY2 negotiates our `Yuyv` mapping (the pod
/// prefers I420 but offers the whole table), and the frames are 2 bytes/pixel.
#[tokio::test]
async fn a_yuy2_node_negotiates_the_packed_format() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn("yuy2", "YUY2", 320, 240, 30) {
        Some(p) => p,
        None => return,
    };
    assert!(pub_.wait_until_registered().await, "node never registered");

    let (packets, result) = capture(&pub_.node, (320, 240, 30), 8, None).await;
    let frames = frame_sizes(&packets);
    eprintln!(
        "caps change {:?}, first frame {:?}",
        first_caps_change(&packets).map(|(_, c)| c),
        frames.first()
    );
    assert_eq!(result, Ok(8));
    let (_, caps) = first_caps_change(&packets).expect("YUY2 is not the advertised format");
    assert_eq!(caps, raw_caps(RawVideoFormat::Yuyv, 320, 240, 30));
    assert!(
        frames.iter().all(|n| *n == 320 * 240 * 2),
        "packed YUYV 320x240 frames are 153600 bytes: {frames:?}"
    );
}

/// `format=yuy2` against a YUY2 node: the advertised caps already carry the
/// pinned format, so the capture starts on it and nothing renegotiates.
#[tokio::test]
async fn a_pinned_format_negotiates_without_a_caps_change() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn("pin-yuy2", "YUY2", 320, 240, 30) {
        Some(p) => p,
        None => return,
    };
    assert!(pub_.wait_until_registered().await, "node never registered");

    let (packets, result) = capture(&pub_.node, (320, 240, 30), 8, Some("yuy2")).await;
    let frames = frame_sizes(&packets);
    eprintln!(
        "pinned yuy2: caps change {:?}, first frame {:?}",
        first_caps_change(&packets).map(|(_, c)| c),
        frames.first()
    );
    assert_eq!(result, Ok(8));
    assert_eq!(
        first_caps_change(&packets),
        None,
        "the pinned format is what was advertised, so nothing renegotiates"
    );
    assert!(
        frames.iter().all(|n| *n == 320 * 240 * 2),
        "packed YUYV 320x240 frames are 153600 bytes: {frames:?}"
    );
}

/// `format=rgba` against a node that only offers I420: the connect pod carries
/// RGBA alone, so the capture fails instead of quietly taking I420 or waiting for
/// frames that never come.
#[tokio::test]
async fn a_pinned_format_the_node_cannot_produce_fails_loud() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn("pin-mismatch", "I420", 320, 240, 30) {
        Some(p) => p,
        None => return,
    };
    assert!(pub_.wait_until_registered().await, "node never registered");

    // `capture` bounds the run, so a hang fails the test rather than blocking it
    let (packets, result) = capture(&pub_.node, (320, 240, 30), 8, Some("rgba")).await;
    eprintln!("pinned rgba against an I420 node: {result:?}");
    assert!(
        result.is_err(),
        "an unproducible pinned format must fail the capture: {result:?}"
    );
    assert!(
        frame_sizes(&packets).is_empty(),
        "no frames are pushed under a format the node never agreed to"
    );
}

/// The audio source's runtime properties reach a live stream: `target-object`
/// picks the published node and `format` / `samplerate` / `channels` decide the
/// format the stream connects with, `num-buffers` ends the capture. The publisher
/// is mono F32LE 44.1 kHz, none of which is a `PipeWireSrc` default, and two
/// client nodes link without a converter between them, so frames only arrive if
/// every one of those properties reached the connect pod.
#[tokio::test]
async fn audio_properties_drive_a_live_capture() {
    if !host_can_run() {
        return;
    }
    let pub_ = match Publisher::spawn_audio("audio-props", "F32LE", 44_100, 1) {
        Some(p) => p,
        None => return,
    };
    assert!(pub_.wait_until_registered().await, "node never registered");

    let mut src = PipeWireSrc::new();
    for (name, value) in [
        ("target-object", PropValue::Str(pub_.node.clone())),
        ("format", PropValue::Str("F32LE".to_string())),
        ("samplerate", PropValue::Uint(44_100)),
        ("channels", PropValue::Uint(1)),
        ("num-buffers", PropValue::Int(8)),
    ] {
        src.set_property(name, value).expect("known property");
    }
    let advertised = src.intercept_caps().await.expect("advertised caps");
    assert_eq!(
        advertised,
        Caps::Audio {
            format: AudioFormat::PcmF32Le,
            channels: 1,
            sample_rate: 44_100,
        },
        "the properties are what the element advertises"
    );
    src.configure_pipeline(&advertised).expect("configure");

    let mut out = Collect::default();
    let result = tokio::time::timeout(Duration::from_secs(30), src.run(&mut out))
        .await
        .expect("capture finishes within 30s");
    let sizes = frame_sizes(&out.packets);
    eprintln!("audio buffers {sizes:?}, run {result:?}");
    assert_eq!(result, Ok(8), "the num-buffers limit ends the capture");
    assert_eq!(sizes.len(), 8);
    // mono F32LE is 4 bytes per sample frame
    assert!(
        sizes.iter().all(|n| *n % 4 == 0 && *n > 0),
        "mono F32LE buffers are whole 4-byte sample frames: {sizes:?}"
    );
    assert!(
        matches!(out.packets.last(), Some(PipelinePacket::Eos)),
        "a bounded capture ends with EOS"
    );
}
