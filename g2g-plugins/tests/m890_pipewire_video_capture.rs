//! Live interop test for `PipeWireVideoSrc` (M890): capture from a node
//! published by GStreamer's `pipewiresink`, the reference PipeWire peer.
//!
//! Needs a running user PipeWire daemon (plus its session manager, which makes
//! the link), `gst-launch-1.0` with the `pipewire` plugin, and `pw-dump`. Each
//! test publishes its own uniquely named node, waits for it to show up in the
//! registry, captures from it by name and checks the negotiated format, the
//! plane-exact frame size and the frame count. Without those host pieces the
//! tests report what is missing and pass vacuously.
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
use g2g_core::{Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat};
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
        let node = format!("g2g-m890-{}-{}", std::process::id(), tag);
        let child = Command::new("gst-launch-1.0")
            .args([
                "-q",
                "videotestsrc",
                "is-live=true",
                "!",
                &format!(
                    "video/x-raw,format={format},width={width},height={height},framerate={fps}/1"
                ),
                "!",
                "pipewiresink",
                "mode=provide",
                &format!("stream-properties=p,node.name={node},media.class=Video/Source"),
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

/// Capture `limit` frames from `node` with the requested geometry / rate.
async fn capture(
    node: &str,
    req: (u32, u32, u32),
    limit: u64,
) -> (Vec<PipelinePacket>, Result<u64, G2gError>) {
    let (w, h, fps) = req;
    let mut src = PipeWireVideoSrc::new()
        .with_target(node)
        .with_size(w, h)
        .with_fps(fps)
        .with_frame_limit(limit);
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

    let (packets, result) = capture(&pub_.node, (320, 240, 30), 12).await;
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
    let (packets, result) = capture(&pub_.node, (640, 480, 15), 8).await;
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

    let (packets, result) = capture(&pub_.node, (320, 240, 30), 8).await;
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
