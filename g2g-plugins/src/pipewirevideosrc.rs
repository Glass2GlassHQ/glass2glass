//! Linux PipeWire video capture source. Connects an input stream to a PipeWire
//! video node (camera, another client's published node, a screen-capture node)
//! and streams raw frames downstream in system memory. The video sibling of the
//! audio [`PipeWireSrc`](crate::pipewiresrc::PipeWireSrc).
//!
//! Like the audio element, PipeWire's callback-driven main loop is pinned to one
//! thread, so it runs on a dedicated worker that feeds the async `run` loop over
//! a channel.
//!
//! ## Negotiation
//!
//! `intercept_caps` publishes fixed caps built from the properties (`I420` at the
//! requested geometry / rate), but the connect pod offers our whole format table
//! and an open size / framerate range, so a node with its own fixed mode still
//! links. What the node settled on arrives in `param_changed`, and when it
//! differs from the advertised caps the element emits `CapsChanged` before the
//! first frame. A format outside the table fails the capture instead of being
//! reinterpreted.
//!
//! ## Scope
//!
//! System memory only: the stream connects with `MAP_BUFFERS` and copies each
//! frame out of the mapped block, de-striding padded rows. DMABUF import is not
//! wired up. There is no xdg-desktop-portal integration either, so screen
//! capture means driving the portal elsewhere and naming the node it opens in
//! `target-object`, which takes either form PipeWire's `target.object` resolves:
//! a node name or an object serial.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

use pipewire as pw;
use pw::spa;

use crate::videoconvert::{raw_format_from_str, raw_format_to_str};

use crate::pwvideo::{
    format_pod_bytes, rate_q16, spa_format, supported_formats, PlaneLayout, VideoInfo, MAX_DIM,
};

/// Requested capture geometry / rate when the caller does not specify one.
const DEFAULT_WIDTH: u32 = 640;
const DEFAULT_HEIGHT: u32 = 480;
const DEFAULT_FPS: u32 = 30;
/// The format the advertised caps carry and the connect pod prefers.
const PREFERRED_FORMAT: RawVideoFormat = RawVideoFormat::I420;

/// Control message to the loop thread (quit on teardown).
enum Ctrl {
    Terminate,
}

/// Loop-thread to element messages.
enum FromWorker {
    /// The negotiated format, sent on every change (so before the first frame).
    Format(VideoInfo),
    /// One tightly packed frame.
    Frame(Vec<u8>),
    /// The stream negotiated something we cannot carry, handed us a buffer that
    /// disagrees with the negotiated geometry, or went to the error state (a
    /// pinned format the node cannot produce lands here).
    Failed(G2gError),
}

#[derive(Debug)]
pub struct PipeWireVideoSrc {
    /// Node to capture from (`node.name` or object serial); empty = the default
    /// video source the session manager picks.
    target: String,
    req_width: u32,
    req_height: u32,
    req_fps: u32,
    /// Pinned capture format: the connect pod then offers this one alone, so the
    /// node either produces it or negotiation fails. `None` = offer the whole
    /// table and take what the node settles on.
    pin_format: Option<RawVideoFormat>,
    /// `None` = run until error or downstream shutdown; else stop after N frames
    /// and emit EOS. The bounded-capture / test path.
    frame_limit: Option<u64>,
    configured: bool,
}

impl Default for PipeWireVideoSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeWireVideoSrc {
    /// Capture from the default video node at 640x480 / 30.
    pub fn new() -> Self {
        Self {
            target: String::new(),
            req_width: DEFAULT_WIDTH,
            req_height: DEFAULT_HEIGHT,
            req_fps: DEFAULT_FPS,
            pin_format: None,
            frame_limit: None,
            configured: false,
        }
    }

    /// Capture from a specific node: its `node.name` or its object serial.
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = target.into();
        self
    }

    /// Request a capture size. The node may keep its own; the negotiated caps
    /// (and a `CapsChanged`) reflect what it actually produces.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.req_width = width;
        self.req_height = height;
        self
    }

    /// Request a frame rate in fps. Best-effort, same as the size.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.req_fps = fps;
        self
    }

    /// Pin the capture format. Unlike the size and rate this is not best effort:
    /// the connect pod offers this format alone, so a node that cannot produce it
    /// fails the capture and no mid-stream format change can arrive.
    pub fn with_format(mut self, format: RawVideoFormat) -> Self {
        self.pin_format = Some(format);
        self
    }

    /// Stop after `n` frames and emit EOS (`0` = no limit). Without this the
    /// source runs until an error or until downstream drops.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = (n > 0).then_some(n);
        self
    }

    /// The format the caps advertise and the connect pod leads with: the pinned
    /// one, or the default preference. A pin the element cannot map fails here.
    fn format(&self) -> Result<RawVideoFormat, G2gError> {
        let format = self.pin_format.unwrap_or(PREFERRED_FORMAT);
        spa_format(format).ok_or(G2gError::CapsMismatch)?;
        Ok(format)
    }

    /// The caps the element advertises before the node has answered. Fixed, so
    /// `fixate()` has nothing left to do.
    fn caps(&self) -> Result<Caps, G2gError> {
        if self.req_width == 0
            || self.req_height == 0
            || self.req_width > MAX_DIM
            || self.req_height > MAX_DIM
        {
            return Err(G2gError::CapsMismatch);
        }
        Ok(Caps::RawVideo {
            format: self.format()?,
            width: Dim::Fixed(self.req_width),
            height: Dim::Fixed(self.req_height),
            framerate: Rate::Fixed(rate_q16(self.req_fps, 1)),
            interlace: g2g_core::Interlace::Any,
        })
    }
}

impl SourceLoop for PipeWireVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.caps())
    }

    /// Produces the requested caps, so a chain built on a PipeWire camera takes
    /// the native arc-consistency path (mirrors `V4l2Src`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(
            self.caps()
                .map(|c| CapsConstraint::Produces(CapsSet::one(c))),
        )
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "PipeWire video source",
            "Source/Video",
            "Captures raw video from a PipeWire node",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "target-object",
                PropKind::Str,
                "node name or object serial to capture from (empty = default)",
            ),
            PropertySpec::new(
                "width",
                PropKind::Uint,
                "requested capture width (the node may keep its own)",
            )
            .with_default("640"),
            PropertySpec::new(
                "height",
                PropKind::Uint,
                "requested capture height (the node may keep its own)",
            )
            .with_default("480"),
            PropertySpec::new(
                "framerate",
                PropKind::Uint,
                "requested capture rate, fps (best effort)",
            )
            .with_default("30"),
            PropertySpec::new(
                "format",
                PropKind::Str,
                "pin the capture format: I420 | NV12 | YUY2 | RGBA | BGRA (empty = whatever the node offers)",
            )
            .with_default(""),
            PropertySpec::new(
                "num-buffers",
                PropKind::Int,
                "frames to capture then EOS (-1 = forever)",
            )
            .with_default("-1"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "target-object" => {
                self.target = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "width" => {
                self.req_width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.req_height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.req_fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "format" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.pin_format = if s.is_empty() {
                    None
                } else {
                    // a name outside the SPA mapping table is an error, never a
                    // silent fall back to the default format
                    let format = raw_format_from_str(s).ok_or(PropError::Value)?;
                    spa_format(format).ok_or(PropError::Value)?;
                    Some(format)
                };
                Ok(())
            }
            "num-buffers" => {
                let n = value.as_int().ok_or(PropError::Type)?;
                self.frame_limit = (n >= 0).then_some(n as u64);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "target-object" => Some(PropValue::Str(self.target.clone())),
            "width" => Some(PropValue::Uint(self.req_width as u64)),
            "height" => Some(PropValue::Uint(self.req_height as u64)),
            "framerate" => Some(PropValue::Uint(self.req_fps as u64)),
            "format" => Some(PropValue::Str(
                self.pin_format.map_or("", raw_format_to_str).into(),
            )),
            "num-buffers" => Some(PropValue::Int(self.frame_limit.map_or(-1, |n| n as i64))),
            _ => None,
        }
    }

    /// Live source: contributes one frame period so the sink keeps a frame in
    /// hand and never runs dry waiting on capture (same as `V4l2Src`).
    fn latency(&self) -> LatencyReport {
        let period_ns = if self.req_fps > 0 {
            1_000_000_000 / u64::from(self.req_fps)
        } else {
            0
        };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut advertised = self.caps()?;
            let pod = format_pod_bytes(
                self.format()?,
                self.pin_format.is_some(),
                self.req_width,
                self.req_height,
                self.req_fps,
            )?;
            let target = self.target.clone();
            let limit = self.frame_limit;

            // Frames and format changes cross from the loop thread to here.
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<FromWorker>();
            // Control + a setup-result handshake (surface a connect failure).
            let (ctrl_tx, ctrl_rx) = pw::channel::channel::<Ctrl>();
            let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<Result<(), i32>>(1);

            let handle = std::thread::Builder::new()
                .name(String::from("g2g-pipewirevideosrc"))
                .spawn(move || {
                    if let Err(code) = build_and_run(&target, &pod, tx, ctrl_rx, &ready_tx) {
                        let _ = ready_tx.send(Err(code));
                    }
                })
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

            // Block briefly for the stream to connect (the daemon round-trip).
            let connected = match ready_rx.recv_timeout(std::time::Duration::from_secs(5)) {
                Ok(Ok(())) => Ok(()),
                Ok(Err(code)) => Err(G2gError::Hardware(HardwareError::PipeWire(code))),
                Err(_) => Err(G2gError::Hardware(HardwareError::PipeWire(-1))),
            };
            if let Err(e) = connected {
                let _ = ctrl_tx.send(Ctrl::Terminate);
                let _ = handle.join();
                return Err(e);
            }

            let mut period_ns = if self.req_fps > 0 {
                1_000_000_000 / u64::from(self.req_fps)
            } else {
                0
            };
            let mut seq = 0u64;
            let mut pts = 0u64;
            let mut downstream_open = true;
            let mut failure = None;

            while limit.is_none_or(|n| seq < n) {
                let Some(msg) = rx.recv().await else {
                    break; // worker ended
                };
                match msg {
                    FromWorker::Failed(e) => {
                        failure = Some(e);
                        break;
                    }
                    FromWorker::Format(info) => {
                        period_ns = info.frame_period_ns();
                        let caps = info.caps();
                        if caps != advertised {
                            if out
                                .push(PipelinePacket::CapsChanged(caps.clone()))
                                .await
                                .is_err()
                            {
                                downstream_open = false;
                                break;
                            }
                            advertised = caps;
                        }
                    }
                    FromWorker::Frame(bytes) => {
                        let arrival_ns = g2g_core::metrics::monotonic_ns();
                        let frame = Frame {
                            domain: MemoryDomain::System(SystemSlice::from_boxed(
                                bytes.into_boxed_slice(),
                            )),
                            timing: FrameTiming {
                                pts_ns: pts,
                                dts_ns: pts,
                                duration_ns: period_ns,
                                capture_ns: pts,
                                arrival_ns,
                                // raw frames are each independently presentable
                                keyframe: true,
                            },
                            sequence: seq,
                            meta: Default::default(),
                        };
                        if out.push(PipelinePacket::DataFrame(frame)).await.is_err() {
                            downstream_open = false;
                            break;
                        }
                        pts += period_ns;
                        seq += 1;
                    }
                }
            }

            // Stop the loop and reap the worker.
            let _ = ctrl_tx.send(Ctrl::Terminate);
            let _ = handle.join();

            if let Some(e) = failure {
                return Err(e);
            }
            if downstream_open {
                out.push(PipelinePacket::Eos).await?;
            }
            Ok(seq)
        })
    }
}

impl PadTemplates for PipeWireVideoSrc {
    /// Produces any raw format the element can map; a constructed instance fixes
    /// the geometry / rate from its properties and the format at negotiation.
    fn pad_templates() -> Vec<PadTemplate> {
        let alternatives: Vec<Caps> = supported_formats()
            .map(|format| Caps::RawVideo {
                format,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            })
            .collect();
        Vec::from([PadTemplate::source(CapsSet::from_alternatives(
            alternatives,
        ))])
    }
}

// =================================================================
// Worker thread: the PipeWire capture main loop
// =================================================================

/// What the `process` callback needs to turn a buffer into a frame, refreshed by
/// `param_changed`. `None` until the format is negotiated (or after a failure,
/// which stops further buffers being interpreted).
struct Negotiated {
    info: VideoInfo,
    layout: PlaneLayout,
}

struct UserData {
    negotiated: Option<Negotiated>,
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
}

fn build_and_run(
    target: &str,
    pod: &[u8],
    tx: tokio::sync::mpsc::UnboundedSender<FromWorker>,
    ctrl_rx: pw::channel::Receiver<Ctrl>,
    ready: &std::sync::mpsc::SyncSender<Result<(), i32>>,
) -> Result<(), i32> {
    pw::init();
    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|_| -1)?;
    let context = pw::context::Context::new(&mainloop).map_err(|_| -1)?;
    let core = context.connect(None).map_err(|_| -1)?;

    // media.type is what the session manager's policy matches on, so it has to
    // be here for the link to be made at all.
    let mut props = pw::properties::properties! {
        *pw::keys::MEDIA_TYPE => "Video",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Camera",
    };
    if !target.is_empty() {
        // spelled out because pipewire-rs gates its TARGET_OBJECT constant
        // behind a crate feature this build does not enable
        props.insert("target.object", target);
    }
    let stream = pw::stream::Stream::new(&core, "g2g-pipewirevideosrc", props).map_err(|_| -1)?;

    let user_data = UserData {
        negotiated: None,
        tx,
    };

    let _listener = stream
        .add_local_listener_with_user_data(user_data)
        // A format the node cannot produce (a pinned one it does not offer) is
        // reported as a stream error, not a failed connect: surface it so the
        // capture fails instead of waiting for frames that never come.
        .state_changed(|_, user_data, _old, new| {
            if let pw::stream::StreamState::Error(_) = new {
                let _ = user_data
                    .tx
                    .send(FromWorker::Failed(G2gError::Hardware(
                        HardwareError::PipeWire(-1),
                    )));
            }
        })
        .param_changed(|_, user_data, id, param| {
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                user_data.negotiated = None;
                return;
            };
            match parse_format(param) {
                Ok(info) => {
                    let changed = user_data.negotiated.as_ref().map(|n| n.info) != Some(info);
                    // geometry already bounded by VideoInfo, so a layout exists
                    let Some(layout) = PlaneLayout::new(info.format, info.width, info.height)
                    else {
                        user_data.negotiated = None;
                        let _ = user_data
                            .tx
                            .send(FromWorker::Failed(G2gError::CapsMismatch));
                        return;
                    };
                    user_data.negotiated = Some(Negotiated { info, layout });
                    if changed {
                        let _ = user_data.tx.send(FromWorker::Format(info));
                    }
                }
                Err(e) => {
                    user_data.negotiated = None;
                    let _ = user_data.tx.send(FromWorker::Failed(e));
                }
            }
        })
        .process(|stream, user_data| {
            let Some(negotiated) = user_data.negotiated.as_ref() else {
                return; // no usable format: drop the buffer rather than guess
            };
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let data = &mut datas[0];
            let (offset, size, stride) = {
                let chunk = data.chunk();
                (
                    chunk.offset() as usize,
                    chunk.size() as usize,
                    usize::try_from(chunk.stride()).unwrap_or(0),
                )
            };
            // An empty chunk is a normal tick (the node produced nothing), not a
            // malformed buffer.
            if size == 0 {
                return;
            }
            let Some(mapped) = data.data() else {
                return;
            };
            let fits = offset
                .checked_add(size)
                .is_some_and(|end| end <= mapped.len());
            let mut frame = Vec::with_capacity(negotiated.layout.frame_bytes());
            if !fits
                || negotiated
                    .layout
                    .copy_tight(&mapped[offset..offset + size], stride, &mut frame)
                    .is_none()
            {
                // The buffer disagrees with the negotiated geometry: fail the
                // capture instead of pushing a malformed frame downstream.
                let _ = user_data
                    .tx
                    .send(FromWorker::Failed(G2gError::CapsMismatch));
                user_data.negotiated = None;
                return;
            }
            let _ = user_data.tx.send(FromWorker::Frame(frame));
        })
        .register()
        .map_err(|_| -1)?;

    let mut params = [spa::pod::Pod::from_bytes(pod).ok_or(-1)?];
    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|_| -1)?;

    let weak = mainloop.downgrade();
    let _recv = ctrl_rx.attach(mainloop.loop_(), move |_ctrl| {
        if let Some(ml) = weak.upgrade() {
            ml.quit();
        }
    });

    let _ = ready.send(Ok(()));
    mainloop.run();
    Ok(())
}

/// Parse a fixated `Format` param into the negotiated geometry / format.
fn parse_format(param: &spa::pod::Pod) -> Result<VideoInfo, G2gError> {
    let (media_type, media_subtype) =
        spa::param::format_utils::parse_format(param).map_err(|_| G2gError::CapsMismatch)?;
    if media_type != spa::param::format::MediaType::Video
        || media_subtype != spa::param::format::MediaSubtype::Raw
    {
        return Err(G2gError::CapsMismatch);
    }
    let mut raw = spa::param::video::VideoInfoRaw::new();
    raw.parse(param).map_err(|_| G2gError::CapsMismatch)?;
    VideoInfo::from_spa(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_requested_config() {
        let src = PipeWireVideoSrc::new()
            .with_target("g2g-node")
            .with_size(1280, 720)
            .with_fps(60)
            .with_format(RawVideoFormat::Nv12)
            .with_frame_limit(7);
        assert_eq!(src.target, "g2g-node");
        assert_eq!(
            (src.req_width, src.req_height, src.req_fps),
            (1280, 720, 60)
        );
        assert_eq!(src.pin_format, Some(RawVideoFormat::Nv12));
        assert_eq!(src.frame_limit, Some(7));
        // no limit is the default and what a 0 limit means
        assert_eq!(
            PipeWireVideoSrc::new().with_frame_limit(0).frame_limit,
            None
        );
    }

    #[test]
    fn caps_are_fixed_and_reject_unusable_geometry() {
        let src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(25);
        assert_eq!(
            src.caps(),
            Ok(Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(25 << 16),
                interlace: g2g_core::Interlace::Any,
            })
        );
        // never advertise Dim::Any / a zero dimension: fixate has to be a no-op
        assert_eq!(
            PipeWireVideoSrc::new().with_size(0, 240).caps(),
            Err(G2gError::CapsMismatch)
        );
        assert_eq!(
            PipeWireVideoSrc::new().with_size(320, MAX_DIM + 1).caps(),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn properties_round_trip_through_the_launch_path() {
        let mut src = PipeWireVideoSrc::new();
        for (name, value) in [
            ("width", PropValue::Uint(1920)),
            ("height", PropValue::Uint(1080)),
            ("framerate", PropValue::Uint(50)),
            ("target-object", PropValue::Str("cam0".to_string())),
            ("format", PropValue::Str("YUY2".to_string())),
            ("num-buffers", PropValue::Int(30)),
        ] {
            src.set_property(name, value.clone()).expect("known prop");
            assert_eq!(src.get_property(name), Some(value));
        }
        assert_eq!(
            (src.req_width, src.req_height, src.req_fps),
            (1920, 1080, 50)
        );
        assert_eq!(src.target, "cam0");
        assert_eq!(src.pin_format, Some(RawVideoFormat::Yuyv));
        assert_eq!(src.frame_limit, Some(30));
        // -1 is no limit, in both directions
        src.set_property("num-buffers", PropValue::Int(-1))
            .expect("known prop");
        assert_eq!(src.frame_limit, None);
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(-1)));
        assert_eq!(
            src.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
        // every declared property is handled by both halves
        for spec in src.properties() {
            assert!(
                src.get_property(spec.name).is_some(),
                "unhandled property {}",
                spec.name
            );
        }
    }

    /// A pinned format is what the caps advertise and the only format the connect
    /// pod offers, so the node cannot hand back something else mid-stream.
    #[test]
    fn a_pinned_format_drives_the_caps_and_the_connect_pod() {
        use crate::pwvideo::spa_format;
        use spa::pod::deserialize::PodDeserializer;
        use spa::pod::{ChoiceValue, Value};
        use spa::utils::{Choice, ChoiceEnum, Id};

        let mut src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(30);
        src.set_property("format", PropValue::Str("yuy2".to_string()))
            .expect("known prop");
        assert_eq!(
            src.caps(),
            Ok(Caps::RawVideo {
                format: RawVideoFormat::Yuyv,
                width: Dim::Fixed(320),
                height: Dim::Fixed(240),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
            })
        );

        let bytes = format_pod_bytes(src.format().unwrap(), true, 320, 240, 30).expect("pod");
        let (_, value) = PodDeserializer::deserialize_any_from(&bytes).expect("pod deserializes");
        let Value::Object(obj) = value else {
            panic!("expected an object pod");
        };
        let format = obj
            .properties
            .iter()
            .find(|p| p.key == spa::param::format::FormatProperties::VideoFormat.as_raw())
            .map(|p| p.value.clone())
            .expect("pod carries a format");
        let Value::Choice(ChoiceValue::Id(Choice(_, ChoiceEnum::Enum { alternatives, .. }))) =
            format
        else {
            panic!("format is an enum choice");
        };
        let pinned = spa_format(RawVideoFormat::Yuyv).unwrap();
        assert_eq!(alternatives, Vec::from([Id(pinned.as_raw())]));

        // an empty pin is the open negotiation the element defaults to
        src.set_property("format", PropValue::Str(String::new()))
            .expect("known prop");
        assert_eq!(src.pin_format, None);
        // and a name the SPA table has no entry for never silently defaults
        for bad in ["y42b", "p010", "nonsense"] {
            assert_eq!(
                src.set_property("format", PropValue::Str(bad.to_string())),
                Err(PropError::Value),
                "{bad} is not a capture format"
            );
        }
        assert_eq!(src.pin_format, None);
    }

    #[test]
    fn configure_marks_the_element_ready() {
        let mut src = PipeWireVideoSrc::new();
        assert!(!src.configured);
        let caps = src.caps().expect("default caps");
        src.configure_pipeline(&caps).expect("fixed caps accepted");
        assert!(src.configured);
    }

    #[test]
    fn pad_template_is_a_raw_video_source() {
        use g2g_core::PadDirection;
        assert!(PipeWireVideoSrc::pad_template(PadDirection::Source).is_some());
        assert!(PipeWireVideoSrc::pad_template(PadDirection::Sink).is_none());
    }

    /// The advertised caps and the negotiated caps compare equal when the node
    /// hands back exactly what was asked for, so no `CapsChanged` is emitted;
    /// any difference produces one.
    #[test]
    fn negotiated_caps_only_differ_when_the_node_differs() {
        let src = PipeWireVideoSrc::new().with_size(320, 240).with_fps(30);
        let same = VideoInfo {
            format: RawVideoFormat::I420,
            width: 320,
            height: 240,
            fps_num: 30,
            fps_denom: 1,
        };
        assert_eq!(same.caps(), src.caps().unwrap());
        let other = VideoInfo {
            format: RawVideoFormat::Yuyv,
            ..same
        };
        assert_ne!(other.caps(), src.caps().unwrap());
    }

    /// The `param_changed` parse: a fixated `Format` param yields the negotiated
    /// info, and a format the element cannot carry fails instead of defaulting.
    #[test]
    fn param_changed_parse_accepts_a_fixated_format_and_rejects_the_rest() {
        use crate::pwvideo::fixed_format_pod_bytes;
        use spa::param::video::VideoFormat;

        let parse = |bytes: &[u8]| {
            let pod = spa::pod::Pod::from_bytes(bytes).expect("pod bytes parse");
            parse_format(pod)
        };
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::YUY2, 640, 360, 25)),
            Ok(VideoInfo {
                format: RawVideoFormat::Yuyv,
                width: 640,
                height: 360,
                fps_num: 25,
                fps_denom: 1,
            })
        );
        // YV12 is a real SPA format the element does not carry
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::YV12, 640, 360, 25)),
            Err(G2gError::CapsMismatch)
        );
        // and a nonsense geometry never reaches an allocation
        assert_eq!(
            parse(&fixed_format_pod_bytes(VideoFormat::I420, 0, 360, 25)),
            Err(G2gError::CapsMismatch)
        );
    }
}
