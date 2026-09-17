//! Alert recorder (M1176): records a clip of the frames around each alert an
//! upstream [`AnalyticsAlert`](crate::analyticsalert::AnalyticsAlert) attached,
//! the gst-python-ml `pyml_alertrecorder` analog. Frames pass through unchanged.
//!
//! A ring holds the last `seconds-before` of frames. The first frame carrying the
//! `alert` blob opens a clip: a child graph
//! `appsrc ! <encoder> ! filesink location=<location>` parsed from the launch
//! registry and run on its own thread, fed the ring and then every frame that
//! follows, with the pts rebased to the clip's first frame. Each further alerted
//! frame pushes the end out to `seconds-after`; once the stream passes it the
//! feed is ended and the clip's thread joined.
//!
//! The ring holds a reference-counted handle on each frame's bytes, so a frame
//! crossing the element is not copied. A frame with no presentation time is
//! forwarded without entering the ring: it has no place in a clip's timeline, and
//! ringing it would grow the ring without bound.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource, Target};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    frame_carries, g2g_error, g2g_info, AsyncElement, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::analyticsalert::ALERT_BLOB;
use crate::appsrc::{register_appsrc, AppSrcFeed};
use crate::clock::WallClock;

/// Where each clip is written. `%s` stands for the time the alert fired.
const DEFAULT_LOCATION: &str = "alert-%s.avi";
/// The encoder and muxer a clip goes through. Motion JPEG in AVI: both halves
/// are in every std build, so a bare `alertrecorder` writes a playable file.
const DEFAULT_ENCODER: &str = "mjpegenc ! avimux";
/// The `%s` the location's time is substituted for.
const TIME_PLACEHOLDER: &str = "%s";
const DEFAULT_SECONDS_BEFORE: f64 = 3.0;
const DEFAULT_SECONDS_BEFORE_TEXT: &str = "3.0";
const DEFAULT_SECONDS_AFTER: f64 = 5.0;
const DEFAULT_SECONDS_AFTER_TEXT: &str = "5.0";

const NS_PER_SECOND: f64 = 1_000_000_000.0;
const CLIP_THREAD_NAME: &str = "g2g-alert-clip";
/// Depth of the clip graph's links. Two frames in flight is enough to keep the
/// encoder busy without buffering the clip in memory.
const CLIP_LINK_CAPACITY: usize = 2;

/// Names one clip's `appsrc` feed. Global, so two recorders (or two clips of one
/// recorder) never claim each other's channel.
static CLIP_SERIAL: AtomicU64 = AtomicU64::new(0);

const SECONDS_PER_MINUTE: u64 = 60;
const SECONDS_PER_HOUR: u64 = 3600;
const SECONDS_PER_DAY: u64 = 86400;
const DAYS_PER_ERA: u64 = 146097;
const DAYS_FROM_0000_03_01_TO_EPOCH: u64 = 719468;

/// One frame held for a clip: the bytes as a shared handle, and the stream time
/// the clip rebases against.
#[derive(Debug)]
struct RingFrame {
    pts_ns: u64,
    domain: MemoryDomain,
}

impl RingFrame {
    /// A second handle on the same bytes, for a second clip.
    fn slice(&self) -> Option<SystemSlice> {
        match self.domain.share() {
            MemoryDomain::System(slice) => Some(slice),
            _ => None,
        }
    }
}

/// One open clip: the feed its child graph reads, the thread that graph runs on,
/// and the stream window it covers.
#[derive(Debug)]
struct Clip {
    path: String,
    feed: AppSrcFeed,
    thread: Option<std::thread::JoinHandle<()>>,
    start_pts_ns: u64,
    end_pts_ns: u64,
}

impl Clip {
    fn open(
        path: String,
        encoder: &str,
        caps: &Caps,
        start_pts_ns: u64,
        end_pts_ns: u64,
    ) -> Result<Clip, G2gError> {
        let channel = format!(
            "{CLIP_THREAD_NAME}-{}",
            CLIP_SERIAL.fetch_add(1, Ordering::Relaxed)
        );
        let feed = register_appsrc(&channel);
        let line = format!(
            "appsrc channel={channel} caps={} ! {encoder} ! filesink location={path}",
            caps.to_gst_string()
        );
        // The graph's elements are not `Send` in a single-threaded build, so it
        // is built on the thread that runs it. `report` carries the build's
        // verdict back, so a bad `encoder` fails the alert instead of leaving a
        // feed no one drains.
        let (report, start_result) = std::sync::mpsc::sync_channel(1);
        let mut thread = Some(
            std::thread::Builder::new()
                .name(CLIP_THREAD_NAME.to_string())
                .spawn(move || {
                    let target = Target::category(short_type_name::<AlertRecorder>());
                    let runtime = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(runtime) => runtime,
                        Err(error) => {
                            let _ = report.send(Err(format!("no clip runtime: {error}")));
                            return;
                        }
                    };
                    let graph = match parse_launch(&crate::registry::default_registry(), &line) {
                        Ok(graph) => graph,
                        Err(error) => {
                            let _ = report.send(Err(format!("`{line}`: {error}")));
                            return;
                        }
                    };
                    if report.send(Ok(())).is_err() {
                        return;
                    }
                    let clock = WallClock::new();
                    if let Err(error) =
                        runtime.block_on(run_graph(graph, &clock, CLIP_LINK_CAPACITY))
                    {
                        g2g_error!(target, "the clip pipeline failed: {error:?}");
                    }
                })
                .map_err(g2g_core::log::io_err)?,
        );
        let started = start_result
            .recv()
            .unwrap_or_else(|_| Err(String::from("the clip thread stopped before it ran")));
        if let Err(reason) = started {
            g2g_error!(
                Target::category(short_type_name::<AlertRecorder>()),
                "cannot record a clip: {reason}"
            );
            if let Some(thread) = thread.take() {
                let _ = thread.join();
            }
            return Err(G2gError::NotConfigured);
        }
        Ok(Clip {
            path,
            feed,
            thread,
            start_pts_ns,
            end_pts_ns,
        })
    }

    /// Feed one frame, rebased onto the clip's own timeline.
    fn push(&self, pts_ns: u64, slice: SystemSlice) {
        self.feed
            .push_slice_blocking(slice, pts_ns.saturating_sub(self.start_pts_ns));
    }

    /// End the feed and wait for the file to be written.
    fn finish(&mut self) {
        self.feed.end_of_stream_blocking();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Records a clip around each alert on the stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::alertrecorder::AlertRecorder;
///
/// // gst-launch equivalent: alertrecorder location=/tmp/alert-%s.avi seconds-before=1.0
/// let recorder = AlertRecorder::new();
/// assert_eq!(recorder.clips_written(), 0);
/// ```
#[derive(Debug)]
pub struct AlertRecorder {
    location: String,
    encoder: String,
    seconds_before: f64,
    seconds_after: f64,
    caps: Option<Caps>,
    ring: VecDeque<RingFrame>,
    clip: Option<Clip>,
    clips_written: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for AlertRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertRecorder {
    pub fn new() -> Self {
        Self {
            location: DEFAULT_LOCATION.to_string(),
            encoder: DEFAULT_ENCODER.to_string(),
            seconds_before: DEFAULT_SECONDS_BEFORE,
            seconds_after: DEFAULT_SECONDS_AFTER,
            caps: None,
            ring: VecDeque::new(),
            clip: None,
            clips_written: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Write the clips here (the `location` property); `%s` stands for the alert
    /// time.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// The launch fragment a clip is encoded and muxed through (the `encoder`
    /// property). It must match the location's extension.
    pub fn with_encoder(mut self, encoder: impl Into<String>) -> Self {
        self.encoder = encoder.into();
        self
    }

    /// Seconds of video kept from before an alert (the `seconds-before`
    /// property).
    pub fn with_seconds_before(mut self, seconds: f64) -> Self {
        self.seconds_before = seconds;
        self
    }

    /// Seconds of video recorded after the last alert (the `seconds-after`
    /// property).
    pub fn with_seconds_after(mut self, seconds: f64) -> Self {
        self.seconds_after = seconds;
        self
    }

    /// Clips finished so far.
    pub fn clips_written(&self) -> u64 {
        self.clips_written
    }

    fn seconds_to_ns(seconds: f64) -> u64 {
        (seconds.max(0.0) * NS_PER_SECOND) as u64
    }

    /// The clip path for an alert now: the location with `%s` replaced by the
    /// UTC time.
    fn clip_path(&self) -> String {
        self.location.replace(
            TIME_PLACEHOLDER,
            &utc_stamp(g2g_core::log::unix_time_source()),
        )
    }

    fn open_clip(&mut self, alert_pts_ns: u64) -> Result<(), G2gError> {
        let caps = self.caps.clone().ok_or(G2gError::NotConfigured)?;
        let start = self.ring.front().map_or(alert_pts_ns, |held| held.pts_ns);
        let end = alert_pts_ns.saturating_add(Self::seconds_to_ns(self.seconds_after));
        let path = self.clip_path();
        let clip = Clip::open(path, &self.encoder, &caps, start, end)?;
        g2g_info!(
            self,
            "recording {} from the alert at {alert_pts_ns}",
            clip.path
        );
        for held in &self.ring {
            let Some(slice) = held.slice() else {
                return Err(G2gError::UnsupportedDomain);
            };
            clip.push(held.pts_ns, slice);
        }
        self.clip = Some(clip);
        Ok(())
    }

    fn finish_clip(&mut self) {
        if let Some(mut clip) = self.clip.take() {
            clip.finish();
            self.clips_written += 1;
            g2g_info!(self, "wrote {}", clip.path);
        }
    }
}

impl AsyncElement for AlertRecorder {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Alert recorder",
            "Filter/Analytics",
            "Records a clip of the frames around each alert on the stream",
            "g2g",
        )
    }

    /// The clip is fed from host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if matches!(upstream_caps, Caps::RawVideo { .. }) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if matches!(input, Caps::RawVideo { .. }) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let Some(pts_ns) = frame.timing.pts() else {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    };
                    if !matches!(frame.domain, MemoryDomain::System(_)) {
                        return Err(G2gError::UnsupportedDomain);
                    }
                    // One conversion to a refcounted handle, so the ring, the
                    // clip and the frame going downstream share the same bytes.
                    frame.domain.make_shareable();
                    let held = RingFrame {
                        pts_ns,
                        domain: frame.domain.share(),
                    };
                    let alerted = frame_carries(&frame.meta, ALERT_BLOB);
                    let before_ns = Self::seconds_to_ns(self.seconds_before);
                    self.ring.push_back(held);
                    while self
                        .ring
                        .front()
                        .is_some_and(|oldest| pts_ns.saturating_sub(oldest.pts_ns) > before_ns)
                    {
                        self.ring.pop_front();
                    }
                    match &self.clip {
                        // The ring, which now ends with this frame, is the clip's
                        // first seconds.
                        None if alerted => self.open_clip(pts_ns)?,
                        None => {}
                        Some(clip) => {
                            let Some(slice) = self.ring.back().and_then(RingFrame::slice) else {
                                return Err(G2gError::UnsupportedDomain);
                            };
                            clip.push(pts_ns, slice);
                        }
                    }
                    if let Some(clip) = &mut self.clip {
                        if alerted {
                            clip.end_pts_ns =
                                pts_ns.saturating_add(Self::seconds_to_ns(self.seconds_after));
                        }
                        if pts_ns >= clip.end_pts_ns {
                            self.finish_clip();
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.caps = Some(caps.clone());
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                PipelinePacket::Flush => {
                    self.finish_clip();
                    self.ring.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner's transform arm forwards EOS; close the open clip
                // first so its file is complete when the run ends.
                PipelinePacket::Eos => {
                    self.finish_clip();
                    self.ring.clear();
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ALERTRECORDER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "encoder" => {
                self.encoder = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            "seconds-before" => {
                self.seconds_before = value.as_double().ok_or(PropError::Type)?;
                Ok(())
            }
            "seconds-after" => {
                self.seconds_after = value.as_double().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "encoder" => Some(PropValue::Str(self.encoder.clone())),
            "seconds-before" => Some(PropValue::Double(self.seconds_before)),
            "seconds-after" => Some(PropValue::Double(self.seconds_after)),
            _ => None,
        }
    }
}

impl Drop for AlertRecorder {
    fn drop(&mut self) {
        self.finish_clip();
    }
}

impl LogSource for AlertRecorder {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// `AlertRecorder`'s settable properties, named as the Python element's.
static ALERTRECORDER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "file each clip is written to, %s standing for the alert time",
    )
    .with_default(DEFAULT_LOCATION),
    PropertySpec::new(
        "encoder",
        PropKind::Str,
        "launch fragment a clip is encoded and muxed through",
    )
    .with_default(DEFAULT_ENCODER),
    PropertySpec::new(
        "seconds-before",
        PropKind::Double,
        "seconds of video kept from before the alert",
    )
    .with_default(DEFAULT_SECONDS_BEFORE_TEXT),
    PropertySpec::new(
        "seconds-after",
        PropKind::Double,
        "seconds of video recorded after the last alert",
    )
    .with_default(DEFAULT_SECONDS_AFTER_TEXT),
];

/// `YYYYmmdd-HHMMSS` in UTC for a UNIX time in nanoseconds, the stamp the clip
/// name carries. Written out here because the workspace has no calendar crate.
fn utc_stamp(unix_ns: u64) -> String {
    let seconds = unix_ns / 1_000_000_000;
    let (year, month, day) = civil_from_days(seconds / SECONDS_PER_DAY);
    let time = seconds % SECONDS_PER_DAY;
    format!(
        "{year:04}{month:02}{day:02}-{:02}{:02}{:02}",
        time / SECONDS_PER_HOUR,
        (time % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE,
        time % SECONDS_PER_MINUTE
    )
}

/// The civil date of a day count since 1970-01-01, by the shift-to-March
/// era arithmetic (no lookup tables, no leap-year special cases).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let shifted = days + DAYS_FROM_0000_03_01_TO_EPOCH;
    let era = shifted / DAYS_PER_ERA;
    let day_of_era = shifted % DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    // The year starts in March in this arithmetic, so January and February
    // belong to the next calendar year.
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    (year + u64::from(month <= 2), month, day)
}
