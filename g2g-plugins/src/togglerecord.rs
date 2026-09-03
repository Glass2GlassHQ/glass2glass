//! Keyframe-aligned start/stop recording across several streams (`togglerecord`).
//!
//! gst's `togglerecord` is one element with a main `sink`/`src` pair plus
//! `sink_%u`/`src_%u` request pads, N in and N out. g2g's graph has no N-in N-out
//! node kind, so the port is one [`ToggleRecord`] per stream, all sharing a
//! [`RecordGroup`]: the `group` property (or `RecordGroup::new()` from Rust) is
//! what a gst request pad is here.
//!
//! The main stream decides. `record=true` moves the group to `Starting`, and the
//! main forwards nothing until its next keyframe, which opens a recorded span at
//! that frame's timestamp. `record=false` moves it to `Stopping`, and the main
//! keeps forwarding until its next keyframe, which closes the span just before
//! that frame. A secondary forwards a frame only when its timestamp falls inside a
//! recorded span, and it may only ask once the main stream has passed the end of
//! that frame, so it never decides ahead of the main.
//!
//! Two differences from gst. gst blocks a non-live upstream while paused; g2g
//! drops, because a g2g source keeps producing. And gst rejects any delta frame on
//! a secondary pad, which in g2g would reject every raw stream (raw frames carry
//! `keyframe: false`), so the check applies only to a secondary whose negotiated
//! caps are compressed video, where cutting mid-GOP really is undecodable.
//!
//! Each stream is its own arm, and a secondary parks until the main advances, so
//! every branch needs enough link capacity to hold the frames in flight, the same
//! reason gst asks for a `queue` per pad.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, Weak};

use tokio::sync::Notify;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// The stop timestamp of the span the group is still recording into.
const SPAN_OPEN: u64 = u64::MAX;

/// Where the group is in the keyframe handshake, gst's `RecordingState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordState {
    Stopped,
    Starting,
    Recording,
    Stopping,
}

#[derive(Debug)]
struct GroupState {
    record: bool,
    state: RecordState,
    /// End timestamp of the newest main-stream frame, forwarded or dropped. A
    /// secondary can decide any frame ending at or before this.
    main_position_ns: u64,
    main_ended: bool,
    /// The keyframe-aligned `[start, stop)` spans the main stream decided, in
    /// order. The last is [`SPAN_OPEN`] while recording.
    spans: Vec<(u64, u64)>,
    main_claimed: bool,
}

/// The shared record decision one or more [`ToggleRecord`] elements act on.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::togglerecord::{RecordGroup, ToggleRecord};
///
/// // gst-launch equivalent: togglerecord group=take main=true / main=false
/// let group = RecordGroup::new();
/// let video = ToggleRecord::main(group.clone());
/// let audio = ToggleRecord::secondary(group.clone());
/// group.set_record(true);
/// ```
#[derive(Debug)]
pub struct RecordGroup {
    state: Mutex<GroupState>,
    /// Woken every time the main stream advances, so a parked secondary retries.
    advanced: Notify,
}

impl RecordGroup {
    /// The group a `togglerecord group=<name>` launch line joined, created on
    /// first ask. The only way an application that built its pipeline from text
    /// can reach the `record` flag.
    pub fn named(name: &str) -> Arc<Self> {
        named_group(name)
    }

    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GroupState {
                record: false,
                state: RecordState::Stopped,
                main_position_ns: 0,
                main_ended: false,
                spans: Vec::new(),
                main_claimed: false,
            }),
            advanced: Notify::new(),
        })
    }

    /// Start or stop recording. Takes effect on the main stream's next keyframe.
    pub fn set_record(&self, record: bool) {
        self.state.lock().unwrap().record = record;
    }

    pub fn record(&self) -> bool {
        self.state.lock().unwrap().record
    }

    /// Whether frames are still going into a recording. `Stopping` counts: the
    /// span stays open until the main stream's next keyframe closes it.
    pub fn recording(&self) -> bool {
        matches!(
            self.state.lock().unwrap().state,
            RecordState::Recording | RecordState::Stopping
        )
    }

    /// The recorded spans as `[start, stop)` timestamps, the last stop
    /// [`SPAN_OPEN`] while recording.
    fn spans(&self) -> Vec<(u64, u64)> {
        self.state.lock().unwrap().spans.clone()
    }

    fn claim_main(&self) -> Result<(), PropError> {
        let mut state = self.state.lock().unwrap();
        if state.main_claimed {
            return Err(PropError::Value);
        }
        state.main_claimed = true;
        Ok(())
    }

    fn release_main(&self) {
        self.state.lock().unwrap().main_claimed = false;
    }

    /// Run the main stream's state machine over one frame. `Some(output_pts)`
    /// forwards it, `None` drops it.
    fn main_frame(&self, pts_ns: u64, end_ns: u64, keyframe: bool) -> Option<u64> {
        let mut state = self.state.lock().unwrap();
        match (state.state, state.record) {
            (RecordState::Stopped, true) => state.state = RecordState::Starting,
            (RecordState::Recording, false) => state.state = RecordState::Stopping,
            _ => {}
        }
        let forward = match state.state {
            RecordState::Stopped => false,
            RecordState::Starting => {
                // Recording can only begin on an independently decodable frame.
                if keyframe {
                    state.spans.push((pts_ns, SPAN_OPEN));
                    state.state = RecordState::Recording;
                    true
                } else {
                    false
                }
            }
            RecordState::Recording => true,
            RecordState::Stopping => {
                // The span ends just before the keyframe, so the recording is a
                // whole number of GOPs.
                if keyframe {
                    if let Some(span) = state.spans.last_mut() {
                        span.1 = pts_ns;
                    }
                    state.state = RecordState::Stopped;
                    false
                } else {
                    true
                }
            }
        };
        state.main_position_ns = state.main_position_ns.max(end_ns);
        let out_pts = forward
            .then(|| recorded_before(&state.spans, pts_ns))
            .flatten();
        drop(state);
        self.advanced.notify_waiters();
        out_pts
    }

    /// The main stream will not advance again, so an open span ends where it
    /// stopped and every parked secondary can decide.
    fn main_ended(&self) {
        let mut state = self.state.lock().unwrap();
        state.main_ended = true;
        let position = state.main_position_ns;
        if let Some(span) = state.spans.last_mut() {
            if span.1 == SPAN_OPEN {
                span.1 = position;
            }
        }
        state.state = RecordState::Stopped;
        drop(state);
        self.advanced.notify_waiters();
    }

    /// Whether the main stream has passed `end_ns`, so a secondary frame ending
    /// there can be decided.
    fn decidable(&self, end_ns: u64) -> bool {
        let state = self.state.lock().unwrap();
        state.main_ended || state.main_position_ns >= end_ns
    }

    /// Park until [`decidable`](Self::decidable). `enable` registers this waiter
    /// before the flag is read, so a `notify_waiters` in between is not lost.
    async fn await_main(&self, end_ns: u64) {
        loop {
            let notified = self.advanced.notified();
            let mut notified = core::pin::pin!(notified);
            notified.as_mut().enable();
            if self.decidable(end_ns) {
                return;
            }
            notified.await;
        }
    }
}

/// The output timestamp of a frame at `pts_ns`, which is the recorded time before
/// it, or `None` when no span covers it. Every member derives its timeline from
/// this one function, so the streams stay aligned across a pause.
fn recorded_before(spans: &[(u64, u64)], pts_ns: u64) -> Option<u64> {
    let mut before = 0u64;
    for &(start, stop) in spans {
        if pts_ns < start {
            return None;
        }
        if pts_ns < stop {
            return Some(before.saturating_add(pts_ns - start));
        }
        before = before.saturating_add(stop - start);
    }
    None
}

/// The `group=` name table. A launch factory builds its element from a plain
/// `fn`, so a name is the only way two elements in a text pipeline can find the
/// same group. Weak, so a group disappears with its last member.
fn group_table() -> &'static Mutex<HashMap<String, Weak<RecordGroup>>> {
    static TABLE: OnceLock<Mutex<HashMap<String, Weak<RecordGroup>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn named_group(name: &str) -> Arc<RecordGroup> {
    let mut table = group_table().lock().unwrap();
    table.retain(|_, weak| weak.strong_count() > 0);
    if let Some(group) = table.get(name).and_then(Weak::upgrade) {
        return group;
    }
    let group = RecordGroup::new();
    table.insert(name.to_string(), Arc::downgrade(&group));
    group
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::togglerecord::{RecordGroup, ToggleRecord};
///
/// let element = ToggleRecord::main(RecordGroup::new());
/// assert!(!element.is_live());
/// ```
#[derive(Debug)]
pub struct ToggleRecord {
    group: Arc<RecordGroup>,
    /// The `group=` name this element joined, empty for a private group.
    group_name: String,
    main: bool,
    /// Whether this element holds its group's single main slot.
    claimed_main: bool,
    is_live: bool,
    /// Whether every frame on this stream begins its own decodable unit, which
    /// only compressed video does not. A recording may start or stop anywhere on
    /// such a stream, and a secondary needs no keyframe flags at all.
    all_keyframes: bool,
    forwarded: u64,
    configured: bool,
    log_name: LogName,
}

impl ToggleRecord {
    /// The main stream of a private group, which is what a lone `togglerecord`
    /// in a launch line is until `group=` moves it.
    pub fn new() -> Self {
        Self::default()
    }

    /// The stream whose keyframes decide when the group records.
    pub fn main(group: Arc<RecordGroup>) -> Self {
        Self::build(group, String::new(), true)
    }

    /// A stream that follows the main stream's decision.
    pub fn secondary(group: Arc<RecordGroup>) -> Self {
        Self::build(group, String::new(), false)
    }

    fn build(group: Arc<RecordGroup>, group_name: String, main: bool) -> Self {
        Self {
            group,
            group_name,
            main,
            claimed_main: false,
            is_live: false,
            all_keyframes: false,
            forwarded: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Pass timestamps through untouched instead of pulling each recording back
    /// onto a continuous timeline.
    pub fn with_is_live(mut self, is_live: bool) -> Self {
        self.is_live = is_live;
        self
    }

    pub fn is_live(&self) -> bool {
        self.is_live
    }

    /// Frames this element let through.
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }

    pub fn group(&self) -> &Arc<RecordGroup> {
        &self.group
    }

    /// Take the group's main slot, unless another element already has it.
    fn take_main_slot(&mut self) -> Result<(), PropError> {
        if self.claimed_main {
            return Ok(());
        }
        self.group.claim_main()?;
        self.claimed_main = true;
        Ok(())
    }

    fn drop_main_slot(&mut self) {
        if self.claimed_main {
            self.group.release_main();
            self.claimed_main = false;
        }
    }
}

impl Default for ToggleRecord {
    fn default() -> Self {
        Self::main(RecordGroup::new())
    }
}

/// A group outlives the graph that used it when a caller holds it by name, so a
/// rebuilt pipeline has to find the main slot free again.
impl Drop for ToggleRecord {
    fn drop(&mut self) {
        self.drop_main_slot();
    }
}

impl AsyncElement for ToggleRecord {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ToggleRecord",
            "Generic",
            "Starts and stops several streams together on the main stream's keyframes",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Whatever the surrounding endpoints settle on flows through: the element
    /// decides when a frame passes, never what shape it has.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.main {
            // The launch line may have set `group=` after `main=`, so the slot is
            // taken here as well as in `set_property`.
            self.take_main_slot().map_err(|_| {
                g2g_error!(
                    self,
                    "group `{}` already has a main stream: exactly one member may be main",
                    self.group_name
                );
                G2gError::CapsMismatch
            })?;
        }
        // Only compressed video carries delta frames. A raw or compressed-audio
        // stream can be cut anywhere, which is also why its frames never set
        // `keyframe`.
        self.all_keyframes = !matches!(absolute_caps, Caps::CompressedVideo { .. });
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                        g2g_error!(
                            self,
                            "frame without a timestamp cannot be placed in or out of a recording"
                        );
                        return Err(G2gError::CapsMismatch);
                    };
                    let keyframe = self.all_keyframes || frame.timing.keyframe;
                    if !self.main && !keyframe {
                        g2g_error!(
                            self,
                            "compressed secondary stream carries delta frames, which cannot be cut at a recording boundary"
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                    let end_ns = pts_ns.saturating_add(frame.timing.duration_ns);
                    let out_pts = if self.main {
                        self.group.main_frame(pts_ns, end_ns, keyframe)
                    } else {
                        self.group.await_main(end_ns).await;
                        recorded_before(&self.group.spans(), pts_ns)
                    };
                    let Some(out_pts) = out_pts else {
                        return Ok(());
                    };
                    if !self.is_live {
                        // The gaps are eaten, so both timestamps move back by
                        // however much never made it into a recording.
                        let eaten = pts_ns - out_pts;
                        frame.timing.pts_ns = out_pts;
                        frame.timing.dts_ns = frame.timing.dts_ns.saturating_sub(eaten);
                    }
                    self.forwarded += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner forwards the EOS sentinel; the group only needs to
                // know the main stream will not advance again, so a secondary
                // still parked stops waiting.
                PipelinePacket::Eos => {
                    if self.main {
                        self.group.main_ended();
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn properties(&self) -> &'static [PropertySpec] {
        TOGGLERECORD_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "record" => {
                self.group
                    .set_record(value.as_bool().ok_or(PropError::Type)?);
                Ok(())
            }
            "is-live" => {
                self.is_live = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "group" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                if name == self.group_name {
                    return Ok(());
                }
                let held_main = self.claimed_main;
                self.drop_main_slot();
                self.group = if name.is_empty() {
                    RecordGroup::new()
                } else {
                    named_group(name)
                };
                self.group_name = name.to_string();
                if held_main {
                    self.take_main_slot()?;
                }
                Ok(())
            }
            "main" => {
                self.main = value.as_bool().ok_or(PropError::Type)?;
                if self.main {
                    self.take_main_slot()?;
                } else {
                    self.drop_main_slot();
                }
                Ok(())
            }
            "recording" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "record" => Some(PropValue::Bool(self.group.record())),
            "recording" => Some(PropValue::Bool(self.group.recording())),
            "is-live" => Some(PropValue::Bool(self.is_live)),
            "group" => Some(PropValue::Str(self.group_name.clone())),
            "main" => Some(PropValue::Bool(self.main)),
            _ => None,
        }
    }
}

impl LogSource for ToggleRecord {
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

/// `ToggleRecord`'s properties. `record`, `recording` and `is-live` are gst's;
/// `group` and `main` are what gst spells with request pads on one element.
static TOGGLERECORD_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "record",
        PropKind::Bool,
        "record, from the main stream's next keyframe; settable on any group member",
    )
    .with_default("false"),
    PropertySpec::new(
        "recording",
        PropKind::Bool,
        "whether the group is inside a recorded span",
    )
    .with_default("false")
    .read_only(),
    PropertySpec::new(
        "is-live",
        PropKind::Bool,
        "pass timestamps through instead of eating the not-recording gaps",
    )
    .with_default("false"),
    PropertySpec::new(
        "group",
        PropKind::Str,
        "name shared by the elements that start and stop together, empty for a group of its own",
    )
    .with_default(""),
    PropertySpec::new(
        "main",
        PropKind::Bool,
        "this stream's keyframes decide the recording; exactly one member per group",
    )
    .with_default("true"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{Dim, Frame, FrameTiming, PushOutcome, Rate, RawVideoFormat};

    /// Long enough that a keyframe every 4th frame gives readable timestamps.
    const FRAME_NS: u64 = 40_000_000;

    /// A main stream whose keyframe flags the element has to honour.
    fn h264() -> Caps {
        Caps::CompressedVideo {
            codec: g2g_core::VideoCodec::H264,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(25 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn raw_video() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(25 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn frame(index: u64, keyframe: bool) -> PipelinePacket {
        let pts_ns = index * FRAME_NS;
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
            FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: FRAME_NS,
                keyframe,
                ..FrameTiming::default()
            },
            index,
        ))
    }

    /// Collects the timestamp of every frame pushed into it.
    #[derive(Default)]
    struct Collect {
        pts: Vec<u64>,
    }

    impl OutputSink for Collect {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            if let PipelinePacket::DataFrame(frame) = packet {
                self.pts.push(frame.timing.pts_ns);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn configured_as(mut element: ToggleRecord, caps: &Caps) -> ToggleRecord {
        element
            .configure_pipeline(caps)
            .expect("passthrough caps are accepted");
        element
    }

    /// A stream whose keyframe flags matter, which is what the state machine is
    /// about.
    fn configured(element: ToggleRecord) -> ToggleRecord {
        configured_as(element, &h264())
    }

    #[tokio::test]
    async fn the_main_stream_starts_recording_on_its_next_keyframe() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut out = Collect::default();

        // Not recording yet: the keyframe at 0 is dropped like everything else.
        main.process(frame(0, true), &mut out).await.unwrap();
        group.set_record(true);
        // Recording asked for at frame 1, but frames 1..3 are delta frames.
        for index in 1..4 {
            main.process(frame(index, false), &mut out).await.unwrap();
        }
        assert!(out.pts.is_empty(), "no keyframe yet, so nothing recorded");
        assert!(!group.recording());

        main.process(frame(4, true), &mut out).await.unwrap();
        assert!(group.recording(), "the keyframe opened the span");
        main.process(frame(5, false), &mut out).await.unwrap();

        // The gap before the recording is eaten, so the output starts at 0.
        assert_eq!(out.pts, vec![0, FRAME_NS]);
        assert_eq!(main.forwarded(), 2);
    }

    #[tokio::test]
    async fn the_main_stream_stops_just_before_its_next_keyframe() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut out = Collect::default();

        group.set_record(true);
        main.process(frame(0, true), &mut out).await.unwrap();
        main.process(frame(1, false), &mut out).await.unwrap();
        group.set_record(false);
        // Still recording: the span only closes at a keyframe.
        main.process(frame(2, false), &mut out).await.unwrap();
        main.process(frame(3, false), &mut out).await.unwrap();
        assert!(group.recording());
        main.process(frame(4, true), &mut out).await.unwrap();
        assert!(!group.recording(), "the keyframe closed the span");
        main.process(frame(5, false), &mut out).await.unwrap();

        assert_eq!(
            out.pts,
            vec![0, FRAME_NS, 2 * FRAME_NS, 3 * FRAME_NS],
            "frames 0..3 recorded, the keyframe at 4 and everything after dropped"
        );
    }

    #[tokio::test]
    async fn a_second_recording_lands_continuously_after_the_first() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut out = Collect::default();

        // Record frames 0..3, skip 4..7, record 8..9.
        group.set_record(true);
        for index in 0..4 {
            main.process(frame(index, index % 4 == 0), &mut out)
                .await
                .unwrap();
        }
        group.set_record(false);
        for index in 4..8 {
            main.process(frame(index, index % 4 == 0), &mut out)
                .await
                .unwrap();
        }
        group.set_record(true);
        for index in 8..10 {
            main.process(frame(index, index % 4 == 0), &mut out)
                .await
                .unwrap();
        }

        // Four recorded frames, then the second recording continues at the 5th
        // frame period rather than at its own input timestamp.
        let expected: Vec<u64> = (0..6).map(|i| i * FRAME_NS).collect();
        assert_eq!(out.pts, expected, "the eaten gap leaves no hole");
    }

    #[tokio::test]
    async fn is_live_leaves_the_timestamps_alone() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()).with_is_live(true));
        let mut out = Collect::default();

        group.set_record(false);
        for index in 0..4 {
            main.process(frame(index, index % 4 == 0), &mut out)
                .await
                .unwrap();
        }
        group.set_record(true);
        for index in 4..6 {
            main.process(frame(index, index % 4 == 0), &mut out)
                .await
                .unwrap();
        }

        assert_eq!(
            out.pts,
            vec![4 * FRAME_NS, 5 * FRAME_NS],
            "the recording keeps its own timestamps"
        );
    }

    #[tokio::test]
    async fn a_secondary_forwards_exactly_the_span_the_main_stream_decided() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut secondary = configured_as(ToggleRecord::secondary(group.clone()), &raw_video());
        let mut main_out = Collect::default();
        let mut secondary_out = Collect::default();

        // Drive the main stream far enough ahead that the secondary never parks:
        // it records frames 4..7 and stops at the keyframe at 8.
        group.set_record(true);
        for index in 0..4 {
            main.process(frame(index, false), &mut main_out)
                .await
                .unwrap();
        }
        main.process(frame(4, true), &mut main_out).await.unwrap();
        for index in 5..8 {
            main.process(frame(index, false), &mut main_out)
                .await
                .unwrap();
        }
        group.set_record(false);
        main.process(frame(8, true), &mut main_out).await.unwrap();
        main.process(frame(9, false), &mut main_out).await.unwrap();

        for index in 0..10 {
            secondary
                .process(frame(index, false), &mut secondary_out)
                .await
                .unwrap();
        }

        assert_eq!(
            main_out.pts,
            vec![0, FRAME_NS, 2 * FRAME_NS, 3 * FRAME_NS],
            "the main stream recorded its frames 4..7"
        );
        assert_eq!(
            secondary_out.pts, main_out.pts,
            "the secondary forwarded the same span on the same timeline"
        );
    }

    #[tokio::test]
    async fn a_secondary_parks_until_the_main_stream_passes_its_frame() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut secondary = configured_as(ToggleRecord::secondary(group.clone()), &raw_video());
        let mut main_out = Collect::default();
        let mut secondary_out = Collect::default();

        // The main stream has not produced anything, so the secondary's first
        // frame is undecidable and it must wait.
        assert!(!group.decidable(FRAME_NS));

        group.set_record(true);
        main.process(frame(0, true), &mut main_out).await.unwrap();
        assert!(group.decidable(FRAME_NS), "the main stream covers frame 0");

        secondary
            .process(frame(0, false), &mut secondary_out)
            .await
            .unwrap();
        assert_eq!(secondary_out.pts, vec![0]);
    }

    #[tokio::test]
    async fn a_secondary_stops_waiting_when_the_main_stream_ends() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut secondary = configured_as(ToggleRecord::secondary(group.clone()), &raw_video());
        let mut out = Collect::default();

        group.set_record(true);
        main.process(frame(0, true), &mut out).await.unwrap();
        main.process(PipelinePacket::Eos, &mut out).await.unwrap();

        // Frame 0 is inside the span the main stream recorded; frame 5 is past
        // where the main stream stopped, and would park forever if EOS did not
        // both release the wait and close the span.
        secondary.process(frame(0, false), &mut out).await.unwrap();
        secondary.process(frame(5, false), &mut out).await.unwrap();
        assert_eq!(
            out.pts,
            vec![0, 0],
            "the main stream's frame and the secondary's frame over the same span"
        );
    }

    #[tokio::test]
    async fn a_compressed_secondary_refuses_delta_frames() {
        let group = RecordGroup::new();
        let mut secondary = configured(ToggleRecord::secondary(group));
        let mut out = Collect::default();
        assert!(
            secondary.process(frame(0, false), &mut out).await.is_err(),
            "a compressed secondary cannot be cut at a delta frame"
        );
    }

    #[tokio::test]
    async fn a_raw_main_stream_records_from_its_first_frame() {
        // Raw frames set no keyframe flag, so the element has to read "every
        // frame is a cut point" off the caps or a raw pipeline never records.
        let group = RecordGroup::new();
        let mut main = configured_as(ToggleRecord::main(group.clone()), &raw_video());
        let mut out = Collect::default();

        group.set_record(true);
        main.process(frame(0, false), &mut out).await.unwrap();
        main.process(frame(1, false), &mut out).await.unwrap();
        assert_eq!(out.pts, vec![0, FRAME_NS]);

        group.set_record(false);
        main.process(frame(2, false), &mut out).await.unwrap();
        assert_eq!(
            out.pts,
            vec![0, FRAME_NS],
            "the stop lands on the next frame too"
        );
    }

    #[tokio::test]
    async fn a_group_takes_one_main_stream() {
        let group = RecordGroup::new();
        let mut first = ToggleRecord::main(group.clone());
        let mut second = ToggleRecord::main(group);
        assert!(first.configure_pipeline(&raw_video()).is_ok());
        assert!(
            second.configure_pipeline(&raw_video()).is_err(),
            "the second main stream has nothing to decide against"
        );
    }

    #[tokio::test]
    async fn the_group_name_joins_two_elements() {
        let mut first = ToggleRecord::new();
        let mut second = ToggleRecord::new();
        first
            .set_property("group", PropValue::Str("take".into()))
            .unwrap();
        second.set_property("main", PropValue::Bool(false)).unwrap();
        second
            .set_property("group", PropValue::Str("take".into()))
            .unwrap();

        // `record` is the group's, so setting it on the secondary reaches the main.
        second
            .set_property("record", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            first.get_property("record"),
            Some(PropValue::Bool(true)),
            "both elements read the same flag"
        );
        assert!(Arc::ptr_eq(first.group(), second.group()));
    }

    #[tokio::test]
    async fn recording_reads_back_and_refuses_a_write() {
        let group = RecordGroup::new();
        let mut main = configured(ToggleRecord::main(group.clone()));
        let mut out = Collect::default();
        assert_eq!(main.get_property("recording"), Some(PropValue::Bool(false)));
        group.set_record(true);
        assert_eq!(
            main.get_property("recording"),
            Some(PropValue::Bool(false)),
            "asking to record is not yet recording"
        );
        main.process(frame(0, true), &mut out).await.unwrap();
        assert_eq!(main.get_property("recording"), Some(PropValue::Bool(true)));
        assert_eq!(
            main.set_property("recording", PropValue::Bool(false)),
            Err(PropError::ReadOnly)
        );
    }
}
