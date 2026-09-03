//! Fallback switch (`fallbackswitch`). Forwards the highest-priority input that
//! is still delivering and falls back to the next one when it stalls. The input
//! index IS the priority: input 0 is the primary, 1 the first fallback, and so
//! on, matching gst's default where a request pad's `priority` is its serial.
//!
//! An input is healthy while it delivered a `DataFrame` within `timeout`
//! nanoseconds of now, so a pad that stops producing loses the output to the next
//! index down. Health is re-evaluated on every incoming packet and on every
//! `Tick`, which is why the element declares a tick interval: a stalled primary
//! has to be noticed even while every other input is silent.
//!
//! `std` only: the health rule measures against
//! [`g2g_core::metrics::monotonic_ns`].

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MultiInputElement,
    OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

/// gst's `fallbackswitch` timeout default: one second without a buffer and the
/// pad is stalled.
const DEFAULT_TIMEOUT_NS: u64 = 1_000_000_000;

/// # Example
///
/// ```no_run
/// use g2g_plugins::fallbackswitch::FallbackSwitch;
///
/// // primary plus one fallback, switching after 200 ms of silence
/// let switch = FallbackSwitch::new(2).with_timeout_ns(200_000_000);
/// ```
#[derive(Debug)]
pub struct FallbackSwitch {
    inputs: usize,
    active: usize,
    auto_switch: bool,
    immediate_fallback: bool,
    timeout_ns: u64,
    stop_on_eos: bool,
    configured: Vec<Option<Caps>>,
    /// When each input last delivered a `DataFrame`, `None` until its first one.
    last_frame_ns: Vec<Option<u64>>,
    /// When any input first delivered, the anchor the startup hold measures from.
    first_frame_ns: Option<u64>,
    /// The caps last pushed downstream, so a switch re-emits only on a real change.
    emitted_caps: Option<Caps>,
    /// Set by an `Eos` under `stop-on-eos`; forwarding is over from then on.
    stopped: bool,
}

impl FallbackSwitch {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "FallbackSwitch needs at least one input");
        Self {
            inputs,
            active: 0,
            auto_switch: true,
            immediate_fallback: false,
            timeout_ns: DEFAULT_TIMEOUT_NS,
            stop_on_eos: false,
            configured: vec![None; inputs],
            last_frame_ns: vec![None; inputs],
            first_frame_ns: None,
            emitted_caps: None,
            stopped: false,
        }
    }

    /// Nanoseconds an input may go without a `DataFrame` before it counts as
    /// stalled (the `timeout` property).
    pub fn with_timeout_ns(mut self, timeout_ns: u64) -> Self {
        self.timeout_ns = timeout_ns;
        self
    }

    /// Pick the forwarded input by hand instead of by health (`auto-switch=false`
    /// plus `active-pad`).
    pub fn with_auto_switch(mut self, auto_switch: bool) -> Self {
        self.auto_switch = auto_switch;
        self
    }

    /// Forward whatever arrives first instead of holding a fallback frame until
    /// the primary has had `timeout` to show up (the `immediate-fallback`
    /// property).
    pub fn with_immediate_fallback(mut self, immediate_fallback: bool) -> Self {
        self.immediate_fallback = immediate_fallback;
        self
    }

    /// Stop forwarding once any input ends (the `stop-on-eos` property).
    pub fn with_stop_on_eos(mut self, stop_on_eos: bool) -> Self {
        self.stop_on_eos = stop_on_eos;
        self
    }

    /// The input forwarded right now.
    pub fn active(&self) -> usize {
        self.active
    }

    fn healthy(&self, input: usize, now_ns: u64) -> bool {
        match self.last_frame_ns[input] {
            Some(last) => now_ns.saturating_sub(last) <= self.timeout_ns,
            None => false,
        }
    }

    /// Whether a lower-priority frame is still being held back at startup: the
    /// primary has delivered nothing and it has had less than `timeout` since the
    /// element saw its first frame on any input.
    fn startup_hold(&self, now_ns: u64) -> bool {
        if self.immediate_fallback || self.last_frame_ns[0].is_some() {
            return false;
        }
        match self.first_frame_ns {
            Some(first) => now_ns.saturating_sub(first) < self.timeout_ns,
            None => true,
        }
    }

    /// Re-pick the forwarded input. The lowest-index healthy one wins; when none
    /// is healthy the current one stays, so a stall does not blank the output.
    fn select(&mut self, now_ns: u64) {
        if !self.auto_switch {
            return;
        }
        if self.startup_hold(now_ns) {
            self.active = 0;
            return;
        }
        if let Some(input) = (0..self.inputs).find(|&i| self.healthy(i, now_ns)) {
            self.active = input;
        }
    }

    /// The body of [`process`](MultiInputElement::process) with "now" passed in,
    /// so the tests drive the health rule without waiting on a real clock.
    async fn handle(
        &mut self,
        input: usize,
        packet: PipelinePacket,
        now_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        // The runner aggregates input ends and emits the merged Eos itself, so
        // this only records the stop and never forwards.
        if let PipelinePacket::Eos = packet {
            if self.stop_on_eos {
                self.stopped = true;
            }
            return Ok(());
        }
        if self.stopped {
            return Ok(());
        }
        if let PipelinePacket::DataFrame(_) = &packet {
            self.last_frame_ns[input] = Some(now_ns);
            self.first_frame_ns.get_or_insert(now_ns);
        }
        self.select(now_ns);
        // A tick only re-checks health; there is nothing of its own to forward.
        if let PipelinePacket::Tick = packet {
            return Ok(());
        }
        // Caps from an inactive input are recorded for the moment it becomes
        // active, not forwarded: downstream is negotiated for the active one.
        if let PipelinePacket::CapsChanged(caps) = &packet {
            self.configured[input] = Some(caps.clone());
        }
        if input != self.active {
            return Ok(());
        }
        if let PipelinePacket::DataFrame(_) = &packet {
            // The newly active input's caps may differ from what downstream last
            // saw, so announce them ahead of its first frame.
            let caps = self.configured[input].clone();
            if let Some(caps) = caps.filter(|c| self.emitted_caps.as_ref() != Some(c)) {
                self.emitted_caps = Some(caps.clone());
                out.push(PipelinePacket::CapsChanged(caps)).await?;
            }
        }
        if let PipelinePacket::CapsChanged(caps) = &packet {
            self.emitted_caps = Some(caps.clone());
        }
        out.push(packet).await?;
        Ok(())
    }
}

impl MultiInputElement for FallbackSwitch {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    /// A stalled primary has to be noticed while every input is silent, so the
    /// arm ticks once per timeout. A zero timeout would tick without pause, so it
    /// declines the timer and re-checks on arriving packets alone.
    fn tick_interval_ns(&self) -> Option<u64> {
        (self.timeout_ns > 0).then_some(self.timeout_ns)
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.configured[0].clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Fallback switch",
            "Generic",
            "Forwards the highest-priority input that is still delivering",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FALLBACKSWITCH_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "active-pad" => {
                let index = value.as_uint().ok_or(PropError::Type)? as usize;
                if index >= self.inputs {
                    return Err(PropError::Value);
                }
                self.active = index;
            }
            "auto-switch" => self.auto_switch = value.as_bool().ok_or(PropError::Type)?,
            "immediate-fallback" => {
                self.immediate_fallback = value.as_bool().ok_or(PropError::Type)?
            }
            "timeout" => self.timeout_ns = value.as_uint().ok_or(PropError::Type)?,
            "stop-on-eos" => self.stop_on_eos = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "active-pad" => Some(PropValue::Uint(self.active as u64)),
            "auto-switch" => Some(PropValue::Bool(self.auto_switch)),
            "immediate-fallback" => Some(PropValue::Bool(self.immediate_fallback)),
            "timeout" => Some(PropValue::Uint(self.timeout_ns)),
            "stop-on-eos" => Some(PropValue::Bool(self.stop_on_eos)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let now_ns = g2g_core::metrics::monotonic_ns();
            self.handle(input, packet, now_ns, out).await
        })
    }
}

/// `FallbackSwitch`'s settable properties, named as gst's `fallbackswitch`.
static FALLBACKSWITCH_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "active-pad",
        PropKind::Uint,
        "index of the input being forwarded",
    )
    .with_default("0"),
    PropertySpec::new(
        "auto-switch",
        PropKind::Bool,
        "pick the input by health instead of by active-pad",
    )
    .with_default("true"),
    PropertySpec::new(
        "immediate-fallback",
        PropKind::Bool,
        "forward a fallback at once instead of waiting timeout for the primary",
    )
    .with_default("false"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "nanoseconds without a buffer before an input counts as stalled",
    )
    .with_default("1000000000"),
    PropertySpec::new(
        "stop-on-eos",
        PropKind::Bool,
        "stop forwarding once any input ends",
    )
    .with_default("false"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{
        AudioFormat, Dim, Frame, FrameTiming, MemoryDomain, PushOutcome, Rate, RawVideoFormat,
        SystemSlice,
    };

    /// Every packet reaching the output, in order: the frame sequences and the
    /// caps announced ahead of them.
    #[derive(Default)]
    struct CollectSink {
        seq: Vec<u64>,
        caps: Vec<Caps>,
    }
    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::DataFrame(f) => self.seq.push(f.sequence),
                PipelinePacket::CapsChanged(c) => self.caps.push(c),
                _ => {}
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn frame(seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        })
    }

    fn video(width: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(width),
            height: Dim::Fixed(240),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn audio() -> Caps {
        Caps::Audio {
            format: AudioFormat::PcmS16Le,
            channels: 2,
            sample_rate: 48_000,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        }
    }

    /// 100 ms timeout, so the tests can name arrival times in whole tens of ms.
    const TIMEOUT_NS: u64 = 100_000_000;
    const MS: u64 = 1_000_000;

    fn switch(inputs: usize) -> FallbackSwitch {
        FallbackSwitch::new(inputs).with_timeout_ns(TIMEOUT_NS)
    }

    #[tokio::test]
    async fn falls_back_when_the_primary_stalls() {
        let mut s = switch(2).with_immediate_fallback(true);
        let mut out = CollectSink::default();
        // Primary delivering: it wins even while the fallback also delivers.
        s.handle(0, frame(1), 0, &mut out).await.unwrap();
        s.handle(1, frame(101), 10 * MS, &mut out).await.unwrap();
        s.handle(0, frame(2), 20 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![1, 2], "the primary owns the output");

        // The fallback keeps delivering while the primary goes silent. Nothing
        // arrives on pad 0 to trigger a re-check, so the tick is what notices the
        // primary is 110 ms stale and hands the output to the fallback.
        s.handle(1, frame(102), 60 * MS, &mut out).await.unwrap();
        s.handle(0, PipelinePacket::Tick, 130 * MS, &mut out)
            .await
            .unwrap();
        assert_eq!(s.active(), 1, "the tick made the switch");
        s.handle(1, frame(103), 135 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![1, 2, 103], "the fallback took over");
    }

    #[tokio::test]
    async fn switches_back_when_the_primary_resumes() {
        let mut s = switch(2).with_immediate_fallback(true);
        let mut out = CollectSink::default();
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        assert_eq!(s.active(), 1, "only the fallback is delivering");
        s.handle(0, frame(1), 10 * MS, &mut out).await.unwrap();
        assert_eq!(s.active(), 0, "the primary outranks a healthy fallback");
        s.handle(1, frame(102), 20 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![101, 1], "pad 1 is dropped once pad 0 is back");
    }

    #[tokio::test]
    async fn no_healthy_input_keeps_the_current_one() {
        let mut s = switch(2).with_immediate_fallback(true);
        let mut out = CollectSink::default();
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        // Both pads are stale now. The active pad must not be reset to 0, or a
        // resuming fallback would be dropped.
        s.handle(0, PipelinePacket::Tick, 500 * MS, &mut out)
            .await
            .unwrap();
        assert_eq!(s.active(), 1);
        s.handle(1, frame(102), 510 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![101, 102]);
    }

    #[tokio::test]
    async fn immediate_fallback_off_holds_the_fallback_for_one_timeout() {
        let mut s = switch(2);
        let mut out = CollectSink::default();
        // The fallback is the first thing the element ever sees, so it is held
        // back to give the primary its timeout to arrive.
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        s.handle(1, frame(102), 50 * MS, &mut out).await.unwrap();
        assert!(out.seq.is_empty(), "held during the startup window");
        assert_eq!(s.active(), 0);
        // A primary that shows up inside the window keeps the output.
        s.handle(0, frame(1), 60 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![1]);
    }

    #[tokio::test]
    async fn immediate_fallback_off_gives_up_after_the_timeout() {
        let mut s = switch(2);
        let mut out = CollectSink::default();
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        assert!(out.seq.is_empty());
        // A whole timeout with no primary: the fallback is released.
        s.handle(1, frame(102), TIMEOUT_NS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![102]);
        assert_eq!(s.active(), 1);
    }

    #[tokio::test]
    async fn immediate_fallback_on_forwards_the_first_arrival() {
        let mut s = switch(2).with_immediate_fallback(true);
        let mut out = CollectSink::default();
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![101], "no startup hold");
    }

    #[tokio::test]
    async fn auto_switch_off_follows_active_pad() {
        let mut s = switch(2).with_auto_switch(false);
        let mut out = CollectSink::default();
        // Pad 1 is the only one delivering, but health is ignored.
        s.handle(1, frame(101), 0, &mut out).await.unwrap();
        s.handle(0, PipelinePacket::Tick, 500 * MS, &mut out)
            .await
            .unwrap();
        s.handle(1, frame(102), 510 * MS, &mut out).await.unwrap();
        assert!(out.seq.is_empty(), "active-pad still names pad 0");
        s.set_property("active-pad", PropValue::Uint(1)).unwrap();
        s.handle(1, frame(103), 520 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![103]);
    }

    #[tokio::test]
    async fn stop_on_eos_ends_forwarding() {
        let mut s = switch(2)
            .with_immediate_fallback(true)
            .with_stop_on_eos(true);
        let mut out = CollectSink::default();
        s.handle(0, frame(1), 0, &mut out).await.unwrap();
        s.handle(1, PipelinePacket::Eos, 10 * MS, &mut out)
            .await
            .unwrap();
        s.handle(0, frame(2), 20 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![1], "one input ending stops the output");
    }

    #[tokio::test]
    async fn eos_without_stop_on_eos_keeps_forwarding() {
        let mut s = switch(2).with_immediate_fallback(true);
        let mut out = CollectSink::default();
        s.handle(0, frame(1), 0, &mut out).await.unwrap();
        s.handle(1, PipelinePacket::Eos, 10 * MS, &mut out)
            .await
            .unwrap();
        s.handle(0, frame(2), 20 * MS, &mut out).await.unwrap();
        assert_eq!(out.seq, vec![1, 2]);
        assert!(out.caps.is_empty(), "Eos is never forwarded from here");
    }

    #[tokio::test]
    async fn switching_re_announces_the_new_input_caps() {
        let mut s = switch(2).with_immediate_fallback(true);
        s.configure_pipeline(0, &video(1920)).unwrap();
        s.configure_pipeline(1, &video(320)).unwrap();
        let mut out = CollectSink::default();
        s.handle(0, frame(1), 0, &mut out).await.unwrap();
        s.handle(0, frame(2), 10 * MS, &mut out).await.unwrap();
        assert_eq!(out.caps, vec![video(1920)], "announced once, not per frame");

        s.handle(0, PipelinePacket::Tick, 200 * MS, &mut out)
            .await
            .unwrap();
        s.handle(1, frame(101), 205 * MS, &mut out).await.unwrap();
        assert_eq!(
            out.caps,
            vec![video(1920), video(320)],
            "the fallback's caps precede its first frame"
        );
    }

    #[tokio::test]
    async fn runtime_caps_flow_only_from_the_active_input() {
        let mut s = switch(2).with_immediate_fallback(true);
        s.configure_pipeline(0, &video(1920)).unwrap();
        s.configure_pipeline(1, &video(320)).unwrap();
        let mut out = CollectSink::default();
        s.handle(0, frame(1), 0, &mut out).await.unwrap();
        // Pad 1 re-types mid-stream while inactive: recorded, not forwarded.
        s.handle(1, PipelinePacket::CapsChanged(audio()), 10 * MS, &mut out)
            .await
            .unwrap();
        assert_eq!(out.caps, vec![video(1920)]);
        // Pad 0 re-types while active: forwarded.
        s.handle(
            0,
            PipelinePacket::CapsChanged(video(640)),
            20 * MS,
            &mut out,
        )
        .await
        .unwrap();
        assert_eq!(out.caps, vec![video(1920), video(640)]);
        // And pad 1's recorded caps are the ones announced when it takes over.
        s.handle(1, frame(101), 300 * MS, &mut out).await.unwrap();
        assert_eq!(out.caps, vec![video(1920), video(640), audio()]);
    }

    #[test]
    fn active_pad_out_of_range_rejected() {
        let mut s = switch(2);
        assert_eq!(
            s.set_property("active-pad", PropValue::Uint(2))
                .unwrap_err(),
            PropError::Value
        );
        assert_eq!(
            s.set_property("timeout", PropValue::Bool(true))
                .unwrap_err(),
            PropError::Type
        );
        assert_eq!(
            s.set_property("priority", PropValue::Uint(0)).unwrap_err(),
            PropError::Unknown
        );
    }

    #[test]
    fn tick_interval_follows_timeout() {
        assert_eq!(
            MultiInputElement::tick_interval_ns(&switch(2)),
            Some(TIMEOUT_NS)
        );
        assert_eq!(
            MultiInputElement::tick_interval_ns(&FallbackSwitch::new(2).with_timeout_ns(0)),
            None
        );
    }
}
