use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;

use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, ConfigureOutcome, DriftClock, ElementMetadata, G2gError, MonotonicClock, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    Segment,
};

use crate::mpegts::{
    ticks_90khz_to_ns, PcrSample, TsDemuxer, TsPacketSplitter, SYSTEM_CLOCK_HZ,
    SYSTEM_CLOCK_TICKS_PER_90KHZ,
};

const NANOS_PER_SECOND: u64 = 1_000_000_000;
// the PCR's 33-bit base times its extension values
const PCR_MODULUS: u64 = (1 << 33) * SYSTEM_CLOCK_TICKS_PER_90KHZ;
// gst mpegtslivesrc: a PCR further than 15 s from the last one starts a new timeline
const PCR_WRAP_TOLERANCE: u64 = 15 * SYSTEM_CLOCK_HZ as u64;
// gst mpegtslivesrc: a PCR more than a second off the clock's projection starts a new timeline
const DISCONTINUITY_NS: u64 = NANOS_PER_SECOND;
// gst mpegtslivesrc window-size default and range
const DEFAULT_WINDOW_SIZE: usize = 32;
const MIN_WINDOW_SIZE: usize = 2;
const MAX_WINDOW_SIZE: usize = 1024;

fn ts_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

// the element checks discontinuities itself, so the drift clock never rejects an observation
fn pcr_clock(window_size: usize) -> Arc<DriftClock> {
    Arc::new(DriftClock::with_window_and_gate(
        Arc::new(MonotonicClock),
        window_size,
        DISCONTINUITY_NS,
    ))
}

// in tsdemux's PTS nanoseconds, the extension's 300ths of a 90 kHz tick dropped
fn pcr_to_pts_ns(pcr_27mhz: u64) -> u64 {
    ticks_90khz_to_ns(pcr_27mhz / SYSTEM_CLOCK_TICKS_PER_90KHZ)
}

// the wrap count that puts `raw` within the tolerance of the last unwrapped PCR, none on a jump
fn unwrap_pcr(raw: u64, last: u64) -> Option<u64> {
    let wraps = last / PCR_MODULUS;
    [wraps.checked_sub(1), Some(wraps), wraps.checked_add(1)]
        .into_iter()
        .flatten()
        .map(|wrap| wrap * PCR_MODULUS + raw)
        .find(|candidate| candidate.abs_diff(last) <= PCR_WRAP_TOLERANCE)
}

#[derive(Debug, Clone, Copy)]
struct PcrTimeline {
    anchor_pcr: u64,
    anchor_clock_ns: u64,
    last_pcr: u64,
}

impl PcrTimeline {
    fn clock_ns(&self, pcr: u64) -> i128 {
        let offset_27mhz = i128::from(pcr) - i128::from(self.anchor_pcr);
        i128::from(self.anchor_clock_ns)
            + offset_27mhz * i128::from(NANOS_PER_SECOND) / SYSTEM_CLOCK_HZ as i128
    }
}

#[derive(Debug)]
pub struct MpegTsLiveSrc {
    reader: TsDemuxer,
    splitter: TsPacketSplitter,
    window_size: usize,
    clock: Arc<DriftClock>,
    timeline: Option<PcrTimeline>,
    sync: Option<ClockSync>,
    configured: bool,
}

impl Default for MpegTsLiveSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl MpegTsLiveSrc {
    pub fn new() -> Self {
        Self {
            reader: TsDemuxer::pcr_reader(),
            splitter: TsPacketSplitter::new(),
            window_size: DEFAULT_WINDOW_SIZE,
            clock: pcr_clock(DEFAULT_WINDOW_SIZE),
            timeline: None,
            sync: None,
            configured: false,
        }
    }

    pub fn clock(&self) -> Arc<DriftClock> {
        self.clock.clone()
    }

    // the segment that maps stream time onto running time, when this PCR starts a new timeline
    fn observe(&mut self, sample: PcrSample, arrival_ns: u64) -> Result<Option<Segment>, G2gError> {
        let projected_ns = self.clock.project_ns(arrival_ns);
        let continued = self
            .timeline
            .filter(|_| !sample.discontinuity)
            .and_then(|timeline| {
                let pcr = unwrap_pcr(sample.pcr_27mhz, timeline.last_pcr)?;
                let clock_ns = u64::try_from(timeline.clock_ns(pcr)).ok()?;
                let near_projection = clock_ns.abs_diff(projected_ns) < DISCONTINUITY_NS;
                near_projection.then_some((timeline, pcr, clock_ns))
            });
        if let Some((timeline, pcr, clock_ns)) = continued {
            self.clock.observe(arrival_ns, clock_ns);
            self.timeline = Some(PcrTimeline {
                last_pcr: pcr,
                ..timeline
            });
            return Ok(None);
        }
        self.clock.observe(arrival_ns, projected_ns);
        self.timeline = Some(PcrTimeline {
            anchor_pcr: sample.pcr_27mhz,
            anchor_clock_ns: projected_ns,
            last_pcr: sample.pcr_27mhz,
        });
        let sync = self.sync.as_ref().ok_or(G2gError::NotConfigured)?;
        let start = pcr_to_pts_ns(sample.pcr_27mhz);
        Ok(Some(Segment {
            start,
            base: projected_ns.saturating_sub(sync.base_time()),
            time: start as i64,
            position: start,
            ..Segment::new()
        }))
    }

    async fn forward_bytes(
        &mut self,
        frame: g2g_core::frame::Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let bytes = frame
            .domain
            .as_system_slice()
            .ok_or(G2gError::UnsupportedDomain)?;
        self.splitter.push(bytes, &mut self.reader);
        let arrival_ns = frame.timing.arrival_ns;
        // tsdemux keeps a segment that follows the runner's opening one
        for sample in self.reader.take_pcr_samples() {
            if let Some(segment) = self.observe(sample, arrival_ns)? {
                out.push(PipelinePacket::Segment(segment)).await?;
            }
        }
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }
}

impl AsyncElement for MpegTsLiveSrc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MPEG-TS live clock",
            "Source/Network",
            "Provides a pipeline clock from the PCR of a live MPEG-TS stream and maps its timestamps onto it",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if *upstream_caps != ts_caps() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(ts_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(ts_caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::LiveSource,
            self.clock.clone(),
        ))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.sync = Some(sync);
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "window-size",
            PropKind::Uint,
            "PCR observations the clock's rate and offset are fitted over",
        )
        .with_default("32")
        .with_range("2", "1024")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "window-size" => {
                let size = value.as_uint().ok_or(PropError::Type)?;
                let size = usize::try_from(size).map_err(|_| PropError::Value)?;
                if !(MIN_WINDOW_SIZE..=MAX_WINDOW_SIZE).contains(&size) {
                    return Err(PropError::Value);
                }
                self.window_size = size;
                self.clock = pcr_clock(size);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "window-size" => Some(PropValue::Uint(self.window_size as u64)),
            _ => None,
        }
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
                PipelinePacket::DataFrame(frame) => self.forward_bytes(frame, out).await,
                PipelinePacket::Flush => {
                    // a restarted source starts a new PCR timeline
                    self.splitter.clear();
                    self.reader = TsDemuxer::pcr_reader();
                    self.timeline = None;
                    out.push(PipelinePacket::Flush).await?;
                    Ok(())
                }
                PipelinePacket::Eos => Ok(()),
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

impl PadTemplates for MpegTsLiveSrc {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        alloc::vec::Vec::from([
            PadTemplate::sink(CapsSet::one(ts_caps())),
            PadTemplate::source(CapsSet::one(ts_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpegts::{TsMuxer, STREAM_TYPE_H264};
    use crate::testutil::{frame, CollectSink};
    use g2g_core::frame::Frame;

    const TS_PACKET_LEN: usize = 188;
    const TABLE_PACKETS: usize = 2;
    const BASE_TIME_NS: u64 = 1_000_000;
    // a 25 fps sender's frame period on its 90 kHz clock
    const FRAME_90KHZ: u64 = 3_600;
    const FRAME_NS: u64 = 40_000_000;
    const FIRST_DTS_90KHZ: u64 = 900_000;
    const IDR_ACCESS_UNIT: [u8; 6] = [0, 0, 0, 1, 0x65, 0x88];
    const FIRST_ARRIVAL_NS: u64 = 5_000_000_000;

    fn element() -> MpegTsLiveSrc {
        let mut element = MpegTsLiveSrc::new();
        element.configure_pipeline(&ts_caps()).unwrap();
        element.set_clock_sync(ClockSync::new(Arc::new(MonotonicClock), BASE_TIME_NS));
        element
    }

    // the muxer puts a PCR equal to a keyframe's DTS in its first packet
    fn keyframe_bytes(muxer: &mut TsMuxer, dts_90khz: u64) -> alloc::vec::Vec<u8> {
        muxer.push_au(&IDR_ACCESS_UNIT, Some(dts_90khz), Some(dts_90khz), true)
    }

    fn arriving(bytes: alloc::vec::Vec<u8>, arrival_ns: u64) -> Frame {
        let mut frame = frame(bytes);
        frame.timing.arrival_ns = arrival_ns;
        frame
    }

    async fn push(element: &mut MpegTsLiveSrc, frame: Frame) -> alloc::vec::Vec<PipelinePacket> {
        let mut sink = CollectSink::default();
        element
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        sink.packets
    }

    fn segments(packets: &[PipelinePacket]) -> alloc::vec::Vec<Segment> {
        packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::Segment(segment) => Some(*segment),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn the_first_pcr_maps_its_stream_time_to_its_arrival() {
        let mut element = element();
        let mut muxer = TsMuxer::new(STREAM_TYPE_H264);
        let bytes = keyframe_bytes(&mut muxer, FIRST_DTS_90KHZ);
        let (tables, rest) = bytes.split_at(TABLE_PACKETS * TS_PACKET_LEN);
        let arrival_ns = FIRST_ARRIVAL_NS;

        let before_pcr = push(&mut element, arriving(tables.to_vec(), arrival_ns)).await;
        let [PipelinePacket::DataFrame(_)] = before_pcr.as_slice() else {
            panic!("the tables pass through alone: {before_pcr:?}");
        };

        let packets = push(&mut element, arriving(rest.to_vec(), arrival_ns)).await;
        let [PipelinePacket::Segment(segment), PipelinePacket::DataFrame(_)] = packets.as_slice()
        else {
            panic!("expected segment, data: {packets:?}");
        };
        let pcr_ns = pcr_to_pts_ns(FIRST_DTS_90KHZ * SYSTEM_CLOCK_TICKS_PER_90KHZ);
        assert_eq!(segment.start, pcr_ns);
        assert_eq!(segment.base, arrival_ns - BASE_TIME_NS);
        assert_eq!(
            segment.to_running_time(pcr_ns),
            Some(arrival_ns - BASE_TIME_NS)
        );
    }

    #[tokio::test]
    async fn the_clock_runs_at_the_senders_rate() {
        const SENDER_RATE: f64 = 1.0001;
        const FRAMES: u64 = 200;
        let mut element = element();
        let mut muxer = TsMuxer::new(STREAM_TYPE_H264);
        let first_arrival_ns = FIRST_ARRIVAL_NS;
        let mut opened = 0;
        for index in 0..FRAMES {
            let bytes = keyframe_bytes(&mut muxer, FIRST_DTS_90KHZ + index * FRAME_90KHZ);
            let arrival_ns =
                first_arrival_ns + (index as f64 * FRAME_NS as f64 / SENDER_RATE) as u64;
            opened += segments(&push(&mut element, arriving(bytes, arrival_ns)).await).len();
        }
        assert_eq!(opened, 1, "one timeline for a steady sender");
        let slope = element.clock().slope();
        assert!((slope - SENDER_RATE).abs() < 1e-6, "slope {slope}");
    }

    #[tokio::test]
    async fn a_pcr_jump_starts_a_new_timeline() {
        const JUMP_90KHZ: u64 = 10 * 90_000;
        let mut element = element();
        let mut muxer = TsMuxer::new(STREAM_TYPE_H264);
        let first_arrival_ns = FIRST_ARRIVAL_NS;
        for index in 0..3 {
            let bytes = keyframe_bytes(&mut muxer, FIRST_DTS_90KHZ + index * FRAME_90KHZ);
            push(
                &mut element,
                arriving(bytes, first_arrival_ns + index * FRAME_NS),
            )
            .await;
        }
        let jumped_dts = FIRST_DTS_90KHZ + 3 * FRAME_90KHZ + JUMP_90KHZ;
        let arrival_ns = first_arrival_ns + 3 * FRAME_NS;
        let packets = push(
            &mut element,
            arriving(keyframe_bytes(&mut muxer, jumped_dts), arrival_ns),
        )
        .await;

        let [segment] = segments(&packets)[..] else {
            panic!("expected one segment: {packets:?}");
        };
        assert_eq!(
            segment.start,
            pcr_to_pts_ns(jumped_dts * SYSTEM_CLOCK_TICKS_PER_90KHZ)
        );
        assert_eq!(segment.base, arrival_ns - BASE_TIME_NS);
    }

    #[tokio::test]
    async fn the_pcr_wrapping_keeps_the_timeline() {
        const PCR_BASE_MODULUS_90KHZ: u64 = 1 << 33;
        let mut element = element();
        let mut muxer = TsMuxer::new(STREAM_TYPE_H264);
        let first_arrival_ns = FIRST_ARRIVAL_NS;
        let mut opened = 0;
        for index in 0..6 {
            let dts = PCR_BASE_MODULUS_90KHZ - 3 * FRAME_90KHZ + index * FRAME_90KHZ;
            let bytes = keyframe_bytes(&mut muxer, dts);
            let arrival_ns = first_arrival_ns + index * FRAME_NS;
            opened += segments(&push(&mut element, arriving(bytes, arrival_ns)).await).len();
        }
        assert_eq!(opened, 1, "the wrap is not a new timeline");
        let slope = element.clock().slope();
        assert!((slope - 1.0).abs() < 1e-6, "slope {slope}");
    }
}
