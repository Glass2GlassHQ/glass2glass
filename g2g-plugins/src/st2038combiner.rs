// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::runtime::{PadKind, PadRequest};
use g2g_core::{
    g2g_warn, st2038_any_alignment, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, Frame,
    FrameTiming, G2gError, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, St2038Alignment,
};

use crate::st2038::{framerate_period_ns, parse_st2038};
use crate::st2038ancmux::{LiveAnchor, MuxInput, QueuedPayload};
use crate::st291::{sort_by_position, AncPacket, AncillaryMeta};

// gst st2038combiner gives a frame with no known end this long, and ticks at it with no frame rate
const FALLBACK_FRAME_DURATION_NS: u64 = 50_000_000;

static ST2038COMBINER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "latency",
        PropKind::Uint,
        "nanoseconds a live combiner waits past a frame's end for late ST 2038 data",
    )
    .with_default("0"),
    PropertySpec::new(
        "drop-late-st2038",
        PropKind::Bool,
        "drop ST 2038 data timed before the current video frame instead of attaching it",
    )
    .with_default("false"),
];

#[derive(Clone, Copy, Debug)]
struct VideoWindow {
    start_ns: u64,
    end_ns: u64,
}

#[derive(Debug, Default)]
pub struct St2038Combiner {
    video_caps: Option<Caps>,
    held_video: VecDeque<PipelinePacket>,
    video_ended: bool,
    st2038: MuxInput,
    st2038_alignment: Option<St2038Alignment>,
    live: bool,
    live_anchor: Option<LiveAnchor>,
    latency_ns: u64,
    drop_late_st2038: bool,
    log_name: LogName,
}

impl St2038Combiner {
    pub const VIDEO: usize = 0;
    pub const ST2038: usize = 1;

    pub fn new() -> Self {
        Self::default()
    }

    fn frame_period_ns(&self) -> Option<u64> {
        let framerate = self.video_caps.as_ref()?.framerate();
        framerate_period_ns(&framerate)
    }

    // a frame's end comes from its duration, the frame rate, the next held frame or the end of video
    fn window_of(&self, front: FrameTiming) -> Option<VideoWindow> {
        let start_ns = front.pts_ns;
        let next_held_pts_ns = self.held_video.iter().find_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
            _ => None,
        });
        let end_ns = if front.duration_ns > 0 {
            start_ns + front.duration_ns
        } else if let Some(period_ns) = self.frame_period_ns() {
            start_ns + period_ns
        } else if let Some(next_pts_ns) = next_held_pts_ns {
            next_pts_ns
        } else if self.video_ended {
            start_ns + FALLBACK_FRAME_DURATION_NS
        } else {
            return None;
        };
        Some(VideoWindow { start_ns, end_ns })
    }

    // frame-aligned ST 2038 sends one payload per frame, so the frame is complete once it is in
    fn frame_payload_arrived(&self, window: VideoWindow) -> bool {
        self.st2038_alignment == Some(St2038Alignment::Frame)
            && self
                .st2038
                .queue
                .iter()
                .any(|q| (window.start_ns..window.end_ns).contains(&q.timing.pts_ns))
    }

    fn window_ready(&self, window: VideoWindow, now_ns: u64) -> bool {
        let deadline_passed = self
            .live_anchor
            .filter(|_| self.live)
            .is_some_and(|anchor| anchor.deadline_passed(window.end_ns, self.latency_ns, now_ns));
        self.st2038.has_nothing_more_before(window.end_ns)
            || self.frame_payload_arrived(window)
            || deadline_passed
    }

    fn take_window_packets(&mut self, window: VideoWindow) -> Vec<AncPacket> {
        let mut packets = Vec::new();
        while self
            .st2038
            .queue
            .front()
            .is_some_and(|q| q.timing.pts_ns < window.end_ns)
        {
            let Some(queued) = self.st2038.queue.pop_front() else {
                break;
            };
            if self.drop_late_st2038 && queued.timing.pts_ns < window.start_ns {
                continue;
            }
            packets.extend(queued.packets);
        }
        sort_by_position(&mut packets);
        packets
    }

    fn attach(frame: &mut Frame, packets: Vec<AncPacket>) {
        if packets.is_empty() {
            return;
        }
        match frame.meta.get_mut::<AncillaryMeta>() {
            Some(meta) => meta.packets.extend(packets),
            None => frame.meta.attach(AncillaryMeta { packets }),
        }
    }

    async fn release_ready_video(
        &mut self,
        now_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        while let Some(packet) = self.held_video.pop_front() {
            let PipelinePacket::DataFrame(mut frame) = packet else {
                out.push(packet).await?;
                continue;
            };
            let window = self.window_of(frame.timing);
            let Some(window) = window.filter(|w| self.window_ready(*w, now_ns)) else {
                self.held_video.push_front(PipelinePacket::DataFrame(frame));
                return Ok(());
            };
            let packets = self.take_window_packets(window);
            Self::attach(&mut frame, packets);
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }

    async fn handle(
        &mut self,
        input: usize,
        packet: PipelinePacket,
        now_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        match (input, packet) {
            (_, PipelinePacket::Tick) => {}
            // the runner sends the merged Eos once every input has ended
            (Self::VIDEO, PipelinePacket::Eos) => self.video_ended = true,
            (_, PipelinePacket::Eos) => self.st2038.ended = true,
            (Self::VIDEO, PipelinePacket::Flush) => {
                self.held_video.clear();
                self.st2038 = MuxInput::default();
                self.live_anchor = None;
                return out.push(PipelinePacket::Flush).await.map(|_| ());
            }
            (Self::VIDEO, packet) => {
                if let PipelinePacket::DataFrame(frame) = &packet {
                    self.live_anchor.get_or_insert(LiveAnchor {
                        pts_ns: frame.timing.pts_ns,
                        wall_ns: now_ns,
                    });
                }
                self.held_video.push_back(packet);
            }
            (_, PipelinePacket::DataFrame(frame)) => {
                let payload = frame
                    .domain
                    .require_system_slice(short_type_name::<Self>())?;
                let Some(packets) = parse_st2038(payload) else {
                    g2g_warn!(self, "dropped an ST 2038 payload that does not parse");
                    return Ok(());
                };
                self.st2038.queue.push_back(QueuedPayload {
                    timing: frame.timing,
                    packets,
                });
            }
            (
                _,
                PipelinePacket::Gap {
                    pts_ns,
                    duration_ns,
                },
            ) => self.st2038.note_gap(pts_ns, duration_ns),
            (_, _) => return Ok(()),
        }
        self.release_ready_video(now_ns, out).await
    }
}

impl MultiInputElement for St2038Combiner {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        Some(self.frame_period_ns().unwrap_or(FALLBACK_FRAME_DURATION_NS))
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(Self::VIDEO)
    }

    fn forwards_caps_changes(&self) -> bool {
        true
    }

    // gst names the pads sink and st2038, and `st2038` parses as a text pad
    fn input_pad_index(&self, req: &PadRequest, _ordinal: usize) -> Option<usize> {
        match req.kind {
            PadKind::Video | PadKind::Any => Some(Self::VIDEO),
            PadKind::Text => Some(Self::ST2038),
            PadKind::Audio => None,
        }
    }

    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match (input, upstream_caps) {
            (Self::VIDEO, _) | (Self::ST2038, Caps::St2038 { .. }) => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_> {
        match input {
            Self::ST2038 => CapsConstraint::Accepts(st2038_any_alignment(Rate::Any)),
            _ => CapsConstraint::AcceptsAny,
        }
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        match (input, absolute_caps) {
            (Self::VIDEO, _) => self.video_caps = Some(absolute_caps.clone()),
            (Self::ST2038, Caps::St2038 { alignment, .. }) => {
                self.st2038_alignment = Some(*alignment)
            }
            _ => return Err(G2gError::CapsMismatch),
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.video_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn configure_liveness(&mut self, live: bool) {
        self.live = live;
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ST-2038 Combiner",
            "Filter",
            "Combines video input stream and ST2038 stream in ancillary meta",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ST2038COMBINER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "latency" => self.latency_ns = value.as_uint().ok_or(PropError::Type)?,
            "drop-late-st2038" => self.drop_late_st2038 = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ns)),
            "drop-late-st2038" => Some(PropValue::Bool(self.drop_late_st2038)),
            _ => None,
        }
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
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

impl LogSource for St2038Combiner {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::st2038::write_st2038;
    use crate::testutil::{data_pts_ns, frame_with_bytes, CollectSink};
    use alloc::vec;

    const FRAME_NS: u64 = 40_000_000;
    const ST2038_INPUT_CAPS: Caps = Caps::St2038 {
        alignment: St2038Alignment::Line,
        framerate: Rate::Any,
    };

    fn packet_at_line(line_number: u16) -> AncPacket {
        let mut packet = AncPacket::from_bytes(0x60, 0x60, &[line_number as u8]);
        packet.line_number = line_number;
        packet.horizontal_offset = 0;
        packet
    }

    fn video_at(pts_ns: u64) -> PipelinePacket {
        let mut frame = frame_with_bytes(0, vec![0; 4]);
        frame.timing.pts_ns = pts_ns;
        frame.timing.duration_ns = FRAME_NS;
        PipelinePacket::DataFrame(frame)
    }

    fn st2038_at(pts_ns: u64, packets: &[AncPacket]) -> PipelinePacket {
        let mut frame = frame_with_bytes(0, write_st2038(packets).unwrap());
        frame.timing.pts_ns = pts_ns;
        PipelinePacket::DataFrame(frame)
    }

    fn combiner(live: bool) -> St2038Combiner {
        let mut combiner = St2038Combiner::new();
        combiner
            .configure_pipeline(St2038Combiner::ST2038, &ST2038_INPUT_CAPS)
            .unwrap();
        combiner.configure_liveness(live);
        combiner
    }

    fn feed(
        combiner: &mut St2038Combiner,
        steps: Vec<(usize, PipelinePacket, u64)>,
    ) -> CollectSink {
        let mut out = CollectSink::default();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for (input, packet, now_ns) in steps {
            runtime
                .block_on(combiner.handle(input, packet, now_ns, &mut out))
                .unwrap();
        }
        out
    }

    fn attached(out: &CollectSink) -> Vec<Option<Vec<AncPacket>>> {
        out.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(
                    f.meta
                        .get::<AncillaryMeta>()
                        .map(|meta| meta.packets.clone()),
                ),
                _ => None,
            })
            .collect()
    }

    const VIDEO: usize = St2038Combiner::VIDEO;
    const ST2038: usize = St2038Combiner::ST2038;

    #[test]
    fn a_frame_carries_its_window_sorted_by_line_once_st2038_moves_past_it() {
        let mut combiner = combiner(false);
        let out = feed(
            &mut combiner,
            vec![
                (VIDEO, video_at(0), 0),
                (ST2038, st2038_at(0, &[packet_at_line(12)]), 0),
                (ST2038, st2038_at(FRAME_NS / 2, &[packet_at_line(9)]), 0),
            ],
        );
        assert!(
            attached(&out).is_empty(),
            "ST 2038 may still add to frame 0"
        );
        let out = feed(
            &mut combiner,
            vec![
                (ST2038, st2038_at(FRAME_NS, &[packet_at_line(10)]), 0),
                (VIDEO, video_at(FRAME_NS), 0),
                (ST2038, PipelinePacket::Eos, 0),
            ],
        );
        assert_eq!(
            attached(&out),
            vec![
                Some(vec![packet_at_line(9), packet_at_line(12)]),
                Some(vec![packet_at_line(10)])
            ]
        );
    }

    #[test]
    fn a_gap_releases_a_frame_with_no_meta() {
        let mut combiner = combiner(false);
        let gap = PipelinePacket::Gap {
            pts_ns: 0,
            duration_ns: FRAME_NS,
        };
        let out = feed(
            &mut combiner,
            vec![(VIDEO, video_at(0), 0), (ST2038, gap, 0)],
        );
        assert_eq!(attached(&out), vec![None]);
    }

    #[test]
    fn late_st2038_is_attached_unless_dropped() {
        let late_then_frame = || {
            vec![
                (ST2038, st2038_at(0, &[packet_at_line(9)]), 0),
                (VIDEO, video_at(FRAME_NS), 0),
                (ST2038, PipelinePacket::Eos, 0),
            ]
        };
        let mut kept = combiner(false);
        let out = feed(&mut kept, late_then_frame());
        assert_eq!(attached(&out), vec![Some(vec![packet_at_line(9)])]);
        let mut dropping = combiner(false);
        dropping
            .set_property("drop-late-st2038", PropValue::Bool(true))
            .unwrap();
        let out = feed(&mut dropping, late_then_frame());
        assert_eq!(attached(&out), vec![None]);
    }

    #[test]
    fn live_combiner_sends_the_frame_once_the_deadline_passes() {
        let wall_ns = 5_000_000_000;
        let latency_ns = 1_000;
        let mut combiner = combiner(true);
        combiner
            .set_property("latency", PropValue::Uint(latency_ns))
            .unwrap();
        let out = feed(
            &mut combiner,
            vec![
                (VIDEO, video_at(0), wall_ns),
                (VIDEO, PipelinePacket::Tick, wall_ns + FRAME_NS),
            ],
        );
        assert!(attached(&out).is_empty(), "latency not yet past");
        let out = feed(
            &mut combiner,
            vec![(VIDEO, PipelinePacket::Tick, wall_ns + FRAME_NS + latency_ns)],
        );
        assert_eq!(data_pts_ns(&out.packets), vec![0]);
    }
}
