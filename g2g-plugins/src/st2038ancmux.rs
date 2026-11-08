use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::caps::framerate_q16_period_ns;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, st2038_any_alignment, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, Frame, FrameTiming, G2gError, MemoryDomain, MultiInputElement, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, St2038Alignment,
};

use crate::st2038::{parse_st2038, write_st2038};
use crate::st291::AncPacket;

// gst's aggregator advances 40 ms at a time while it knows no frame rate
const UNKNOWN_FRAME_PERIOD_TICK_NS: u64 = 40_000_000;

static ST2038ANCMUX_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "latency",
    PropKind::Uint,
    "nanoseconds a live mux waits past a frame's end for late inputs",
)
.with_default("0")];

// gst st2038ancmux writes alignment=packet unless downstream asks for another
fn output_alternatives() -> CapsSet {
    CapsSet::from_alternatives(
        St2038Alignment::ALL
            .into_iter()
            .map(|alignment| Caps::St2038 {
                alignment,
                framerate: Rate::Any,
            })
            .collect(),
    )
}

#[derive(Debug)]
struct QueuedPayload {
    timing: FrameTiming,
    packets: Vec<AncPacket>,
}

#[derive(Debug, Default)]
struct MuxInput {
    queue: VecDeque<QueuedPayload>,
    gap_until_ns: Option<u64>,
    ended: bool,
}

impl MuxInput {
    fn has_nothing_more_before(&self, end_ns: u64) -> bool {
        self.ended
            || self.queue.back().is_some_and(|q| q.timing.pts_ns >= end_ns)
            || self.gap_until_ns.is_some_and(|gap_end| gap_end >= end_ns)
    }
}

// a frame's packets have pts below end_ns
#[derive(Clone, Copy, Debug)]
struct MuxWindow {
    start_ns: u64,
    end_ns: u64,
}

#[derive(Clone, Copy, Debug)]
struct LiveAnchor {
    pts_ns: u64,
    wall_ns: u64,
}

#[derive(Debug)]
pub struct St2038AncMux {
    inputs: Vec<MuxInput>,
    output_caps: Option<Caps>,
    caps_emitted: bool,
    live: bool,
    latency_ns: u64,
    next_start_ns: Option<u64>,
    live_anchor: Option<LiveAnchor>,
    sequence: u64,
    log_name: LogName,
}

impl St2038AncMux {
    pub fn new(inputs: usize) -> Self {
        Self {
            inputs: (0..inputs).map(|_| MuxInput::default()).collect(),
            output_caps: None,
            caps_emitted: false,
            live: false,
            latency_ns: 0,
            next_start_ns: None,
            live_anchor: None,
            sequence: 0,
            log_name: LogName::default(),
        }
    }

    fn output_alignment_and_period(&self) -> Result<(St2038Alignment, Option<u64>), G2gError> {
        match &self.output_caps {
            Some(Caps::St2038 {
                alignment,
                framerate,
            }) => {
                let period_ns = match framerate {
                    Rate::Fixed(q16) => framerate_q16_period_ns(*q16),
                    _ => None,
                };
                Ok((*alignment, period_ns))
            }
            _ => Err(G2gError::NotConfigured),
        }
    }

    fn earliest_queued_ns(&self) -> Option<u64> {
        self.inputs
            .iter()
            .filter_map(|input| input.queue.front())
            .map(|queued| queued.timing.pts_ns)
            .min()
    }

    fn next_window(&self, frame_period_ns: Option<u64>) -> Option<MuxWindow> {
        let earliest_ns = self.earliest_queued_ns();
        let all_ended = self.inputs.iter().all(|input| input.ended);
        if earliest_ns.is_none() && (!self.live || all_ended) {
            return None;
        }
        let Some(period_ns) = frame_period_ns else {
            let start_ns = earliest_ns?;
            return Some(MuxWindow {
                start_ns,
                end_ns: start_ns + 1,
            });
        };
        let start_ns = self.next_start_ns.or(earliest_ns)?;
        Some(MuxWindow {
            start_ns,
            end_ns: start_ns + period_ns,
        })
    }

    fn live_deadline_passed(&self, end_ns: u64, now_ns: u64) -> bool {
        let Some(anchor) = self.live_anchor.filter(|_| self.live) else {
            return false;
        };
        let due_ns = anchor.wall_ns + end_ns.saturating_sub(anchor.pts_ns) + self.latency_ns;
        now_ns >= due_ns
    }

    async fn drain(&mut self, now_ns: u64, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (alignment, frame_period_ns) = self.output_alignment_and_period()?;
        while let Some(window) = self.next_window(frame_period_ns) {
            let inputs_done = self
                .inputs
                .iter()
                .all(|input| input.has_nothing_more_before(window.end_ns));
            if !inputs_done && !self.live_deadline_passed(window.end_ns, now_ns) {
                return Ok(());
            }
            self.emit_window(window, alignment, frame_period_ns, out)
                .await?;
        }
        Ok(())
    }

    // later inputs and later payloads win a line position two packets both cover, as in gst
    fn take_window_lines(
        &mut self,
        end_ns: u64,
    ) -> (Option<FrameTiming>, BTreeMap<u16, BTreeMap<u16, AncPacket>>) {
        let mut first_timing: Option<FrameTiming> = None;
        let mut lines: BTreeMap<u16, BTreeMap<u16, AncPacket>> = BTreeMap::new();
        for input in self.inputs.iter_mut().rev() {
            let mut taken = Vec::new();
            while input
                .queue
                .front()
                .is_some_and(|queued| queued.timing.pts_ns < end_ns)
            {
                taken.extend(input.queue.pop_front());
            }
            for queued in taken.into_iter().rev() {
                if first_timing.is_none_or(|t| queued.timing.pts_ns < t.pts_ns) {
                    first_timing = Some(queued.timing);
                }
                for packet in queued.packets {
                    let line = lines.entry(packet.line_number).or_default();
                    let start = packet.horizontal_offset;
                    let end = start.saturating_add(packet.user_data.len() as u16);
                    let overlaps = line.values().any(|kept| {
                        let kept_start = kept.horizontal_offset;
                        let kept_end = kept_start.saturating_add(kept.user_data.len() as u16);
                        (start >= kept_start && start < kept_end)
                            || (kept_start >= start && kept_start < end)
                    });
                    if !overlaps {
                        line.insert(start, packet);
                    }
                }
            }
        }
        (first_timing, lines)
    }

    async fn emit_window(
        &mut self,
        window: MuxWindow,
        alignment: St2038Alignment,
        frame_period_ns: Option<u64>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let (first_timing, lines) = self.take_window_lines(window.end_ns);
        self.next_start_ns = frame_period_ns.map(|_| window.end_ns);
        let Some(first_timing) = first_timing else {
            let resume_ns = self
                .earliest_queued_ns()
                .map_or(window.end_ns, |earliest| earliest.max(window.end_ns));
            self.next_start_ns = frame_period_ns.map(|_| resume_ns);
            let gap = PipelinePacket::Gap {
                pts_ns: window.start_ns,
                duration_ns: resume_ns - window.start_ns,
            };
            return out.push(gap).await.map(|_| ());
        };
        let timing = match frame_period_ns {
            Some(period_ns) => FrameTiming {
                pts_ns: window.start_ns,
                dts_ns: window.start_ns,
                duration_ns: period_ns,
                ..first_timing
            },
            None => first_timing,
        };
        let groups: Vec<Vec<AncPacket>> = match alignment {
            St2038Alignment::Frame => Vec::from([lines
                .into_values()
                .flat_map(BTreeMap::into_values)
                .collect()]),
            St2038Alignment::Line => lines
                .into_values()
                .map(|line| line.into_values().collect())
                .collect(),
            St2038Alignment::Packet => lines
                .into_values()
                .flat_map(BTreeMap::into_values)
                .map(|packet| Vec::from([packet]))
                .collect(),
        };
        if !self.caps_emitted {
            let caps = self.output_caps.clone().ok_or(G2gError::NotConfigured)?;
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            self.caps_emitted = true;
        }
        for group in groups {
            let payload = write_st2038(&group).expect("parsed packets fit their fields");
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                timing,
                self.sequence,
            );
            self.sequence += 1;
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
        match packet {
            PipelinePacket::DataFrame(frame) => {
                let payload = frame
                    .domain
                    .require_system_slice(short_type_name::<Self>())?;
                let Some(packets) = parse_st2038(payload) else {
                    g2g_warn!(
                        self,
                        "dropped an ST 2038 payload on input {input} that does not parse"
                    );
                    return Ok(());
                };
                let pts_ns = frame.timing.pts_ns;
                self.live_anchor.get_or_insert(LiveAnchor {
                    pts_ns,
                    wall_ns: now_ns,
                });
                self.inputs[input].queue.push_back(QueuedPayload {
                    timing: frame.timing,
                    packets,
                });
                self.drain(now_ns, out).await
            }
            PipelinePacket::Gap {
                pts_ns,
                duration_ns,
            } => {
                let gap_end_ns = pts_ns.saturating_add(duration_ns);
                let gap_until_ns = &mut self.inputs[input].gap_until_ns;
                *gap_until_ns = Some(gap_until_ns.map_or(gap_end_ns, |end| end.max(gap_end_ns)));
                self.drain(now_ns, out).await
            }
            // the runner aggregates the inputs' Eos and sends the merged one
            PipelinePacket::Eos => {
                self.inputs[input].ended = true;
                self.drain(now_ns, out).await
            }
            PipelinePacket::Tick => self.drain(now_ns, out).await,
            PipelinePacket::CapsChanged(_) => Ok(()),
            PipelinePacket::Segment(_) | PipelinePacket::Flush if input != 0 => Ok(()),
            PipelinePacket::Flush => {
                for input in &mut self.inputs {
                    *input = MuxInput::default();
                }
                self.next_start_ns = None;
                self.live_anchor = None;
                out.push(PipelinePacket::Flush).await.map(|_| ())
            }
            other => out.push(other).await.map(|_| ()),
        }
    }
}

impl MultiInputElement for St2038AncMux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn tick_interval_ns(&self) -> Option<u64> {
        let frame_period_ns = self
            .output_alignment_and_period()
            .ok()
            .and_then(|(_, period)| period);
        Some(frame_period_ns.unwrap_or(UNKNOWN_FRAME_PERIOD_TICK_NS))
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::St2038 { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(st2038_any_alignment(Rate::Any))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.intercept_caps(input, absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    // the runner re-derives the output from this when an input's caps change mid-stream
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError> {
        let produced = match &self.output_caps {
            Some(caps) => CapsSet::one(caps.clone()),
            None => output_alternatives(),
        };
        Ok(CapsConstraint::Produces(produced))
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.output_caps.clone().ok_or(G2gError::NotConfigured)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::St2038 { .. } = output_caps else {
            return Err(G2gError::CapsMismatch);
        };
        self.output_caps = Some(output_caps.clone());
        Ok(())
    }

    fn configure_liveness(&mut self, live: bool) {
        self.live = live;
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SMPTE ST-2038 ancillary metadata muxer",
            "Metadata/Video/Muxer",
            "Muxes SMPTE ST-2038 ancillary metadata streams into a single stream",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ST2038ANCMUX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "latency" => self.latency_ns = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "latency" => Some(PropValue::Uint(self.latency_ns)),
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

impl LogSource for St2038AncMux {
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
    use crate::testutil::{data_bytes, frame_with_bytes, CollectSink};
    use alloc::vec;
    use g2g_core::caps::framerate_q16_from_fraction;

    const FRAME_NS: u64 = 40_000_000;

    fn st2038(alignment: St2038Alignment) -> Caps {
        Caps::St2038 {
            alignment,
            framerate: Rate::Fixed(framerate_q16_from_fraction(25, 1).unwrap()),
        }
    }

    fn packet_at(line_number: u16, horizontal_offset: u16, byte: u8) -> AncPacket {
        let mut packet = AncPacket::from_bytes(0x60, 0x60, &[byte, byte]);
        packet.line_number = line_number;
        packet.horizontal_offset = horizontal_offset;
        packet
    }

    fn payload_at(pts_ns: u64, packets: &[AncPacket]) -> PipelinePacket {
        let mut frame = frame_with_bytes(0, write_st2038(packets).unwrap());
        frame.timing.pts_ns = pts_ns;
        PipelinePacket::DataFrame(frame)
    }

    fn mux(output: Caps, live: bool) -> St2038AncMux {
        let mut mux = St2038AncMux::new(2);
        mux.configure_output(&output).unwrap();
        mux.configure_liveness(live);
        mux
    }

    fn feed(mux: &mut St2038AncMux, steps: Vec<(usize, PipelinePacket, u64)>) -> CollectSink {
        let mut out = CollectSink::default();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for (input, packet, now_ns) in steps {
            runtime
                .block_on(mux.handle(input, packet, now_ns, &mut out))
                .unwrap();
        }
        out
    }

    fn data_frames(out: &CollectSink) -> Vec<(u64, Vec<u8>)> {
        out.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some((
                    f.timing.pts_ns,
                    f.domain.as_system_slice().unwrap().to_vec(),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn frame_alignment_merges_a_frame_by_line_once_every_input_moves_on() {
        let (line_9, line_12, line_12_later) = (
            packet_at(9, 0, 1),
            packet_at(12, 0, 2),
            packet_at(12, 50, 3),
        );
        let mut mux = mux(st2038(St2038Alignment::Frame), false);
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(0, core::slice::from_ref(&line_12)), 0),
                (
                    1,
                    payload_at(0, &[line_12_later.clone(), line_9.clone()]),
                    0,
                ),
            ],
        );
        assert!(
            data_frames(&out).is_empty(),
            "both inputs may still add to frame 0"
        );
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(FRAME_NS, core::slice::from_ref(&line_9)), 0),
                (1, PipelinePacket::Eos, 0),
            ],
        );
        let merged = write_st2038(&[line_9.clone(), line_12, line_12_later]).unwrap();
        assert_eq!(data_frames(&out)[0], (0, merged));
    }

    #[test]
    fn line_and_packet_alignment_split_the_frame() {
        let packets = [packet_at(9, 0, 1), packet_at(9, 50, 2), packet_at(12, 0, 3)];
        let ended = |alignment| {
            let mut mux = mux(st2038(alignment), false);
            let out = feed(
                &mut mux,
                vec![
                    (0, payload_at(0, &packets), 0),
                    (0, PipelinePacket::Eos, 0),
                    (1, PipelinePacket::Eos, 0),
                ],
            );
            data_frames(&out)
                .into_iter()
                .map(|(_, b)| b)
                .collect::<Vec<_>>()
        };
        let one = |p: &[AncPacket]| write_st2038(p).unwrap();
        assert_eq!(
            ended(St2038Alignment::Line),
            vec![one(&packets[..2]), one(&packets[2..])]
        );
        assert_eq!(
            ended(St2038Alignment::Packet),
            packets
                .iter()
                .map(|p| one(core::slice::from_ref(p)))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_later_input_wins_an_overlapping_position() {
        let earlier = packet_at(9, 0, 1);
        let later = packet_at(9, 1, 2);
        let mut mux = mux(st2038(St2038Alignment::Frame), false);
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(0, &[earlier]), 0),
                (1, payload_at(0, core::slice::from_ref(&later)), 0),
                (0, PipelinePacket::Eos, 0),
                (1, PipelinePacket::Eos, 0),
            ],
        );
        assert_eq!(data_bytes(&out.packets), write_st2038(&[later]).unwrap());
    }

    #[test]
    fn a_gap_on_the_quiet_input_releases_the_frame() {
        let packet = packet_at(9, 0, 1);
        let mut mux = mux(st2038(St2038Alignment::Frame), false);
        let gap = PipelinePacket::Gap {
            pts_ns: 0,
            duration_ns: FRAME_NS,
        };
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(0, core::slice::from_ref(&packet)), 0),
                (0, payload_at(FRAME_NS, core::slice::from_ref(&packet)), 0),
                (1, gap, 0),
            ],
        );
        assert_eq!(
            data_frames(&out),
            vec![(0, write_st2038(&[packet]).unwrap())]
        );
    }

    #[test]
    fn live_mux_times_out_on_a_quiet_input_and_fills_silence_with_gaps() {
        let packet = packet_at(9, 0, 1);
        let wall_ns = 5_000_000_000;
        let mut mux = mux(st2038(St2038Alignment::Frame), true);
        mux.set_property("latency", PropValue::Uint(1_000)).unwrap();
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(0, core::slice::from_ref(&packet)), wall_ns),
                (0, PipelinePacket::Tick, wall_ns + FRAME_NS),
            ],
        );
        assert!(data_frames(&out).is_empty(), "latency not yet past");
        let out = feed(
            &mut mux,
            vec![(0, PipelinePacket::Tick, wall_ns + 2 * FRAME_NS + 1_000)],
        );
        assert_eq!(
            data_frames(&out),
            vec![(0, write_st2038(&[packet]).unwrap())]
        );
        let gap = out.packets.iter().find_map(|p| match p {
            PipelinePacket::Gap {
                pts_ns,
                duration_ns,
            } => Some((*pts_ns, *duration_ns)),
            _ => None,
        });
        assert_eq!(gap, Some((FRAME_NS, FRAME_NS)));
    }

    #[test]
    fn without_a_framerate_payloads_group_by_equal_pts() {
        let (a, b) = (packet_at(9, 0, 1), packet_at(12, 0, 2));
        let mut mux = mux(
            Caps::St2038 {
                alignment: St2038Alignment::Frame,
                framerate: Rate::Any,
            },
            false,
        );
        let out = feed(
            &mut mux,
            vec![
                (0, payload_at(7, core::slice::from_ref(&a)), 0),
                (1, payload_at(7, core::slice::from_ref(&b)), 0),
                (0, payload_at(9, core::slice::from_ref(&a)), 0),
                (1, PipelinePacket::Eos, 0),
            ],
        );
        assert_eq!(data_frames(&out)[0], (7, write_st2038(&[a, b]).unwrap()));
    }
}
