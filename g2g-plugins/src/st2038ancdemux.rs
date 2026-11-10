// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    g2g_warn, st2038_any_alignment, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, Frame, FrameTiming, G2gError, MemoryDomain, MultiOutputElement,
    MultiOutputSink, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, St2038Alignment,
};

use crate::st2038::{parse_st2038, write_st2038};
use crate::st291::AncPacket;

const COMBINED_PAD: &str = "src";
// gst st2038ancdemux sends a gap on a stream once it falls this far behind the input
const LATE_STREAM_NS: u64 = 500_000_000;

fn any_st2038() -> CapsSet {
    st2038_any_alignment(Rate::Any)
}

fn packet_caps(framerate: Rate) -> Caps {
    Caps::St2038 {
        alignment: St2038Alignment::Packet,
        framerate,
    }
}

fn input_framerate(caps: &Caps) -> Result<Rate, G2gError> {
    match caps {
        Caps::St2038 { framerate, .. } => Ok(framerate.clone()),
        _ => Err(G2gError::CapsMismatch),
    }
}

// each packet of one ST 2038 payload with its bytes, None when the payload does not parse
fn split_packets(payload: &[u8]) -> Option<Vec<(AncPacket, Vec<u8>)>> {
    parse_st2038(payload)?
        .into_iter()
        .map(|packet| {
            let bytes = write_st2038(core::slice::from_ref(&packet)).ok()?;
            Some((packet, bytes))
        })
        .collect()
}

fn packet_frame(bytes: Vec<u8>, timing: FrameTiming, sequence: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        sequence,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AncStreamId {
    did: u8,
    sdid: u8,
    line_number: u16,
    horizontal_offset: u16,
}

impl AncStreamId {
    fn of(packet: &AncPacket) -> Self {
        Self {
            did: packet.did,
            sdid: packet.sdid,
            line_number: packet.line_number,
            horizontal_offset: packet.horizontal_offset,
        }
    }

    // gst st2038ancdemux names a stream's pad anc_%02x_%02x_at_%u_%u from DID, SDID, line and offset
    fn from_pad_name(name: &str) -> Option<Self> {
        let rest = name.strip_prefix("anc_")?;
        let (ids, location) = rest.split_once("_at_")?;
        let (did, sdid) = ids.split_once('_')?;
        let (line_number, horizontal_offset) = location.split_once('_')?;
        Some(Self {
            did: u8::from_str_radix(did, 16).ok()?,
            sdid: u8::from_str_radix(sdid, 16).ok()?,
            line_number: line_number.parse().ok()?,
            horizontal_offset: horizontal_offset.parse().ok()?,
        })
    }
}

#[derive(Debug, Default)]
pub struct St2038AncDemux {
    output_caps: Option<Caps>,
    caps_emitted: bool,
    sequence: u64,
    log_name: LogName,
}

impl St2038AncDemux {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AsyncElement for St2038AncDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        input_framerate(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input| match input_framerate(input) {
            Ok(framerate) => CapsSet::one(packet_caps(framerate)),
            Err(_) => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.output_caps = Some(packet_caps(input_framerate(absolute_caps)?));
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SMPTE ST-2038 ancillary metadata demuxer",
            "Metadata/Video/Demuxer",
            "Splits individual ancillary metadata streams from an SMPTE ST-2038 stream",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let output_caps = self.output_caps.clone().ok_or(G2gError::NotConfigured)?;
                    let payload = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let Some(packets) = split_packets(payload) else {
                        g2g_warn!(self, "dropped an ST 2038 payload that does not parse");
                        return Ok(());
                    };
                    if !self.caps_emitted && !packets.is_empty() {
                        out.push(PipelinePacket::CapsChanged(output_caps)).await?;
                        self.caps_emitted = true;
                    }
                    for (_, bytes) in packets {
                        let frame = packet_frame(bytes, frame.timing, self.sequence);
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    Ok(())
                }
                // the runner hands a transform its own solved output caps, upstream caps stop here
                PipelinePacket::CapsChanged(caps) => {
                    if self.output_caps.as_ref() != Some(&caps) {
                        return Ok(());
                    }
                    self.caps_emitted = true;
                    out.push(PipelinePacket::CapsChanged(caps))
                        .await
                        .map(|_| ())
                }
                // the runner arm forwards the trailing Eos
                PipelinePacket::Eos => Ok(()),
                other => out.push(other).await.map(|_| ()),
            }
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for St2038AncDemux {
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

impl PadTemplates for St2038AncDemux {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(any_st2038()),
            PadTemplate::source(CapsSet::one(packet_caps(Rate::Any))),
        ])
    }
}

pub(crate) async fn push_to_ports(
    ports: impl IntoIterator<Item = usize>,
    packet: impl Fn() -> PipelinePacket,
    out: &mut dyn MultiOutputSink,
) -> Result<(), G2gError> {
    for port in ports {
        out.push_to(port, packet()).await?;
    }
    Ok(())
}

#[derive(Debug)]
struct DemuxPort {
    // None for the combined `src` pad
    stream: Option<AncStreamId>,
    caps_emitted: bool,
    last_pts_ns: Option<u64>,
}

impl DemuxPort {
    fn new(stream: Option<AncStreamId>) -> Self {
        Self {
            stream,
            caps_emitted: false,
            last_pts_ns: None,
        }
    }

    fn carries(&self, packet: &AncPacket) -> bool {
        self.stream.is_none_or(|id| id == AncStreamId::of(packet))
    }
}

#[derive(Debug)]
pub struct St2038AncDemuxN {
    ports: Vec<DemuxPort>,
    framerate: Rate,
    sequence: u64,
    log_name: LogName,
}

impl St2038AncDemuxN {
    pub fn new(ports: usize) -> Self {
        Self {
            ports: (0..ports).map(|_| DemuxPort::new(None)).collect(),
            framerate: Rate::Any,
            sequence: 0,
            log_name: LogName::default(),
        }
    }

    async fn push_gaps_for_late_streams(
        &mut self,
        pts_ns: u64,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        for (index, port) in self.ports.iter_mut().enumerate() {
            if port.stream.is_none() {
                continue;
            }
            let last_pts_ns = *port.last_pts_ns.get_or_insert(pts_ns);
            let behind_ns = pts_ns.saturating_sub(last_pts_ns);
            if behind_ns <= LATE_STREAM_NS {
                continue;
            }
            port.last_pts_ns = Some(pts_ns);
            let gap = PipelinePacket::Gap {
                pts_ns: last_pts_ns,
                duration_ns: behind_ns,
            };
            out.push_to(index, gap).await?;
        }
        Ok(())
    }
}

impl MultiOutputElement for St2038AncDemuxN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        input_framerate(upstream_caps)?;
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(any_st2038())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.framerate = input_framerate(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        (port < self.ports.len()).then(|| packet_caps(Rate::Any))
    }

    fn name_output_ports(&mut self, pad_names: &[&str]) -> Result<(), usize> {
        let mut ports = Vec::with_capacity(pad_names.len());
        for (index, name) in pad_names.iter().enumerate() {
            let stream = match *name {
                "" | COMBINED_PAD => None,
                _ => Some(AncStreamId::from_pad_name(name).ok_or(index)?),
            };
            ports.push(DemuxPort::new(stream));
        }
        self.ports = ports;
        Ok(())
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SMPTE ST-2038 ancillary metadata demuxer",
            "Metadata/Video/Demuxer",
            "Splits individual ancillary metadata streams from an SMPTE ST-2038 stream",
            crate::AARON_BOXER,
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let payload = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let Some(packets) = split_packets(payload) else {
                        g2g_warn!(self, "dropped an ST 2038 payload that does not parse");
                        return Ok(());
                    };
                    let pts_ns = frame.timing.pts_ns;
                    for (packet, bytes) in &packets {
                        for index in 0..self.ports.len() {
                            if !self.ports[index].carries(packet) {
                                continue;
                            }
                            if !self.ports[index].caps_emitted {
                                let caps = packet_caps(self.framerate.clone());
                                out.push_to(index, PipelinePacket::CapsChanged(caps))
                                    .await?;
                                self.ports[index].caps_emitted = true;
                            }
                            self.ports[index].last_pts_ns = Some(pts_ns);
                            let frame = packet_frame(bytes.clone(), frame.timing, self.sequence);
                            self.sequence += 1;
                            out.push_to(index, PipelinePacket::DataFrame(frame)).await?;
                        }
                    }
                    self.push_gaps_for_late_streams(pts_ns, out).await
                }
                PipelinePacket::Segment(segment) => {
                    push_to_ports(
                        0..self.ports.len(),
                        || PipelinePacket::Segment(segment),
                        out,
                    )
                    .await
                }
                PipelinePacket::Flush => {
                    for port in &mut self.ports {
                        port.last_pts_ns = None;
                    }
                    push_to_ports(0..self.ports.len(), || PipelinePacket::Flush, out).await
                }
                PipelinePacket::Gap {
                    pts_ns,
                    duration_ns,
                } => {
                    let gap = || PipelinePacket::Gap {
                        pts_ns,
                        duration_ns,
                    };
                    push_to_ports(0..self.ports.len(), gap, out).await
                }
                // the runner broadcasts the trailing Eos, and each port announces its own caps
                _ => Ok(()),
            }
        })
    }
}

impl LogSource for St2038AncDemuxN {
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
    use crate::st291::{CAPTION_DID, CEA608_SDID, CEA708_CDP_SDID};
    use crate::testutil::{data_bytes, first_caps, frame_with_bytes, run, PortCollectSink};
    use alloc::vec;
    use g2g_core::caps::framerate_q16_from_fraction;

    const GST_CEA708_PACKET: &[u8] = include_bytes!("../tests/fixtures/st2038_cea708_gst.bin");
    const CEA608_S334: [u8; 3] = [0x80, 0x94, 0x2C];

    fn rate_25() -> Rate {
        Rate::Fixed(framerate_q16_from_fraction(25, 1).unwrap())
    }

    fn line_caps() -> Caps {
        Caps::St2038 {
            alignment: St2038Alignment::Line,
            framerate: rate_25(),
        }
    }

    fn cea608_at_line_9() -> AncPacket {
        let mut packet = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
        packet.line_number = 9;
        packet.horizontal_offset = 200;
        packet
    }

    fn timecode() -> AncPacket {
        AncPacket::from_bytes(0x60, 0x60, &[0x01, 0x02])
    }

    fn payload() -> Vec<u8> {
        let mut payload = GST_CEA708_PACKET.to_vec();
        payload.extend(write_st2038(&[cea608_at_line_9(), timecode()]).unwrap());
        payload
    }

    fn st2038(packet: AncPacket) -> Vec<u8> {
        write_st2038(&[packet]).unwrap()
    }

    fn run_ports(demux: &mut St2038AncDemuxN, frames: Vec<Frame>) -> PortCollectSink {
        demux.configure_pipeline(&line_caps()).unwrap();
        let mut out = PortCollectSink::new(demux.ports.len());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        for frame in frames {
            runtime
                .block_on(demux.process(PipelinePacket::DataFrame(frame), &mut out))
                .unwrap();
        }
        out
    }

    #[test]
    fn single_output_sends_each_packet_on_its_own() {
        let mut demux = St2038AncDemux::new();
        let out = run(&mut demux, &line_caps(), &[&payload()]);
        assert_eq!(first_caps(&out.packets), Some(packet_caps(rate_25())));
        let frames: Vec<Vec<u8>> = out
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.domain.as_system_slice().unwrap().to_vec()),
                _ => None,
            })
            .collect();
        assert_eq!(
            frames,
            vec![
                GST_CEA708_PACKET.to_vec(),
                st2038(cea608_at_line_9()),
                st2038(timecode())
            ]
        );
    }

    #[test]
    fn named_pads_carry_their_stream_and_src_carries_all() {
        let mut demux = St2038AncDemuxN::new(3);
        demux
            .name_output_ports(&["anc_61_02_at_9_200", "src", "anc_61_01_at_9_0"])
            .unwrap();
        let out = run_ports(&mut demux, vec![frame_with_bytes(0, payload())]);
        assert_eq!(data_bytes(&out.ports[0]), st2038(cea608_at_line_9()));
        assert_eq!(data_bytes(&out.ports[1]), payload());
        assert_eq!(data_bytes(&out.ports[2]), GST_CEA708_PACKET);
        assert_eq!(first_caps(&out.ports[2]), Some(packet_caps(rate_25())));
        assert_eq!(
            AncStreamId::from_pad_name("anc_61_01_at_9_0"),
            Some(AncStreamId {
                did: CAPTION_DID,
                sdid: CEA708_CDP_SDID,
                line_number: 9,
                horizontal_offset: 0
            })
        );
    }

    #[test]
    fn rejects_pad_names_that_name_no_stream() {
        let mut demux = St2038AncDemuxN::new(2);
        assert_eq!(
            demux.name_output_ports(&["src", "anc_zz_01_at_9_0"]),
            Err(1)
        );
        assert_eq!(demux.name_output_ports(&["video_0", "src"]), Err(0));
    }

    #[test]
    fn a_stream_that_falls_behind_gets_a_gap() {
        let mut demux = St2038AncDemuxN::new(2);
        demux
            .name_output_ports(&["src", "anc_61_02_at_9_200"])
            .unwrap();
        let late_ns = LATE_STREAM_NS + 1;
        let mut quiet = frame_with_bytes(0, st2038(timecode()));
        quiet.timing.pts_ns = late_ns;
        let out = run_ports(
            &mut demux,
            vec![frame_with_bytes(0, st2038(timecode())), quiet],
        );
        let gaps: Vec<(u64, u64)> = out.ports[1]
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::Gap {
                    pts_ns,
                    duration_ns,
                } => Some((*pts_ns, *duration_ns)),
                _ => None,
            })
            .collect();
        assert_eq!(gaps, vec![(0, late_ns)]);
        assert!(data_bytes(&out.ports[1]).is_empty());
    }
}
