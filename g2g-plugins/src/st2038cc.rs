use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    g2g_warn, st2038_any_alignment, AsyncElement, Caps, CapsConstraint, CapsSet,
    ClosedCaptionFormat, ConfigureOutcome, ElementMetadata, Frame, FrameTiming, G2gError,
    MemoryDomain, MultiOutputElement, MultiOutputSink, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, St2038Alignment,
};

use crate::st2038::{parse_st2038, write_st2038};
use crate::st291::{
    word_with_parity, AncPacket, CAPTION_DID, CEA608_SDID, CEA708_CDP_SDID, HORIZONTAL_OFFSET_BITS,
    LINE_NUMBER_BITS,
};

#[derive(Debug)]
struct CaptionCarriage {
    format: ClosedCaptionFormat,
    sdid: u8,
    demux_pad: &'static str,
}

impl CaptionCarriage {
    fn caps(&self) -> Caps {
        Caps::ClosedCaption {
            format: self.format,
        }
    }
}

// gst-plugins-rs st2038anctocc names its pads src_cea708 and src_cea608
static CAPTION_CARRIAGES: [CaptionCarriage; 2] = [
    CaptionCarriage {
        format: ClosedCaptionFormat::Cea708Cdp,
        sdid: CEA708_CDP_SDID,
        demux_pad: "src_cea708",
    },
    CaptionCarriage {
        format: ClosedCaptionFormat::Cea608S334,
        sdid: CEA608_SDID,
        demux_pad: "src_cea608",
    },
];

fn carriage_of_caps(caps: &Caps) -> Option<&'static CaptionCarriage> {
    CAPTION_CARRIAGES.iter().find(|c| c.caps() == *caps)
}

fn caption_caps_set() -> CapsSet {
    CapsSet::from_alternatives(
        CAPTION_CARRIAGES
            .iter()
            .map(CaptionCarriage::caps)
            .collect(),
    )
}

fn caption_frame(packet: &AncPacket, timing: FrameTiming, sequence: u64) -> Frame {
    let bytes = packet.user_data_bytes().into_boxed_slice();
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes)),
        timing,
        sequence,
    )
}

fn accept_st2038(caps: &Caps) -> Result<Caps, G2gError> {
    match caps {
        Caps::St2038 { .. } => Ok(caps.clone()),
        _ => Err(G2gError::CapsMismatch),
    }
}

fn any_st2038() -> CapsSet {
    st2038_any_alignment(Rate::Any)
}

// gst cctost2038anc writes alignment=frame unless downstream asks for another
fn cctost2038anc_output() -> CapsSet {
    let alignments = [
        St2038Alignment::Frame,
        St2038Alignment::Line,
        St2038Alignment::Packet,
    ];
    CapsSet::from_alternatives(
        alignments
            .into_iter()
            .map(|alignment| Caps::St2038 {
                alignment,
                framerate: Rate::Any,
            })
            .collect(),
    )
}

// gst cctost2038anc writes line 9 unless told otherwise
const DEFAULT_LINE_NUMBER: u16 = 9;

static CCTOST2038ANC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "c-not-y-channel",
        PropKind::Bool,
        "mark the written packets as colour-difference channel data",
    )
    .with_default("false"),
    PropertySpec::new(
        "line-number",
        PropKind::Uint,
        "SDI line number of the written packets",
    )
    .with_range("0", "2047")
    .with_default("9"),
    PropertySpec::new(
        "horizontal-offset",
        PropKind::Uint,
        "horizontal offset of the written packets",
    )
    .with_range("0", "4095")
    .with_default("0"),
];

#[derive(Debug)]
pub struct CcToSt2038Anc {
    c_not_y_channel: bool,
    line_number: u16,
    horizontal_offset: u16,
    sdid: Option<u8>,
    output_caps: Option<Caps>,
    caps_emitted: bool,
    log_name: LogName,
}

impl Default for CcToSt2038Anc {
    fn default() -> Self {
        Self {
            c_not_y_channel: false,
            line_number: DEFAULT_LINE_NUMBER,
            horizontal_offset: 0,
            sdid: None,
            output_caps: None,
            caps_emitted: false,
            log_name: LogName::default(),
        }
    }
}

impl CcToSt2038Anc {
    pub fn new() -> Self {
        Self::default()
    }

    fn st2038_payload(&self, sdid: u8, captions: &[u8]) -> Option<Vec<u8>> {
        let packet = AncPacket {
            c_not_y_channel: self.c_not_y_channel,
            line_number: self.line_number,
            horizontal_offset: self.horizontal_offset,
            did: CAPTION_DID,
            sdid,
            user_data: captions.iter().map(|&b| word_with_parity(b)).collect(),
        };
        match write_st2038(&[packet]) {
            Ok(payload) => Some(payload),
            Err(overflow) => {
                g2g_warn!(
                    self,
                    "dropped a {} byte caption payload: {overflow:?}",
                    captions.len()
                );
                None
            }
        }
    }
}

fn bit_width_value(value: PropValue, bits: u32) -> Result<u16, PropError> {
    let PropValue::Uint(v) = value else {
        return Err(PropError::Type);
    };
    if v >> bits != 0 {
        return Err(PropError::Value);
    }
    Ok(v as u16)
}

impl AsyncElement for CcToSt2038Anc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        carriage_of_caps(upstream_caps)
            .map(CaptionCarriage::caps)
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Vec::from([(caption_caps_set(), cctost2038anc_output())]))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let carriage = carriage_of_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.sdid = Some(carriage.sdid);
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        self.output_caps = Some(accept_st2038(output_caps)?);
        Ok(())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "CC to ST-2038 ANC",
            "Generic",
            "Wraps CEA-708 CDPs and CEA-608 S334-1A captions in SMPTE ST 2038 ancillary packets",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CCTOST2038ANC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "c-not-y-channel" => {
                let PropValue::Bool(flag) = value else {
                    return Err(PropError::Type);
                };
                self.c_not_y_channel = flag;
            }
            "line-number" => self.line_number = bit_width_value(value, LINE_NUMBER_BITS)?,
            "horizontal-offset" => {
                self.horizontal_offset = bit_width_value(value, HORIZONTAL_OFFSET_BITS)?
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "c-not-y-channel" => Some(PropValue::Bool(self.c_not_y_channel)),
            "line-number" => Some(PropValue::Uint(u64::from(self.line_number))),
            "horizontal-offset" => Some(PropValue::Uint(u64::from(self.horizontal_offset))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let sdid = self.sdid.ok_or(G2gError::NotConfigured)?;
                    let output_caps = self.output_caps.clone().ok_or(G2gError::NotConfigured)?;
                    let captions = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let Some(payload) = self.st2038_payload(sdid, captions) else {
                        return Ok(());
                    };
                    if !self.caps_emitted {
                        out.push(PipelinePacket::CapsChanged(output_caps)).await?;
                        self.caps_emitted = true;
                    }
                    let st2038 = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                        frame.timing,
                        frame.sequence,
                    );
                    out.push(PipelinePacket::DataFrame(st2038))
                        .await
                        .map(|_| ())
                }
                // the runner hands a transform its own solved output caps, upstream caption caps stop here
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

impl LogSource for CcToSt2038Anc {
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

impl PadTemplates for CcToSt2038Anc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(caption_caps_set()),
            PadTemplate::source(any_st2038()),
        ])
    }
}

#[derive(Debug, Default)]
pub struct St2038AncToCc {
    carriage: Option<&'static CaptionCarriage>,
    caps_emitted: bool,
    sequence: u64,
    log_name: LogName,
}

impl St2038AncToCc {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AsyncElement for St2038AncToCc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        accept_st2038(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Vec::from([(any_st2038(), caption_caps_set())]))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        accept_st2038(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        self.carriage = Some(carriage_of_caps(output_caps).ok_or(G2gError::CapsMismatch)?);
        Ok(())
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "ST-2038 ANC to CC",
            "Generic",
            "Extracts CEA-708 CDPs or CEA-608 S334-1A captions from SMPTE ST 2038 ancillary packets",
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
                    let carriage = self.carriage.ok_or(G2gError::NotConfigured)?;
                    let payload = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let Some(packets) = parse_st2038(payload) else {
                        g2g_warn!(self, "dropped an ST 2038 payload that does not parse");
                        return Ok(());
                    };
                    let captions = packets
                        .iter()
                        .filter(|p| p.did == CAPTION_DID && p.sdid == carriage.sdid);
                    for caption in captions {
                        if !self.caps_emitted {
                            out.push(PipelinePacket::CapsChanged(carriage.caps()))
                                .await?;
                            self.caps_emitted = true;
                        }
                        let frame = caption_frame(caption, frame.timing, self.sequence);
                        self.sequence += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    Ok(())
                }
                // the runner hands a transform its own solved output caps, the upstream St2038 caps stop here
                PipelinePacket::CapsChanged(caps) => {
                    if self.carriage.is_none_or(|c| c.caps() != caps) {
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

impl LogSource for St2038AncToCc {
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

impl PadTemplates for St2038AncToCc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(any_st2038()),
            PadTemplate::source(caption_caps_set()),
        ])
    }
}

#[derive(Debug)]
pub struct St2038AncToCcDemux {
    port_carriages: Vec<&'static CaptionCarriage>,
    caps_emitted: Vec<bool>,
    sequence: u64,
    log_name: LogName,
}

impl Default for St2038AncToCcDemux {
    fn default() -> Self {
        Self {
            port_carriages: CAPTION_CARRIAGES.iter().collect(),
            caps_emitted: alloc::vec![false; CAPTION_CARRIAGES.len()],
            sequence: 0,
            log_name: LogName::default(),
        }
    }
}

impl St2038AncToCcDemux {
    pub fn new() -> Self {
        Self::default()
    }

    async fn broadcast(
        &self,
        packet: impl Fn() -> PipelinePacket,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        for port in 0..self.port_carriages.len() {
            out.push_to(port, packet()).await?;
        }
        Ok(())
    }
}

impl MultiOutputElement for St2038AncToCcDemux {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        accept_st2038(upstream_caps)
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(any_st2038())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        accept_st2038(absolute_caps)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        self.port_carriages.get(port).map(|c| c.caps())
    }

    fn name_output_ports(&mut self, pad_names: &[&str]) -> Result<(), usize> {
        if pad_names.len() > CAPTION_CARRIAGES.len() {
            return Err(CAPTION_CARRIAGES.len());
        }
        let mut named: Vec<&'static CaptionCarriage> = Vec::new();
        for (port, name) in pad_names.iter().enumerate() {
            let carriage = CAPTION_CARRIAGES
                .iter()
                .find(|c| c.demux_pad == *name)
                .ok_or(port)?;
            if named.iter().any(|c| c.sdid == carriage.sdid) {
                return Err(port);
            }
            named.push(carriage);
        }
        self.caps_emitted = alloc::vec![false; named.len()];
        self.port_carriages = named;
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
            "ST-2038 ANC to CC",
            "Generic",
            "Splits CEA-708 CDPs and CEA-608 S334-1A captions out of SMPTE ST 2038 ancillary packets",
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
                    let Some(packets) = parse_st2038(payload) else {
                        g2g_warn!(self, "dropped an ST 2038 payload that does not parse");
                        return Ok(());
                    };
                    for caption in packets.iter().filter(|p| p.did == CAPTION_DID) {
                        let Some(port) = self
                            .port_carriages
                            .iter()
                            .position(|c| c.sdid == caption.sdid)
                        else {
                            continue;
                        };
                        if !self.caps_emitted[port] {
                            let caps = self.port_carriages[port].caps();
                            out.push_to(port, PipelinePacket::CapsChanged(caps)).await?;
                            self.caps_emitted[port] = true;
                        }
                        let frame = caption_frame(caption, frame.timing, self.sequence);
                        self.sequence += 1;
                        out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    }
                    Ok(())
                }
                PipelinePacket::Segment(segment) => {
                    self.broadcast(|| PipelinePacket::Segment(segment), out)
                        .await
                }
                PipelinePacket::Flush => self.broadcast(|| PipelinePacket::Flush, out).await,
                // the runner broadcasts the trailing Eos, and each port announces its own caps
                _ => Ok(()),
            }
        })
    }
}

impl LogSource for St2038AncToCcDemux {
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
    use crate::cea::parse_cdp;
    use crate::testutil::{data_bytes, first_caps, frame, run, PortCollectSink};
    use alloc::vec;

    // gst-plugins-rs's st2038 test made this with mccparse ! st2038anctocc ! cctost2038anc at default properties
    const GST_CEA708_PACKET: &[u8] = include_bytes!("../tests/fixtures/st2038_cea708_gst.bin");
    const CEA608_S334: [u8; 3] = [0x80, 0x94, 0x2C];

    fn caps(format: ClosedCaptionFormat) -> Caps {
        Caps::ClosedCaption { format }
    }

    fn frame_st2038() -> Caps {
        Caps::St2038 {
            alignment: St2038Alignment::Frame,
            framerate: Rate::Any,
        }
    }

    fn cctost2038anc_to(output: &Caps) -> CcToSt2038Anc {
        let mut to_st2038 = CcToSt2038Anc::new();
        to_st2038.configure_output(output).unwrap();
        to_st2038
    }

    #[test]
    fn caption_round_trip_matches_gst_cctost2038anc() {
        let cea708 = caps(ClosedCaptionFormat::Cea708Cdp);
        let mut to_captions = St2038AncToCc::new();
        to_captions.configure_output(&cea708).unwrap();
        let captions = run(&mut to_captions, &frame_st2038(), &[GST_CEA708_PACKET]);
        assert_eq!(first_caps(&captions.packets), Some(cea708.clone()));
        let cdp = data_bytes(&captions.packets);
        assert!(parse_cdp(&cdp).is_some());

        let st2038 = run(&mut cctost2038anc_to(&frame_st2038()), &cea708, &[&cdp]);
        assert_eq!(first_caps(&st2038.packets), Some(frame_st2038()));
        assert_eq!(data_bytes(&st2038.packets), GST_CEA708_PACKET);
    }

    #[test]
    fn cea608_packet_takes_the_location_properties() {
        let mut to_st2038 = cctost2038anc_to(&frame_st2038());
        to_st2038
            .set_property("line-number", PropValue::Uint(21))
            .unwrap();
        to_st2038
            .set_property("horizontal-offset", PropValue::Uint(5))
            .unwrap();
        to_st2038
            .set_property("c-not-y-channel", PropValue::Bool(true))
            .unwrap();
        let cea608 = caps(ClosedCaptionFormat::Cea608S334);
        let st2038 = run(&mut to_st2038, &cea608, &[&CEA608_S334]);
        let packets = parse_st2038(&data_bytes(&st2038.packets)).unwrap();
        let mut expected = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
        expected.line_number = 21;
        expected.horizontal_offset = 5;
        expected.c_not_y_channel = true;
        assert_eq!(packets, vec![expected]);

        let mut to_cea708 = St2038AncToCc::new();
        to_cea708
            .configure_output(&caps(ClosedCaptionFormat::Cea708Cdp))
            .unwrap();
        let cea708 = run(
            &mut to_cea708,
            &frame_st2038(),
            &[&data_bytes(&st2038.packets)],
        );
        assert!(data_bytes(&cea708.packets).is_empty());
    }

    #[test]
    fn location_properties_reject_values_wider_than_the_field() {
        let mut to_st2038 = CcToSt2038Anc::new();
        assert_eq!(
            to_st2038.set_property("line-number", PropValue::Uint(1 << LINE_NUMBER_BITS)),
            Err(PropError::Value)
        );
        assert_eq!(
            to_st2038.set_property(
                "horizontal-offset",
                PropValue::Uint(1 << HORIZONTAL_OFFSET_BITS)
            ),
            Err(PropError::Value)
        );
        assert_eq!(
            to_st2038.get_property("line-number"),
            Some(PropValue::Uint(u64::from(DEFAULT_LINE_NUMBER)))
        );
    }

    #[test]
    fn demux_routes_each_caption_type_to_its_named_pad() {
        let timecode = AncPacket::from_bytes(0x60, 0x60, &[0x01]);
        let cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
        let cdp = parse_st2038(GST_CEA708_PACKET).unwrap().remove(0);
        let payload = write_st2038(&[timecode, cea608, cdp.clone()]).unwrap();

        let mut demux = St2038AncToCcDemux::new();
        demux
            .name_output_ports(&["src_cea608", "src_cea708"])
            .unwrap();
        demux.configure_pipeline(&frame_st2038()).unwrap();
        let mut out = PortCollectSink::new(2);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime
            .block_on(demux.process(PipelinePacket::DataFrame(frame(payload)), &mut out))
            .unwrap();

        assert_eq!(
            first_caps(&out.ports[0]),
            Some(caps(ClosedCaptionFormat::Cea608S334))
        );
        assert_eq!(data_bytes(&out.ports[0]), CEA608_S334);
        assert_eq!(
            first_caps(&out.ports[1]),
            Some(caps(ClosedCaptionFormat::Cea708Cdp))
        );
        assert_eq!(data_bytes(&out.ports[1]), cdp.user_data_bytes());
    }

    #[test]
    fn demux_with_one_named_pad_carries_only_that_caption_type() {
        let cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
        let mut payload = GST_CEA708_PACKET.to_vec();
        payload.extend(write_st2038(&[cea608]).unwrap());
        let mut demux = St2038AncToCcDemux::new();
        demux.name_output_ports(&["src_cea608"]).unwrap();
        assert_eq!(
            demux.port_output_caps(0),
            Some(caps(ClosedCaptionFormat::Cea608S334))
        );
        assert_eq!(demux.port_output_caps(1), None);
        let mut out = PortCollectSink::new(1);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime
            .block_on(demux.process(PipelinePacket::DataFrame(frame(payload)), &mut out))
            .unwrap();
        assert_eq!(data_bytes(&out.ports[0]), CEA608_S334);
    }

    #[test]
    fn demux_rejects_unknown_and_repeated_pads() {
        let mut demux = St2038AncToCcDemux::new();
        assert_eq!(demux.name_output_ports(&["src_cea708", ""]), Err(1));
        assert_eq!(
            demux.name_output_ports(&["src_cea708", "src_cea708"]),
            Err(1)
        );
        assert_eq!(
            demux.name_output_ports(&["src_cea608", "src_cea708", "src_cea608"]),
            Err(2)
        );
    }
}
