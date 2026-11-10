use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    Frame, G2gError, MemoryDomain, MultiOutputElement, MultiOutputSink, OutputSink, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, St2038Alignment,
};

use crate::st2038::{framerate_period_ns, video_framerate, write_st2038};
use crate::st2038ancdemux::push_to_ports;
use crate::st291::{sort_by_position, AncillaryMeta};

const VIDEO_PAD: &str = "src";
const ST2038_PAD: &str = "st2038";

static ST2038EXTRACTOR_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "remove-ancillary-meta",
    PropKind::Bool,
    "remove the ancillary meta from the video frames sent on",
)
.with_default("false")];

fn st2038_caps(framerate: Rate) -> Caps {
    Caps::St2038 {
        alignment: St2038Alignment::Frame,
        framerate,
    }
}

fn set_remove_ancillary_meta(
    remove: &mut bool,
    name: &str,
    value: PropValue,
) -> Result<(), PropError> {
    match name {
        "remove-ancillary-meta" => *remove = value.as_bool().ok_or(PropError::Type)?,
        _ => return Err(PropError::Unknown),
    }
    Ok(())
}

fn get_remove_ancillary_meta(remove: bool, name: &str) -> Option<PropValue> {
    (name == "remove-ancillary-meta").then_some(PropValue::Bool(remove))
}

fn element_metadata() -> ElementMetadata {
    ElementMetadata::new(
        "Ancillary meta to ST-2038",
        "Generic",
        "Extracts ST2038 stream in ancillary meta from video input stream",
        crate::AARON_BOXER,
    )
}

#[derive(Debug, Default)]
pub struct St2038Extractor {
    remove_ancillary_meta: bool,
}

impl St2038Extractor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl AsyncElement for St2038Extractor {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        element_metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ST2038EXTRACTOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        set_remove_ancillary_meta(&mut self.remove_ancillary_meta, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        get_remove_ancillary_meta(self.remove_ancillary_meta, name)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) if self.remove_ancillary_meta => {
                    frame.meta.remove::<AncillaryMeta>();
                    out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
                }
                // the runner arm forwards the trailing Eos
                PipelinePacket::Eos => Ok(()),
                other => out.push(other).await.map(|_| ()),
            }
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtractorPort {
    Video,
    St2038 { caps_emitted: bool },
}

#[derive(Debug)]
pub struct St2038ExtractorN {
    ports: Vec<ExtractorPort>,
    framerate: Rate,
    remove_ancillary_meta: bool,
    sequence: u64,
    log_name: LogName,
}

impl St2038ExtractorN {
    // port 0 sends the video on and every other port the ST 2038 stream
    pub fn new(ports: usize) -> Self {
        Self {
            ports: (0..ports)
                .map(|port| match port {
                    0 => ExtractorPort::Video,
                    _ => ExtractorPort::St2038 {
                        caps_emitted: false,
                    },
                })
                .collect(),
            framerate: Rate::Any,
            remove_ancillary_meta: false,
            sequence: 0,
            log_name: LogName::default(),
        }
    }

    fn st2038_payload(&self, frame: &Frame) -> Vec<u8> {
        let Some(meta) = frame.meta.get::<AncillaryMeta>() else {
            return Vec::new();
        };
        let mut packets = meta.packets.clone();
        sort_by_position(&mut packets);
        let mut payload = Vec::new();
        for packet in &packets {
            match write_st2038(core::slice::from_ref(packet)) {
                Ok(bytes) => payload.extend(bytes),
                Err(overflow) => g2g_warn!(
                    self,
                    "dropped an ancillary packet whose field does not fit ST 2038: {overflow:?}"
                ),
            }
        }
        payload
    }

    // a frame with no ancillary packets sends a gap as long as the frame
    fn empty_frame_gap(&self, frame: &Frame) -> PipelinePacket {
        let period_ns = framerate_period_ns(&self.framerate);
        let duration_ns = match frame.timing.duration_ns {
            0 => period_ns.unwrap_or(0),
            duration_ns => duration_ns,
        };
        PipelinePacket::Gap {
            pts_ns: frame.timing.pts_ns,
            duration_ns,
        }
    }

    async fn push_st2038(
        &mut self,
        frame: &Frame,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let payload = self.st2038_payload(frame);
        let st2038_frame = (!payload.is_empty()).then(|| {
            let mut st2038_frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
                frame.timing,
                self.sequence,
            );
            st2038_frame.domain.make_shareable();
            st2038_frame
        });
        if st2038_frame.is_some() {
            self.sequence += 1;
        }
        for index in 0..self.ports.len() {
            let ExtractorPort::St2038 { caps_emitted } = &mut self.ports[index] else {
                continue;
            };
            let Some(st2038_frame) = &st2038_frame else {
                out.push_to(index, self.empty_frame_gap(frame)).await?;
                continue;
            };
            if !*caps_emitted {
                *caps_emitted = true;
                let caps = st2038_caps(self.framerate.clone());
                out.push_to(index, PipelinePacket::CapsChanged(caps))
                    .await?;
            }
            out.push_to(index, PipelinePacket::DataFrame(st2038_frame.share()))
                .await?;
        }
        Ok(())
    }

    fn ports_where(&self, kind: impl Fn(&ExtractorPort) -> bool) -> Vec<usize> {
        (0..self.ports.len())
            .filter(|&index| kind(&self.ports[index]))
            .collect()
    }

    async fn push_video(
        &self,
        mut frame: Frame,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        frame.domain.make_shareable();
        let video_ports = self.ports_where(|port| *port == ExtractorPort::Video);
        push_to_ports(
            video_ports,
            || PipelinePacket::DataFrame(frame.share()),
            out,
        )
        .await
    }

    async fn handle_caps(
        &mut self,
        caps: Caps,
        out: &mut dyn MultiOutputSink,
    ) -> Result<(), G2gError> {
        let framerate = video_framerate(&caps);
        let framerate_changed = framerate != self.framerate;
        self.framerate = framerate;
        let video_ports = self.ports_where(|port| *port == ExtractorPort::Video);
        push_to_ports(
            video_ports,
            || PipelinePacket::CapsChanged(caps.clone()),
            out,
        )
        .await?;
        if !framerate_changed {
            return Ok(());
        }
        let announced_st2038_ports =
            self.ports_where(|port| *port == ExtractorPort::St2038 { caps_emitted: true });
        let caps = st2038_caps(self.framerate.clone());
        push_to_ports(
            announced_st2038_ports,
            || PipelinePacket::CapsChanged(caps.clone()),
            out,
        )
        .await
    }
}

impl MultiOutputElement for St2038ExtractorN {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.framerate = video_framerate(absolute_caps);
        Ok(ConfigureOutcome::Accepted)
    }

    fn port_caps_constraint(&self, port: usize) -> Option<CapsConstraint<'_>> {
        match self.ports.get(port)? {
            ExtractorPort::Video => Some(CapsConstraint::IdentityAny),
            ExtractorPort::St2038 { .. } => {
                Some(CapsConstraint::DerivedOutput(Box::new(|input| {
                    CapsSet::one(st2038_caps(video_framerate(input)))
                })))
            }
        }
    }

    fn name_output_ports(&mut self, pad_names: &[&str]) -> Result<(), usize> {
        let mut ports = Vec::with_capacity(pad_names.len());
        for (index, name) in pad_names.iter().enumerate() {
            ports.push(match *name {
                "" | VIDEO_PAD => ExtractorPort::Video,
                ST2038_PAD => ExtractorPort::St2038 {
                    caps_emitted: false,
                },
                _ => return Err(index),
            });
        }
        self.ports = ports;
        Ok(())
    }

    fn metadata(&self) -> ElementMetadata {
        element_metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ST2038EXTRACTOR_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        set_remove_ancillary_meta(&mut self.remove_ancillary_meta, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        get_remove_ancillary_meta(self.remove_ancillary_meta, name)
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
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    self.push_st2038(&frame, out).await?;
                    if self.remove_ancillary_meta {
                        frame.meta.remove::<AncillaryMeta>();
                    }
                    self.push_video(frame, out).await
                }
                PipelinePacket::CapsChanged(caps) => self.handle_caps(caps, out).await,
                PipelinePacket::Segment(segment) => {
                    push_to_ports(
                        0..self.ports.len(),
                        || PipelinePacket::Segment(segment),
                        out,
                    )
                    .await
                }
                PipelinePacket::Flush => {
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
                // the runner broadcasts the trailing Eos
                _ => Ok(()),
            }
        })
    }
}

impl LogSource for St2038ExtractorN {
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
    use crate::st291::AncPacket;
    use crate::testutil::{frame_with_bytes, PortCollectSink};
    use alloc::vec;

    const FRAME_NS: u64 = 40_000_000;
    const VIDEO_PORT: usize = 0;
    const ST2038_PORT: usize = 1;

    fn packet_at_line(line_number: u16) -> AncPacket {
        let mut packet = AncPacket::from_bytes(0x60, 0x60, &[line_number as u8]);
        packet.line_number = line_number;
        packet.horizontal_offset = 0;
        packet
    }

    fn video_frame(packets: Option<Vec<AncPacket>>) -> Frame {
        let mut frame = frame_with_bytes(0, vec![0; 4]);
        frame.timing.pts_ns = FRAME_NS;
        frame.timing.duration_ns = FRAME_NS;
        if let Some(packets) = packets {
            frame.meta.attach(AncillaryMeta { packets });
        }
        frame
    }

    fn extract(extractor: &mut St2038ExtractorN, frame: Frame) -> PortCollectSink {
        let mut out = PortCollectSink::new(extractor.ports.len());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime
            .block_on(extractor.process(PipelinePacket::DataFrame(frame), &mut out))
            .unwrap();
        out
    }

    fn video_meta(out: &PortCollectSink) -> Option<AncillaryMeta> {
        match &out.ports[VIDEO_PORT][..] {
            [PipelinePacket::DataFrame(frame)] => frame.meta.get::<AncillaryMeta>().cloned(),
            other => panic!("expected one video frame, got {other:?}"),
        }
    }

    #[test]
    fn the_meta_goes_out_as_one_frame_aligned_payload_sorted_by_line() {
        let mut extractor = St2038ExtractorN::new(2);
        extractor
            .name_output_ports(&[VIDEO_PAD, ST2038_PAD])
            .unwrap();
        let packets = vec![packet_at_line(12), packet_at_line(9)];
        let out = extract(&mut extractor, video_frame(Some(packets.clone())));
        let [PipelinePacket::CapsChanged(caps), PipelinePacket::DataFrame(st2038)] =
            &out.ports[ST2038_PORT][..]
        else {
            panic!(
                "expected caps then one payload: {:?}",
                out.ports[ST2038_PORT]
            );
        };
        assert_eq!(*caps, st2038_caps(Rate::Any));
        assert_eq!(st2038.timing.pts_ns, FRAME_NS);
        assert_eq!(
            st2038.domain.as_system_slice().unwrap(),
            &write_st2038(&[packet_at_line(9), packet_at_line(12)]).unwrap()[..]
        );
        assert_eq!(video_meta(&out), Some(AncillaryMeta { packets }));
    }

    #[test]
    fn remove_ancillary_meta_strips_the_video_frames() {
        let mut extractor = St2038ExtractorN::new(2);
        extractor
            .set_property("remove-ancillary-meta", PropValue::Bool(true))
            .unwrap();
        let out = extract(&mut extractor, video_frame(Some(vec![packet_at_line(9)])));
        assert_eq!(video_meta(&out), None);
    }

    #[test]
    fn a_frame_without_meta_sends_a_gap_as_long_as_the_frame() {
        let mut extractor = St2038ExtractorN::new(2);
        let out = extract(&mut extractor, video_frame(None));
        assert!(matches!(
            out.ports[ST2038_PORT][..],
            [PipelinePacket::Gap {
                pts_ns: FRAME_NS,
                duration_ns: FRAME_NS
            }]
        ));
    }

    #[test]
    fn an_unknown_pad_name_is_rejected() {
        let mut extractor = St2038ExtractorN::new(2);
        assert_eq!(extractor.name_output_ports(&[VIDEO_PAD, "anc"]), Err(1));
    }
}
