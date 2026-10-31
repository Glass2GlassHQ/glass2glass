use alloc::boxed::Box;
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;
use core::marker::PhantomData;
use core::pin::Pin;

use g2g_core::memory::SystemSlice;
use g2g_core::meta::{SeiUserData, SeiUserDataMeta};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::annexb::vcl_start;
use crate::h264parse::H264Codec;
use crate::h265parse::H265Codec;
use crate::nalparse::NalCodec;
use crate::sei::{build_sei_nal, PAYLOAD_USER_DATA_REGISTERED, PAYLOAD_USER_DATA_UNREGISTERED};

pub type H264SeiInserter = SeiInsert<H264Codec>;
pub type H265SeiInserter = SeiInsert<H265Codec>;

pub struct SeiInsert<C: NalCodec> {
    codec: PhantomData<C>,
}

impl<C: NalCodec> fmt::Debug for SeiInsert<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SeiInsert")
            .field("codec", &C::CODEC)
            .finish()
    }
}

impl<C: NalCodec> Default for SeiInsert<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: NalCodec> SeiInsert<C> {
    pub fn new() -> Self {
        Self { codec: PhantomData }
    }

    fn any_caps() -> Caps {
        Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::CompressedVideo { codec, .. } if *codec == C::CODEC)
    }
}

// `au` with one SEI NAL per message placed before its first slice, `None` when it has no slice to precede
pub fn insert_user_data(au: &[u8], codec: VideoCodec, messages: &[SeiUserData]) -> Option<Vec<u8>> {
    let first_slice = vcl_start(au, codec)?;
    let nals: Vec<Vec<u8>> = messages
        .iter()
        .map(|message| match message {
            SeiUserData::Registered { t35 } => {
                build_sei_nal(PAYLOAD_USER_DATA_REGISTERED, t35, codec)
            }
            SeiUserData::Unregistered { uuid, payload } => {
                let mut body = Vec::with_capacity(uuid.len() + payload.len());
                body.extend_from_slice(uuid);
                body.extend_from_slice(payload);
                build_sei_nal(PAYLOAD_USER_DATA_UNREGISTERED, &body, codec)
            }
        })
        .collect();
    let mut bytes = Vec::with_capacity(au.len() + nals.iter().map(Vec::len).sum::<usize>());
    bytes.extend_from_slice(&au[..first_slice]);
    for nal in &nals {
        bytes.extend_from_slice(nal);
    }
    bytes.extend_from_slice(&au[first_slice..]);
    Some(bytes)
}

impl<C: NalCodec> AsyncElement for SeiInsert<C> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(CapsSet::one(Self::any_caps()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if Self::accepts(absolute_caps) {
            Ok(ConfigureOutcome::Accepted)
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "SEI user data inserter",
            "Codec/Video/Filter",
            "Writes each frame's SeiUserDataMeta into its access unit as SEI user data messages",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            let PipelinePacket::DataFrame(mut frame) = packet else {
                return out.push(packet).await.map(|_| ());
            };
            let with_user_data = frame
                .meta
                .get::<SeiUserDataMeta>()
                .zip(frame.domain.as_system_slice())
                .and_then(|(meta, au)| insert_user_data(au, C::CODEC, &meta.messages));
            if let Some(bytes) = with_user_data {
                frame.domain =
                    MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
            }
            out.push(PipelinePacket::DataFrame(frame)).await.map(|_| ())
        })
    }
}

impl<C: NalCodec> PadTemplates for SeiInsert<C> {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::any_caps())),
            PadTemplate::source(CapsSet::one(Self::any_caps())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sei::for_each_message;

    const UUID: [u8; 16] = *b"g2g-sei-unreg-16";
    // United States country code, then the ATSC provider code and a user identifier
    const T35: [u8; 7] = [0xB5, 0x00, 0x31, b'T', b'E', b'S', b'T'];

    fn messages() -> Vec<SeiUserData> {
        Vec::from([
            SeiUserData::Registered { t35: T35.to_vec() },
            // three zero bytes force emulation prevention inside the NAL
            SeiUserData::Unregistered {
                uuid: UUID,
                payload: Vec::from([0, 0, 0, 1, 0xAB]),
            },
        ])
    }

    #[test]
    fn both_kinds_read_back_in_order_before_the_slice() {
        for (codec, slice) in [
            (VideoCodec::H264, &[0, 0, 0, 1, 0x65, 0x88, 0x84][..]),
            (VideoCodec::H265, &[0, 0, 0, 1, 0x26, 0x01, 0xAF][..]),
        ] {
            let au = slice.to_vec();
            let written = insert_user_data(&au, codec, &messages()).expect("the AU has a slice");
            let mut read = Vec::new();
            for_each_message(&written, codec, |payload_type, payload| {
                read.push((payload_type, payload.to_vec()))
            });
            let mut unregistered = UUID.to_vec();
            unregistered.extend_from_slice(&[0, 0, 0, 1, 0xAB]);
            assert_eq!(
                read,
                Vec::from([
                    (PAYLOAD_USER_DATA_REGISTERED, T35.to_vec()),
                    (PAYLOAD_USER_DATA_UNREGISTERED, unregistered),
                ]),
                "{codec:?}"
            );
            assert!(written.ends_with(slice), "{codec:?} slice stays last");
        }
    }

    #[test]
    fn an_access_unit_without_a_slice_is_left_alone() {
        let sps_only = [0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E];
        assert!(insert_user_data(&sps_only, VideoCodec::H264, &messages()).is_none());
    }
}
