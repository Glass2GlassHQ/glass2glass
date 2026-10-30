use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{AudioFormat, Caps, CapsSet, G2gError, OutputSink, PadTemplate, PipelinePacket};

use crate::demuxseek::Admitted;

// a parser or tagger takes and gives its formats at any channel count and rate, which the
// stream headers refine
pub(crate) fn passthrough_caps(formats: &[AudioFormat]) -> CapsSet {
    CapsSet::from_alternatives(
        formats
            .iter()
            .map(|&format| crate::typefind::elementary_audio_caps(format))
            .collect(),
    )
}

pub(crate) fn require_format(formats: &[AudioFormat], caps: &Caps) -> Result<(), G2gError> {
    match caps {
        Caps::Audio { format, .. } if formats.contains(format) => Ok(()),
        _ => Err(G2gError::CapsMismatch),
    }
}

pub(crate) fn pad_templates(formats: &[AudioFormat]) -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(passthrough_caps(formats)),
        PadTemplate::source(passthrough_caps(formats)),
    ])
}

// the decoder adds the previous frame's filter tail to each frame it decodes
pub(crate) const OVERLAP_PREROLL_FRAMES: usize = 1;

pub(crate) async fn push_admitted(
    admitted: Admitted<Frame>,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    if let Some(segment) = admitted.segment {
        out.push(PipelinePacket::Segment(segment)).await?;
    }
    for frame in admitted {
        out.push(PipelinePacket::DataFrame(frame)).await?;
    }
    Ok(())
}
