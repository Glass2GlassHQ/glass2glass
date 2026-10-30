use alloc::vec::Vec;

use g2g_core::{pcm_caps_any_rate_and_channels, AudioFormat, CapsSet, PadTemplate};

// a filter that rewrites samples and keeps the caps takes and gives the same set
pub(crate) fn pad_templates(caps: CapsSet) -> Vec<PadTemplate> {
    Vec::from([PadTemplate::sink(caps.clone()), PadTemplate::source(caps)])
}

pub(crate) fn s16_caps() -> CapsSet {
    pcm_caps_any_rate_and_channels(|format| format == AudioFormat::PcmS16Le)
}
