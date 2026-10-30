use alloc::string::String;

use g2g_core::{Caps, CapsSet};

// a decoder's sink template can pin a channel count and rate the stream states only once
// parsed, so a consumer takes a stream by its codec
pub(crate) fn takes(accepted: &CapsSet, carried: &Caps) -> bool {
    accepted
        .alternatives()
        .iter()
        .any(|caps| same_codec(caps, carried))
}

fn same_codec(a: &Caps, b: &Caps) -> bool {
    match (a, b) {
        (Caps::CompressedVideo { codec: x, .. }, Caps::CompressedVideo { codec: y, .. }) => x == y,
        (Caps::Audio { format: x, .. }, Caps::Audio { format: y, .. }) => x == y,
        _ => a == b,
    }
}

// the first of `streams` the consumer takes, `None` while the demux's default caps
// already fit it, so a line that negotiates today keeps its stream
pub(crate) fn stream_for_consumer<S>(
    default: &CapsSet,
    streams: impl IntoIterator<Item = (S, Caps)>,
    accepted: &CapsSet,
) -> Option<S> {
    if default
        .alternatives()
        .iter()
        .any(|caps| takes(accepted, caps))
    {
        return None;
    }
    streams
        .into_iter()
        .find(|(_, caps)| takes(accepted, caps))
        .map(|(stream, _)| stream)
}

pub fn demux_stream(name: &str, accepted: &CapsSet) -> Option<String> {
    match name {
        "tsdemux" => crate::tsdemux::stream_for_consumer(accepted),
        "matroskademux" => crate::mkvdemux::stream_for_consumer(accepted),
        "qtdemux" => crate::mp4demux::stream_for_consumer(accepted),
        "flvdemux" => crate::flvdemux::stream_for_consumer(accepted),
        _ => None,
    }
}
