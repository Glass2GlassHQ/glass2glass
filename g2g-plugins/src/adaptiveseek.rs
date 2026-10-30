use g2g_core::log::{ClockTime, LogSource};
use g2g_core::{g2g_warn, Seek, Segment};

// `earliest_pts_ns` comes from the first media fetched after the seek, which starts at `presentation_start_ns`
pub(crate) fn opening_segment(
    source: &impl LogSource,
    seek: &Seek,
    presentation_start_ns: u64,
    earliest_pts_ns: Option<u64>,
) -> Segment {
    let media_start_ns = earliest_pts_ns.unwrap_or_else(|| {
        g2g_warn!(
            source,
            "no timestamp found in the segment at {}, the seek plays on presentation time",
            ClockTime(presentation_start_ns)
        );
        presentation_start_ns
    });
    seek_segment(seek, presentation_start_ns, media_start_ns)
}

// the seek target, on the timestamps of the segment it landed in
fn seek_segment(seek: &Seek, presentation_start_ns: u64, media_start_ns: u64) -> Segment {
    let position_ns = seek.start.max(presentation_start_ns);
    let to_media = |presentation_ns: u64| {
        media_start_ns.saturating_add(presentation_ns.saturating_sub(presentation_start_ns))
    };
    let media_seek = Seek {
        start: to_media(position_ns),
        stop: to_media(seek.stop),
        ..*seek
    };
    Segment {
        time: position_ns,
        ..Segment::for_flush_seek(&media_seek, None)
    }
}
