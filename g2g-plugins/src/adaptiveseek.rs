use g2g_core::log::{ClockTime, LogSource};
use g2g_core::{g2g_warn, Seek, Segment};

// the Segments an adaptive source sends ahead of its media, so running time follows presentation time
#[derive(Debug, Default)]
pub(crate) struct AdaptiveTimeline {
    // a flushing seek taken, its Segment goes out with the first media fetched after it
    seek_to_open: Option<Seek>,
    // the presentation time at running time 0, set by the first media out and by a seek
    origin_ns: Option<u64>,
}

impl AdaptiveTimeline {
    pub(crate) fn seek(&mut self, seek: Seek) {
        self.seek_to_open = Some(seek);
    }

    // `earliest_pts_ns` reads the first presented timestamp of the media, which starts at `presentation_start_ns`
    pub(crate) fn segment_ahead(
        &mut self,
        source: &impl LogSource,
        presentation_start_ns: u64,
        discontinuity: bool,
        earliest_pts_ns: impl FnOnce() -> Option<u64>,
    ) -> Option<Segment> {
        if let Some(seek) = self.seek_to_open.take() {
            let media_start_ns = media_start_ns(source, presentation_start_ns, earliest_pts_ns());
            let segment = seek_segment(&seek, presentation_start_ns, media_start_ns);
            self.origin_ns = Some(segment.time);
            return Some(segment);
        }
        let Some(origin_ns) = self.origin_ns else {
            // the demuxer opens the stream
            self.origin_ns = Some(presentation_start_ns);
            return None;
        };
        if !discontinuity {
            return None;
        }
        let media_start_ns = media_start_ns(source, presentation_start_ns, earliest_pts_ns());
        Some(Segment {
            base: presentation_start_ns.saturating_sub(origin_ns),
            start: media_start_ns,
            position: media_start_ns,
            time: presentation_start_ns,
            ..Segment::new()
        })
    }
}

fn media_start_ns(
    source: &impl LogSource,
    presentation_start_ns: u64,
    earliest_pts_ns: Option<u64>,
) -> u64 {
    earliest_pts_ns.unwrap_or_else(|| {
        g2g_warn!(
            source,
            "no timestamp found in the segment at {}, it plays on presentation time",
            ClockTime(presentation_start_ns)
        );
        presentation_start_ns
    })
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
