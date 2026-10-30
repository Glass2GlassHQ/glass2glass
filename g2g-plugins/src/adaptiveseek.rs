use g2g_core::log::{ClockTime, LogSource};
use g2g_core::runtime::SeekRefusal;
use g2g_core::{g2g_warn, Seek, SeekFlags, SeekType, Segment};

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

// the presentation times a seek on a live stream can reach
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LiveWindow {
    pub(crate) start_ns: u64,
    pub(crate) stop_ns: u64,
}

impl LiveWindow {
    // gst adaptivedemux's rules: an edge past the window moves onto it, except the start of an accurate seek
    pub(crate) fn admit(self, mut seek: Seek) -> Result<Seek, SeekRefusal> {
        let forward = seek.rate > 0.0;
        let accurate = seek.flags.contains(SeekFlags::ACCURATE);
        for (edge_type, edge) in [
            (&mut seek.start_type, &mut seek.start),
            (&mut seek.stop_type, &mut seek.stop),
        ] {
            if *edge_type == SeekType::End {
                *edge = self.stop_ns.saturating_sub(*edge);
                *edge_type = SeekType::Set;
            }
        }
        if seek.start_type == SeekType::Set && seek.start < self.start_ns && (!forward || !accurate)
        {
            seek.start = self.start_ns;
        }
        if seek.stop_type == SeekType::Set && seek.stop > self.stop_ns && (forward || !accurate) {
            seek.stop = self.stop_ns;
        }
        let outside = |edge_type: SeekType, edge: u64| {
            edge_type == SeekType::Set && (edge < self.start_ns || edge > self.stop_ns)
        };
        if outside(seek.start_type, seek.start) || outside(seek.stop_type, seek.stop) {
            return Err(SeekRefusal::OutsideLiveWindow {
                start_ns: self.start_ns,
                stop_ns: self.stop_ns,
            });
        }
        Ok(seek)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: LiveWindow = LiveWindow {
        start_ns: 10_000_000_000,
        stop_ns: 40_000_000_000,
    };
    const BEFORE_WINDOW_NS: u64 = 4_000_000_000;
    const INSIDE_WINDOW_NS: u64 = 25_000_000_000;
    const PAST_WINDOW_NS: u64 = 50_000_000_000;

    fn accurate(seek: Seek) -> Seek {
        Seek {
            flags: seek.flags | SeekFlags::ACCURATE,
            ..seek
        }
    }

    fn refused() -> Result<Seek, SeekRefusal> {
        Err(SeekRefusal::OutsideLiveWindow {
            start_ns: WINDOW.start_ns,
            stop_ns: WINDOW.stop_ns,
        })
    }

    #[test]
    fn a_target_inside_the_window_is_kept() {
        let seek = Seek::flush_to(INSIDE_WINDOW_NS);
        assert_eq!(WINDOW.admit(seek), Ok(seek));
    }

    #[test]
    fn a_target_before_the_window_moves_onto_its_start() {
        let admitted = WINDOW.admit(Seek::flush_to(BEFORE_WINDOW_NS)).unwrap();
        assert_eq!(admitted.start, WINDOW.start_ns);
    }

    #[test]
    fn an_accurate_target_before_the_window_is_refused() {
        assert_eq!(
            WINDOW.admit(accurate(Seek::flush_to(BEFORE_WINDOW_NS))),
            refused()
        );
    }

    #[test]
    fn a_target_past_the_window_is_refused() {
        assert_eq!(WINDOW.admit(Seek::flush_to(PAST_WINDOW_NS)), refused());
    }

    #[test]
    fn a_target_from_the_end_counts_back_from_the_window_stop() {
        let from_end = Seek {
            start_type: SeekType::End,
            start: WINDOW.stop_ns - INSIDE_WINDOW_NS,
            ..Seek::flush_to(0)
        };
        let admitted = WINDOW.admit(from_end).unwrap();
        assert_eq!(
            (admitted.start_type, admitted.start),
            (SeekType::Set, INSIDE_WINDOW_NS)
        );
    }

    #[test]
    fn a_stop_past_the_window_moves_onto_its_stop() {
        let admitted = WINDOW
            .admit(Seek {
                stop_type: SeekType::Set,
                stop: PAST_WINDOW_NS,
                ..Seek::flush_to(INSIDE_WINDOW_NS)
            })
            .unwrap();
        assert_eq!(admitted.stop, WINDOW.stop_ns);
    }

    #[test]
    fn a_reverse_seek_moves_its_start_onto_the_window_even_when_accurate() {
        let admitted = WINDOW
            .admit(accurate(Seek::reverse(BEFORE_WINDOW_NS, INSIDE_WINDOW_NS)))
            .unwrap();
        assert_eq!(admitted.start, WINDOW.start_ns);
    }
}
