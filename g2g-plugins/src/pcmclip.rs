use alloc::vec::Vec;
use core::num::NonZeroU32;

use g2g_core::log::{ClockTime, Target};
use g2g_core::time::ticks_to_ns;
use g2g_core::{FrameTiming, G2gError, Segment};

// gst's audio decoder base class clips under this category too.
const LOG_CATEGORY: &str = "audiodecoder";

// drops the decoded samples outside the segment, `None` when none are left
pub(crate) fn clip_pcm_to_segment(
    segment: &Segment,
    timing: FrameTiming,
    mut pcm: Vec<u8>,
    bytes_per_sample_frame: usize,
    sample_rate: u32,
) -> Result<Option<(FrameTiming, Vec<u8>)>, G2gError> {
    let rate = NonZeroU32::new(sample_rate).ok_or(G2gError::CapsMismatch)?;
    let sample_count = pcm
        .len()
        .checked_div(bytes_per_sample_frame)
        .ok_or(G2gError::CapsMismatch)?;
    let Some(kept) = segment.kept_samples(timing.pts_ns, sample_count, rate) else {
        g2g_core::g2g_log!(
            Target::category(LOG_CATEGORY),
            "dropping buffer pts {} outside the segment",
            ClockTime(timing.pts_ns)
        );
        return Ok(None);
    };
    if kept.range == (0..sample_count) {
        return Ok(Some((timing, pcm)));
    }
    g2g_core::g2g_log!(
        Target::category(LOG_CATEGORY),
        "clipping buffer pts {} to samples {:?} of {sample_count}",
        ClockTime(timing.pts_ns),
        kept.range
    );
    pcm.truncate(kept.range.end * bytes_per_sample_frame);
    pcm.drain(..kept.range.start * bytes_per_sample_frame);
    let clipped = FrameTiming {
        pts_ns: kept.pts_ns,
        duration_ns: ticks_to_ns(kept.range.len() as u64, sample_rate),
        ..timing
    };
    Ok(Some((clipped, pcm)))
}

// narrows a segment to the stream time a container or codec header presents
pub(crate) fn within_presentation(mut segment: Segment, presented: &Segment) -> Segment {
    segment.start = segment.start.max(presented.start);
    segment.stop = match (segment.stop, presented.stop) {
        (Some(upstream), Some(presented)) => Some(upstream.min(presented)),
        (upstream, presented) => upstream.or(presented),
    };
    segment
}
