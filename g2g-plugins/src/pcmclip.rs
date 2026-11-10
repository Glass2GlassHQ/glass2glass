// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use core::num::NonZeroU32;

use g2g_core::log::{ClockTime, Target};
use g2g_core::time::{ns_to_ticks, ticks_to_ns};
use g2g_core::{FrameTiming, G2gError, Segment};

// gst's audio decoder base class clips under this category too.
const LOG_CATEGORY: &str = "audiodecoder";

// drops the decoded samples the frame discards from its start and end and those outside the segment, `None` when none are left
pub(crate) fn clip_pcm_to_segment(
    segment: &Segment,
    mut timing: FrameTiming,
    mut pcm: Vec<u8>,
    bytes_per_sample_frame: usize,
    sample_rate: u32,
) -> Result<Option<(FrameTiming, Vec<u8>)>, G2gError> {
    let rate = NonZeroU32::new(sample_rate).ok_or(G2gError::CapsMismatch)?;
    let mut sample_count = pcm
        .len()
        .checked_div(bytes_per_sample_frame)
        .ok_or(G2gError::CapsMismatch)?;
    if timing.start_discard_ns > 0 || timing.end_discard_ns > 0 {
        let head_samples = (ns_to_ticks(u64::from(timing.start_discard_ns), sample_rate) as usize)
            .min(sample_count);
        let tail_samples = ns_to_ticks(u64::from(timing.end_discard_ns), sample_rate) as usize;
        let end = sample_count.saturating_sub(tail_samples).max(head_samples);
        pcm.truncate(end * bytes_per_sample_frame);
        pcm.drain(..head_samples * bytes_per_sample_frame);
        sample_count = end - head_samples;
        timing.pts_ns = timing
            .pts_ns
            .saturating_add(ticks_to_ns(head_samples as u64, sample_rate));
        timing.duration_ns = ticks_to_ns(sample_count as u64, sample_rate);
        timing.start_discard_ns = 0;
        timing.end_discard_ns = 0;
    }
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

// the presentation of a stream whose decoder outputs `priming_ns` of encoder priming first
pub(crate) fn presentation_after(first_pts_ns: u64, priming_ns: u64) -> Segment {
    // open-GOP B frames present before the first unit out
    if priming_ns == 0 {
        return Segment::new();
    }
    Segment {
        start: first_pts_ns.saturating_add(priming_ns),
        ..Segment::new()
    }
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
