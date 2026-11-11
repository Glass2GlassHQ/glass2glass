// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

/// Per-coded-picture metadata for random access, built by
/// `H264DpbDecoder::index_pictures` without decoding: whether the picture is a
/// keyframe (an IDR, a valid seek point to decode forward from), its `frame_num`,
/// and its picture-order-count (the presentation-order key).
#[derive(Debug, Clone, Copy)]
pub struct PictureMeta {
    /// The picture resets the picture-order-count (an IDR, and for H.265 an IRAP
    /// with `NoRaslOutputFlag`): the boundary of a coded video sequence, so POC is only comparable within
    /// the run since the last one. Used to group pictures for display-order sort.
    pub is_keyframe: bool,
    /// The picture is an IRAP random-access point decoding can (re)start at: an
    /// IDR always, and for H.265 also a CRA / BLA. A superset of `is_keyframe`
    /// (a CRA does not reset POC but is a valid seek target). A seek to a CRA
    /// discards its RASL leading pictures (they reference now-absent pre-CRA
    /// frames), so a consumer seeking to a leading picture must use an earlier
    /// random-access point instead.
    pub is_random_access: bool,
    pub frame_num: u32,
    /// Picture-order-count: pictures are presented in ascending POC order.
    pub poc: i32,
    // false for an H.265 picture decoded only as a reference (`pic_output_flag`)
    pub pic_output_flag: bool,
    // an H.265 `no_output_of_prior_pics_flag` at a sequence start: the pictures
    // still waiting for output are dropped (C.5.2.2)
    pub discards_prior_pictures: bool,
    // the H.265 DPB once this picture is decoded, which bumps waiting pictures
    // out when it is full (C.5.2.3)
    pub dpb: Option<H265DpbOccupancy>,
}

/// The reference pictures an H.265 DPB holds right after a picture decodes,
/// the decoded picture included, and the `sps_max_dec_pic_buffering` it may hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H265DpbOccupancy {
    pub(crate) reference_pocs: [i32; MAX_DPB_SLOTS],
    pub(crate) reference_count: usize,
    pub(crate) capacity: usize,
}

impl H265DpbOccupancy {
    pub(crate) fn references(&self) -> &[i32] {
        &self.reference_pocs[..self.reference_count]
    }
}

// the `num-dpb-slots` ceiling, past every codec's reference count
pub(crate) const MAX_DPB_SLOTS: usize = 32;

/// A streaming display-order reorder buffer for the element's system (NV12)
/// output. Hardware decode retires pictures in coding order; a viewer needs
/// presentation (POC) order, which differs for any stream with B-frames. Each
/// decoded picture is pushed tagged with its (coded-video-sequence, POC) key
/// (the same key the whole-stream `reorder_to_display_order` sorts on) and the
/// buffer releases pictures in ascending key order once the earliest-display one
/// of the group is settled.
///
/// POC only compares within one coded video sequence (it resets at each
/// keyframe), so a keyframe both closes the previous CVS (all its pictures are
/// submitted by then, so they release in POC order) and opens a new one. Within a
/// CVS the buffer bumps the lowest-key picture once it holds more than `max_hold`
/// (the DPB slot count, a safe upper bound on the reorder depth), so a long GOP
/// does not buffer without bound.
///
/// Only H.264 / H.265 use this; AV1 (whose display order comes from
/// `show_existing_frame` / `order_hint`, not a monotonic POC) stays in coding
/// order. Generic over the payload so the same logic could serve the GPU-texture
/// path if that becomes AU-streamed.
/// What one [`ReorderBuffer::push`] releases: pictures to emit in display order,
/// and pictures dropped without output.
#[derive(Debug)]
pub(crate) struct ReorderRelease<T> {
    pub(crate) output: alloc::vec::Vec<T>,
    pub(crate) dropped: alloc::vec::Vec<T>,
}

pub(crate) struct ReorderBuffer<T> {
    /// Held pictures, each `(cvs, poc, payload)`. Small (<= `max_hold` within a
    /// CVS), so a Vec + linear min-scan beats a heap.
    pending: alloc::vec::Vec<(u32, i32, T)>,
    /// Current coded-video-sequence index, bumped at each keyframe after the first.
    cvs: u32,
    /// Whether any picture has been pushed (the first keyframe is not a boundary).
    started: bool,
    /// Bump once the current CVS holds more than this many pictures. Set to the
    /// DPB slot count when the decoder is built.
    pub(crate) max_hold: usize,
}

impl<T> ReorderBuffer<T> {
    pub(crate) fn new() -> Self {
        Self {
            pending: alloc::vec::Vec::new(),
            cvs: 0,
            started: false,
            max_hold: 0,
        }
    }

    /// Push one decoded picture (in decode order) and return the pictures now
    /// ready to emit, in display order. A keyframe first releases the whole
    /// previous CVS (all submitted), or drops it on an H.265
    /// `no_output_of_prior_pics_flag`, then joins the new CVS. Within a CVS the
    /// buffer bumps the lowest-key picture while it holds more than `max_hold`,
    /// or while an H.265 DPB holding the waiting pictures besides its references
    /// is over capacity (C.5.2.3). A picture with `pic_output_flag` 0 is dropped.
    pub(crate) fn push(&mut self, meta: &PictureMeta, payload: T) -> ReorderRelease<T> {
        let mut release = ReorderRelease {
            output: alloc::vec::Vec::new(),
            dropped: alloc::vec::Vec::new(),
        };
        if meta.is_keyframe && self.started {
            let previous = self.take_sorted();
            if meta.discards_prior_pictures {
                release.dropped = previous;
            } else {
                release.output = previous;
            }
            self.cvs += 1;
        }
        self.started = true;
        if meta.pic_output_flag {
            self.pending.push((self.cvs, meta.poc, payload));
        } else {
            release.dropped.push(payload);
        }
        while !self.pending.is_empty()
            && (self.pending.len() > self.max_hold || self.dpb_over_capacity(meta.dpb))
        {
            release.output.push(self.pop_min());
        }
        release
    }

    // waiting pictures that are no longer references still take DPB room
    fn dpb_over_capacity(&self, dpb: Option<H265DpbOccupancy>) -> bool {
        let Some(dpb) = dpb else {
            return false;
        };
        let references = dpb.references();
        let waiting_only = self
            .pending
            .iter()
            .filter(|(cvs, poc, _)| *cvs != self.cvs || !references.contains(poc))
            .count();
        references.len() + waiting_only > dpb.capacity
    }

    /// Release everything held, in display order (end of stream / a reconfig
    /// boundary).
    pub(crate) fn flush(&mut self) -> alloc::vec::Vec<T> {
        self.take_sorted()
    }

    /// Discard all held pictures and the CVS state (a seek / flush drops pre-flush
    /// data; the next keyframe starts a fresh sequence).
    pub(crate) fn reset(&mut self) {
        self.pending.clear();
        self.cvs = 0;
        self.started = false;
    }

    /// Remove and return the lowest-`(cvs, poc)` held picture.
    fn pop_min(&mut self) -> T {
        let mut min_i = 0;
        for i in 1..self.pending.len() {
            let a = (self.pending[i].0, self.pending[i].1);
            let b = (self.pending[min_i].0, self.pending[min_i].1);
            if a < b {
                min_i = i;
            }
        }
        self.pending.remove(min_i).2
    }

    /// Drain all held pictures, sorted into display order. A stable sort keeps
    /// decode order among any equal keys (which should not occur).
    pub(crate) fn take_sorted(&mut self) -> alloc::vec::Vec<T> {
        let mut items = core::mem::take(&mut self.pending);
        items.sort_by_key(|a| (a.0, a.1));
        items.into_iter().map(|(_, _, p)| p).collect()
    }
}
