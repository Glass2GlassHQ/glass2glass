// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::{BTreeMap, BinaryHeap};
use core::cmp::Reverse;

use g2g_core::frame::Frame;
use g2g_core::meta::FrameMetaSet;

// what an output picture inherits from the access unit it decodes
#[derive(Debug, Default)]
pub(crate) struct InputStamp {
    pub(crate) arrival_ns: u64,
    pub(crate) meta: FrameMetaSet,
}

impl InputStamp {
    pub(crate) fn of(frame: &Frame) -> Self {
        Self {
            arrival_ns: frame.timing.arrival_ns,
            meta: frame.meta.clone(),
        }
    }
}

// a unit the decoder drops never has its stamp taken
const MAX_PENDING_INPUT_STAMPS: usize = 1024;

// keyed by pts, the one value a decoder carries from a unit to the picture it decodes
#[derive(Debug, Default)]
pub(crate) struct PendingInputStamps(BTreeMap<u64, InputStamp>);

impl PendingInputStamps {
    pub(crate) fn insert(&mut self, pts_ns: u64, stamp: InputStamp) {
        if stamp.arrival_ns == 0 && stamp.meta.is_empty() {
            return;
        }
        self.0.insert(pts_ns, stamp);
        while self.0.len() > MAX_PENDING_INPUT_STAMPS {
            self.0.pop_first();
        }
    }

    pub(crate) fn take(&mut self, pts_ns: u64) -> InputStamp {
        self.0.remove(&pts_ns).unwrap_or_default()
    }
}

// the pts of the pictures a decoder holds for display order, which pictures leaving in display order take earliest first
#[derive(Debug, Default)]
pub(crate) struct HeldPictureTimes(BinaryHeap<Reverse<u64>>);

impl HeldPictureTimes {
    pub(crate) fn insert(&mut self, pts_ns: u64) {
        self.0.push(Reverse(pts_ns));
    }

    // mp4s with no composition offsets stamp B-frame streams in decode order, which this puts back in display order
    pub(crate) fn take_earliest(&mut self) -> u64 {
        let Reverse(pts_ns) = self
            .0
            .pop()
            .expect("every picture leaving the reorder buffer entered it with a pts");
        pts_ns
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}
