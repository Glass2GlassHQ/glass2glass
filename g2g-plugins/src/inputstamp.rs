// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::frame::Frame;
use g2g_core::meta::FrameMetaSet;

use crate::ptsmap::PendingByPts;

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

#[derive(Debug, Default)]
pub(crate) struct PendingInputStamps(PendingByPts<InputStamp>);

impl PendingInputStamps {
    pub(crate) fn insert(&mut self, pts_ns: u64, stamp: InputStamp) {
        if stamp.arrival_ns == 0 && stamp.meta.is_empty() {
            return;
        }
        self.0.insert(pts_ns, stamp);
    }

    pub(crate) fn take(&mut self, pts_ns: u64) -> InputStamp {
        self.0.take(pts_ns).unwrap_or_default()
    }
}
