// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::BTreeMap;

// a unit the codec drops never has its value taken
const MAX_PENDING_BY_PTS: usize = 1024;

// keyed by pts, the one value a codec carries from its input to what it emits
#[derive(Debug)]
pub(crate) struct PendingByPts<T>(BTreeMap<u64, T>);

impl<T> Default for PendingByPts<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<T> PendingByPts<T> {
    pub(crate) fn insert(&mut self, pts_ns: u64, value: T) {
        self.0.insert(pts_ns, value);
        while self.0.len() > MAX_PENDING_BY_PTS {
            self.0.pop_first();
        }
    }

    pub(crate) fn take(&mut self, pts_ns: u64) -> Option<T> {
        self.0.remove(&pts_ns)
    }
}
