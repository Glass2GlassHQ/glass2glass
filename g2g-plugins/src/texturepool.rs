// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::sync::Arc;
use alloc::vec::Vec;

pub(crate) fn take_from_pool<T, E>(
    pool: &mut Vec<Arc<T>>,
    create: impl FnOnce() -> Result<T, E>,
) -> Result<Arc<T>, E> {
    if let Some(free) = pool.iter().find(|item| Arc::strong_count(item) == 1) {
        return Ok(free.clone());
    }
    let item = Arc::new(create()?);
    pool.push(item.clone());
    Ok(item)
}
