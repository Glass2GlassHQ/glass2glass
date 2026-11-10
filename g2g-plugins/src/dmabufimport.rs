// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use std::os::fd::BorrowedFd;

use g2g_core::memory::OwnedDmaBuf;
use g2g_core::{G2gError, HardwareError};

// a dma-buf not seen once this many are held means the producer's pool changed
const MAX_CACHED_IMPORTS: usize = 16;

#[derive(Debug)]
pub(crate) struct DmaBufImportCache<T> {
    imports: Vec<((u64, u64), T)>,
}

impl<T> DmaBufImportCache<T> {
    pub(crate) const fn new() -> Self {
        Self {
            imports: Vec::new(),
        }
    }
}

impl<T> Default for DmaBufImportCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> DmaBufImportCache<T> {
    // a producer pool handing its buffers round imports each one once
    pub(crate) fn get_or_import(
        &mut self,
        dmabuf: &OwnedDmaBuf,
        size: u64,
        import: impl FnOnce() -> Result<T, G2gError>,
    ) -> Result<T, G2gError> {
        // SAFETY: the fd is a live dma-buf owned by `dmabuf` for this call.
        let fd = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) };
        let key = (dmabuf_inode(fd)?, size);
        if let Some((_, imported)) = self.imports.iter().find(|(known, _)| *known == key) {
            return Ok(imported.clone());
        }
        let imported = import()?;
        if self.imports.len() >= MAX_CACHED_IMPORTS {
            self.imports.clear();
        }
        self.imports.push((key, imported.clone()));
        Ok(imported)
    }
}

// every fd of one dma-buf shares its inode, which is not reused while an import holds the buffer open
fn dmabuf_inode(fd: BorrowedFd<'_>) -> Result<u64, G2gError> {
    use std::os::unix::fs::MetadataExt;
    let error = |_| G2gError::Hardware(HardwareError::Other);
    let file = std::fs::File::from(fd.try_clone_to_owned().map_err(error)?);
    Ok(file.metadata().map_err(error)?.ino())
}
