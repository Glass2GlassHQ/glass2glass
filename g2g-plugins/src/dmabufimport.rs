// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use std::os::fd::BorrowedFd;

use g2g_core::memory::OwnedDmaBuf;
use g2g_core::{G2gError, HardwareError};

// producer pools hand round fewer buffers than this
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
        release: impl FnOnce(T) -> Result<(), G2gError>,
    ) -> Result<T, G2gError> {
        // SAFETY: the fd is a live dma-buf owned by `dmabuf` for this call.
        let fd = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) };
        let key = (dmabuf_inode(fd)?, size);
        if let Some(position) = self.imports.iter().position(|(known, _)| *known == key) {
            let entry = self.imports.remove(position);
            let imported = entry.1.clone();
            self.imports.push(entry);
            return Ok(imported);
        }
        let imported = import()?;
        if self.imports.len() >= MAX_CACHED_IMPORTS {
            let (_, oldest) = self.imports.remove(0);
            release(oldest)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::IntoRawFd;

    // sockets stand in for dma-bufs, each with an inode of its own
    fn buffer() -> OwnedDmaBuf {
        let (socket, _) = std::os::unix::net::UnixStream::pair().unwrap();
        // SAFETY: the fd is a freshly opened socket this call gives up.
        unsafe { OwnedDmaBuf::from_raw(socket.into_raw_fd(), 0, 0) }
    }

    #[test]
    fn a_full_cache_releases_the_least_recently_used_import() {
        let buffers: Vec<OwnedDmaBuf> = (0..=MAX_CACHED_IMPORTS).map(|_| buffer()).collect();
        let mut cache = DmaBufImportCache::new();
        let mut released = Vec::new();
        let mut get = |cache: &mut DmaBufImportCache<usize>, index: usize| {
            cache
                .get_or_import(
                    &buffers[index],
                    0,
                    || Ok(index),
                    |oldest| {
                        released.push(oldest);
                        Ok(())
                    },
                )
                .unwrap()
        };
        for index in 0..MAX_CACHED_IMPORTS {
            assert_eq!(get(&mut cache, index), index);
        }
        assert_eq!(get(&mut cache, 0), 0);
        assert_eq!(get(&mut cache, MAX_CACHED_IMPORTS), MAX_CACHED_IMPORTS);
        drop(get);
        assert_eq!(released, [1], "the oldest import not used since goes");
    }
}
