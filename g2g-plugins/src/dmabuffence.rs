// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::memory::{OwnedDmaBuf, SyncFile};
use g2g_core::{G2gError, HardwareError};

// returns once the producer's GPU writes into `dmabuf` have finished
pub(crate) async fn wait_for_producer(dmabuf: &OwnedDmaBuf) -> Result<(), G2gError> {
    if let Some(fence) = dmabuf.sync_file() {
        let fence = fence.clone();
        return g2g_core::offload::run_blocking(move || wait_signaled(&fence)).await;
    }
    // a vulkan timeline semaphore needs a vulkan device to wait on
    if dmabuf.sync_fd().is_some() {
        return Err(G2gError::UnsupportedDomain);
    }
    Ok(())
}

fn wait_signaled(fence: &SyncFile) -> Result<(), G2gError> {
    let mut poll_fd = libc::pollfd {
        fd: fence.as_raw(),
        events: libc::POLLIN,
        revents: 0,
    };
    loop {
        // SAFETY: `poll_fd` is one live pollfd on an fd `fence` keeps open.
        let ready = unsafe { libc::poll(&mut poll_fd, 1, -1) };
        if ready > 0 {
            return Ok(());
        }
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};
    use std::time::Duration;

    use g2g_core::memory::SyncFd;

    use super::*;

    const UNSIGNALLED_WAIT: Duration = Duration::from_millis(100);

    fn unsignalled_fence() -> OwnedFd {
        // eventfd stands in for a sync_file: readable once something writes to it
        // SAFETY: eventfd takes no pointers and returns a fresh descriptor or -1.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC) };
        assert!(fd >= 0, "eventfd");
        // SAFETY: `fd` is the fresh descriptor above.
        unsafe { OwnedFd::from_raw_fd(fd) }
    }

    fn signal(fence: &OwnedFd) {
        let one = 1u64.to_ne_bytes();
        // SAFETY: writes eight bytes from a live buffer to an open eventfd.
        let written = unsafe {
            libc::write(
                std::os::fd::AsRawFd::as_raw_fd(fence),
                one.as_ptr().cast(),
                8,
            )
        };
        assert_eq!(written, 8, "eventfd write");
    }

    fn dmabuf() -> OwnedDmaBuf {
        // SAFETY: memfd_create takes a valid C string and returns a fresh descriptor or -1.
        let fd = unsafe { libc::memfd_create(c"g2g-dmabuffence".as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0, "memfd_create");
        // SAFETY: `fd` is the fresh descriptor above, handed over whole.
        unsafe { OwnedDmaBuf::from_raw(fd, 0, 0) }
    }

    #[tokio::test]
    async fn the_wait_ends_only_once_the_sync_file_signals() {
        let fence = unsignalled_fence();
        let held = fence.try_clone().expect("dup the eventfd");
        // SAFETY: `held` is a descriptor this call hands over whole.
        let dmabuf = dmabuf().with_sync_file(unsafe { SyncFile::from_raw(held.into_raw_fd()) });
        let early = tokio::time::timeout(UNSIGNALLED_WAIT, wait_for_producer(&dmabuf)).await;
        assert!(early.is_err(), "the wait ended before the fence signalled");
        signal(&fence);
        wait_for_producer(&dmabuf).await.expect("signalled");
    }

    #[tokio::test]
    async fn a_frame_fenced_only_by_a_timeline_semaphore_is_refused() {
        let semaphore = unsignalled_fence();
        // SAFETY: `semaphore` is a descriptor this call hands over whole.
        let fenced = dmabuf().with_sync(unsafe { SyncFd::from_raw(semaphore.into_raw_fd()) }, 1);
        assert_eq!(
            wait_for_producer(&fenced).await,
            Err(G2gError::UnsupportedDomain)
        );
        assert_eq!(wait_for_producer(&dmabuf()).await, Ok(()));
    }
}
