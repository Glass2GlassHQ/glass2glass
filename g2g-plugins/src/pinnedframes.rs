// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::ffi::c_void;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::{Mutex, PoisonError};

use g2g_core::{G2gError, HostAllocator, SystemSlice};

pub(crate) trait PinnedAllocation: Sized + core::fmt::Debug + Send + 'static {
    type Device: core::fmt::Debug + Send + Sync + 'static;

    fn allocate(device: &Self::Device, size: usize) -> Result<Self, G2gError>;

    fn size(&self) -> usize;

    fn as_mut_slice(&mut self) -> &mut [u8];
}

#[derive(Debug)]
pub(crate) struct PinnedFrames<M: PinnedAllocation> {
    pub(crate) device: M::Device,
    free: Mutex<Vec<M>>,
    // address and size of every buffer this pool allocated
    pub(crate) allocations: Mutex<Vec<(usize, usize)>>,
}

#[derive(Debug)]
struct PinnedLease<M: PinnedAllocation> {
    memory: M,
    frames: Arc<PinnedFrames<M>>,
}

unsafe extern "C" fn return_lease<M: PinnedAllocation>(user: *mut c_void) {
    // SAFETY: `user` is the `Box` that `PinnedFrames::lease` leaked, and SystemSlice calls this once.
    let lease = unsafe { Box::from_raw(user.cast::<PinnedLease<M>>()) };
    let PinnedLease { memory, frames } = *lease;
    frames
        .free
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(memory);
}

impl<M: PinnedAllocation> PinnedFrames<M> {
    pub(crate) fn lease(self: &Arc<Self>, len: usize) -> Result<SystemSlice, G2gError> {
        let recycled = {
            let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
            free.iter()
                .position(|memory| memory.size() >= len)
                .map(|index| free.swap_remove(index))
        };
        let mut memory = match recycled {
            Some(memory) => memory,
            None => {
                let mut memory = M::allocate(&self.device, len)?;
                let address = memory.as_mut_slice().as_mut_ptr() as usize;
                self.allocations
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((address, memory.size()));
                memory
            }
        };
        let pointer = memory.as_mut_slice().as_mut_ptr();
        let lease = Box::into_raw(Box::new(PinnedLease {
            memory,
            frames: Arc::clone(self),
        }));
        // SAFETY: the lease owns `len` or more bytes at `pointer` until `return_lease` runs, and only this slice reaches them.
        Ok(unsafe {
            SystemSlice::from_foreign_exclusive(pointer, len, Some(return_lease::<M>), lease.cast())
        })
    }

    pub(crate) fn allocated(&self, bytes: &[u8]) -> bool {
        let start = bytes.as_ptr() as usize;
        self.allocations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .any(|&(address, size)| start >= address && start + bytes.len() <= address + size)
    }
}

#[derive(Debug)]
pub(crate) struct PinnedOffer<M: PinnedAllocation> {
    pub(crate) frames: Arc<PinnedFrames<M>>,
    pub(crate) allocator: HostAllocator,
}

impl<M: PinnedAllocation> PinnedOffer<M> {
    pub(crate) fn new(device: M::Device) -> Self {
        let frames = Arc::new(PinnedFrames {
            device,
            free: Mutex::default(),
            allocations: Mutex::default(),
        });
        let leased_from = Arc::clone(&frames);
        Self {
            frames,
            allocator: HostAllocator::new(move |len| leased_from.lease(len)),
        }
    }
}
