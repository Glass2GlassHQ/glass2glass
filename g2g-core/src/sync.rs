//! Concurrency-primitive compat layer for loom.
//!
//! Under `--cfg loom` the atomics and `UnsafeCell` come from loom, so its model
//! checker can explore every thread interleaving of the [`SpscFrameRing`]
//! protocol. In every normal build they are the `core` primitives, with the cell
//! wrapped to loom's `.with` / `.with_mut` API at zero cost (a transparent
//! newtype over `core::cell::UnsafeCell` the optimizer erases).
//!
//! [`SpscFrameRing`]: crate::spsc::SpscFrameRing

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// `core::cell::UnsafeCell` presented with loom's `.with` / `.with_mut` API so
/// one primitive source compiles against loom or `core` unchanged. Zero cost:
/// a transparent newtype whose accessors forward to `UnsafeCell::get`.
#[cfg(not(loom))]
#[repr(transparent)]
pub(crate) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    pub(crate) const fn new(value: T) -> Self {
        Self(core::cell::UnsafeCell::new(value))
    }

    pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }

    /// Raw pointer to the contents, for the consumer's zero-copy lend (which
    /// outlives any scoped access). loom has no equivalent, so the lend path is
    /// `#[cfg(not(loom))]`; the loom consumer reads through `.with` instead.
    pub(crate) fn get(&self) -> *mut T {
        self.0.get()
    }
}
