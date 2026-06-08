//! Buffer pools — implementation deferred to M4.
//!
//! Will provide three backing strategies:
//! - `Arc`-recycled `PooledFrame` on `std` / multi-thread targets
//! - Index-into-array `PooledFrame` on `no_std + alloc` single-thread targets
//! - Compile-time sized `BufferPool<T, const N: usize>` for strict no-heap RTOS
