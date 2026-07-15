//! Link-time no-heap proof (M625) + panic-free proof (M626): a real g2g
//! pipeline that links for a bare Cortex-M target with NO global allocator, NO
//! `alloc` crate dependency, and NO reachable panic path.
//!
//! The pipeline itself lives in the shared `noalloc-pipeline` crate (also
//! booted on an emulated Cortex-M by `examples/g2g-qemu`); this staticlib
//! wraps it behind a C entry point so `tools/noalloc-check.sh` can assert the
//! linked archive references zero allocator symbols and zero `core::panicking`
//! symbols. If any reachable path needed the heap, this crate would fail to
//! build (there is no allocator to satisfy it, and the `alloc` crate is not
//! even linked); that it builds for `thumbv7em-none-eabihf` is the guarantee,
//! and the panic-free symbol check proves the `#[panic_handler]` below is dead
//! code.

#![no_std]

// No `#[global_allocator]`: this is the point. With `g2g-core` built
// `default-features = false` the `alloc` crate is not a dependency, so a heap use
// anywhere on the reachable path would be a compile / link error, not a runtime
// surprise.

// Required by `no_std`, but unreachable: the archive has no `core::panicking`
// symbols (checked by `tools/noalloc-check.sh`), so nothing can call this.
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Run the shared source -> transform -> SPI-display pipeline and return its
/// wire checksum. `#[no_mangle]` + the returned value keep it (and everything
/// it reaches) from being eliminated, so the linked archive contains the real,
/// heap-free pipeline code.
#[no_mangle]
pub extern "C" fn g2g_noalloc_run() -> u64 {
    noalloc_pipeline::run()
}

/// The checksum a correct run produces, so harnesses (host-harness.c) compare
/// against the pipeline's own constant instead of hardcoding it.
#[no_mangle]
pub extern "C" fn g2g_noalloc_expected() -> u64 {
    noalloc_pipeline::EXPECTED_CHECKSUM
}

/// Run the flagship audio graph (M644: capture -> convert -> resample ->
/// mix -> encode -> RTP) and return its wire checksum, extending every
/// symbol proof over the whole audio element set.
#[no_mangle]
pub extern "C" fn g2g_audio_run() -> u64 {
    noalloc_pipeline::audio::run_audio()
}

/// The audio graph's pinned checksum (see `noalloc-pipeline`'s
/// `AUDIO_EXPECTED_CHECKSUM`).
#[no_mangle]
pub extern "C" fn g2g_audio_expected() -> u64 {
    noalloc_pipeline::audio::AUDIO_EXPECTED_CHECKSUM
}
