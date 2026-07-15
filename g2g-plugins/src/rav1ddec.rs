//! Pure-Rust AV1 decode element (Rav1dDec, `rav1d` feature): `CompressedVideo{Av1}`
//! in, `RawVideo{I420}` out, via `re_rav1d`, the Rust port of dav1d. Same codec, no
//! C: `re_rav1d` is a line-for-line safe-Rust reimplementation of the dav1d decoder
//! and re-exports dav1d-rs's safe API, so this element is `Dav1dDec` with the backend
//! swapped from libdav1d (FFI) to `re_rav1d` (Rust). It builds with no system deps
//! and no NASM (`default-features = false`), so it reaches the pure-Rust / wasm
//! targets that the libdav1d path cannot.
//!
//! Decoded geometry is recovered per picture and carried in the `CapsChanged`; the
//! decoded picture is packed into the matching fully-planar [`RawVideoFormat`]:
//! 4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 / 12-bit (`I420` / `I422` / `I444` and their
//! `p10` / `p12` variants), recovered per picture from the layout + bit depth;
//! 10/12-bit samples pass through as native little-endian 2-byte words. Monochrome
//! (I400) is rejected. System memory. Pure Rust, but it pays a speed cost versus
//! libdav1d's hand-written asm.
//!
//! The element body is shared with `dav1ddec` via [`av1_decoder!`](crate::av1dec).

use crate::av1dec::av1_decoder;

av1_decoder!(
    Rav1dDec,
    re_rav1d,
    "AV1 decoder (rav1d)",
    "Decodes AV1 to I420 via the pure-Rust re_rav1d"
);
