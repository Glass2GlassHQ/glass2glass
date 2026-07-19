//! AV1 decode element (Dav1dDec, `dav1d` feature): `CompressedVideo{Av1}` in,
//! `RawVideo{I420}` out, via the `dav1d` crate's safe bindings to libdav1d (the C
//! AV1 decoder with hand-written assembly, the speed reference).
//!
//! The decoder is stateful (frame threading / reordering), so the send/drain
//! protocol is: hand each AV1 temporal unit to `send_data`, and on a `Try again`
//! drain the ready pictures via `get_picture` and push the pending data with
//! `send_pending_data`. Decoded geometry is recovered per picture, so a
//! `CapsChanged` carries it before the first frame and on any mid-stream change
//! (the source may negotiate `Any` dims). The decoded picture is packed into the
//! matching fully-planar [`RawVideoFormat`]: 4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 /
//! 12-bit (`I420` / `I422` / `I444` and their `p10` / `p12` variants), the format
//! recovered per picture from dav1d's layout + bit depth and carried in the
//! `CapsChanged`. 10/12-bit samples pass through as the native little-endian
//! 2-byte words. Monochrome (I400) is rejected (no planar-YUV format for it).
//! System memory. NOT pure Rust (links libdav1d); for a pure-Rust AV1 decoder see
//! `rav1ddec.rs` (`Rav1dDec`, the `re_rav1d` port) behind the `rav1d` feature.
//!
//! The element body is shared with `rav1ddec` via [`av1_decoder!`](crate::av1dec).

use crate::av1dec::av1_decoder;

av1_decoder!(
    Dav1dDec,
    dav1d,
    "AV1 decoder (dav1d)",
    "Decodes AV1 to I420 via libdav1d"
);
