// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Timestamp burn-in (`timestampburn`, `latency-bench` feature): overwrites the
//! top-left of an 8-bit YUV frame's luma with a 64-block strip carrying the
//! `CLOCK_MONOTONIC` nanoseconds at which the frame passed through. A consumer
//! that decodes the strip and subtracts it from its own `CLOCK_MONOTONIC` gets
//! the elapsed time across everything in between, measured by one clock.
//!
//! This is the instrument behind `tools/latency-bench-e2e.sh`, which puts the
//! same burned stream through a g2g consumer and a GStreamer one and reads both
//! with the same binary, so the two numbers cover the same span. The encode and
//! decode halves live here together, and the `g2g-latency-reader` binary calls
//! the decode half, so the layout cannot drift between writer and reader.
//!
//! The strip is one bit per block, most significant first, blocks laid left to
//! right along the top edge. A block is a solid square of video-range white for
//! a 1 and video-range black for a 0, sized so a lossy encoder at a normal
//! bitrate cannot blur one into its neighbour. Chroma is untouched, so the strip
//! renders grey on a colour frame. The layout is fixed: no properties.
//!
//! Sub-nanosecond framing detail: the timestamp is read inside `apply`, after
//! the frame arrives, so the span a reader computes starts at burn time and not
//! at capture.
//!
//! Unix only: it reads `clock_gettime(CLOCK_MONOTONIC)` through `libc`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat,
};

use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 8] = [
    RawVideoFormat::I420,
    RawVideoFormat::Nv12,
    RawVideoFormat::I422,
    RawVideoFormat::I444,
    RawVideoFormat::A420,
    RawVideoFormat::Gray8,
    RawVideoFormat::Yuyv,
    RawVideoFormat::Uyvy,
];

// where the 8-bit luma samples of one row of plane 0 sit: the byte of the first, then every `step` bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LumaSamples {
    pub first: usize,
    pub step: usize,
}

impl LumaSamples {
    // the bytes of a row the strip spans
    fn strip_row_bytes(self) -> usize {
        self.first + (STRIP_WIDTH_PX - 1) * self.step + 1
    }
}

pub fn luma_samples(format: RawVideoFormat) -> Option<LumaSamples> {
    match format {
        RawVideoFormat::I420
        | RawVideoFormat::Nv12
        | RawVideoFormat::I422
        | RawVideoFormat::I444
        | RawVideoFormat::A420
        | RawVideoFormat::Gray8 => Some(LumaSamples { first: 0, step: 1 }),
        // YUY2 interleaves Y0 U Y1 V
        RawVideoFormat::Yuyv => Some(LumaSamples { first: 0, step: 2 }),
        // UYVY interleaves U Y0 V Y1
        RawVideoFormat::Uyvy => Some(LumaSamples { first: 1, step: 2 }),
        _ => None,
    }
}

/// Bits in the burned timestamp, one per block.
pub const BLOCK_COUNT: usize = 64;
/// Side of one block, in luma samples. Large enough that an ultrafast H.264
/// encode keeps each block's interior flat.
pub const BLOCK_PX: usize = 16;
/// Luma of a block standing for a 1 bit (video-range white).
pub const LUMA_ONE: u8 = 235;
/// Luma of a block standing for a 0 bit (video-range black).
pub const LUMA_ZERO: u8 = 16;
/// Luma samples the strip spans horizontally. A frame narrower than this cannot
/// carry a timestamp.
pub const STRIP_WIDTH_PX: usize = BLOCK_COUNT * BLOCK_PX;

/// Side of the square sampled at each block's centre when decoding. Compression
/// smears the block edges, so the reader keeps clear of them.
const SAMPLE_PX: usize = 8;
/// How far a sampled block mean may sit from [`LUMA_ZERO`] or [`LUMA_ONE`] and
/// still count as legible.
const SAMPLE_TOLERANCE: u8 = 48;

// the bytes of plane 0 the strip spans, the last row only as far as the strip
fn strip_bytes(stride: usize, luma: LumaSamples) -> usize {
    stride * (BLOCK_PX - 1) + luma.strip_row_bytes()
}

/// Write `ns` across the top-left of plane 0 as [`BLOCK_COUNT`] blocks,
/// most significant bit first. `plane` must hold [`BLOCK_PX`] rows of
/// `stride` bytes, each wide enough for [`STRIP_WIDTH_PX`] luma samples.
pub fn burn(plane: &mut [u8], stride: usize, luma: LumaSamples, ns: u64) {
    debug_assert!(stride >= luma.strip_row_bytes() && plane.len() >= strip_bytes(stride, luma));
    for row in plane.chunks_mut(stride).take(BLOCK_PX) {
        let samples = row.iter_mut().skip(luma.first).step_by(luma.step);
        for (x, sample) in samples.take(STRIP_WIDTH_PX).enumerate() {
            let bit = (ns >> (BLOCK_COUNT - 1 - x / BLOCK_PX)) & 1;
            *sample = if bit == 1 { LUMA_ONE } else { LUMA_ZERO };
        }
    }
}

/// Read back a timestamp [`burn`] wrote, or `None` when the strip is too
/// degraded to trust: at least half the blocks must sample close to one of the
/// two levels.
pub fn decode(plane: &[u8], stride: usize, luma: LumaSamples) -> Option<u64> {
    if stride < luma.strip_row_bytes() || plane.len() < strip_bytes(stride, luma) {
        return None;
    }
    const OFFSET: usize = (BLOCK_PX - SAMPLE_PX) / 2;
    const MIDPOINT: u8 = LUMA_ZERO / 2 + LUMA_ONE / 2;

    let mut ns = 0u64;
    let mut legible = 0usize;
    for block in 0..BLOCK_COUNT {
        let left = block * BLOCK_PX + OFFSET;
        let mut sum = 0u32;
        for row in OFFSET..OFFSET + SAMPLE_PX {
            let start = row * stride + luma.first + left * luma.step;
            sum += plane[start..]
                .iter()
                .step_by(luma.step)
                .take(SAMPLE_PX)
                .map(|&s| u32::from(s))
                .sum::<u32>();
        }
        let mean = (sum / (SAMPLE_PX * SAMPLE_PX) as u32) as u8;
        if mean.abs_diff(LUMA_ZERO) <= SAMPLE_TOLERANCE
            || mean.abs_diff(LUMA_ONE) <= SAMPLE_TOLERANCE
        {
            legible += 1;
        }
        ns = (ns << 1) | u64::from(mean >= MIDPOINT);
    }
    (legible * 2 >= BLOCK_COUNT).then_some(ns)
}

/// `CLOCK_MONOTONIC` as nanoseconds. Raw rather than [`std::time::Instant`] so
/// two processes on one machine can subtract each other's readings.
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`; `clock_gettime` only writes
    // into it and returns 0 on success.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::timestampburn::TimestampBurn;
///
/// // videotestsrc ! timestampburn ! x264enc
/// let burn = TimestampBurn::new();
/// ```
#[derive(Debug, Default)]
pub struct TimestampBurn {
    state: FilterState,
}

impl TimestampBurn {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PixelFilter for TimestampBurn {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, _h: u32, pixels: &mut [u8]) {
        let luma = luma_samples(format).expect("FORMATS lists only formats with 8-bit luma");
        let stride = format
            .plane_stride(0, w)
            .expect("plane 0 of a configured frame has a stride") as usize;
        burn(pixels, stride, luma, monotonic_ns());
    }

    #[cfg(target_os = "linux")]
    fn apply_dmabuf(
        &mut self,
        format: RawVideoFormat,
        w: u32,
        h: u32,
        dmabuf: &g2g_core::memory::OwnedDmaBuf,
    ) -> Result<(), G2gError> {
        let luma = luma_samples(format).expect("FORMATS lists only formats with 8-bit luma");
        let (layout, _) = crate::paddedrows::dmabuf_plane_layout(dmabuf, format, w, h)
            .ok_or(G2gError::CapsMismatch)?;
        let plane = layout.plane(0).ok_or(G2gError::CapsMismatch)?;
        let mut map = crate::dmabufmap::DmaBufWriteMap::write(dmabuf)?;
        let strip = map
            .from_fd_start_mut()
            .get_mut(plane.offset..)
            .filter(|strip| {
                plane.stride >= luma.strip_row_bytes()
                    && strip.len() >= strip_bytes(plane.stride, luma)
            })
            .ok_or(G2gError::CapsMismatch)?;
        burn(strip, plane.stride, luma, monotonic_ns());
        Ok(())
    }
}

impl AsyncElement for TimestampBurn {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Monotonic timestamp burn-in",
            "Filter/Effect/Video",
            "Burns the current CLOCK_MONOTONIC time into the luma plane as a block strip",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        let system = g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System);
        // linux burns into a dma-buf through a CPU mapping
        #[cfg(target_os = "linux")]
        let system = system.with(g2g_core::memory::MemoryDomainKind::DmaBuf);
        system
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        videofx::intercept_caps::<Self>(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        videofx::same_caps_constraint::<Self>()
    }

    /// Rejects a frame too small to hold the strip, so a mis-sized bench
    /// pipeline fails at negotiation instead of burning a truncated timestamp.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (_, w, h, _) = videofx::accept_input::<Self>(absolute_caps)?;
        if (w as usize) < STRIP_WIDTH_PX || (h as usize) < BLOCK_PX {
            return Err(G2gError::CapsMismatch);
        }
        videofx::configure(self, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        videofx::drive(self, packet, out)
    }
}

impl PadTemplates for TimestampBurn {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const STRIDE: usize = STRIP_WIDTH_PX;
    const PLANAR: LumaSamples = LumaSamples { first: 0, step: 1 };

    #[test]
    fn burned_timestamps_round_trip_exactly() {
        let mut luma = vec![0u8; STRIDE * BLOCK_PX];
        for ns in [
            0,
            1,
            u64::MAX,
            0xA5A5_5A5A_A5A5_5A5A,
            monotonic_ns(),
            1_234_567_890_123_456_789,
        ] {
            burn(&mut luma, STRIDE, PLANAR, ns);
            assert_eq!(
                decode(&luma, STRIDE, PLANAR),
                Some(ns),
                "round trip of {ns}"
            );
        }
    }

    #[test]
    fn a_wider_frame_keeps_its_pixels_outside_the_strip() {
        const WIDE: usize = STRIP_WIDTH_PX + 64;
        let mut luma = vec![7u8; WIDE * (BLOCK_PX + 4)];
        burn(&mut luma, WIDE, PLANAR, 42);
        assert_eq!(decode(&luma, WIDE, PLANAR), Some(42));
        assert!(luma[STRIP_WIDTH_PX..WIDE].iter().all(|&s| s == 7));
        assert!(luma[WIDE * BLOCK_PX..].iter().all(|&s| s == 7));
    }

    /// Blur of the size an ultrafast encode leaves still decodes: the reader
    /// samples the block centre, and the tolerance covers a shifted level.
    #[test]
    fn a_degraded_strip_still_decodes() {
        let mut luma = vec![0u8; STRIDE * BLOCK_PX];
        burn(&mut luma, STRIDE, PLANAR, 0xDEAD_BEEF_0BAD_F00D);
        for sample in luma.iter_mut() {
            *sample = sample.saturating_add(30).min(LUMA_ONE);
        }
        assert_eq!(decode(&luma, STRIDE, PLANAR), Some(0xDEAD_BEEF_0BAD_F00D));
    }

    #[test]
    fn a_frame_with_no_strip_is_rejected() {
        // mid-grey everywhere: no block sits near either level.
        let luma = vec![128u8; STRIDE * BLOCK_PX];
        assert_eq!(decode(&luma, STRIDE, PLANAR), None);
        // and a frame too narrow to hold the strip.
        assert_eq!(decode(&luma, STRIP_WIDTH_PX - 1, PLANAR), None);
    }

    const WIDTH: u32 = STRIP_WIDTH_PX as u32;
    const HEIGHT: u32 = BLOCK_PX as u32 * 2;
    // a byte no strip block writes, so every changed byte is one the burn wrote
    const UNTOUCHED: u8 = 128;

    #[test]
    fn every_format_burns_a_strip_that_reads_back_from_its_i420_luma() {
        for format in FORMATS {
            let mut frame = vec![UNTOUCHED; crate::pixel::frame_byte_size(format, WIDTH, HEIGHT)];
            let before = monotonic_ns();
            TimestampBurn::new().apply(format, WIDTH, HEIGHT, &mut frame);
            let after = monotonic_ns();
            let changed = frame.iter().filter(|&&byte| byte != UNTOUCHED).count();
            assert_eq!(
                changed,
                STRIP_WIDTH_PX * BLOCK_PX,
                "{format:?} writes one byte per strip sample"
            );
            let i420 = crate::videoconvert::convert(
                &frame,
                format,
                RawVideoFormat::I420,
                WIDTH as usize,
                HEIGHT as usize,
                g2g_core::Colorimetry::UNKNOWN,
            );
            let burned = decode(&i420, WIDTH as usize, PLANAR);
            assert!(
                burned.is_some_and(|ns| (before..=after).contains(&ns)),
                "{format:?} strip reads back from luma as {burned:?}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_dmabuf_frame_is_burned_in_place_at_its_plane_offset() {
        use g2g_core::frame::{Frame, FrameTiming};
        use g2g_core::memory::OwnedDmaBuf;
        use g2g_core::{Caps, Dim, Interlace, MemoryDomain, Rate};
        use std::io::{Read, Seek, Write};
        use std::os::fd::{AsRawFd, FromRawFd};

        let format = RawVideoFormat::Nv12;
        // a capture driver's buffer: rows padded past the width, plane 0 after a header
        let stride = STRIP_WIDTH_PX + 64;
        let offset = 4096;
        let length = offset + stride * HEIGHT as usize * 3 / 2;
        // SAFETY: a fresh memfd with no other owner stands in for a dma-buf.
        let fd = unsafe { libc::memfd_create(c"frame".as_ptr(), 0) };
        assert!(fd >= 0, "memfd_create failed");
        // SAFETY: the fd was just created and this file is its only owner.
        let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
        file.write_all(&vec![UNTOUCHED; length]).unwrap();
        // SAFETY: dup gives the frame its own descriptor of the same memfd.
        let frame_fd = unsafe { libc::dup(file.as_raw_fd()) };
        // SAFETY: the duplicate is owned by nothing else.
        let dmabuf = unsafe { OwnedDmaBuf::from_raw(frame_fd, stride as u32, offset as u32) };

        let mut element = TimestampBurn::new();
        element
            .configure_pipeline(&Caps::RawVideo {
                format,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Any,
                interlace: Interlace::Progressive,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: g2g_core::PixelAspect::SQUARE,
            })
            .unwrap();
        let mut sink = crate::testutil::CollectSink::default();
        let frame = Frame {
            domain: MemoryDomain::DmaBuf(dmabuf),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let before = monotonic_ns();
        g2g_core::runtime::block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink))
            .unwrap();
        let forwarded = sink.packets.iter().any(|packet| {
            matches!(packet, PipelinePacket::DataFrame(frame) if matches!(frame.domain, MemoryDomain::DmaBuf(_)))
        });
        assert!(forwarded, "the frame stays a dma-buf");

        let mut bytes = Vec::new();
        file.rewind().unwrap();
        file.read_to_end(&mut bytes).unwrap();
        assert!(bytes[..offset].iter().all(|&byte| byte == UNTOUCHED));
        let burned = decode(&bytes[offset..], stride, PLANAR).expect("the strip reads back");
        assert!(burned >= before && burned <= monotonic_ns());
    }
}
