//! Linux KMS/DRM display sink for NV12 frames.
//!
//! M14: direct scanout to a connected display via the primary plane on the
//! first usable CRTC. Consumes NV12 `DataFrame`s (the layout
//! [`crate::ffmpegdec::FfmpegH264Dec`] emits when configured with
//! [`crate::ffmpegdec::OutputFormat::Nv12`]) and presents them with vsync
//! page-flips.
//!
//! Pipeline:
//!
//! ```text
//! RtspSrc ─► FfmpegH264Dec(Nv12) ─► KmsSink
//!                                     │
//!                                     └─► primary plane scanout
//! ```
//!
//! ## Operational constraints
//!
//! - **Needs DRM master.** A running Wayland/X11 compositor already holds
//!   master on the GPU; opening `/dev/dri/card0` and calling `set_crtc`
//!   returns `EBUSY` (`PermissionDenied`). Test from a tty
//!   (`Ctrl+Alt+F3`) or via DRM lease. Production embedded deployments
//!   typically run without a compositor.
//! - **NV12 only.** I420 callers must reconfigure the decoder
//!   (`with_output_format(OutputFormat::Nv12)`); the sink rejects everything
//!   else with `CapsMismatch`.
//! - **Fixed input dims.** The dumb-buffer pool is allocated at
//!   negotiation time. Mid-stream `CapsChanged` to a different geometry is
//!   not supported in v1 (decoder/source resolution must stay constant for
//!   the session).
//! - **No letterboxing / scaling.** Buffers scan out at their native
//!   dimensions on the primary plane; if the video is smaller than the
//!   active display mode the result is driver-dependent (commonly the
//!   video at origin with stale framebuffer around it). v2 will add an
//!   overlay-plane path with src/dst rectangles.
//! - **Tearing-free, not low-latency.** Each frame waits for the
//!   `PageFlip` event of the previous one before submitting. Display rate
//!   limits throughput. For pure low-latency you'd want async flips
//!   (`DRM_MODE_PAGE_FLIP_ASYNC`), deferred to v2.
//!
//! ## Threading
//!
//! `KmsSink` holds a `Card` (an owned `std::fs::File` over `/dev/dri/cardN`)
//! plus DRM resource handles. All KMS calls happen on the runner's worker
//! thread via `&mut self`; nothing is shared. No raw pointers, no `unsafe`.

use core::future::Future;
use core::pin::Pin;
use std::os::unix::io::{AsFd, BorrowedFd};
use std::path::{Path, PathBuf};

use alloc::boxed::Box;
use alloc::vec::Vec;

use drm::buffer::{Buffer, DrmFourcc, DrmModifier, Handle as BufferHandle, PlanarBuffer};
use drm::control::{
    connector, crtc, framebuffer, Device as ControlDevice, FbCmd2Flags, Event, Mode,
    PageFlipFlags,
};
use drm::Device;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, HardwareError, MemoryDomain, OutputSink,
    PipelinePacket, VideoFormat,
};

/// Thin wrapper over `/dev/dri/cardN` implementing the `drm` device traits
/// via its owned file's borrowed fd. Same pattern the upstream `drm` crate
/// examples use; we redeclare it here so `g2g-plugins` doesn't take an
/// example-only dependency.
#[derive(Debug)]
struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, G2gError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Card(file))
    }
}

/// One NV12 dumb buffer + the registered KMS framebuffer that points at it.
///
/// NV12 has a Y plane (full res, 8bpp) followed by an interleaved UV plane
/// (half res, 16bpp per pixel pair). We allocate a *single* dumb buffer
/// sized to hold both planes — `w` columns by `h * 3 / 2` rows at 8bpp,
/// which gives `w * h * 3 / 2` bytes total — and treat the same buffer
/// handle as both planes via offsets when adding the framebuffer.
struct Slot {
    db: drm::control::dumbbuffer::DumbBuffer,
    fb: framebuffer::Handle,
    /// Width / height of the *video*, not the dumb buffer. The buffer's
    /// physical extent is (width, height * 3 / 2) at 8bpp.
    width: u32,
    height: u32,
}

/// Planar-buffer adapter so we can call `add_planar_framebuffer` with an
/// NV12 layout backed by a single dumb-buffer handle.
struct Nv12Planar {
    handle: BufferHandle,
    width: u32,
    height: u32,
    /// Bytes per row of the Y plane. NV12's UV plane uses the same pitch
    /// (each row holds `width/2` interleaved UV pairs = `width` bytes).
    pitch: u32,
}

impl PlanarBuffer for Nv12Planar {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    fn format(&self) -> DrmFourcc {
        DrmFourcc::Nv12
    }
    fn modifier(&self) -> Option<DrmModifier> {
        None
    }
    fn pitches(&self) -> [u32; 4] {
        [self.pitch, self.pitch, 0, 0]
    }
    fn handles(&self) -> [Option<BufferHandle>; 4] {
        [Some(self.handle), Some(self.handle), None, None]
    }
    fn offsets(&self) -> [u32; 4] {
        // UV plane sits immediately after the Y plane in the same buffer.
        [0, self.pitch * self.height, 0, 0]
    }
}

/// Pool size. Two is the minimum that lets a flip be in flight while we
/// fill the next buffer; three lets us tolerate one frame of decoder
/// jitter before stalling on vsync. Two keeps memory minimal and matches
/// the simple flip cadence v1 uses.
const POOL_SIZE: usize = 2;

pub struct KmsSink {
    device_path: PathBuf,
    card: Option<Card>,
    connector: Option<connector::Handle>,
    crtc: Option<crtc::Handle>,
    mode: Option<Mode>,
    slots: Vec<Slot>,
    /// Which slot is *currently scanning out*. The next frame is rendered
    /// into the other one and page-flipped to.
    front: usize,
    /// Set after the first frame's `set_crtc`. Until then we drive the
    /// initial mode; afterwards we use `page_flip`.
    crtc_set: bool,
    /// True while a previously-submitted page flip is still in flight.
    /// We drain the corresponding `PageFlip` event before scheduling the
    /// next flip so the buffer we're about to overwrite is off scanout.
    flip_pending: bool,
    /// Bytes per row for the Y plane. Equals video width, which is also
    /// the NV12 UV-pair-row stride; precomputed for the per-frame copy.
    pitch: u32,
    width: u32,
    height: u32,
    frames_presented: u64,
}

impl core::fmt::Debug for KmsSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KmsSink")
            .field("device_path", &self.device_path)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frames_presented", &self.frames_presented)
            .finish()
    }
}

impl Default for KmsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl KmsSink {
    /// Defaults to `/dev/dri/card0`. Override with [`Self::with_device`]
    /// for multi-GPU systems or render-node-only environments.
    pub fn new() -> Self {
        Self {
            device_path: PathBuf::from("/dev/dri/card0"),
            card: None,
            connector: None,
            crtc: None,
            mode: None,
            slots: Vec::new(),
            front: 0,
            crtc_set: false,
            flip_pending: false,
            pitch: 0,
            width: 0,
            height: 0,
            frames_presented: 0,
        }
    }

    pub fn with_device<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.device_path = path.into();
        self
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented
    }

    fn open_and_discover(&mut self) -> Result<(), G2gError> {
        let card = Card::open(&self.device_path)?;

        let res = card
            .resource_handles()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Pick the first *connected* connector. The order is driver-stable
        // for a given hardware topology, so on multi-output setups the
        // caller can rely on it (and we can add explicit selection later).
        let con_handle = res
            .connectors()
            .iter()
            .copied()
            .find_map(|h| {
                let info = card.get_connector(h, true).ok()?;
                (info.state() == connector::State::Connected).then_some(h)
            })
            .ok_or(G2gError::Hardware(HardwareError::Other))?;

        let con_info = card
            .get_connector(con_handle, true)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let mode = *con_info
            .modes()
            .first()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;

        // Pick any CRTC the driver advertises. Atomic configuration would
        // intersect with `possible_crtcs`; v1's set_crtc path is forgiving
        // enough that the first CRTC works on every Linux desktop GPU.
        let crtc_handle = res
            .crtcs()
            .first()
            .copied()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;

        self.card = Some(card);
        self.connector = Some(con_handle);
        self.crtc = Some(crtc_handle);
        self.mode = Some(mode);
        Ok(())
    }

    fn allocate_slots(&mut self, width: u32, height: u32) -> Result<(), G2gError> {
        let card = self.card.as_ref().ok_or(G2gError::NotConfigured)?;
        let mut slots = Vec::with_capacity(POOL_SIZE);
        for _ in 0..POOL_SIZE {
            // Allocate a single dumb buffer covering Y + UV. Total bytes =
            // width * height * 3 / 2, achieved by asking for a buffer that
            // is `width` wide but `height + height/2` tall at 8bpp.
            let buf_height = height + height / 2;
            let db = card
                .create_dumb_buffer((width, buf_height), DrmFourcc::C8, 8)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let pitch = db.pitch();
            let planar = Nv12Planar {
                handle: db.handle(),
                width,
                height,
                pitch,
            };
            let fb = card
                .add_planar_framebuffer(&planar, FbCmd2Flags::empty())
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            slots.push(Slot {
                db,
                fb,
                width,
                height,
            });
        }
        self.slots = slots;
        self.pitch = self.slots[0].db.pitch();
        self.width = width;
        self.height = height;
        self.front = 0;
        Ok(())
    }

    /// Copy an NV12 frame into the back slot's dumb buffer, then either
    /// `set_crtc` (first frame) or `page_flip` (subsequent frames). On
    /// flip submission, drains the previous flip's completion event so we
    /// only overwrite buffers that are off scanout.
    fn present(&mut self, nv12: &[u8]) -> Result<(), G2gError> {
        // Drain a still-pending flip first; we must not overwrite the
        // buffer that the CRTC is still reading from.
        if self.flip_pending {
            self.wait_for_flip()?;
        }

        let back = (self.front + 1) % POOL_SIZE;
        let card = self.card.as_ref().ok_or(G2gError::NotConfigured)?;

        let crtc_handle = self.crtc.ok_or(G2gError::NotConfigured)?;
        let mode = self.mode.ok_or(G2gError::NotConfigured)?;
        let con_handle = self.connector.ok_or(G2gError::NotConfigured)?;

        // Tight scope on the map: drop the DumbMapping before we make any
        // KMS call that might invalidate the kernel's view of the buffer.
        {
            let slot = &mut self.slots[back];
            let mut map = card
                .map_dumb_buffer(&mut slot.db)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            copy_nv12(nv12, slot.width, slot.height, self.pitch, map.as_mut())?;
        }

        let fb = self.slots[back].fb;

        if !self.crtc_set {
            card.set_crtc(crtc_handle, Some(fb), (0, 0), &[con_handle], Some(mode))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            self.crtc_set = true;
        } else {
            card.page_flip(crtc_handle, fb, PageFlipFlags::EVENT, None)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            self.flip_pending = true;
        }

        self.front = back;
        self.frames_presented += 1;
        Ok(())
    }

    /// Block until the kernel reports the in-flight page flip completed.
    /// Drops every event up to and including a `PageFlip`; vblank events
    /// (we don't request them) are ignored.
    fn wait_for_flip(&mut self) -> Result<(), G2gError> {
        let card = self.card.as_ref().ok_or(G2gError::NotConfigured)?;
        loop {
            let events = card
                .receive_events()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let mut saw_flip = false;
            for ev in events {
                if matches!(ev, Event::PageFlip(_)) {
                    saw_flip = true;
                }
            }
            if saw_flip {
                self.flip_pending = false;
                return Ok(());
            }
            // No flip event yet; the read returned vblanks or nothing.
            // Loop and block again.
        }
    }

    fn teardown(&mut self) {
        // Best-effort cleanup. If the card is gone or the kernel has
        // already torn things down (eg tty switch), we silently move on.
        if self.card.is_none() {
            return;
        }
        if self.flip_pending {
            let _ = self.wait_for_flip();
        }
        let slots = core::mem::take(&mut self.slots);
        if let Some(card) = self.card.as_ref() {
            for slot in slots {
                let _ = card.destroy_framebuffer(slot.fb);
                let _ = card.destroy_dumb_buffer(slot.db);
            }
        }
        self.crtc_set = false;
        self.flip_pending = false;
    }
}

impl Drop for KmsSink {
    fn drop(&mut self) {
        self.teardown();
    }
}

impl AsyncElement for KmsSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through. See WaylandSink for the same rationale: Phase-1
        // negotiation carries the decoder's pre-decode (H.264) caps, not
        // the post-decode NV12. We validate NV12 in `configure_pipeline`
        // when the decoder's mid-stream CapsChanged cascades down to us.
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::Video {
                format: VideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*w, *h),
            // Initial negotiation with the decoder's H.264 input caps:
            // accept as no-op, defer real setup until the NV12
            // CapsChanged arrives mid-stream.
            Caps::Video { .. } => return Ok(ConfigureOutcome::Accepted),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w % 2 != 0 || h % 2 != 0 {
            // NV12's UV plane is half-res in both dims; odd extents would
            // need rounded handling we don't bother with in v1.
            return Err(G2gError::CapsMismatch);
        }

        // Mid-stream geometry change after the buffer pool is allocated
        // means we'd need to reallocate. Not handled in v1 — surface as
        // CapsMismatch so the operator notices.
        if !self.slots.is_empty() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            return Err(G2gError::CapsMismatch);
        }

        if self.card.is_none() {
            self.open_and_discover()?;
        }
        self.allocate_slots(w, h)?;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain, .. }) => {
                    let MemoryDomain::System(slice) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.present(slice.as_slice())?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) | PipelinePacket::Flush => Ok(()),
                PipelinePacket::Eos => {
                    // Drain any in-flight flip so the final frame is fully
                    // presented before the pipeline tears down.
                    if self.flip_pending {
                        self.wait_for_flip()?;
                    }
                    Ok(())
                }
            }
        })
    }
}

/// Copy an `nv12` source buffer (Y plane of `width * height` followed by
/// UV plane of `width * height / 2`) into a destination dumb buffer with
/// the given `pitch`. The destination layout is the kernel's NV12 dumb
/// buffer: Y plane first at offset 0, UV plane at offset `pitch * height`,
/// both at `pitch` bytes per row.
fn copy_nv12(src: &[u8], width: u32, height: u32, pitch: u32, dst: &mut [u8]) -> Result<(), G2gError> {
    let w = width as usize;
    let h = height as usize;
    let stride = pitch as usize;
    let y_bytes = w * h;
    let uv_rows = h / 2;
    let uv_bytes = w * uv_rows;
    if src.len() < y_bytes + uv_bytes {
        return Err(G2gError::CapsMismatch);
    }
    if dst.len() < stride * (h + uv_rows) {
        return Err(G2gError::Hardware(HardwareError::Other));
    }

    let (y_src, uv_src) = src.split_at(y_bytes);

    // Y plane: rows of `w` bytes copied into rows of `stride` bytes.
    for row in 0..h {
        let dst_off = row * stride;
        let src_off = row * w;
        dst[dst_off..dst_off + w].copy_from_slice(&y_src[src_off..src_off + w]);
    }
    // UV plane: same row count as half-height, same w bytes per row (one
    // U/V pair per chroma sample = 2 bytes per chroma pair, `w/2` pairs =
    // `w` bytes). Lives in dst starting at row `h`.
    for row in 0..uv_rows {
        let dst_off = (h + row) * stride;
        let src_off = row * w;
        dst[dst_off..dst_off + w].copy_from_slice(&uv_src[src_off..src_off + w]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::Rate;

    #[test]
    fn intercept_passes_through_h264_for_deferred_configure() {
        let sink = KmsSink::new();
        let h264 = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn intercept_passes_through_nv12() {
        let sink = KmsSink::new();
        let nv12 = Caps::Video {
            format: VideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&nv12), Ok(nv12));
    }

    #[test]
    fn configure_accepts_h264_as_deferred_noop() {
        let mut sink = KmsSink::new();
        let h264 = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        match sink.configure_pipeline(&h264) {
            Ok(_) => {}
            Err(e) => panic!("deferred H.264 accept failed: {e:?}"),
        }
        assert!(sink.card.is_none(), "no DRM device opened on H.264 caps");
        assert!(sink.slots.is_empty(), "no buffers allocated on H.264 caps");
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = KmsSink::new();
        let odd = Caps::Video {
            format: VideoFormat::Nv12,
            width: Dim::Fixed(641),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        // `ConfigureOutcome` doesn't implement `PartialEq`, so match the
        // Err arm explicitly rather than using assert_eq!.
        match sink.configure_pipeline(&odd) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn copy_nv12_packs_planes_at_pitch() {
        // 4x2 NV12: Y = 8 bytes, UV = 4 bytes (1 chroma row of 4 bytes).
        // Destination pitch = 8 to exercise the row-stride path.
        let src: Vec<u8> = (0..12).collect();
        let mut dst = vec![0u8; 8 * 3];
        copy_nv12(&src, 4, 2, 8, &mut dst).unwrap();
        // Y plane rows at pitch 8: rows are width=4 bytes wide, rest unset.
        assert_eq!(&dst[0..4], &[0, 1, 2, 3]);
        assert_eq!(&dst[8..12], &[4, 5, 6, 7]);
        // UV plane begins at row=2 (offset 16); one row of 4 bytes.
        assert_eq!(&dst[16..20], &[8, 9, 10, 11]);
    }

    #[test]
    fn copy_nv12_rejects_truncated_src() {
        let src = vec![0u8; 10]; // Need 12 for 4x2 NV12.
        let mut dst = vec![0u8; 24];
        assert!(copy_nv12(&src, 4, 2, 8, &mut dst).is_err());
    }

    fn fmt_unused(_: &dyn core::fmt::Debug) {}

    #[test]
    fn debug_impl_does_not_touch_card() {
        // The Debug derive on KmsSink must work on an unopened sink so we
        // can print/log it before the first negotiation. Card is None at
        // this point — if Debug tried to format it as a Card we'd panic.
        let sink = KmsSink::new();
        fmt_unused(&sink);
    }
}
