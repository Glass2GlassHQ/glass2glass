//! Linux H.264 decode element using the `cros-codecs` VAAPI backend.
//!
//! M13: consumes Annex-B H.264 `DataFrame`s (the bitstream `RtspSrc` /
//! `H264Parse` already emit, `MemoryDomain::System`) and produces decoded NV12
//! frames, also `MemoryDomain::System` (CPU copy out of the GBM-allocated
//! surface). A `CapsChanged(Nv12, w, h)` is emitted before the first decoded
//! frame and again whenever the decoder signals a resolution change.
//!
//! Pipeline:
//!
//! ```text
//! RtspSrc ─► H264Parse ─► VaapiH264Dec ─► [downstream sink / ML]
//!  (System/H264 Annex-B)       (System/NV12)
//! ```
//!
//! Threading: `cros_codecs::libva::Display` is `Rc<Display>` and therefore
//! `!Send`. The element is moved between worker threads but never shared
//! (the runner holds at most one `&mut self` reference at a time), so an
//! `unsafe impl Send` is sound on the same grounds as `MfDecode`: ownership
//! transfer, never aliasing.
//!
//! Deferred:
//! - Zero-copy `MemoryDomain::DmaBuf` output. The GBM-allocated surface is
//!   already a DMA-buf; exposing its fd via `OwnedDmaBuf` is a follow-up that
//!   needs a refcount story to keep the surface alive until downstream
//!   consumers release it. This element copies pixels into `System` memory
//!   to match `MfDecode`'s shape.
//! - H.265 decode. The same stateless decoder framework supports it; a sibling
//!   element keyed on `VideoFormat::H265` is straightforward.
//! - Mid-stream resolution change is observed (`DecoderEvent::FormatChanged`)
//!   but resolution-driven `Reconfigure` upstream is not yet plumbed.
//!
//! Known runtime limitations (cros-codecs 0.0.6, not g2g):
//! - On AMD desktop GPUs (radeonsi), `libva::Display::open()` and bitstream
//!   parsing both succeed (the SPS / first frames are decoded as far as the
//!   parameter-set stage), but **frame allocation fails**: cros-codecs's
//!   `GbmDevice::new_frame(NV12, ...)` calls `gbm_bo_create` with the
//!   ChromeOS-specific `GBM_BO_USE_HW_VIDEO_DECODER` flag (1 << 13), which
//!   radeonsi does not honour for `NV12`. The standard `GBM_BO_USE_LINEAR`
//!   fallback also fails — Mesa's radeonsi GBM provider does not expose
//!   `NV12` contiguous allocations at all. This is a cros-codecs assumption
//!   inherited from ChromeOS hardware, not a g2g bug; the implementation is
//!   correct against the cros-codecs API. On Intel iGPUs with the iHD VAAPI
//!   driver, the same code is expected to work end-to-end. The recommended
//!   path on AMD desktop is to wait for a cros-codecs surface backend that
//!   uses libva-managed surfaces (no GBM), or to fall back to ffmpeg's
//!   `h264_vaapi` decoder behind a separate feature.
//! - cros-codecs hard-codes a 16x16 initial VAContext at decoder construction
//!   time, which AMD rejects with `VA_STATUS_ERROR_RESOLUTION_NOT_SUPPORTED`
//!   before any bitstream is even fed. A larger initial size (e.g. 1920x1088)
//!   accepts on every driver in the field and is resized by `new_sequence()`
//!   once the SPS lands. Upstream patch pending.

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use alloc::boxed::Box;
use alloc::vec::Vec;

use cros_codecs::bitstream_utils::NalIterator;
use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::decoder::stateless::h264::H264;
use cros_codecs::decoder::stateless::{
    DecodeError, DynStatelessVideoDecoder, StatelessDecoder, StatelessVideoDecoder,
};
use cros_codecs::decoder::{DecodedHandle, DecoderEvent, StreamInfo};
use cros_codecs::libva;
use cros_codecs::video_frame::gbm_video_frame::{GbmDevice, GbmUsage};
use cros_codecs::video_frame::generic_dma_video_frame::GenericDmaVideoFrame;
use cros_codecs::video_frame::{VideoFrame, UV_PLANE, Y_PLANE};
use cros_codecs::{BlockingMode, Fourcc};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, VideoFormat,
};

/// Default DRM render node. The user can pick a different device via
/// [`VaapiH264Dec::with_render_node`] for multi-GPU systems.
const DEFAULT_RENDER_NODE: &str = "/dev/dri/renderD128";

/// One decoded picture, pixels already copied out of the GBM surface.
struct DecodedNv12 {
    bytes: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
}

pub struct VaapiH264Dec {
    render_node: PathBuf,
    gbm: Option<std::sync::Arc<GbmDevice>>,
    decoder: Option<DynStatelessVideoDecoder<GenericDmaVideoFrame>>,
    info: Option<StreamInfo>,
    last_caps: Option<Caps>,
    configured: bool,
    emitted: u64,
}

// SAFETY: `DynStatelessVideoDecoder` owns an `Rc<libva::Display>` (`!Send`).
// The framework's `multi-thread` runner requires `Send` elements so it can hand
// a task between worker threads. We uphold that by construction and contract:
// libva is callable from any thread (driver-level locking), the runner drives
// the element through `&mut self` (never concurrently), and the contained `Rc`
// is moved with the element — no clone is shared across the move boundary, so
// the non-atomic refcount is never raced.
unsafe impl Send for VaapiH264Dec {}

impl core::fmt::Debug for VaapiH264Dec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VaapiH264Dec")
            .field("render_node", &self.render_node)
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl Default for VaapiH264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl VaapiH264Dec {
    pub fn new() -> Self {
        Self::with_render_node(DEFAULT_RENDER_NODE)
    }

    pub fn with_render_node<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            render_node: path.into(),
            gbm: None,
            decoder: None,
            info: None,
            last_caps: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Iterate Annex-B NAL units out of one access unit and feed each one.
    fn feed_access_unit(
        &mut self,
        bitstream: &[u8],
        pts_ns: u64,
        decoded: &mut Vec<DecodedNv12>,
    ) -> Result<(), G2gError> {
        // cros-codecs takes timestamps as `u64`. The unit is opaque to the
        // backend — it's echoed back unchanged on the decoded handle — so we
        // feed nanoseconds straight through to avoid lossy conversions.
        for nal in NalIterator::<H264Nalu>::new(bitstream) {
            self.feed_nal(nal.as_ref(), pts_ns, decoded)?;
        }
        Ok(())
    }

    fn feed_nal(
        &mut self,
        nal: &[u8],
        pts_ns: u64,
        decoded: &mut Vec<DecodedNv12>,
    ) -> Result<(), G2gError> {
        let mut offset = 0usize;
        let mut guard = 0u32;
        // `decode()` may consume the whole NAL or just a prefix; loop until
        // we've drained it. `CheckEvents` / `NotEnoughOutputBuffers` mean the
        // backend wants us to dequeue events (which may include format change
        // or returning a finished frame to the pool) before retrying.
        while offset < nal.len() {
            match self.try_decode(pts_ns, &nal[offset..]) {
                Ok(consumed) => {
                    if consumed == 0 {
                        // Defensive: should not happen but avoid infinite loop.
                        self.drain_events(decoded)?;
                    }
                    offset += consumed;
                }
                Err(DecodeError::CheckEvents)
                | Err(DecodeError::NotEnoughOutputBuffers(_)) => {
                    self.drain_events(decoded)?;
                }
                Err(_) => return Err(G2gError::Hardware(HardwareError::V4l2(0))),
            }
            guard += 1;
            if guard > 128 {
                return Err(G2gError::Hardware(HardwareError::V4l2(0)));
            }
        }
        self.drain_events(decoded)
    }

    fn try_decode(&mut self, timestamp: u64, bytes: &[u8]) -> Result<usize, DecodeError> {
        // The stream may not have produced a StreamInfo yet — that arrives via
        // a `FormatChanged` event after the SPS is parsed. The allocator
        // closure handles that by returning `None`, which `decode()` surfaces
        // as `DecodeError::CheckEvents`, prompting the caller to drain.
        let info = self.info.clone();
        let gbm = self.gbm.as_ref().cloned();
        let mut alloc_cb = move || -> Option<GenericDmaVideoFrame> {
            let info = info.as_ref()?;
            let gbm = gbm.as_ref()?.clone();
            gbm.new_frame(
                Fourcc::from(b"NV12"),
                info.display_resolution,
                info.coded_resolution,
                GbmUsage::Decode,
            )
            .ok()?
            .to_generic_dma_video_frame()
            .ok()
        };
        // `decoder` is `Some` whenever `configured` is true; the caller checks
        // `configured` before reaching the decode loop.
        let decoder = self.decoder.as_mut().expect("decoder must be initialised");
        decoder.decode(timestamp, bytes, &mut alloc_cb)
    }

    fn drain_events(&mut self, decoded: &mut Vec<DecodedNv12>) -> Result<(), G2gError> {
        loop {
            let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
            let Some(event) = decoder.next_event() else {
                return Ok(());
            };
            match event {
                DecoderEvent::FormatChanged => {
                    // Re-borrow after consuming the event.
                    let decoder = self.decoder.as_mut().expect("decoder still present");
                    self.info = decoder.stream_info().cloned();
                }
                DecoderEvent::FrameReady(handle) => {
                    let pts_ns = handle.timestamp();
                    let frame = handle.video_frame();
                    let bytes = copy_nv12(&*frame)?;
                    let res = frame.resolution();
                    decoded.push(DecodedNv12 {
                        bytes,
                        width: res.width,
                        height: res.height,
                        pts_ns,
                    });
                }
            }
        }
    }

    fn drain_eos(&mut self, decoded: &mut Vec<DecodedNv12>) -> Result<(), G2gError> {
        if let Some(d) = self.decoder.as_mut() {
            d.flush()
                .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?;
        }
        self.drain_events(decoded)
    }
}

impl AsyncElement for VaapiH264Dec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes H.264 at any geometry; intersecting narrows the proposal
        // and rejects non-H.264 inputs.
        let supported = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Video {
                format: VideoFormat::H264,
                ..
            } => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        let display = libva::Display::open()
            .ok_or(G2gError::Hardware(HardwareError::V4l2(0)))?;
        let gbm = GbmDevice::open(&self.render_node)
            .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?;
        let decoder = StatelessDecoder::<H264, _>::new_vaapi(display, BlockingMode::Blocking)
            .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?
            .into_trait_object();
        self.gbm = Some(gbm);
        self.decoder = Some(decoder);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut decoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed_access_unit(slice.as_slice(), frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(_) => {
                    // Upstream H.264 caps are swallowed; we emit our own NV12
                    // CapsChanged from the decoder's stream info before each
                    // decoded frame whose geometry differs from the last one
                    // we advertised.
                }
                PipelinePacket::Flush => {
                    if let Some(d) = self.decoder.as_mut() {
                        d.flush()
                            .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?;
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
            }

            for d in decoded {
                let new_caps = nv12_caps(d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.bytes)),
                    caps: new_caps,
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                    },
                    sequence: self.emitted,
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Copy NV12 pixels out of a decoded VAAPI surface into a packed
/// `width * height * 3 / 2` buffer (Y plane followed by interleaved UV).
/// The source plane pitch may exceed `width` due to hardware alignment, so
/// each row is copied individually.
fn copy_nv12<F: VideoFrame>(frame: &F) -> Result<Box<[u8]>, G2gError> {
    let res = frame.resolution();
    let w = res.width as usize;
    let h = res.height as usize;
    let y_size = w * h;
    let uv_size = w * h / 2;

    let pitches = frame.get_plane_pitch();
    if pitches.len() < 2 {
        return Err(G2gError::Hardware(HardwareError::V4l2(0)));
    }
    let mapping = frame
        .map()
        .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?;
    let planes = mapping.get();
    if planes.len() < 2 {
        return Err(G2gError::Hardware(HardwareError::V4l2(0)));
    }

    let y_pitch = pitches[Y_PLANE];
    let uv_pitch = pitches[UV_PLANE];
    let y_src = planes[Y_PLANE];
    let uv_src = planes[UV_PLANE];

    let mut out = alloc::vec![0u8; y_size + uv_size];

    for row in 0..h {
        let src_start = row * y_pitch;
        let dst_start = row * w;
        out[dst_start..dst_start + w].copy_from_slice(&y_src[src_start..src_start + w]);
    }
    for row in 0..(h / 2) {
        let src_start = row * uv_pitch;
        let dst_start = y_size + row * w;
        out[dst_start..dst_start + w].copy_from_slice(&uv_src[src_start..src_start + w]);
    }

    Ok(out.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_caps_are_fixed() {
        assert_eq!(
            nv12_caps(640, 480),
            Caps::Video {
                format: VideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
            }
        );
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = VaapiH264Dec::new();
        let vp9 = Caps::Video {
            format: VideoFormat::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_narrows_h264_geometry() {
        let dec = VaapiH264Dec::new();
        let proposal = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn unconfigured_decoder_reports_zero_decoded() {
        let dec = VaapiH264Dec::new();
        assert_eq!(dec.decoded_count(), 0);
    }
}
