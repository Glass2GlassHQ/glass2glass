//! Linux H.264 / H.265 decode elements using the `cros-codecs` VAAPI backend.
//!
//! M13: consumes Annex-B `DataFrame`s (the bitstream `RtspSrc` / `H264Parse` /
//! `H265Parse` already emit, `MemoryDomain::System`) and produces decoded NV12
//! frames, also `MemoryDomain::System` (CPU copy out of the GBM-allocated
//! surface). A `CapsChanged(Nv12, w, h)` is emitted before the first decoded
//! frame and again whenever the decoder signals a resolution change.
//!
//! [`VaapiDec`] holds the whole decode path; the codec picks the cros-codecs
//! stateless decoder and the NAL splitter through the [`VaapiCodec`] binding,
//! so [`VaapiH264Dec`] and [`VaapiH265Dec`] (M1036) are the same element with
//! a different binding.
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
//! A mid-stream resolution change (`DecoderEvent::FormatChanged` reporting a
//! new display resolution) both re-emits the output `CapsChanged` and stashes
//! a `Reconfigure::Propose` carrying the new input geometry, which the runner
//! collects through `take_reconfigure` and relays up the input link.
//!
//! Deferred:
//! - Zero-copy `MemoryDomain::DmaBuf` output. The GBM-allocated surface is
//!   already a DMA-buf; exposing its fd via `OwnedDmaBuf` is a follow-up that
//!   needs a refcount story to keep the surface alive until downstream
//!   consumers release it. This element copies pixels into `System` memory
//!   to match `MfDecode`'s shape.
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
use core::marker::PhantomData;
use core::pin::Pin;
use std::path::PathBuf;
use std::rc::Rc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;

use cros_codecs::bitstream_utils::NalIterator;
use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::decoder::stateless::h264::H264;
use cros_codecs::decoder::stateless::h265::H265;
use cros_codecs::decoder::stateless::{
    DecodeError, DynStatelessVideoDecoder, StatelessDecoder, StatelessVideoDecoder,
};
use cros_codecs::decoder::{DecodedHandle, DecoderEvent, StreamInfo};
use cros_codecs::libva;
use cros_codecs::video_frame::gbm_video_frame::{GbmDevice, GbmUsage};
use cros_codecs::video_frame::generic_dma_video_frame::GenericDmaVideoFrame;
use cros_codecs::video_frame::{VideoFrame, UV_PLANE, Y_PLANE};
use cros_codecs::{BlockingMode, Fourcc, Resolution};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    Reconfigure, VideoCodec,
};

/// Default DRM render node. The user can pick a different device via
/// [`VaapiDec::with_render_node`] for multi-GPU systems.
const DEFAULT_RENDER_NODE: &str = "/dev/dri/renderD128";

/// The codec half of the VAAPI decode path: which cros-codecs stateless
/// decoder to build, how to split an access unit into NAL units, and the
/// names this element goes by. Implemented by [`H264Codec`] / [`H265Codec`];
/// cros-codecs keys both the decoder type and the NAL parser on the codec, so
/// they have to travel together.
pub trait VaapiCodec: 'static {
    /// The compressed format this decoder consumes.
    const CODEC: VideoCodec;
    /// `G2G_DEBUG` filtering key, also the name in error messages.
    const LOG_CATEGORY: &'static str;
    /// `gst-inspect` long name.
    const LONG_NAME: &'static str;
    /// `gst-inspect` description.
    const DESCRIPTION: &'static str;

    /// Build the cros-codecs stateless decoder over an open VA display.
    fn open_decoder(
        display: Rc<libva::Display>,
    ) -> Result<DynStatelessVideoDecoder<GenericDmaVideoFrame>, G2gError>;

    /// Split one Annex-B access unit into its NAL units.
    fn nal_units(bitstream: &[u8]) -> Box<dyn Iterator<Item = Cow<'_, [u8]>> + '_>;
}

/// H.264 binding for [`VaapiDec`].
#[derive(Debug)]
pub struct H264Codec;

impl VaapiCodec for H264Codec {
    const CODEC: VideoCodec = VideoCodec::H264;
    const LOG_CATEGORY: &'static str = "VaapiH264Dec";
    const LONG_NAME: &'static str = "VA-API H.264 decoder";
    const DESCRIPTION: &'static str = "Hardware H.264 decode via VA-API";

    fn open_decoder(
        display: Rc<libva::Display>,
    ) -> Result<DynStatelessVideoDecoder<GenericDmaVideoFrame>, G2gError> {
        Ok(
            StatelessDecoder::<H264, _>::new_vaapi(display, BlockingMode::Blocking)
                .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?
                .into_trait_object(),
        )
    }

    fn nal_units(bitstream: &[u8]) -> Box<dyn Iterator<Item = Cow<'_, [u8]>> + '_> {
        Box::new(NalIterator::<H264Nalu>::new(bitstream))
    }
}

/// H.265 binding for [`VaapiDec`].
#[derive(Debug)]
pub struct H265Codec;

impl VaapiCodec for H265Codec {
    const CODEC: VideoCodec = VideoCodec::H265;
    const LOG_CATEGORY: &'static str = "VaapiH265Dec";
    const LONG_NAME: &'static str = "VA-API H.265 decoder";
    const DESCRIPTION: &'static str = "Hardware H.265 decode via VA-API";

    fn open_decoder(
        display: Rc<libva::Display>,
    ) -> Result<DynStatelessVideoDecoder<GenericDmaVideoFrame>, G2gError> {
        Ok(
            StatelessDecoder::<H265, _>::new_vaapi(display, BlockingMode::Blocking)
                .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?
                .into_trait_object(),
        )
    }

    fn nal_units(bitstream: &[u8]) -> Box<dyn Iterator<Item = Cow<'_, [u8]>> + '_> {
        Box::new(NalIterator::<H265Nalu>::new(bitstream))
    }
}

/// VA-API H.264 decoder.
pub type VaapiH264Dec = VaapiDec<H264Codec>;

/// VA-API H.265 decoder.
pub type VaapiH265Dec = VaapiDec<H265Codec>;

/// One decoded picture, pixels already copied out of the GBM surface.
struct DecodedNv12 {
    bytes: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::vaapidec::VaapiH264Dec;
///
/// let decoder = VaapiH264Dec::with_render_node("/dev/dri/renderD128");
/// ```
pub struct VaapiDec<C: VaapiCodec> {
    render_node: PathBuf,
    gbm: Option<std::sync::Arc<GbmDevice>>,
    decoder: Option<DynStatelessVideoDecoder<GenericDmaVideoFrame>>,
    info: Option<StreamInfo>,
    last_caps: Option<Caps>,
    /// M16 workaround #3 Phase A: most recent input caps received via
    /// `PipelinePacket::CapsChanged`. See `ffmpegdec.rs` for the full
    /// notes; same shape across all three decoders.
    input_caps: Option<Caps>,
    /// Upstream request parked by a mid-stream resolution change, handed to
    /// the runner by `take_reconfigure`.
    pending_reconfigure: Option<Reconfigure>,
    configured: bool,
    emitted: u64,
    codec: PhantomData<C>,
}

// SAFETY: `DynStatelessVideoDecoder` owns an `Rc<libva::Display>` (`!Send`).
// The framework's `multi-thread` runner requires `Send` elements so it can hand
// a task between worker threads. We uphold that by construction and contract:
// libva is callable from any thread (driver-level locking), the runner drives
// the element through `&mut self` (never concurrently), and the contained `Rc`
// is moved with the element — no clone is shared across the move boundary, so
// the non-atomic refcount is never raced.
unsafe impl<C: VaapiCodec> Send for VaapiDec<C> {}

impl<C: VaapiCodec> core::fmt::Debug for VaapiDec<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(C::LOG_CATEGORY)
            .field("render_node", &self.render_node)
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl<C: VaapiCodec> Default for VaapiDec<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: VaapiCodec> VaapiDec<C> {
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
            input_caps: None,
            pending_reconfigure: None,
            configured: false,
            emitted: 0,
            codec: PhantomData,
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
        for nal in C::nal_units(bitstream) {
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
                Err(DecodeError::CheckEvents) | Err(DecodeError::NotEnoughOutputBuffers(_)) => {
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
                    let previous = self.info.as_ref().map(|i| i.display_resolution);
                    // Re-borrow after consuming the event.
                    let decoder = self.decoder.as_mut().expect("decoder still present");
                    let info = decoder.stream_info().cloned();
                    if let Some(current) = info.as_ref().map(|i| i.display_resolution) {
                        if let Some(request) =
                            geometry_reconfigure::<C>(previous, current, self.input_caps.as_ref())
                        {
                            self.pending_reconfigure = Some(request);
                        }
                    }
                    self.info = info;
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

impl<C: VaapiCodec> PadTemplates for VaapiDec<C> {
    /// Static superset for auto-plug: the codec in (any geometry), raw NV12 out
    /// (the only format the VAAPI path produces, copied from the GBM surface).
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(Caps::CompressedVideo {
                codec: C::CODEC,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            })),
            PadTemplate::source(CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            })),
        ])
    }
}

impl<C: VaapiCodec> AsyncElement for VaapiDec<C> {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes its codec at any geometry; intersecting narrows the proposal
        // and rejects every other format.
        let supported = Caps::CompressedVideo {
            codec: C::CODEC,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Annex-B input the decoder reads with the CPU before handing it to
    /// libva: system memory only.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    /// A mid-stream resolution change parks a counter-proposal for the input
    /// link here; the runner relays it to the upstream producer.
    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        self.pending_reconfigure.take()
    }

    fn log_category(&self) -> &'static str {
        C::LOG_CATEGORY
    }

    /// M16 step 5m: native `DerivedOutput` — accepts the codec at any
    /// geometry and produces NV12 at the same dims/framerate. The closure
    /// validates the input format and returns an empty set on mismatch, so
    /// the solver rejects a foreign upstream at negotiation time instead of
    /// via the dynamic `intercept_caps` callback. Mixed chains get real
    /// per-link caps from the solver: the codec to the decoder, NV12 to the
    /// sink. Mirrors `FfmpegH264Dec` (step 5k); the VAAPI backend only ever
    /// emits NV12, so there is no output-format choice.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| derive_output_caps::<C>(input)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec, .. } if *codec == C::CODEC => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        let display = libva::Display::open().ok_or(G2gError::Hardware(HardwareError::V4l2(0)))?;
        let gbm = GbmDevice::open(&self.render_node)
            .map_err(|_| G2gError::Hardware(HardwareError::V4l2(0)))?;
        let decoder = C::open_decoder(display)?;
        self.gbm = Some(gbm);
        self.decoder = Some(decoder);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            C::LONG_NAME,
            "Codec/Decoder/Video/Hardware",
            C::DESCRIPTION,
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "device",
            PropKind::Str,
            "DRM render node for VA-API (e.g. /dev/dri/renderD128)",
        )];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                self.render_node = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device" => Some(PropValue::Str(
                self.render_node.to_string_lossy().into_owned(),
            )),
            _ => None,
        }
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
                    let slice = frame.domain.require_system_slice(C::LOG_CATEGORY)?;
                    self.feed_access_unit(slice, frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // M16 workaround #3 Phase A: validate + record.
                    // Reject an incompatible mid-stream format change
                    // (e.g. H.264 -> VP9) loud; previously dropped
                    // silently. Output `CapsChanged` is still emitted
                    // from decoded stream info at the decode boundary
                    // so the ordering invariant from §3 is preserved.
                    match &c {
                        Caps::CompressedVideo { codec, .. } if *codec == C::CODEC => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    self.input_caps = Some(c);
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
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            for d in decoded {
                let new_caps = nv12_caps(d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    // M16 workaround #3 Phase A debug assertion. See
                    // `ffmpegdec.rs` for the full rationale.
                    #[cfg(debug_assertions)]
                    if let Some(input) = self.input_caps.as_ref() {
                        let expected = derive_output_caps::<C>(input);
                        debug_assert!(
                            !expected
                                .intersect(&CapsSet::one(new_caps.clone()))
                                .is_empty(),
                            "vaapidec decode-time output {new_caps:?} inconsistent with derive_output_caps({input:?}) = {expected:?}"
                        );
                    }
                    out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                        .await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.bytes)),
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                    meta: Default::default(),
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

/// Single source of truth for the decoder's output-side caps derivation.
/// Shared by the `DerivedOutput` constraint closure and the
/// workaround-#3 Phase A debug assertion. VAAPI only emits NV12, so
/// there's no output-format choice.
fn derive_output_caps<C: VaapiCodec>(input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
        } if *codec == C::CODEC => CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
            interlace: g2g_core::Interlace::Any,
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

/// The upstream request a `FormatChanged` warrants. Only a real change earns
/// one: the first format the stream reports is the geometry negotiation
/// already solved for. The proposal is input-side caps (the link it travels
/// up), carrying the geometry the bitstream turned out to have.
fn geometry_reconfigure<C: VaapiCodec>(
    previous: Option<Resolution>,
    current: Resolution,
    input_caps: Option<&Caps>,
) -> Option<Reconfigure> {
    if previous? == current {
        return None;
    }
    let framerate = match input_caps {
        Some(Caps::CompressedVideo { framerate, .. }) => framerate.clone(),
        _ => Rate::Any,
    };
    Some(Reconfigure::Propose(Caps::CompressedVideo {
        codec: C::CODEC,
        width: Dim::Fixed(current.width),
        height: Dim::Fixed(current.height),
        framerate,
    }))
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
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
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
                interlace: g2g_core::Interlace::Any,
            }
        );
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = VaapiH264Dec::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_narrows_h264_geometry() {
        let dec = VaapiH264Dec::new();
        let proposal = Caps::CompressedVideo {
            codec: VideoCodec::H264,
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

    #[test]
    fn geometry_reconfigure_only_on_a_real_change() {
        let r = |w, h| Resolution {
            width: w,
            height: h,
        };
        let input = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Fixed(30 << 16),
        };
        // First format: nothing to renegotiate.
        assert_eq!(
            geometry_reconfigure::<H265Codec>(None, r(1280, 720), Some(&input)),
            None
        );
        // Same geometry again: nothing either.
        assert_eq!(
            geometry_reconfigure::<H265Codec>(Some(r(1280, 720)), r(1280, 720), Some(&input)),
            None
        );
        // A real change proposes the new input geometry, framerate kept.
        assert_eq!(
            geometry_reconfigure::<H265Codec>(Some(r(1280, 720)), r(1920, 1080), Some(&input)),
            Some(Reconfigure::Propose(Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
            }))
        );
    }

    #[test]
    fn caps_constraint_is_derived_output_h264_to_nv12() {
        // M16 step 5m: DerivedOutput closure validates H.264 input and
        // emits NV12 at the same dims/rate; non-H.264 yields an empty set.
        let dec = VaapiH264Dec::new();
        let CapsConstraint::DerivedOutput(f) = dec.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        assert_eq!(
            f(&h264).alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
            }]
        );

        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(f(&vp9).is_empty());
    }
}
