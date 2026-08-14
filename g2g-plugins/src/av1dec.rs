//! Shared AV1 decoder element, generated for each backend by [`av1_decoder!`].
//!
//! [`Dav1dDec`](crate::dav1ddec) (libdav1d, the C reference with hand-written
//! assembly) and [`Rav1dDec`](crate::rav1ddec) (`re_rav1d`, the pure-Rust port)
//! are the same element: `CompressedVideo{Av1}` in, fully-planar `RawVideo` out,
//! driving the stateful send/drain protocol and recovering per-picture geometry.
//! `re_rav1d` re-exports dav1d-rs's safe API, so the two backends expose an
//! identical `Decoder` / `Picture` surface, differing only in the crate they live
//! in. They are distinct foreign types with no shared trait, so the shared body is
//! factored as a macro parameterized on the backend crate rather than a generic.

/// Generate an AV1 decoder element `$ty` over backend crate `$backend` (whose
/// `Decoder` / `Picture` / `PixelLayout` / `PlanarImageComponent` mirror the
/// dav1d-rs API). `$long_name` / `$description` fill [`ElementMetadata`].
macro_rules! av1_decoder {
    ($ty:ident, $backend:ident, $long_name:expr, $description:expr) => {
        use core::future::Future;
        use core::pin::Pin;

        use alloc::boxed::Box;
        use alloc::vec::Vec;

        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{
            AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
            G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
            RawVideoFormat, Rate, VideoCodec,
        };

        use $backend::{Decoder, PixelLayout, PlanarImageComponent, Picture};

        /// Decoded pictures from one fed unit: each `(format, packed pixels, (width, height))`.
        type DecodedFrames = Vec<(RawVideoFormat, Vec<u8>, (u32, u32))>;

        #[doc = concat!("Decodes an AV1 stream into a fully-planar YUV format via `", stringify!($backend), "`.")]
        pub struct $ty {
            decoder: Option<Decoder>,
            framerate: Rate,
            /// Last emitted (format, width, height), so `CapsChanged` is sent only on change.
            out: Option<(RawVideoFormat, u32, u32)>,
            sequence: u64,
            configured: bool,
            /// Timing of the newest input unit, stamped onto the reorder-delayed
            /// pictures the EOS drain emits (their own units' stamps are gone).
            last_timing: g2g_core::frame::FrameTiming,
        }

        impl core::fmt::Debug for $ty {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                // The backend `Decoder` is not Debug; report only the element's own state.
                f.debug_struct(stringify!($ty))
                    .field("out", &self.out)
                    .field("sequence", &self.sequence)
                    .field("configured", &self.configured)
                    .finish_non_exhaustive()
            }
        }

        impl Default for $ty {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $ty {
            pub fn new() -> Self {
                Self {
                    decoder: None,
                    framerate: Rate::Any,
                    out: None,
                    sequence: 0,
                    configured: false,
                    last_timing: g2g_core::frame::FrameTiming::default(),
                }
            }

            fn input_template() -> Caps {
                Caps::CompressedVideo {
                    codec: VideoCodec::Av1,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                }
            }

            fn output_caps(&self, format: RawVideoFormat, w: u32, h: u32) -> Caps {
                Caps::RawVideo {
                    format,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    framerate: self.framerate.clone(),
                 interlace: g2g_core::Interlace::Any, }
            }

            /// Feed one AV1 temporal unit and collect every picture now decodable,
            /// each as `(format, packed pixels, (width, height))`. Drives the
            /// send/drain protocol so a frame-threading / reordering delay does not
            /// strand input (the `Try again` -> drain -> `send_pending_data` cycle).
            fn feed(decoder: &mut Decoder, unit: Vec<u8>) -> Result<DecodedFrames, G2gError> {
                let mut frames = Vec::new();
                let mut send = decoder.send_data(unit, None, None, None);
                loop {
                    Self::drain_ready(decoder, &mut frames)?;
                    match send {
                        Ok(()) => break, // input fully consumed
                        Err(e) if e.is_again() => send = decoder.send_pending_data(),
                        Err(_) => return Err(G2gError::CapsMismatch),
                    }
                }
                Ok(frames)
            }

            /// Collect every picture decodable right now. With no new input this
            /// is the end-of-stream drain: past the last fed unit the decoder
            /// hands out its reorder-delayed tail until `Try again` means empty
            /// (the dav1d draining contract).
            fn drain_ready(
                decoder: &mut Decoder,
                frames: &mut DecodedFrames,
            ) -> Result<(), G2gError> {
                loop {
                    match decoder.get_picture() {
                        Ok(pic) => {
                            let format = pic_format(&pic)?;
                            frames.push((
                                format,
                                pack_planar(&pic, format)?,
                                (pic.width(), pic.height()),
                            ));
                        }
                        Err(e) if e.is_again() => return Ok(()),
                        Err(_) => return Err(G2gError::CapsMismatch),
                    }
                }
            }

            /// Push `frames` as caps-checked output at `timing`, the shared tail
            /// of the per-unit decode and the EOS drain.
            async fn emit(
                &mut self,
                frames: DecodedFrames,
                timing: g2g_core::frame::FrameTiming,
                out: &mut dyn OutputSink,
            ) -> Result<(), G2gError> {
                for (format, pixels, (w, h)) in frames {
                    if self.out != Some((format, w, h)) {
                        out.push(PipelinePacket::CapsChanged(self.output_caps(format, w, h)))
                            .await?;
                        self.out = Some((format, w, h));
                    }
                    let decoded = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
                        timing,
                        self.sequence,
                    );
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                Ok(())
            }
        }

        /// The fully-planar [`RawVideoFormat`] matching a decoded picture's chroma
        /// layout and bit depth. Rejects monochrome (I400), which has no
        /// planar-YUV format.
        fn pic_format(pic: &Picture) -> Result<RawVideoFormat, G2gError> {
            use RawVideoFormat as F;
            Ok(match (pic.pixel_layout(), pic.bit_depth()) {
                (PixelLayout::I420, 8) => F::I420,
                (PixelLayout::I420, 10) => F::I420p10,
                (PixelLayout::I420, 12) => F::I420p12,
                (PixelLayout::I422, 8) => F::I422,
                (PixelLayout::I422, 10) => F::I422p10,
                (PixelLayout::I422, 12) => F::I422p12,
                (PixelLayout::I444, 8) => F::I444,
                (PixelLayout::I444, 10) => F::I444p10,
                (PixelLayout::I444, 12) => F::I444p12,
                _ => return Err(G2gError::CapsMismatch),
            })
        }

        /// Pack a decoded picture into the tight planar layout of `format` (Y then U
        /// then V), copying each plane row honoring its stride. The chroma plane
        /// dimensions and per-sample byte size come from the format itself, so
        /// 4:2:0 / 4:2:2 / 4:4:4 at 8 / 10 / 12-bit share one path; 10/12-bit
        /// samples are the native LE 2-byte words.
        fn pack_planar(pic: &Picture, format: RawVideoFormat) -> Result<Vec<u8>, G2gError> {
            let bps = format.bytes_per_sample();
            // `chroma_shift` is always `Some` here: `pic_format` only yields planar formats.
            let (hs, vs) = format.chroma_shift().ok_or(G2gError::CapsMismatch)?;
            let (w, h) = (pic.width(), pic.height());
            let (cw, ch) = (w.div_ceil(1 << hs), h.div_ceil(1 << vs));
            let mut out = Vec::with_capacity(((w * h + 2 * cw * ch) as usize) * bps);
            for (comp, pw, ph) in [
                (PlanarImageComponent::Y, w, h),
                (PlanarImageComponent::U, cw, ch),
                (PlanarImageComponent::V, cw, ch),
            ] {
                let plane = pic.plane(comp);
                let (stride, _) = pic.plane_data_geometry(comp);
                let bytes: &[u8] = &plane;
                let row_len = pw as usize * bps;
                for row in 0..ph {
                    let start = (row * stride) as usize;
                    let end = start + row_len;
                    // The plane buffer is stride*height; a row never overruns it.
                    out.extend_from_slice(bytes.get(start..end).ok_or(G2gError::CapsMismatch)?);
                }
            }
            Ok(out)
        }

        impl AsyncElement for $ty {
            type ProcessFuture<'a>
                = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
            where
                Self: 'a;

            /// Reads host memory, so it takes system frames only. The allocation
            /// cascade turns that into a download demand on a GPU producer.
            fn input_domains(&self) -> g2g_core::memory::DomainSet {
                g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
            }

            fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
                upstream_caps.intersect(&Self::input_template())
            }

            fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
                CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
                    Caps::CompressedVideo { codec: VideoCodec::Av1, width, height, framerate } => {
                        CapsSet::one(Caps::RawVideo {
                            format: RawVideoFormat::I420,
                            width: width.clone(),
                            height: height.clone(),
                            framerate: framerate.clone(),
                         interlace: g2g_core::Interlace::Any, })
                    }
                    _ => CapsSet::from_alternatives(Vec::new()),
                }))
            }

            fn configure_pipeline(
                &mut self,
                absolute_caps: &Caps,
            ) -> Result<ConfigureOutcome, G2gError> {
                let Caps::CompressedVideo { codec: VideoCodec::Av1, framerate, .. } = absolute_caps
                else {
                    return Err(G2gError::CapsMismatch);
                };
                self.framerate = framerate.clone();
                self.decoder = Some(Decoder::new().map_err(|_| G2gError::CapsMismatch)?);
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }

            fn metadata(&self) -> ElementMetadata {
                ElementMetadata::new($long_name, "Codec/Decoder/Video", $description, "g2g")
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
                    match packet {
                        PipelinePacket::DataFrame(frame) => {
                            let slice = frame.domain.require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                            let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                            let unit = slice.to_vec();
                            let frames = Self::feed(decoder, unit)?;
                            self.last_timing = frame.timing;
                            self.emit(frames, frame.timing, out).await?;
                        }
                        PipelinePacket::CapsChanged(_) => {}
                        PipelinePacket::Eos => {
                            // A reordering stream holds its tail pictures back
                            // (M1003): drain them before the sentinel, so the
                            // last frames of the stream are not lost.
                            if let Some(decoder) = self.decoder.as_mut() {
                                let mut frames = DecodedFrames::new();
                                Self::drain_ready(decoder, &mut frames)?;
                                let timing = self.last_timing;
                                self.emit(frames, timing, out).await?;
                            }
                            out.push(PipelinePacket::Eos).await?;
                        }
                        other => {
                            out.push(other).await?;
                        }
                    }
                    Ok(())
                })
            }
        }

        impl PadTemplates for $ty {
            fn pad_templates() -> Vec<PadTemplate> {
                let av1 = Self::input_template();
                let raw = Caps::RawVideo {
                    format: RawVideoFormat::I420,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                 interlace: g2g_core::Interlace::Any, };
                Vec::from([
                    PadTemplate::sink(CapsSet::one(av1)),
                    PadTemplate::source(CapsSet::one(raw)),
                ])
            }
        }
    };
}

pub(crate) use av1_decoder;
