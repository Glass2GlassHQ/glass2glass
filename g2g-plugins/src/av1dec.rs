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

        use crate::inputstamp::{InputStamp, PendingInputStamps};
        use g2g_core::frame::{Frame, FrameTiming};
        use g2g_core::memory::SystemSlice;
        use g2g_core::{
            AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
            G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
            RawVideoFormat, Rate, Segment, VideoCodec,
        };

        use $backend::{Decoder, PixelLayout, PlanarImageComponent, Picture, Settings};

        /// The output shape one decoded picture fixes: pixel format, geometry, and
        /// how its samples map to colour.
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct OutputShape {
            format: RawVideoFormat,
            width: u32,
            height: u32,
            colorimetry: g2g_core::Colorimetry,
        }

        struct DecodedPicture {
            shape: OutputShape,
            pts_ns: u64,
            pixels: Vec<u8>,
        }

        type DecodedFrames = Vec<DecodedPicture>;

        // every format `pic_format` yields, I420 first so an unconstrained link fixes to it
        const OUTPUT_FORMATS: [RawVideoFormat; 9] = [
            RawVideoFormat::I420,
            RawVideoFormat::I420p10,
            RawVideoFormat::I420p12,
            RawVideoFormat::I422,
            RawVideoFormat::I422p10,
            RawVideoFormat::I422p12,
            RawVideoFormat::I444,
            RawVideoFormat::I444p10,
            RawVideoFormat::I444p12,
        ];

        fn output_alternatives(
            width: &Dim,
            height: &Dim,
            framerate: &Rate,
            colorimetry: g2g_core::Colorimetry,
        ) -> CapsSet {
            CapsSet::from_alternatives(
                OUTPUT_FORMATS
                    .iter()
                    .map(|format| Caps::RawVideo {
                        format: *format,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                        interlace: g2g_core::Interlace::Any,
                        colorimetry,
                    })
                    .collect(),
            )
        }

        #[doc = concat!("Decodes an AV1 stream into a fully-planar YUV format via `", stringify!($backend), "`.")]
        pub struct $ty {
            decoder: Option<Decoder>,
            framerate: Rate,
            /// Colorimetry of the negotiated input caps, which an upstream parser
            /// or demuxer may already have refined. Preferred over what the
            /// backend reports per picture, whose enum mapping cannot tell the
            /// two BT.601 primaries apart; it fills in whatever the caps leave
            /// unknown.
            input_colorimetry: g2g_core::Colorimetry,
            /// Last emitted output shape, so `CapsChanged` is sent only on change.
            out: Option<OutputShape>,
            sequence: u64,
            configured: bool,
            /// Timing of the newest input unit, whose fields other than pts the
            /// EOS drain's pictures take.
            last_timing: FrameTiming,
            input_stamps: PendingInputStamps,
            segment: Segment,
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
                    input_colorimetry: g2g_core::Colorimetry::UNKNOWN,
                    out: None,
                    sequence: 0,
                    configured: false,
                    last_timing: FrameTiming::default(),
                    input_stamps: PendingInputStamps::default(),
                    segment: Segment::new(),
                }
            }

            fn input_template() -> Caps {
                Caps::CompressedVideo {
                    codec: VideoCodec::Av1,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                }
            }

            fn output_caps(&self, shape: OutputShape) -> Caps {
                Caps::RawVideo {
                    format: shape.format,
                    width: Dim::Fixed(shape.width),
                    height: Dim::Fixed(shape.height),
                    framerate: self.framerate.clone(),
                    interlace: g2g_core::Interlace::Any,
                    colorimetry: shape.colorimetry,
                }
            }

            /// Feed one AV1 temporal unit and collect every picture now decodable,
            /// each as its output shape plus packed pixels. Drives the
            /// send/drain protocol so a frame-threading / reordering delay does not
            /// strand input (the `Try again` -> drain -> `send_pending_data` cycle).
            fn feed(
                decoder: &mut Decoder,
                unit: Vec<u8>,
                pts_ns: u64,
                input_colorimetry: g2g_core::Colorimetry,
            ) -> Result<DecodedFrames, G2gError> {
                let mut frames = Vec::new();
                // the backend echoes this timestamp on the picture the unit decodes to
                let mut send = decoder.send_data(unit, None, Some(pts_ns as i64), None);
                loop {
                    Self::drain_ready(decoder, &mut frames, input_colorimetry)?;
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
                input_colorimetry: g2g_core::Colorimetry,
            ) -> Result<(), G2gError> {
                loop {
                    match decoder.get_picture() {
                        Ok(pic) => {
                            let format = pic_format(&pic)?;
                            let shape = OutputShape {
                                format,
                                width: pic.width(),
                                height: pic.height(),
                                colorimetry: merge_colorimetry(input_colorimetry, &pic),
                            };
                            let pts_ns = pic.timestamp().ok_or(G2gError::CapsMismatch)? as u64;
                            frames.push(DecodedPicture {
                                shape,
                                pts_ns,
                                pixels: pack_planar(&pic, format)?,
                            });
                        }
                        Err(e) if e.is_again() => return Ok(()),
                        Err(_) => return Err(G2gError::CapsMismatch),
                    }
                }
            }

            /// Push `frames` as caps-checked output, each at its own pts with the
            /// other fields of `timing`, the shared tail of the per-unit decode
            /// and the EOS drain.
            async fn emit(
                &mut self,
                frames: DecodedFrames,
                timing: FrameTiming,
                out: &mut dyn OutputSink,
            ) -> Result<(), G2gError> {
                for picture in frames {
                    let input = self.input_stamps.take(picture.pts_ns);
                    if !self.segment.presents(picture.pts_ns) {
                        continue;
                    }
                    if self.out != Some(picture.shape) {
                        out.push(PipelinePacket::CapsChanged(self.output_caps(picture.shape)))
                            .await?;
                        self.out = Some(picture.shape);
                    }
                    let decoded = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            picture.pixels.into_boxed_slice(),
                        )),
                        timing: FrameTiming {
                            pts_ns: picture.pts_ns,
                            dts_ns: picture.pts_ns,
                            arrival_ns: input.arrival_ns,
                            ..timing
                        },
                        sequence: self.sequence,
                        meta: input.meta,
                    };
                    self.sequence += 1;
                    out.push(PipelinePacket::DataFrame(decoded)).await?;
                }
                Ok(())
            }
        }

        /// How the decoded samples map to colour: the negotiated input caps, with
        /// every field they leave unknown taken from the picture's own sequence
        /// header. The caps win a disagreement (the backend's colour enums fold
        /// the two BT.601 primaries together, so a parser reading the same header
        /// is the more exact of the two).
        fn merge_colorimetry(
            input: g2g_core::Colorimetry,
            pic: &Picture,
        ) -> g2g_core::Colorimetry {
            let coded = g2g_core::Colorimetry::from_cicp(
                pic.color_primaries() as u8,
                pic.transfer_characteristic() as u8,
                pic.matrix_coefficients() as u8,
                matches!(pic.color_range(), $backend::pixel::YUVRange::Full),
            );
            input.intersect(&coded).unwrap_or(input)
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
                    // Decode does not change how the samples map to colour, so
                    // the bitstream's tag rides through.
                    Caps::CompressedVideo { codec: VideoCodec::Av1, width, height, framerate, colorimetry } => {
                        output_alternatives(width, height, framerate, *colorimetry)
                    }
                    _ => CapsSet::from_alternatives(Vec::new()),
                }))
            }

            fn configure_pipeline(
                &mut self,
                absolute_caps: &Caps,
            ) -> Result<ConfigureOutcome, G2gError> {
                let Caps::CompressedVideo { codec: VideoCodec::Av1, framerate, colorimetry, .. } =
                    absolute_caps
                else {
                    return Err(G2gError::CapsMismatch);
                };
                // the first call carries the solve's fixated placeholder rate
                if self.configured && self.framerate != *framerate {
                    self.framerate = framerate.clone();
                    self.out = None;
                }
                self.input_colorimetry = *colorimetry;
                // a mid-stream reconfigure keeps the decoder and the pictures it still holds
                if self.decoder.is_none() {
                    let mut settings = Settings::new();
                    // aomdec and ffmpeg output only the highest spatial layer of each temporal unit
                    settings.set_all_layers(false);
                    self.decoder = Some(
                        Decoder::with_settings(&settings).map_err(|_| G2gError::CapsMismatch)?,
                    );
                }
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
                            self.input_stamps.insert(frame.timing.pts_ns, InputStamp::of(&frame));
                            let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
                            let unit = slice.to_vec();
                            let frames = Self::feed(
                                decoder,
                                unit,
                                frame.timing.pts_ns,
                                self.input_colorimetry,
                            )?;
                            self.last_timing = frame.timing;
                            self.emit(frames, frame.timing, out).await?;
                        }
                        // An upstream parser refines the colour description
                        // mid-stream, after configure saw the container's caps.
                        PipelinePacket::CapsChanged(Caps::CompressedVideo { colorimetry, .. }) => {
                            self.input_colorimetry = colorimetry;
                        }
                        PipelinePacket::CapsChanged(_) => {}
                        PipelinePacket::Segment(segment) => {
                            self.segment = segment;
                            out.push(PipelinePacket::Segment(segment)).await?;
                        }
                        PipelinePacket::Eos => {
                            // A reordering stream holds its tail pictures back
                            // (M1003): drain them before the sentinel, so the
                            // last frames of the stream are not lost.
                            if let Some(decoder) = self.decoder.as_mut() {
                                let mut frames = DecodedFrames::new();
                                Self::drain_ready(decoder, &mut frames, self.input_colorimetry)?;
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
                Vec::from([
                    PadTemplate::sink(CapsSet::one(av1)),
                    PadTemplate::source(output_alternatives(
                        &Dim::Any,
                        &Dim::Any,
                        &Rate::Any,
                        g2g_core::Colorimetry::UNKNOWN,
                    )),
                ])
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use crate::ivfdemux::IvfDemux;
            use crate::testutil::{first_caps, run, CollectSink};
            use g2g_core::ByteStreamEncoding;

            // av1-1-b8-22-svc-L2T2 cut to two temporal units, each a base and an enhancement layer
            const SPATIAL_LAYERS_FIXTURE: &[u8] =
                include_bytes!("../tests/fixtures/av1-1-b8-22-svc-L2T2_2tu.ivf");

            #[test]
            fn a_scalable_stream_decodes_to_its_highest_spatial_layer() {
                let demuxed = run(
                    &mut IvfDemux::new(),
                    &Caps::ByteStream { encoding: ByteStreamEncoding::Ivf },
                    &[SPATIAL_LAYERS_FIXTURE],
                );
                let coded = first_caps(&demuxed.packets).expect("the demuxer reports the stream");
                let Caps::CompressedVideo { width, height, .. } = coded.clone() else {
                    panic!("coded video caps, got {coded:?}");
                };
                let mut decoder = $ty::new();
                decoder.configure_pipeline(&coded).unwrap();
                let mut sink = CollectSink::default();
                let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
                let mut temporal_units = 0;
                for packet in demuxed.packets {
                    temporal_units += usize::from(matches!(packet, PipelinePacket::DataFrame(_)));
                    runtime.block_on(decoder.process(packet, &mut sink)).unwrap();
                }
                runtime.block_on(decoder.process(PipelinePacket::Eos, &mut sink)).unwrap();

                let output_sizes: Vec<(Dim, Dim)> = sink
                    .packets
                    .iter()
                    .filter_map(|packet| match packet {
                        PipelinePacket::CapsChanged(Caps::RawVideo { width, height, .. }) => {
                            Some((width.clone(), height.clone()))
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(output_sizes, alloc::vec![(width, height)]);
                let pictures = sink
                    .packets
                    .iter()
                    .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
                    .count();
                assert_eq!(pictures, temporal_units, "one picture per temporal unit");
            }
        }
    };
}

pub(crate) use av1_decoder;
