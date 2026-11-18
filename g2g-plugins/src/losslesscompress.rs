// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata,
    FrameTiming, G2gError, LosslessCompression, OutputSink, PadCaps, PadDirection, PadTemplate,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};
use zlib_rs::{Deflate, DeflateFlush, Inflate, InflateFlush, Status};

use crate::pixel::frame_byte_size;

const DEFAULT_LEVEL: u32 = 6;
const MAX_LEVEL: u32 = 9;
// zlib's largest window, the one every zlib and raw deflate decoder accepts
const WINDOW_BITS: u8 = 15;

fn zlib_header(compression: LosslessCompression) -> bool {
    match compression {
        LosslessCompression::Zlib => true,
        LosslessCompression::Deflate => false,
    }
}

fn compressed_caps(compression: LosslessCompression, original: Option<&Caps>) -> Caps {
    Caps::LosslesslyCompressed {
        compression,
        original: original.map(|original| Box::new(original.clone())),
    }
}

fn system_frame(bytes: Vec<u8>, timing: FrameTiming, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        sequence,
        meta: Default::default(),
    }
}

fn compress(
    bytes: &[u8],
    compression: LosslessCompression,
    level: u32,
) -> Result<Vec<u8>, G2gError> {
    let mut deflate = Deflate::new(level as i32, zlib_header(compression), WINDOW_BITS);
    let mut out = vec![0; zlib_rs::compress_bound(bytes.len())];
    match deflate.compress(bytes, &mut out, DeflateFlush::Finish) {
        Ok(Status::StreamEnd) => {
            out.truncate(deflate.total_out() as usize);
            Ok(out)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

static COMPRESS_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "level",
    PropKind::Uint,
    "compression level, 0 stores the data and 9 compresses best",
)];

#[derive(Debug)]
pub struct LosslessCompress {
    compression: LosslessCompression,
    level: u32,
    input: Option<Caps>,
    last_caps: Option<Caps>,
}

impl LosslessCompress {
    pub fn new(compression: LosslessCompression) -> Self {
        Self {
            compression,
            level: DEFAULT_LEVEL,
            input: None,
            last_caps: None,
        }
    }

    pub fn with_level(mut self, level: u32) -> Self {
        assert!(
            level <= MAX_LEVEL,
            "compression level {level} is above {MAX_LEVEL}"
        );
        self.level = level;
        self
    }
}

impl AsyncElement for LosslessCompress {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        let (name, description) = match self.compression {
            LosslessCompression::Zlib => (
                "Zlib compressor",
                "Compresses each buffer into one zlib stream",
            ),
            LosslessCompression::Deflate => (
                "Deflate compressor",
                "Compresses each buffer into one raw deflate stream",
            ),
        };
        ElementMetadata::new(name, "Encoder/Generic", description, crate::AARON_BOXER)
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let compression = self.compression;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            CapsSet::one(compressed_caps(compression, Some(input)))
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.input = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        COMPRESS_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "level" => {
                let level = value.as_uint().ok_or(PropError::Type)?;
                self.level = u32::try_from(level)
                    .ok()
                    .filter(|level| *level <= MAX_LEVEL)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "level" => Some(PropValue::Uint(u64::from(self.level))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let input = self.input.as_ref().ok_or(G2gError::NotConfigured)?;
                    let caps = compressed_caps(self.compression, Some(input));
                    let bytes = frame.domain.require_system_bytes("losslesscompress")?;
                    let (compression, level) = (self.compression, self.level);
                    #[cfg(feature = "offload")]
                    let compressed = {
                        let owned = bytes.to_vec();
                        g2g_core::offload::run_blocking(move || {
                            compress(&owned, compression, level)
                        })
                        .await?
                    };
                    #[cfg(not(feature = "offload"))]
                    let compressed = compress(&bytes, compression, level)?;
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    let mut out_frame = system_frame(compressed, frame.timing, frame.sequence);
                    out_frame.meta = frame.meta;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // the runner also sends an element its own solved output caps
                PipelinePacket::CapsChanged(caps) => {
                    let own_output = self
                        .input
                        .as_ref()
                        .map(|input| compressed_caps(self.compression, Some(input)));
                    if own_output.as_ref() == Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    } else {
                        self.input = Some(caps);
                    }
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

pub fn compress_pad_templates(compression: LosslessCompression) -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink_any(),
        PadTemplate::source(CapsSet::one(compressed_caps(compression, None))),
    ])
}

// crafted streams of unknown size inflate to at most this
const UNSIZED_STREAM_LIMIT: usize = 256 << 20;
const FIRST_OUTPUT_BLOCK: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecompressedSize {
    Exact(usize),
    AtMost(usize),
}

impl DecompressedSize {
    fn of(original: &Caps) -> Self {
        match original {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            } => DecompressedSize::Exact(frame_byte_size(*format, *width, *height)),
            _ => DecompressedSize::AtMost(UNSIZED_STREAM_LIMIT),
        }
    }

    fn limit(self) -> usize {
        match self {
            DecompressedSize::Exact(size) | DecompressedSize::AtMost(size) => size,
        }
    }
}

struct Inflating {
    inflate: Inflate,
    output: Vec<u8>,
    written: usize,
    timing: FrameTiming,
}

impl core::fmt::Debug for Inflating {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Inflating")
            .field("written", &self.written)
            .field("timing", &self.timing)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct Decompressed {
    bytes: Vec<u8>,
    timing: FrameTiming,
}

fn inflate_chunk(
    mut pending: Option<Inflating>,
    mut input: &[u8],
    compression: LosslessCompression,
    size: DecompressedSize,
    timing: FrameTiming,
) -> Result<(Option<Inflating>, Vec<Decompressed>), G2gError> {
    let mut finished = Vec::new();
    // streams can span input buffers, and one buffer can end one stream and start the next
    while !input.is_empty() {
        let stream = pending.get_or_insert_with(|| Inflating {
            inflate: Inflate::new(zlib_header(compression), WINDOW_BITS),
            output: Vec::new(),
            written: 0,
            timing,
        });
        if stream.written == stream.output.len() {
            let limit = size.limit();
            if stream.written >= limit {
                return Err(G2gError::CapsMismatch);
            }
            let grown = stream.output.len().max(FIRST_OUTPUT_BLOCK / 2) * 2;
            stream.output.resize(grown.min(limit), 0);
        }
        let (in_before, out_before) = (stream.inflate.total_in(), stream.inflate.total_out());
        let status = stream
            .inflate
            .decompress(
                input,
                &mut stream.output[stream.written..],
                InflateFlush::NoFlush,
            )
            .map_err(|_| G2gError::CapsMismatch)?;
        let consumed = (stream.inflate.total_in() - in_before) as usize;
        let produced = (stream.inflate.total_out() - out_before) as usize;
        input = &input[consumed..];
        stream.written += produced;
        if status == Status::StreamEnd {
            let mut done = pending.take().expect("a stream is being inflated");
            if matches!(size, DecompressedSize::Exact(expected) if expected != done.written) {
                return Err(G2gError::CapsMismatch);
            }
            done.output.truncate(done.written);
            finished.push(Decompressed {
                bytes: done.output,
                timing: done.timing,
            });
            continue;
        }
        let output_full = stream.written == stream.output.len();
        if consumed == 0 && produced == 0 && !output_full {
            return Err(G2gError::CapsMismatch);
        }
    }
    Ok((pending, finished))
}

#[derive(Debug)]
pub struct LosslessDecompress {
    compression: LosslessCompression,
    original: Option<Caps>,
    pending: Option<Inflating>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl LosslessDecompress {
    pub fn new(compression: LosslessCompression) -> Self {
        Self {
            compression,
            original: None,
            pending: None,
            last_caps: None,
            emitted: 0,
        }
    }

    fn original_of(&self, caps: &Caps) -> Result<Caps, G2gError> {
        match caps {
            Caps::LosslesslyCompressed {
                compression,
                original: Some(original),
            } if *compression == self.compression => Ok((**original).clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }
}

impl AsyncElement for LosslessDecompress {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        let name = match self.compression {
            LosslessCompression::Zlib => "Zlib decompressor",
            LosslessCompression::Deflate => "Deflate decompressor",
        };
        ElementMetadata::new(
            name,
            "Decoder/Generic",
            "Restores the buffers and caps a lossless compressor wrapped",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&compressed_caps(self.compression, None))
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let compression = self.compression;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            decompressed_caps(compression, input)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.original = Some(self.original_of(absolute_caps)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let original = self.original.clone().ok_or(G2gError::NotConfigured)?;
                    let size = DecompressedSize::of(&original);
                    let bytes = frame.domain.require_system_bytes("losslessdecompress")?;
                    let (pending, compression) = (self.pending.take(), self.compression);
                    #[cfg(feature = "offload")]
                    let (pending, finished) = {
                        let owned = bytes.to_vec();
                        let timing = frame.timing;
                        g2g_core::offload::run_blocking(move || {
                            inflate_chunk(pending, &owned, compression, size, timing)
                        })
                        .await?
                    };
                    #[cfg(not(feature = "offload"))]
                    let (pending, finished) =
                        inflate_chunk(pending, &bytes, compression, size, frame.timing)?;
                    self.pending = pending;
                    if !finished.is_empty() && self.last_caps.as_ref() != Some(&original) {
                        out.push(PipelinePacket::CapsChanged(original.clone()))
                            .await?;
                        self.last_caps = Some(original);
                    }
                    for decompressed in finished {
                        let out_frame =
                            system_frame(decompressed.bytes, decompressed.timing, self.emitted);
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(out_frame)).await?;
                    }
                }
                // the runner also sends an element its own solved output caps
                PipelinePacket::CapsChanged(caps @ Caps::LosslesslyCompressed { .. }) => {
                    self.original = Some(self.original_of(&caps)?);
                }
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.pending = None;
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos if self.pending.is_some() => {
                    return Err(G2gError::CapsMismatch);
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

fn decompressed_caps(compression: LosslessCompression, input: &Caps) -> CapsSet {
    match input {
        Caps::LosslesslyCompressed {
            compression: incoming,
            original: Some(original),
        } if *incoming == compression => CapsSet::one((**original).clone()),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

pub fn zlib_decompressed_caps(input: &Caps) -> CapsSet {
    decompressed_caps(LosslessCompression::Zlib, input)
}

pub fn deflate_decompressed_caps(input: &Caps) -> CapsSet {
    decompressed_caps(LosslessCompression::Deflate, input)
}

pub fn decompress_pad_templates(compression: LosslessCompression) -> Vec<PadTemplate> {
    Vec::from([
        PadTemplate::sink(CapsSet::one(compressed_caps(compression, None))),
        PadTemplate {
            direction: PadDirection::Source,
            caps: PadCaps::Any,
        },
    ])
}
