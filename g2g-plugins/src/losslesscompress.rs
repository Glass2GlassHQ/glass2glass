// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use brotli::enc::encode::{BrotliEncoderOperation, BrotliEncoderStateStruct};
use brotli::enc::{BrotliAlloc, BrotliEncoderMaxCompressedSize};
use brotli::{
    Allocator, BrotliDecompressStream, BrotliResult, BrotliState, SliceWrapper, SliceWrapperMut,
};
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
// zlib's largest window, the one every zlib and raw deflate decoder accepts
const WINDOW_BITS: u8 = 15;

fn max_level(compression: LosslessCompression) -> u32 {
    match compression {
        LosslessCompression::Zlib | LosslessCompression::Deflate => 9,
        LosslessCompression::Brotli => 11,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct HeapAllocator;

#[derive(Debug)]
struct HeapSlice<T>(Box<[T]>);

impl<T> Default for HeapSlice<T> {
    fn default() -> Self {
        HeapSlice(Box::default())
    }
}

impl<T> SliceWrapper<T> for HeapSlice<T> {
    fn slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> SliceWrapperMut<T> for HeapSlice<T> {
    fn slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T: Clone + Default> Allocator<T> for HeapAllocator {
    type AllocatedMemory = HeapSlice<T>;

    fn alloc_cell(&mut self, len: usize) -> HeapSlice<T> {
        HeapSlice(vec![T::default(); len].into_boxed_slice())
    }

    fn free_cell(&mut self, _data: HeapSlice<T>) {}
}

impl BrotliAlloc for HeapAllocator {}

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

fn deflate(bytes: &[u8], mut deflate: Deflate) -> Result<Vec<u8>, G2gError> {
    let mut out = vec![0; zlib_rs::compress_bound(bytes.len())];
    match deflate.compress(bytes, &mut out, DeflateFlush::Finish) {
        Ok(Status::StreamEnd) => {
            out.truncate(deflate.total_out() as usize);
            Ok(out)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

fn brotli_compress(bytes: &[u8], level: u32) -> Result<Vec<u8>, G2gError> {
    let mut encoder = BrotliEncoderStateStruct::new(HeapAllocator);
    encoder.params.quality = level as i32;
    let mut out = vec![0; BrotliEncoderMaxCompressedSize(bytes.len())];
    let (mut available_in, mut input_offset) = (bytes.len(), 0);
    let (mut available_out, mut output_offset) = (out.len(), 0);
    let accepted = encoder.compress_stream(
        BrotliEncoderOperation::BROTLI_OPERATION_FINISH,
        &mut available_in,
        bytes,
        &mut input_offset,
        &mut available_out,
        &mut out,
        &mut output_offset,
        &mut None,
        &mut |_, _, _, _| (),
    );
    if !accepted || !encoder.is_finished() {
        return Err(G2gError::CapsMismatch);
    }
    out.truncate(output_offset);
    Ok(out)
}

fn compress(
    bytes: &[u8],
    compression: LosslessCompression,
    level: u32,
) -> Result<Vec<u8>, G2gError> {
    match compression {
        LosslessCompression::Zlib => deflate(bytes, Deflate::new(level as i32, true, WINDOW_BITS)),
        LosslessCompression::Deflate => {
            deflate(bytes, Deflate::new(level as i32, false, WINDOW_BITS))
        }
        LosslessCompression::Brotli => brotli_compress(bytes, level),
    }
}

static ZLIB_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "level",
    PropKind::Uint,
    "compression level, 0 stores the data and 9 compresses best",
)];

static BROTLI_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "level",
    PropKind::Uint,
    "compression level, 0 is fastest and 11 compresses best",
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
        let max = max_level(self.compression);
        assert!(level <= max, "compression level {level} is above {max}");
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
            LosslessCompression::Brotli => (
                "Brotli compressor",
                "Compresses each buffer into one brotli stream",
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
        match self.compression {
            LosslessCompression::Zlib | LosslessCompression::Deflate => ZLIB_PROPS,
            LosslessCompression::Brotli => BROTLI_PROPS,
        }
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "level" => {
                let level = value.as_uint().ok_or(PropError::Type)?;
                self.level = u32::try_from(level)
                    .ok()
                    .filter(|level| *level <= max_level(self.compression))
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

enum StreamDecoder {
    Zlib(Inflate),
    Brotli(Box<BrotliState<HeapAllocator, HeapAllocator, HeapAllocator>>),
}

struct DecodeStep {
    consumed: usize,
    produced: usize,
    stream_end: bool,
}

fn inflate_step(inflate: &mut Inflate, input: &[u8], output: &mut [u8]) -> Option<DecodeStep> {
    let (in_before, out_before) = (inflate.total_in(), inflate.total_out());
    let status = inflate
        .decompress(input, output, InflateFlush::NoFlush)
        .ok()?;
    Some(DecodeStep {
        consumed: (inflate.total_in() - in_before) as usize,
        produced: (inflate.total_out() - out_before) as usize,
        stream_end: status == Status::StreamEnd,
    })
}

/// Inflate one complete stream with a caller-supplied limit on its decoded size.
pub(crate) fn inflate_bounded(
    mut input: &[u8],
    expect_zlib_header: bool,
    limit: usize,
) -> Option<Vec<u8>> {
    let mut inflate = Inflate::new(expect_zlib_header, WINDOW_BITS);
    // Leave one extra byte to distinguish exact-limit output from overflow.
    let capacity = limit.checked_add(1)?;
    let mut output = vec![0; input.len().saturating_mul(2).clamp(1, capacity)];
    let mut written = 0;
    loop {
        let step = inflate_step(&mut inflate, input, &mut output[written..])?;
        written += step.produced;
        if written > limit {
            return None;
        }
        if step.stream_end {
            output.truncate(written);
            return Some(output);
        }
        if step.consumed == 0 && step.produced == 0 {
            return None;
        }
        input = &input[step.consumed..];
        if written == output.len() {
            output.resize(output.len().saturating_mul(2).min(capacity), 0);
        }
    }
}

impl StreamDecoder {
    fn new(compression: LosslessCompression) -> Self {
        match compression {
            LosslessCompression::Zlib => StreamDecoder::Zlib(Inflate::new(true, WINDOW_BITS)),
            LosslessCompression::Deflate => StreamDecoder::Zlib(Inflate::new(false, WINDOW_BITS)),
            LosslessCompression::Brotli => {
                let mut state = BrotliState::new(HeapAllocator, HeapAllocator, HeapAllocator);
                // large windows reach 1 GiB and are outside RFC 7932
                state.large_window = false;
                StreamDecoder::Brotli(Box::new(state))
            }
        }
    }

    fn decode(&mut self, input: &[u8], output: &mut [u8]) -> Result<DecodeStep, G2gError> {
        match self {
            StreamDecoder::Zlib(inflate) => {
                inflate_step(inflate, input, output).ok_or(G2gError::CapsMismatch)
            }
            StreamDecoder::Brotli(state) => {
                let (mut available_in, mut consumed) = (input.len(), 0);
                let (mut available_out, mut produced, mut total_out) = (output.len(), 0, 0);
                let result = BrotliDecompressStream(
                    &mut available_in,
                    &mut consumed,
                    input,
                    &mut available_out,
                    &mut produced,
                    output,
                    &mut total_out,
                    state,
                );
                let stream_end = match result {
                    BrotliResult::ResultFailure => return Err(G2gError::CapsMismatch),
                    BrotliResult::ResultSuccess => true,
                    BrotliResult::NeedsMoreInput | BrotliResult::NeedsMoreOutput => false,
                };
                Ok(DecodeStep {
                    consumed,
                    produced,
                    stream_end,
                })
            }
        }
    }
}

struct PartialStream {
    decoder: StreamDecoder,
    output: Vec<u8>,
    written: usize,
    timing: FrameTiming,
}

impl core::fmt::Debug for PartialStream {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PartialStream")
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

fn decompress_chunk(
    mut pending: Option<PartialStream>,
    mut input: &[u8],
    compression: LosslessCompression,
    size: DecompressedSize,
    timing: FrameTiming,
) -> Result<(Option<PartialStream>, Vec<Decompressed>), G2gError> {
    let mut finished = Vec::new();
    // one byte past the bound shows that a stream is too large
    let capacity = size.limit() + 1;
    let mut output_full = false;
    // streams can span input buffers, and one buffer can end one stream and start the next
    while !input.is_empty() || output_full {
        let stream = pending.get_or_insert_with(|| PartialStream {
            decoder: StreamDecoder::new(compression),
            output: Vec::new(),
            written: 0,
            timing,
        });
        if stream.written == stream.output.len() {
            let grown = stream.output.len().max(FIRST_OUTPUT_BLOCK / 2) * 2;
            stream.output.resize(grown.min(capacity), 0);
        }
        let DecodeStep {
            consumed,
            produced,
            stream_end,
        } = stream
            .decoder
            .decode(input, &mut stream.output[stream.written..])?;
        input = &input[consumed..];
        stream.written += produced;
        if stream.written > size.limit() {
            return Err(G2gError::CapsMismatch);
        }
        if stream_end {
            let mut done = pending.take().expect("a stream is being decompressed");
            if matches!(size, DecompressedSize::Exact(expected) if expected != done.written) {
                return Err(G2gError::CapsMismatch);
            }
            done.output.truncate(done.written);
            finished.push(Decompressed {
                bytes: done.output,
                timing: done.timing,
            });
            output_full = false;
            continue;
        }
        if consumed == 0 && produced == 0 && !input.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        // brotli takes all of its input before it has written all of its output
        output_full = stream.written == stream.output.len();
    }
    Ok((pending, finished))
}

#[derive(Debug)]
pub struct LosslessDecompress {
    compression: LosslessCompression,
    original: Option<Caps>,
    pending: Option<PartialStream>,
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
            LosslessCompression::Brotli => "Brotli decompressor",
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
                            decompress_chunk(pending, &owned, compression, size, timing)
                        })
                        .await?
                    };
                    #[cfg(not(feature = "offload"))]
                    let (pending, finished) =
                        decompress_chunk(pending, &bytes, compression, size, frame.timing)?;
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

pub fn brotli_decompressed_caps(input: &Caps) -> CapsSet {
    decompressed_caps(LosslessCompression::Brotli, input)
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
