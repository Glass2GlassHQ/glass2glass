// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, BayerFormat, BayerPattern, Caps, CapsConstraint, CapsSet, CapsTransform,
    ConfigureOutcome, Dim, ElementMetadata, FieldTransform, G2gError, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, Rate, RawVideoFormat, RawVideoShape,
};

use crate::pixel::frame_byte_size;

const DEMOSAIC_FORMATS: [RawVideoFormat; 5] = [
    RawVideoFormat::Rgba8,
    RawVideoFormat::Bgra8,
    RawVideoFormat::Bgrx8,
    RawVideoFormat::Argb8,
    RawVideoFormat::Rgba64Le,
];

const MOSAIC_INPUT_FORMAT: RawVideoFormat = RawVideoFormat::Argb8;

fn bayer_formats() -> Vec<RawVideoFormat> {
    g2g_core::caps::ALL_BAYER_FORMATS
        .map(RawVideoFormat::Bayer)
        .to_vec()
}

fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    }
}

fn any_geometry_set(formats: &[RawVideoFormat]) -> CapsSet {
    CapsSet::from_alternatives(formats.iter().copied().map(any_geometry).collect())
}

fn first_supported(upstream_caps: &Caps, formats: &[RawVideoFormat]) -> Result<Caps, G2gError> {
    formats
        .iter()
        .find_map(|&format| upstream_caps.intersect(&any_geometry(format)).ok())
        .ok_or(G2gError::CapsMismatch)
}

fn format_changing_constraint(
    accept: Vec<RawVideoFormat>,
    produce: Vec<RawVideoFormat>,
) -> CapsConstraint<'static> {
    let shapes = produce
        .iter()
        .map(|&format| RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(format)))
        .collect();
    CapsConstraint::DerivedFields(CapsTransform::RawVideo {
        accept,
        produce,
        shapes,
    })
}

#[derive(Clone, Debug)]
struct Stream {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    pixel_aspect: g2g_core::PixelAspect,
}

impl Stream {
    fn from_caps(caps: &Caps) -> Result<Self, G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            pixel_aspect,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        Ok(Self {
            format: *format,
            width: *width,
            height: *height,
            framerate: framerate.clone(),
            pixel_aspect: *pixel_aspect,
        })
    }

    fn with_format(&self, format: RawVideoFormat) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: self.pixel_aspect,
        }
    }
}

trait BayerConversion {
    const NAME: &'static str;
    fn is_input(format: RawVideoFormat) -> bool;
    fn input_stream(caps: &Caps) -> Result<Stream, G2gError>;
    fn convert(bytes: &[u8], input: &Stream, output: RawVideoFormat) -> Result<Vec<u8>, G2gError>;
}

#[derive(Debug, Default)]
struct ConversionState {
    input: Option<Stream>,
    output_format: Option<RawVideoFormat>,
    last_caps: Option<Caps>,
    emitted: u64,
}

async fn drive<C: BayerConversion + 'static>(
    state: &mut ConversionState,
    packet: PipelinePacket,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    match packet {
        PipelinePacket::DataFrame(frame) => {
            let input = state.input.as_ref().ok_or(G2gError::NotConfigured)?;
            let output_format = state.output_format.ok_or(G2gError::NotConfigured)?;
            let bytes = frame.domain.require_system_bytes(C::NAME)?;
            if bytes.len() < frame_byte_size(input.format, input.width, input.height) {
                return Err(G2gError::CapsMismatch);
            }
            #[cfg(feature = "offload")]
            let converted = {
                let (owned, owned_input) = (bytes.to_vec(), input.clone());
                g2g_core::offload::run_blocking(move || {
                    C::convert(&owned, &owned_input, output_format)
                })
                .await?
            };
            #[cfg(not(feature = "offload"))]
            let converted = C::convert(&bytes, input, output_format)?;
            let caps = input.with_format(output_format);
            if state.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                state.last_caps = Some(caps);
            }
            let out_frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(converted.into_boxed_slice())),
                timing: frame.timing,
                sequence: state.emitted,
                meta: Default::default(),
            };
            state.emitted += 1;
            out.push(PipelinePacket::DataFrame(out_frame)).await?;
            Ok(())
        }
        // the runner also sends an element its own solved output caps, told apart from an input by format
        PipelinePacket::CapsChanged(caps) => {
            let Caps::RawVideo { format, .. } = &caps else {
                return Err(G2gError::CapsMismatch);
            };
            if C::is_input(*format) {
                state.input = Some(C::input_stream(&caps)?);
                return Ok(());
            }
            state.output_format = Some(*format);
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            state.last_caps = Some(caps);
            Ok(())
        }
        PipelinePacket::Flush => {
            state.last_caps = None;
            out.push(PipelinePacket::Flush).await?;
            Ok(())
        }
        other => {
            out.push(other).await?;
            Ok(())
        }
    }
}

#[derive(Debug, Default)]
pub struct Bayer2Rgb {
    state: ConversionState,
}

impl Bayer2Rgb {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BayerConversion for Bayer2Rgb {
    const NAME: &'static str = "bayer2rgb";

    fn is_input(format: RawVideoFormat) -> bool {
        matches!(format, RawVideoFormat::Bayer(_))
    }

    fn input_stream(caps: &Caps) -> Result<Stream, G2gError> {
        let stream = Stream::from_caps(caps)?;
        // a single row or column holds only two of the three colours
        let demosaicable = Self::is_input(stream.format) && stream.width >= 2 && stream.height >= 2;
        match demosaicable {
            true => Ok(stream),
            false => Err(G2gError::CapsMismatch),
        }
    }

    fn convert(bytes: &[u8], input: &Stream, output: RawVideoFormat) -> Result<Vec<u8>, G2gError> {
        let RawVideoFormat::Bayer(bayer) = input.format else {
            return Err(G2gError::CapsMismatch);
        };
        demosaic(
            bytes,
            bayer,
            input.width as usize,
            input.height as usize,
            output,
        )
    }
}

impl AsyncElement for Bayer2Rgb {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Bayer to RGB converter",
            "Filter/Converter/Video",
            "Demosaics video/x-bayer into packed RGB",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        first_supported(upstream_caps, &bayer_formats())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        format_changing_constraint(bayer_formats(), DEMOSAIC_FORMATS.to_vec())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.state.input = Some(Self::input_stream(absolute_caps)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo { format, .. } = output_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !DEMOSAIC_FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        self.state.output_format = Some(*format);
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(drive::<Self>(&mut self.state, packet, out))
    }
}

impl PadTemplates for Bayer2Rgb {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(any_geometry_set(&bayer_formats())),
            PadTemplate::source(any_geometry_set(&DEMOSAIC_FORMATS)),
        ])
    }
}

fn read_samples(src: &[u8], bayer: BayerFormat, count: usize) -> Vec<u32> {
    if bayer.sample.bit_depth() == 8 {
        return src[..count]
            .iter()
            .map(|&sample| u32::from(sample))
            .collect();
    }
    src[..count * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| {
            u32::from(match bayer.sample.is_big_endian() {
                true => u16::from_be_bytes(pair),
                false => u16::from_le_bytes(pair),
            })
        })
        .collect()
}

fn average(a: u32, b: u32) -> u32 {
    (a + b + 1) >> 1
}

struct RowColours {
    even_column_colour: Vec<u32>,
    odd_column_colour: Vec<u32>,
}

fn split_row(samples: &[u32]) -> RowColours {
    let width = samples.len();
    let last = width - 1;
    let mirrored = |index: isize| match index {
        before if before < 0 => samples[before.unsigned_abs()],
        index => samples[index.unsigned_abs().min(2 * last - index.unsigned_abs())],
    };
    let mut even = Vec::with_capacity(width);
    let mut odd = Vec::with_capacity(width);
    for (x, &sample) in samples.iter().enumerate() {
        let left = x as isize - 1;
        // GStreamer copies the left neighbour instead of averaging in the second to last column
        let interpolated = match x + 2 == width {
            true => mirrored(left),
            false => average(mirrored(left), mirrored(left + 2)),
        };
        let (even_colour, odd_colour) = match x % 2 {
            0 => (sample, interpolated),
            _ => (interpolated, sample),
        };
        even.push(even_colour);
        odd.push(odd_colour);
    }
    RowColours {
        even_column_colour: even,
        odd_column_colour: odd,
    }
}

fn to_8_bits(value: u32, depth: u8) -> u8 {
    (value >> (depth - 8)).min(255) as u8
}

// GStreamer widens 8-bit samples by 257 and scales wider ones by 65535 over two to the depth
fn to_16_bits(value: u32, depth: u8) -> u16 {
    match depth {
        8 => (value * 257) as u16,
        _ => ((value * 65535) >> depth).min(65535) as u16,
    }
}

fn demosaic(
    src: &[u8],
    bayer: BayerFormat,
    width: usize,
    height: usize,
    output: RawVideoFormat,
) -> Result<Vec<u8>, G2gError> {
    let write_pixel: fn(&mut Vec<u8>, u8, [u32; 3]) = match output {
        RawVideoFormat::Rgba8 => |out, depth, [r, g, b]| {
            out.extend([
                to_8_bits(r, depth),
                to_8_bits(g, depth),
                to_8_bits(b, depth),
                255,
            ])
        },
        RawVideoFormat::Argb8 => |out, depth, [r, g, b]| {
            out.extend([
                255,
                to_8_bits(r, depth),
                to_8_bits(g, depth),
                to_8_bits(b, depth),
            ])
        },
        RawVideoFormat::Bgra8 | RawVideoFormat::Bgrx8 => |out, depth, [r, g, b]| {
            out.extend([
                to_8_bits(b, depth),
                to_8_bits(g, depth),
                to_8_bits(r, depth),
                255,
            ])
        },
        RawVideoFormat::Rgba64Le => |out, depth, rgb| {
            for channel in rgb {
                out.extend(to_16_bits(channel, depth).to_le_bytes());
            }
            // GStreamer scales the opaque alpha like a sample, which leaves 65534 at 16 bits
            let opaque = if depth == 8 { 255 } else { 65535 };
            out.extend(to_16_bits(opaque, depth).to_le_bytes());
        },
        _ => return Err(G2gError::CapsMismatch),
    };
    let depth = bayer.sample.bit_depth();
    let row_stride = RawVideoFormat::Bayer(bayer)
        .plane_stride(0, width as u32)
        .ok_or(G2gError::CapsMismatch)? as usize;
    let split = |y: usize| split_row(&read_samples(&src[y * row_stride..], bayer, width));
    let first_row_holds_blue_or_red =
        matches!(bayer.pattern, BayerPattern::Bggr | BayerPattern::Rggb);
    let red_where_blue_is = matches!(bayer.pattern, BayerPattern::Rggb | BayerPattern::Gbrg);
    let mut split_rows: [Option<(usize, RowColours)>; 4] = Default::default();
    let mut out = Vec::with_capacity(width * height * 8);
    for y in 0..height {
        let above = if y == 0 { 1 } else { y - 1 };
        // GStreamer takes the row four up as the row below the last one
        let below = match (y + 1 < height, height >= 4) {
            (true, _) => y + 1,
            (false, true) => height - 4,
            (false, false) => height - 2,
        };
        for row in [above, y, below] {
            let slot = &mut split_rows[row % split_rows.len()];
            if !matches!(slot, Some((cached, _)) if *cached == row) {
                *slot = Some((row, split(row)));
            }
        }
        let colours = |row: usize| {
            &split_rows[row % split_rows.len()]
                .as_ref()
                .expect("rows above, at and below y are split")
                .1
        };
        let (above, current, below) = (colours(above), colours(y), colours(below));
        let blue_row = (y % 2 == 0) == first_row_holds_blue_or_red;
        for x in 0..width {
            let (blue, green, red) = match blue_row {
                true => {
                    let green_around = average(
                        average(above.even_column_colour[x], below.even_column_colour[x]),
                        current.odd_column_colour[x],
                    );
                    let green = if x % 2 == 0 {
                        green_around
                    } else {
                        current.odd_column_colour[x]
                    };
                    (
                        current.even_column_colour[x],
                        green,
                        average(above.odd_column_colour[x], below.odd_column_colour[x]),
                    )
                }
                false => {
                    let green_around = average(
                        average(above.odd_column_colour[x], below.odd_column_colour[x]),
                        current.even_column_colour[x],
                    );
                    let green = if x % 2 == 0 {
                        current.even_column_colour[x]
                    } else {
                        green_around
                    };
                    (
                        average(above.even_column_colour[x], below.even_column_colour[x]),
                        green,
                        current.odd_column_colour[x],
                    )
                }
            };
            let (blue, red) = match red_where_blue_is {
                true => (red, blue),
                false => (blue, red),
            };
            write_pixel(&mut out, depth, [red, green, blue]);
        }
    }
    Ok(out)
}

#[derive(Debug, Default)]
pub struct Rgb2Bayer {
    state: ConversionState,
}

impl Rgb2Bayer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BayerConversion for Rgb2Bayer {
    const NAME: &'static str = "rgb2bayer";

    fn is_input(format: RawVideoFormat) -> bool {
        format == MOSAIC_INPUT_FORMAT
    }

    fn input_stream(caps: &Caps) -> Result<Stream, G2gError> {
        let stream = Stream::from_caps(caps)?;
        match Self::is_input(stream.format) && stream.width > 0 && stream.height > 0 {
            true => Ok(stream),
            false => Err(G2gError::CapsMismatch),
        }
    }

    fn convert(bytes: &[u8], input: &Stream, output: RawVideoFormat) -> Result<Vec<u8>, G2gError> {
        let RawVideoFormat::Bayer(bayer) = output else {
            return Err(G2gError::CapsMismatch);
        };
        mosaic(bytes, bayer, input.width as usize, input.height as usize)
    }
}

impl AsyncElement for Rgb2Bayer {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RGB to Bayer converter",
            "Filter/Converter/Video",
            "Converts ARGB video/x-raw to video/x-bayer",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        first_supported(upstream_caps, &[MOSAIC_INPUT_FORMAT])
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        format_changing_constraint(Vec::from([MOSAIC_INPUT_FORMAT]), bayer_formats())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.state.input = Some(Self::input_stream(absolute_caps)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format: format @ RawVideoFormat::Bayer(_),
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.state.output_format = Some(*format);
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(drive::<Self>(&mut self.state, packet, out))
    }
}

impl PadTemplates for Rgb2Bayer {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(any_geometry_set(&[MOSAIC_INPUT_FORMAT])),
            PadTemplate::source(any_geometry_set(&bayer_formats())),
        ])
    }
}

fn blue_site_in_block(pattern: BayerPattern) -> usize {
    match pattern {
        BayerPattern::Bggr => 0,
        BayerPattern::Gbrg => 1,
        BayerPattern::Grbg => 2,
        BayerPattern::Rggb => 3,
    }
}

fn mosaic(
    argb: &[u8],
    bayer: BayerFormat,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, G2gError> {
    const RED: usize = 1;
    const GREEN: usize = 2;
    const BLUE: usize = 3;
    let blue = blue_site_in_block(bayer.pattern);
    let depth = u32::from(bayer.sample.bit_depth());
    let row_stride = RawVideoFormat::Bayer(bayer)
        .plane_stride(0, width as u32)
        .ok_or(G2gError::CapsMismatch)? as usize;
    let mut out = Vec::with_capacity(row_stride * height);
    for (y, row) in argb.chunks_exact(width * 4).take(height).enumerate() {
        for (x, pixel) in row.as_chunks::<4>().0.iter().enumerate() {
            let site = (y % 2) * 2 + x % 2;
            let channel = match site {
                _ if site == blue => BLUE,
                _ if site == blue ^ 3 => RED,
                _ => GREEN,
            };
            let value = pixel[channel];
            if depth == 8 {
                out.push(value);
                continue;
            }
            let wide =
                ((u32::from(value) << (depth - 8)) | (u32::from(value) >> (16 - depth))) as u16;
            out.extend(match bayer.sample.is_big_endian() {
                true => wide.to_be_bytes(),
                false => wide.to_le_bytes(),
            });
        }
        out.resize((y + 1) * row_stride, 0);
    }
    Ok(out)
}
