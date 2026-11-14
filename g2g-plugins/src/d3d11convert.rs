// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D, ID3D11VideoContext1,
    D3D11_BIND_FLAG, D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_MAP_READ, D3D11_TEXTURE2D_DESC, D3D11_USAGE,
    D3D11_USAGE_DEFAULT, D3D11_USAGE_STAGING,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_COLOR_SPACE_TYPE,
    DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601, DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709,
    DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601, DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC,
};

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, CapsTransform, ColorRange,
    Colorimetry, ConfigureOutcome, Dim, ElementMetadata, FieldTransform, G2gError, HardwareError,
    Interlace, MatrixCoefficients, MemoryDomain, OutputSink, OwnedD3D11Texture, PadTemplate,
    PadTemplates, PipelinePacket, PixelAspect, Rate, RawVideoFormat, RawVideoShape,
};

use crate::d3d11texture::{texture_frame, texture_objects, texture_proposal, TextureOwner};
use crate::d3d11video::VideoProcessor;
use crate::mfcommon::mf_err;
use crate::texturepool::take_from_pool;

const OUTPUT_DOMAINS: DomainSet =
    DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::D3D11Texture);

#[derive(Debug, Clone)]
struct InputStream {
    width: u32,
    height: u32,
    framerate: Rate,
    colorimetry: Colorimetry,
    pixel_aspect: PixelAspect,
}

#[derive(Debug)]
struct ConvertGpu {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    processor: VideoProcessor,
    pool: Vec<Arc<TextureOwner>>,
    // the converted frame and its CPU-readable copy, reused when frames leave in system memory
    readback: Option<(ID3D11Texture2D, ID3D11Texture2D)>,
}

#[derive(Debug)]
pub struct D3D11Convert {
    configured: bool,
    out_domain: MemoryDomainKind,
    input: Option<InputStream>,
    output_colorimetry: Colorimetry,
    last_caps: Option<Caps>,
    gpu: Option<ConvertGpu>,
}

impl Default for D3D11Convert {
    fn default() -> Self {
        Self::new()
    }
}

fn output_color_space(colorimetry: Colorimetry) -> Result<DXGI_COLOR_SPACE_TYPE, G2gError> {
    let full = colorimetry.range == ColorRange::Full;
    match colorimetry.matrix {
        MatrixCoefficients::Bt709 if full => Ok(DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709),
        MatrixCoefficients::Bt709 => Ok(DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709),
        // videoconvert writes BT.601 when the stream names no matrix
        MatrixCoefficients::Bt601 | MatrixCoefficients::Unknown if full => {
            Ok(DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601)
        }
        MatrixCoefficients::Bt601 | MatrixCoefficients::Unknown => {
            Ok(DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601)
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

fn create_texture(
    device: &ID3D11Device,
    input: &InputStream,
    usage: D3D11_USAGE,
    bind: D3D11_BIND_FLAG,
    cpu_access: u32,
) -> Result<ID3D11Texture2D, G2gError> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: input.width,
        Height: input.height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_NV12,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: usage,
        BindFlags: bind.0 as u32,
        CPUAccessFlags: cpu_access,
        MiscFlags: 0,
    };
    let mut texture = None;
    // SAFETY: `desc` and the out-param are valid locals for the call.
    unsafe {
        device
            .CreateTexture2D(&desc, None, Some(&mut texture))
            .map_err(mf_err)?;
    }
    texture.ok_or(G2gError::Hardware(HardwareError::Other))
}

fn render_target(device: &ID3D11Device, input: &InputStream) -> Result<ID3D11Texture2D, G2gError> {
    // a video processor output view needs a render target
    create_texture(
        device,
        input,
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        0,
    )
}

impl D3D11Convert {
    pub fn new() -> Self {
        Self {
            configured: false,
            out_domain: MemoryDomainKind::System,
            input: None,
            output_colorimetry: Colorimetry::UNKNOWN,
            last_caps: None,
            gpu: None,
        }
    }

    fn gpu_for(&mut self, device: ID3D11Device) -> Result<&mut ConvertGpu, G2gError> {
        let input = self.input.as_ref().ok_or(G2gError::NotConfigured)?;
        if let Some(gpu) = self.gpu.take().filter(|gpu| gpu.device == device) {
            return Ok(self.gpu.insert(gpu));
        }
        // SAFETY: COM calls on objects from `device`.
        let context = unsafe {
            let context = device.GetImmediateContext().map_err(mf_err)?;
            // d3d11 immediate contexts are not thread safe, and the producer's thread uses this one too
            let multithread: ID3D11Multithread = context.cast().map_err(mf_err)?;
            let _ = multithread.SetMultithreadProtected(true);
            context
        };
        let processor = VideoProcessor::new(&device, &context, input.width, input.height)?;
        let output_space = output_color_space(self.output_colorimetry)?;
        // SAFETY: the processor and its context come from `device`.
        unsafe {
            let video_context: ID3D11VideoContext1 =
                processor.video_context.cast().map_err(mf_err)?;
            video_context.VideoProcessorSetStreamColorSpace1(
                &processor.processor,
                0,
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
            );
            video_context.VideoProcessorSetOutputColorSpace1(&processor.processor, output_space);
        }
        Ok(self.gpu.insert(ConvertGpu {
            device,
            context,
            processor,
            pool: Vec::new(),
            readback: None,
        }))
    }

    fn convert(&mut self, texture: &OwnedD3D11Texture) -> Result<MemoryDomain, G2gError> {
        if texture.dxgi_format != DXGI_FORMAT_B8G8R8A8_UNORM.0 as u32 {
            return Err(G2gError::CapsMismatch);
        }
        let (device, source) = texture_objects(texture)?;
        let input = self.input.clone().ok_or(G2gError::NotConfigured)?;
        if (texture.width, texture.height) != (input.width, input.height) {
            return Err(G2gError::CapsMismatch);
        }
        let out_domain = self.out_domain;
        let gpu = self.gpu_for(device)?;
        if out_domain == MemoryDomainKind::D3D11Texture {
            let converted = take_from_pool(&mut gpu.pool, || {
                render_target(&gpu.device, &input).map(TextureOwner)
            })?;
            gpu.processor
                .blt(&source, texture.subresource, &converted.0)?;
            return Ok(texture_frame(
                &gpu.device,
                converted,
                input.width,
                input.height,
                DXGI_FORMAT_NV12,
            ));
        }
        let (converted, staging) = match &gpu.readback {
            Some(readback) => readback.clone(),
            None => {
                let converted = render_target(&gpu.device, &input)?;
                let staging = create_texture(
                    &gpu.device,
                    &input,
                    D3D11_USAGE_STAGING,
                    D3D11_BIND_FLAG(0),
                    D3D11_CPU_ACCESS_READ.0 as u32,
                )?;
                gpu.readback.insert((converted, staging)).clone()
            }
        };
        gpu.processor
            .blt(&source, texture.subresource, &converted)?;
        let width = input.width as usize;
        let rows = input.height as usize * 3 / 2;
        let mut image = vec![0u8; width * rows];
        // SAFETY: the staging texture is mapped for reading between Map and Unmap, and each row read
        // stays inside RowPitch * rows bytes of the mapping, the luma rows followed by the chroma rows.
        unsafe {
            gpu.context.CopyResource(&staging, &converted);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            gpu.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .map_err(mf_err)?;
            let base = mapped.pData.cast::<u8>().cast_const();
            for (row, target) in image.chunks_exact_mut(width).enumerate() {
                let start = base.add(row * mapped.RowPitch as usize);
                target.copy_from_slice(core::slice::from_raw_parts(start, width));
            }
            gpu.context.Unmap(&staging, 0);
        }
        Ok(MemoryDomain::System(SystemSlice::from_boxed(
            image.into_boxed_slice(),
        )))
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        let input = self.input.as_ref().ok_or(G2gError::NotConfigured)?;
        Ok(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(input.width),
            height: Dim::Fixed(input.height),
            framerate: input.framerate.clone(),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry {
                range: self.output_colorimetry.range,
                matrix: self.output_colorimetry.matrix,
                transfer: input.colorimetry.transfer,
                primaries: input.colorimetry.primaries,
            },
            pixel_aspect: input.pixel_aspect,
        })
    }
}

fn any_geometry(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: Interlace::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
    }
}

impl AsyncElement for D3D11Convert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Direct3D11 colour converter",
            "Filter/Converter/Video/Hardware",
            "Converts a BGRA D3D11 texture to NV12 on the GPU through a D3D11 video processor",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::D3D11Texture)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        self.out_domain
    }

    fn output_domains(&self) -> DomainSet {
        OUTPUT_DOMAINS
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        texture_proposal(caps, self.input_domains())
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.clone().resolve_for_producer(OUTPUT_DOMAINS) {
            self.out_domain = resolved.domain;
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&any_geometry(RawVideoFormat::Bgra8))
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedFields(CapsTransform::RawVideo {
            accept: vec![RawVideoFormat::Bgra8],
            produce: vec![RawVideoFormat::Nv12],
            shapes: vec![
                RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(RawVideoFormat::Nv12))
            ],
        })
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::Bgra8,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            colorimetry,
            pixel_aspect,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        // nv12 textures need even dimensions
        if width % 2 != 0 || height % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let input = InputStream {
            width: *width,
            height: *height,
            framerate: framerate.clone(),
            colorimetry: *colorimetry,
            pixel_aspect: *pixel_aspect,
        };
        if self.input.as_ref().map(|i| (i.width, i.height)) != Some((input.width, input.height)) {
            self.gpu = None;
        }
        self.input = Some(input);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            colorimetry,
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        output_color_space(*colorimetry)?;
        if *colorimetry != self.output_colorimetry {
            self.gpu = None;
        }
        self.output_colorimetry = *colorimetry;
        Ok(())
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
                    let MemoryDomain::D3D11Texture(texture) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let domain = self.convert(texture)?;
                    let caps = self.output_caps()?;
                    if self.last_caps.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.last_caps = Some(caps);
                    }
                    out.push(PipelinePacket::DataFrame(Frame {
                        domain,
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: Default::default(),
                    }))
                    .await?;
                }
                // the runner configures this element before it pushes its own output caps here
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                    self.last_caps = Some(caps);
                }
                PipelinePacket::Flush => {
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                // the runner forwards end-of-stream after `process` returns
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for D3D11Convert {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(any_geometry(RawVideoFormat::Bgra8))),
            PadTemplate::source(CapsSet::one(any_geometry(RawVideoFormat::Nv12))),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
    use windows::Win32::Graphics::Direct3D11::{
        D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA,
    };

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 32;
    const BAR_WIDTH: usize = 8;
    // white, yellow, cyan, green, magenta, red, blue, black as BGRA
    const BARS: [[u8; 4]; 8] = [
        [255, 255, 255, 255],
        [0, 255, 255, 255],
        [255, 255, 0, 255],
        [0, 255, 0, 255],
        [255, 0, 255, 255],
        [0, 0, 255, 255],
        [255, 0, 0, 255],
        [0, 0, 0, 255],
    ];
    // gpu and cpu conversions round differently
    const TOLERANCE: u8 = 2;

    fn video_device() -> Option<ID3D11Device> {
        let mut device = None;
        // SAFETY: the out-param is a valid local, written by the call before it is read.
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
            .ok()?;
        }
        device
    }

    fn colour_bars() -> Vec<u8> {
        (0..HEIGHT as usize * WIDTH as usize)
            .flat_map(|pixel| BARS[(pixel % WIDTH as usize) / BAR_WIDTH])
            .collect()
    }

    fn bgra_frame(device: &ID3D11Device, pixels: &[u8]) -> OwnedD3D11Texture {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: WIDTH,
            Height: HEIGHT,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE).0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast(),
            SysMemPitch: WIDTH * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        // SAFETY: `pixels` holds WIDTH * HEIGHT BGRA pixels at the stated pitch.
        unsafe {
            device
                .CreateTexture2D(&desc, Some(&data), Some(&mut texture))
                .expect("create the bgra texture");
        }
        let MemoryDomain::D3D11Texture(frame) = texture_frame(
            device,
            Arc::new(TextureOwner(texture.expect("texture"))),
            WIDTH,
            HEIGHT,
            DXGI_FORMAT_B8G8R8A8_UNORM,
        ) else {
            unreachable!("texture_frame builds a D3D11 frame");
        };
        frame
    }

    #[test]
    fn gpu_conversion_matches_videoconvert() {
        let Some(device) = video_device() else {
            std::eprintln!("skipped: no D3D11 device with video support");
            return;
        };
        let pixels = colour_bars();
        let frame = bgra_frame(&device, &pixels);
        let input = Caps::RawVideo {
            format: RawVideoFormat::Bgra8,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Progressive,
            colorimetry: Colorimetry::SRGB,
            pixel_aspect: PixelAspect::SQUARE,
        };
        let mut convert = D3D11Convert::new();
        convert.configure_pipeline(&input).expect("bgra input");
        convert
            .configure_output(&any_geometry(RawVideoFormat::Nv12))
            .expect("nv12 output");
        let MemoryDomain::System(converted) = convert.convert(&frame).expect("convert") else {
            panic!("system output was negotiated");
        };
        let expected = crate::videoconvert::convert(
            &pixels,
            RawVideoFormat::Bgra8,
            RawVideoFormat::Nv12,
            WIDTH as usize,
            HEIGHT as usize,
            Colorimetry::UNKNOWN,
        );
        assert_eq!(converted.as_slice().len(), expected.len());
        let worst = converted
            .as_slice()
            .iter()
            .zip(expected.iter())
            .map(|(gpu, cpu)| gpu.abs_diff(*cpu))
            .max()
            .expect("non-empty frame");
        assert!(worst <= TOLERANCE, "largest sample difference {worst}");
    }

    #[test]
    fn untagged_output_converts_with_bt601_limited_like_videoconvert() {
        assert_eq!(
            output_color_space(Colorimetry::UNKNOWN),
            Ok(DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P601)
        );
        let bt709_full = Colorimetry {
            matrix: MatrixCoefficients::Bt709,
            range: ColorRange::Full,
            ..Colorimetry::UNKNOWN
        };
        assert_eq!(
            output_color_space(bt709_full),
            Ok(DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709)
        );
        let bt2020 = Colorimetry {
            matrix: MatrixCoefficients::Bt2020Ncl,
            ..Colorimetry::UNKNOWN
        };
        assert_eq!(output_color_space(bt2020), Err(G2gError::CapsMismatch));
    }
}
