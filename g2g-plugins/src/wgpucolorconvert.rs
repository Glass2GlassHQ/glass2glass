//! `WgpuColorConvert`: NV12 to RGBA on the GPU, texture in, texture out.
//!
//! Takes an NV12 [`MemoryDomain::WgpuTexture`] frame (a
//! [`WgpuNv12Texture`] from `CudaToWgpu` or a GPU decoder, in either NV12
//! layout) and renders it into a fresh `Rgba8Unorm` texture with the same
//! YCbCr -> RGB shaders `WgpuSink` presents NV12 with, so an RGBA-only GPU
//! element such as `VelloAnalyticsOverlay` can follow a GPU decode without the
//! frame passing through system memory.
//!
//! The work runs on the device the incoming texture lives on, and the output
//! texture is made there too, so the next element adopts that device the same
//! way.
//!
//! `wgpu-sink` feature.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedWgpuTexture};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsTransform, Colorimetry, ConfigureOutcome, Dim,
    ElementMetadata, FieldTransform, Frame, G2gError, Interlace, MemoryDomain, OutputSink,
    PipelinePacket, Rate, RawVideoFormat, RawVideoShape,
};

use crate::gpu::{
    texture_holds_picture, texture_layout, WgpuNv12Texture, WgpuTextureKeepAlive, WgpuTextureLayout,
};
use crate::wgpusink::{
    fragment_nv12, BlitPipeline, BlitResources, FRAGMENT_MULTIPLANAR_NV12, FRAGMENT_PACKED_NV12,
};

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn nv12_to_rgba() -> CapsTransform {
    CapsTransform::RawVideo {
        accept: vec![RawVideoFormat::Nv12],
        produce: vec![RawVideoFormat::Rgba8],
        shapes: vec![
            RawVideoShape::PASSTHROUGH.with_format(FieldTransform::Fixed(RawVideoFormat::Rgba8))
        ],
    }
}

/// The NV12 picture the negotiated input caps describe.
#[derive(Debug, Clone, Copy)]
struct Picture {
    width: u32,
    height: u32,
    colorimetry: Colorimetry,
}

fn picture(caps: &Caps) -> Result<Picture, G2gError> {
    let Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        colorimetry,
        ..
    } = caps
    else {
        return Err(G2gError::CapsMismatch);
    };
    if *width == 0 || *height == 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok(Picture {
        width: *width,
        height: *height,
        colorimetry: *colorimetry,
    })
}

/// Pipelines built on the device of the first frame, for one colorimetry.
#[derive(Debug)]
struct ConvertGpu {
    colorimetry: Colorimetry,
    packed: BlitPipeline,
    planes: BlitPipeline,
    resources: BlitResources,
}

impl ConvertGpu {
    fn new(device: &wgpu::Device, colorimetry: Colorimetry) -> Self {
        Self {
            colorimetry,
            packed: BlitPipeline::build(
                device,
                OUTPUT_FORMAT,
                WgpuTextureLayout::PackedNv12,
                &fragment_nv12(colorimetry, FRAGMENT_PACKED_NV12),
            ),
            planes: BlitPipeline::build(
                device,
                OUTPUT_FORMAT,
                WgpuTextureLayout::MultiplanarNv12,
                &fragment_nv12(colorimetry, FRAGMENT_MULTIPLANAR_NV12),
            ),
            resources: BlitResources::new(device),
        }
    }
}

/// Converts NV12 `WgpuTexture` frames to RGBA `WgpuTexture` frames on the GPU.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::wgpucolorconvert::WgpuColorConvert;
///
/// let convert = WgpuColorConvert::new();
/// assert_eq!(convert.converted(), 0);
/// ```
#[derive(Debug, Default)]
pub struct WgpuColorConvert {
    picture: Option<Picture>,
    gpu: Option<ConvertGpu>,
    converted: u64,
}

impl WgpuColorConvert {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames converted so far.
    pub fn converted(&self) -> u64 {
        self.converted
    }

    fn convert(&mut self, source: &WgpuNv12Texture) -> Result<WgpuTextureKeepAlive, G2gError> {
        let picture = self.picture.ok_or(G2gError::NotConfigured)?;
        let texture = source.texture();
        let layout = texture_layout(texture).ok_or(G2gError::UnsupportedDomain)?;
        if layout == WgpuTextureLayout::Rgba {
            return Err(G2gError::UnsupportedDomain);
        }
        if !texture_holds_picture(texture, layout, picture.width, picture.height) {
            return Err(G2gError::CapsMismatch);
        }
        let (device, queue) = (source.device(), source.queue());
        if self
            .gpu
            .as_ref()
            .is_none_or(|gpu| gpu.colorimetry != picture.colorimetry)
        {
            self.gpu = Some(ConvertGpu::new(device, picture.colorimetry));
        }
        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;
        let blit = match layout {
            WgpuTextureLayout::PackedNv12 => &gpu.packed,
            _ => &gpu.planes,
        };
        gpu.resources
            .write_picture_size(queue, picture.width, picture.height);

        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpu-color-convert-output"),
            size: wgpu::Extent3d {
                width: picture.width,
                height: picture.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_FORMAT,
            // COPY_SRC: Vello copies an image texture into its atlas.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-color-convert"),
        });
        blit.encode(device, &mut encoder, texture, layout, &gpu.resources, &view);
        queue.submit([encoder.finish()]);
        Ok(WgpuTextureKeepAlive::new(
            device.clone(),
            queue.clone(),
            output,
        ))
    }
}

impl AsyncElement for WgpuColorConvert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let nv12_any = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        };
        upstream_caps.intersect(&nv12_any)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedFields(nv12_to_rgba())
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuTexture
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.picture = Some(picture(absolute_caps)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu colour converter",
            "Filter/Converter/Video/Hardware",
            "Converts NV12 wgpu textures to RGBA wgpu textures on the GPU",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let source = owned
                        .keep_alive()
                        .as_any()
                        .downcast_ref::<WgpuNv12Texture>()
                        .ok_or(G2gError::UnsupportedDomain)?;
                    let output = self.convert(source)?;
                    let domain = MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                        owned.width,
                        owned.height,
                        Arc::new(output),
                    ));
                    self.converted += 1;
                    let out_frame = Frame {
                        domain,
                        timing: frame.timing,
                        sequence: frame.sequence,
                        meta: frame.meta,
                    };
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                // The runner's transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{read_rgba_texture_dq, shared_ctx, texture_of};
    use alloc::vec::Vec;
    use g2g_core::{FrameTiming, PushOutcome};

    fn nv12_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
        }
    }

    #[derive(Default)]
    struct FrameSink {
        packets: Vec<PipelinePacket>,
    }
    impl OutputSink for FrameSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            self.packets
                .push(packet_slot.take().expect("poll_push without a packet"));
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    /// A packed NV12 texture, top half red and bottom half blue (BT.601 limited),
    /// converts to an RGBA texture on the same device with the colours intact,
    /// and the analytics carried on the frame survive the conversion.
    #[tokio::test]
    async fn packed_nv12_texture_converts_to_rgba_texture() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(ctx) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuColorConvert test");
            return;
        };
        let (w, h) = (64u32, 8u32);
        const RED: (u8, u8, u8) = (81, 90, 240);
        const BLUE: (u8, u8, u8) = (41, 240, 110);
        let mut bytes = Vec::new();
        for y in 0..h {
            let luma = if y < h / 2 { RED.0 } else { BLUE.0 };
            bytes.extend(core::iter::repeat_n(luma, w as usize));
        }
        for y in 0..h / 2 {
            let (_, cb, cr) = if y < h / 4 { RED } else { BLUE };
            for _ in 0..w / 2 {
                bytes.extend_from_slice(&[cb, cr]);
            }
        }
        let packed_rows = h + h / 2;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test-nv12"),
            size: wgpu::Extent3d {
                width: w,
                height: packed_rows,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w),
                rows_per_image: Some(packed_rows),
            },
            wgpu::Extent3d {
                width: w,
                height: packed_rows,
                depth_or_array_layers: 1,
            },
        );
        #[cfg_attr(not(feature = "analytics"), allow(unused_mut))]
        let mut frame = Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                Arc::new(WgpuNv12Texture::new(
                    ctx.device.clone(),
                    ctx.queue.clone(),
                    texture,
                )),
            )),
            FrameTiming::default(),
            0,
        );
        #[cfg(feature = "analytics")]
        frame.meta.attach(g2g_core::AnalyticsMeta::new());

        let mut convert = WgpuColorConvert::new();
        convert.configure_pipeline(&nv12_caps(w, h)).unwrap();
        let mut sink = FrameSink::default();
        convert
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let Some(PipelinePacket::DataFrame(out)) = sink.packets.pop() else {
            panic!("a converted frame was pushed");
        };
        #[cfg(feature = "analytics")]
        assert!(out.meta.get::<g2g_core::AnalyticsMeta>().is_some());
        let MemoryDomain::WgpuTexture(owned) = &out.domain else {
            panic!("the output stays a GPU texture");
        };
        let rgba = texture_of(owned).expect("an RGBA texture keep-alive");
        assert_eq!(rgba.format(), OUTPUT_FORMAT);
        let pixels = read_rgba_texture_dq(&ctx.device, &ctx.queue, rgba).unwrap();
        let px = |x: u32, y: u32| {
            let i = ((y * w + x) * 4) as usize;
            [pixels[i], pixels[i + 1], pixels[i + 2]]
        };
        let top = px(5, 0);
        assert!(
            top[0] > 230 && top[1] < 25 && top[2] < 25,
            "top rows convert to red: {top:?}"
        );
        let bottom = px(5, h - 1);
        assert!(
            bottom[2] > 230 && bottom[0] < 25 && bottom[1] < 25,
            "bottom rows convert to blue: {bottom:?}"
        );
        assert_eq!(convert.converted(), 1);
    }
}
