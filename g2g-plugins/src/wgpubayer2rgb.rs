// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedWgpuTexture};
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, MemoryDomain,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat,
};

use crate::bayer::{
    bayer_formats, demosaic_input, demosaic_pad_templates, drive, first_row_holds_blue_or_red,
    first_supported, format_changing_constraint, red_where_blue_is, BayerConversion,
    ConversionState, Stream,
};
use crate::gpu::{producer_context, publish_producer_context, GpuContext, WgpuTextureKeepAlive};
use crate::pixel::frame_byte_size;

const OUTPUT_FORMAT: RawVideoFormat = RawVideoFormat::Rgba8;
const WORKGROUP_SIDE: u32 = 8;

const DEMOSAIC_SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    row_stride: u32,
    depth: u32,
    big_endian: u32,
    first_row_holds_blue_or_red: u32,
    red_where_blue_is: u32,
    unused: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> bayer: array<u32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;

fn byte_at(offset: u32) -> u32 {
    return (bayer[offset / 4u] >> ((offset % 4u) * 8u)) & 0xffu;
}

fn sample(row: u32, column: u32) -> u32 {
    if (params.depth == 8u) {
        return byte_at(row * params.row_stride + column);
    }
    let offset = row * params.row_stride + column * 2u;
    let first = byte_at(offset);
    let second = byte_at(offset + 1u);
    if (params.big_endian == 1u) {
        return (first << 8u) | second;
    }
    return first | (second << 8u);
}

fn mirrored(row: u32, index: i32) -> u32 {
    let last = i32(params.width) - 1;
    let column = abs(index);
    return sample(row, u32(min(column, 2 * last - column)));
}

fn average(a: u32, b: u32) -> u32 {
    return (a + b + 1u) >> 1u;
}

struct RowColours {
    even_column_colour: u32,
    odd_column_colour: u32,
}

fn split(row: u32, column: u32) -> RowColours {
    let own = sample(row, column);
    let left = i32(column) - 1;
    var interpolated = average(mirrored(row, left), mirrored(row, left + 2));
    if (column + 2u == params.width) {
        interpolated = mirrored(row, left);
    }
    if (column % 2u == 0u) {
        return RowColours(own, interpolated);
    }
    return RowColours(interpolated, own);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    var above = 1u;
    if (y > 0u) {
        above = y - 1u;
    }
    var below = y + 1u;
    if (below >= params.height) {
        if (params.height >= 4u) {
            below = params.height - 4u;
        } else {
            below = params.height - 2u;
        }
    }
    let up = split(above, x);
    let here = split(y, x);
    let down = split(below, x);
    let blue_row = ((y % 2u) == 0u) == (params.first_row_holds_blue_or_red == 1u);
    var blue: u32;
    var green: u32;
    var red: u32;
    if (blue_row) {
        let green_around = average(
            average(up.even_column_colour, down.even_column_colour),
            here.odd_column_colour,
        );
        green = select(here.odd_column_colour, green_around, x % 2u == 0u);
        blue = here.even_column_colour;
        red = average(up.odd_column_colour, down.odd_column_colour);
    } else {
        let green_around = average(
            average(up.odd_column_colour, down.odd_column_colour),
            here.even_column_colour,
        );
        green = select(green_around, here.even_column_colour, x % 2u == 0u);
        blue = average(up.even_column_colour, down.even_column_colour);
        red = here.odd_column_colour;
    }
    if (params.red_where_blue_is == 1u) {
        let swapped = blue;
        blue = red;
        red = swapped;
    }
    let rgb = min(vec3<u32>(red, green, blue) >> vec3<u32>(params.depth - 8u), vec3<u32>(255u));
    textureStore(output, vec2<i32>(i32(x), i32(y)), vec4<f32>(vec3<f32>(rgb) / 255.0, 1.0));
}
"#;

#[derive(Debug)]
struct DemosaicGpu {
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    bayer: Option<wgpu::Buffer>,
}

impl DemosaicGpu {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgpu-bayer2rgb"),
            source: wgpu::ShaderSource::Wgsl(DEMOSAIC_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("wgpu-bayer2rgb"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wgpu-bayer2rgb-params"),
            size: 8 * 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            params,
            bayer: None,
        }
    }

    fn bayer_buffer(&mut self, device: &wgpu::Device, size: u64) -> &wgpu::Buffer {
        if self
            .bayer
            .as_ref()
            .is_some_and(|buffer| buffer.size() != size)
        {
            self.bayer = None;
        }
        self.bayer.get_or_insert_with(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wgpu-bayer2rgb-samples"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }
}

#[derive(Debug, Default)]
pub struct WgpuBayer2Rgb {
    state: ConversionState,
    context: Option<GpuContext>,
    gpu: Option<DemosaicGpu>,
}

impl WgpuBayer2Rgb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: GpuContext) -> Self {
        self.context = Some(context);
        self
    }
}

impl BayerConversion for WgpuBayer2Rgb {
    const NAME: &'static str = "wgpubayer2rgb";

    fn state(&mut self) -> &mut ConversionState {
        &mut self.state
    }

    fn is_input(format: RawVideoFormat) -> bool {
        matches!(format, RawVideoFormat::Bayer(_))
    }

    fn input_stream(caps: &Caps) -> Result<Stream, G2gError> {
        demosaic_input(caps)
    }

    async fn convert(
        &mut self,
        bytes: &[u8],
        input: &Stream,
        output: RawVideoFormat,
    ) -> Result<MemoryDomain, G2gError> {
        let RawVideoFormat::Bayer(bayer) = input.format else {
            return Err(G2gError::CapsMismatch);
        };
        if output != OUTPUT_FORMAT {
            return Err(G2gError::CapsMismatch);
        }
        let context = self.context.as_ref().ok_or(G2gError::NotConfigured)?;
        let (device, queue) = (&context.device, &context.queue);
        let gpu = self.gpu.get_or_insert_with(|| DemosaicGpu::new(device));
        let samples = &bytes[..frame_byte_size(input.format, input.width, input.height)];
        let row_stride = input
            .format
            .plane_stride(0, input.width)
            .ok_or(G2gError::CapsMismatch)?;
        let params: Vec<u8> = [
            input.width,
            input.height,
            row_stride,
            u32::from(bayer.sample.bit_depth()),
            u32::from(bayer.sample.is_big_endian()),
            u32::from(first_row_holds_blue_or_red(bayer.pattern)),
            u32::from(red_where_blue_is(bayer.pattern)),
            0,
        ]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
        queue.write_buffer(&gpu.params, 0, &params);
        let bayer_buffer = gpu.bayer_buffer(device, samples.len() as u64).clone();
        queue.write_buffer(&bayer_buffer, 0, samples);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("wgpu-bayer2rgb-output"),
            size: wgpu::Extent3d {
                width: input.width,
                height: input.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wgpu-bayer2rgb"),
            layout: &gpu.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bayer_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("wgpu-bayer2rgb"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("wgpu-bayer2rgb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                input.width.div_ceil(WORKGROUP_SIDE),
                input.height.div_ceil(WORKGROUP_SIDE),
                1,
            );
        }
        queue.submit([encoder.finish()]);
        Ok(MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
            input.width,
            input.height,
            Arc::new(WgpuTextureKeepAlive::new(
                device.clone(),
                queue.clone(),
                texture,
            )),
        )))
    }
}

impl AsyncElement for WgpuBayer2Rgb {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu Bayer to RGB converter",
            "Filter/Converter/Video/Hardware",
            "Demosaics video/x-bayer into an RGBA wgpu texture on the GPU",
            crate::AARON_BOXER,
        )
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::SystemView)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuTexture
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        first_supported(upstream_caps, &bayer_formats())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        format_changing_constraint(bayer_formats(), Vec::from([OUTPUT_FORMAT]))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.state.input = Some(Self::input_stream(absolute_caps)?);
        if self.context.is_none() {
            let context = match producer_context() {
                Some(context) => context,
                None => block_on(GpuContext::headless())?,
            };
            // display sinks present on the published device, and a texture binds only on the device that made it
            self.context = Some(publish_producer_context(context));
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format: OUTPUT_FORMAT,
            ..
        } = output_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        self.state.output_format = Some(OUTPUT_FORMAT);
        Ok(())
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(drive(self, packet, out))
    }
}

impl PadTemplates for WgpuBayer2Rgb {
    fn pad_templates() -> Vec<PadTemplate> {
        demosaic_pad_templates(&[OUTPUT_FORMAT])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{read_rgba_texture_dq, shared_ctx, texture_of};
    use g2g_core::{CapsSet, Frame, FrameTiming, PushOutcome, SystemSlice};

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

    struct Case {
        input: &'static [u8],
        width: u32,
        height: u32,
        format: &'static str,
        expected_bgra: &'static [u8],
    }

    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!("../tests/fixtures/", $name))
        };
    }

    const CASES: [Case; 11] = [
        Case {
            input: fixture!("bayer_12x8_8bit.raw"),
            width: 12,
            height: 8,
            format: "bggr",
            expected_bgra: fixture!("bayer_12x8_bggr_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_8bit.raw"),
            width: 12,
            height: 8,
            format: "rggb",
            expected_bgra: fixture!("bayer_12x8_rggb_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_8bit.raw"),
            width: 12,
            height: 8,
            format: "grbg",
            expected_bgra: fixture!("bayer_12x8_grbg_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_8bit.raw"),
            width: 12,
            height: 8,
            format: "gbrg",
            expected_bgra: fixture!("bayer_12x8_gbrg_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_grbg10le.raw"),
            width: 12,
            height: 8,
            format: "grbg10le",
            expected_bgra: fixture!("bayer_12x8_grbg10le_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_grbg12le.raw"),
            width: 12,
            height: 8,
            format: "grbg12le",
            expected_bgra: fixture!("bayer_12x8_grbg12le_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_grbg12be.raw"),
            width: 12,
            height: 8,
            format: "grbg12be",
            expected_bgra: fixture!("bayer_12x8_grbg12be_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x8_grbg16be.raw"),
            width: 12,
            height: 8,
            format: "grbg16be",
            expected_bgra: fixture!("bayer_12x8_grbg16be_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x5_8bit.raw"),
            width: 12,
            height: 5,
            format: "gbrg",
            expected_bgra: fixture!("bayer_12x5_gbrg_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_12x3_8bit.raw"),
            width: 12,
            height: 3,
            format: "rggb",
            expected_bgra: fixture!("bayer_12x3_rggb_to_bgra.raw"),
        },
        Case {
            input: fixture!("bayer_6x4_8bit.raw"),
            width: 6,
            height: 4,
            format: "bggr",
            expected_bgra: fixture!("bayer_6x4_bggr_to_bgra.raw"),
        },
    ];

    fn fixed_caps(text: &str) -> Caps {
        let set = CapsSet::from_gst_string(text).expect("caps parse");
        let [caps] = set.alternatives() else {
            panic!("{text} is not fixed");
        };
        caps.clone()
    }

    #[tokio::test]
    async fn gpu_demosaic_matches_the_gstreamer_fixtures() {
        let _gpu = crate::GPU_TEST_LOCK.lock().await;
        let Some(context) = shared_ctx().await else {
            std::eprintln!("no wgpu adapter; skipping WgpuBayer2Rgb test");
            return;
        };
        for case in &CASES {
            let (width, height, format) = (case.width, case.height, case.format);
            let mut element = WgpuBayer2Rgb::new().with_context(context.clone());
            element
                .configure_pipeline(&fixed_caps(&alloc::format!(
                    "video/x-bayer,format={format},width={width},height={height}"
                )))
                .expect("Bayer input");
            element
                .configure_output(&fixed_caps(&alloc::format!(
                    "video/x-raw,format=RGBA,width={width},height={height}"
                )))
                .expect("RGBA output");
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(case.input.into())),
                FrameTiming::default(),
                0,
            );
            let mut sink = FrameSink::default();
            element
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("demosaics");
            let Some(PipelinePacket::DataFrame(out)) = sink.packets.pop() else {
                panic!("{format} {width}x{height}: a frame was pushed");
            };
            let MemoryDomain::WgpuTexture(owned) = &out.domain else {
                panic!("the output is a GPU texture");
            };
            let texture = texture_of(owned).expect("an RGBA texture");
            let rgba = read_rgba_texture_dq(&context.device, &context.queue, texture).unwrap();
            let bgra: Vec<u8> = rgba
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&[red, green, blue, alpha]| [blue, green, red, alpha])
                .collect();
            assert!(
                bgra == case.expected_bgra,
                "{format} {width}x{height} differs from GStreamer"
            );
        }
    }
}
