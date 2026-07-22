//! WebGPU zero-copy presentation sink (browser/wasm). Consumes decoded frames in
//! `MemoryDomain::WebGPUExternalTexture` (the output of
//! [`WebCodecsDecode::with_gpu_output`](crate::webcodecsdecode::WebCodecsDecode::with_gpu_output)),
//! imports each `VideoFrame` as a `GPUExternalTexture`, and samples it in a render
//! pass onto a `<canvas>` WebGPU context, with **no CPU readback** (M541).
//!
//! This is the browser analog of the native Vulkan Video -> `wgpu::Texture` wedge:
//! the browser's hardware decoder keeps the frame GPU-resident, and it is composited
//! straight into the page, never copied into wasm memory. Contrast with
//! [`CanvasSink`](crate::canvassink::CanvasSink), which copies the frame out to
//! system RGBA and paints it via the 2D context.
//!
//! Device setup (`navigator.gpu` -> adapter -> device) is async, so it runs lazily
//! on the first frame inside `process` (`configure_pipeline` only records the canvas
//! id). The render pipeline, sampler and bind-group layout are built once; per frame
//! only the external-texture import and its bind group are rebuilt (a
//! `GPUExternalTexture` is single-frame: it expires when the source `VideoFrame`
//! advances/closes).
//!
//! Build requires `--cfg=web_sys_unstable_apis` (WebGPU + WebCodecs web-sys bindings
//! are unstable).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

use js_sys::JsOption;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    GpuBindGroupDescriptor, GpuBindGroupEntry, GpuBindGroupLayout, GpuBindGroupLayoutDescriptor,
    GpuBindGroupLayoutEntry, GpuCanvasAlphaMode, GpuCanvasConfiguration, GpuCanvasContext,
    GpuColorTargetState, GpuDevice, GpuExternalTextureBindingLayout, GpuExternalTextureDescriptor,
    GpuFragmentState, GpuLoadOp, GpuPipelineLayoutDescriptor, GpuPowerPreference,
    GpuPrimitiveState, GpuPrimitiveTopology, GpuQueue, GpuRenderPassColorAttachment,
    GpuRenderPassDescriptor, GpuRenderPipeline, GpuRenderPipelineDescriptor,
    GpuRequestAdapterOptions, GpuSampler, GpuSamplerBindingLayout, GpuSamplerBindingType,
    GpuShaderModuleDescriptor, GpuStoreOp, GpuTextureFormat, GpuUncapturedErrorEvent,
    GpuVertexState, HtmlCanvasElement,
};

use crate::webcodecsdecode::VideoFrameOwner;

/// WGSL: a fullscreen triangle sampling an external (YUV) texture. `texture_external`
/// is the binding for an imported `VideoFrame`; `textureSampleBaseClampToEdge` is the
/// only sampling function allowed for it (it applies the frame's colour conversion).
const SHADER: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var tex: texture_external;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let xy = corners[i];
    var out: VOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // Clip space -> texture space: [-1,1] -> [0,1], with Y flipped.
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    return textureSampleBaseClampToEdge(tex, samp, in.uv);
}

// Inference variant: a per-pixel nearest-centroid classifier (argmin over K class
// colour centroids) run on the decoded frame, zero-copy from the external texture.
// The smallest honest neural primitive (a 1-NN / prototype classifier), it stands
// in for a real model head; the GPU-compute wiring is identical for a CNN. Output
// is the class index as grey (class/(K-1)) so the segmentation is visible on the
// canvas (this host cannot read a storage buffer back to the CPU; presentation is
// the reliable output path here). On an SMPTE bar test pattern the seven top bars
// classify to a monotonic grey staircase.
const K: u32 = 8u;
fn centroid(k: u32) -> vec3<f32> {
    switch k {
        case 0u: { return vec3<f32>(1.0, 1.0, 1.0); }  // white
        case 1u: { return vec3<f32>(1.0, 1.0, 0.0); }  // yellow
        case 2u: { return vec3<f32>(0.0, 1.0, 1.0); }  // cyan
        case 3u: { return vec3<f32>(0.0, 1.0, 0.0); }  // green
        case 4u: { return vec3<f32>(1.0, 0.0, 1.0); }  // magenta
        case 5u: { return vec3<f32>(1.0, 0.0, 0.0); }  // red
        case 6u: { return vec3<f32>(0.0, 0.0, 1.0); }  // blue
        default: { return vec3<f32>(0.0, 0.0, 0.0); }  // black
    }
}

@fragment
fn fs_infer(in: VOut) -> @location(0) vec4<f32> {
    let rgb = textureSampleBaseClampToEdge(tex, samp, in.uv).rgb;
    var best: u32 = 0u;
    var best_d: f32 = 1.0e9;
    for (var k: u32 = 0u; k < K; k = k + 1u) {
        let d = distance(rgb, centroid(k));
        if (d < best_d) { best_d = d; best = k; }
    }
    let g = f32(best) / f32(K - 1u);
    return vec4<f32>(g, g, g, 1.0);
}

// A real 2-layer convolutional network run per output pixel over the decoded frame
// (conv3x3 -> ReLU -> conv3x3 -> ReLU). Layer 1 has two 3x3 filters (the classic
// Sobel x / y edge detectors, i.e. what a trained first conv layer learns); layer 2
// is a 3x3 average over each feature map, summed into an edge-energy scalar shown as
// grey. The receptive field is 5x5, so one fragment gathers a 5x5 luma neighbourhood
// and evaluates both layers with no intermediate texture. The texel step comes from
// dpdx/dpdy of the interpolated uv (the canvas is sized to the frame, so one screen
// pixel is one texel), so no dimensions uniform is needed. The shader is weight/arch
// agnostic: trained weights or more layers slot straight in.
fn luma_at(uv: vec2<f32>) -> f32 {
    let c = textureSampleBaseClampToEdge(tex, samp, uv).rgb;
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_cnn(in: VOut) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(dpdx(in.uv).x, dpdy(in.uv).y);
    // 5x5 luma support, indices 0..4 == offsets -2..2.
    var lum: array<array<f32, 5>, 5>;
    for (var i: i32 = 0; i < 5; i = i + 1) {
        for (var j: i32 = 0; j < 5; j = j + 1) {
            let off = vec2<f32>(f32(j - 2), f32(i - 2)) * texel;
            lum[i][j] = luma_at(in.uv + off);
        }
    }
    // Layer 1: Sobel x/y at each of the 3x3 positions (p,q index the top-left of the
    // 3x3 window in lum), ReLU.
    var acc: f32 = 0.0;
    for (var p: i32 = 0; p < 3; p = p + 1) {
        for (var q: i32 = 0; q < 3; q = q + 1) {
            let gx = -lum[p][q] + lum[p][q + 2]
                     - 2.0 * lum[p + 1][q] + 2.0 * lum[p + 1][q + 2]
                     - lum[p + 2][q] + lum[p + 2][q + 2];
            let gy = -lum[p][q] - 2.0 * lum[p][q + 1] - lum[p][q + 2]
                     + lum[p + 2][q] + 2.0 * lum[p + 2][q + 1] + lum[p + 2][q + 2];
            // Layer 2: average combine of the ReLU'd layer-1 activations.
            acc = acc + max(gx, 0.0) + max(gy, 0.0);
        }
    }
    let g = clamp(acc / 9.0, 0.0, 1.0);
    return vec4<f32>(g, g, g, 1.0);
}
"#;

/// The GPU objects built once on the first frame and reused every frame after.
struct GpuState {
    device: GpuDevice,
    queue: GpuQueue,
    context: GpuCanvasContext,
    pipeline: GpuRenderPipeline,
    sampler: GpuSampler,
    /// Shared by the pipeline layout and every per-frame bind group, so they are
    /// guaranteed compatible.
    bind_group_layout: GpuBindGroupLayout,
}

pub struct WebGpuCanvasSink {
    canvas_id: String,
    canvas: Option<HtmlCanvasElement>,
    gpu: Option<GpuState>,
    width: u32,
    height: u32,
    configured: bool,
    presented: u64,
    /// Fragment entry point = the inference head run over each decoded frame:
    /// `"fs"` passthrough, `"fs_infer"` per-pixel classifier, `"fs_cnn"` a 2-layer
    /// conv net. GPU inference over the zero-copy frame, visualized to the canvas.
    head: &'static str,
    // Kept alive for the device's lifetime; JS holds a raw reference to it.
    _on_error: Option<Closure<dyn FnMut(GpuUncapturedErrorEvent)>>,
}

impl core::fmt::Debug for WebGpuCanvasSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebGpuCanvasSink")
            .field("canvas_id", &self.canvas_id)
            .field("configured", &self.configured)
            .field("gpu_ready", &self.gpu.is_some())
            .field("width", &self.width)
            .field("height", &self.height)
            .field("presented", &self.presented)
            .finish_non_exhaustive()
    }
}

impl WebGpuCanvasSink {
    /// `canvas_id` is the `id` of an existing `<canvas>`; the WebGPU device and
    /// context are acquired lazily on the first frame (the handshake is async).
    pub fn new(canvas_id: impl Into<String>) -> Self {
        Self {
            canvas_id: canvas_id.into(),
            canvas: None,
            gpu: None,
            width: 0,
            height: 0,
            configured: false,
            presented: 0,
            head: "fs",
            _on_error: None,
        }
    }

    /// Run the per-pixel nearest-centroid classifier over each decoded frame
    /// (zero-copy from the GPU external texture) and present the class map, instead
    /// of the plain passthrough. The browser GPU-inference primitive: a real, if
    /// tiny, classifier whose compute wiring is identical for a full model.
    pub fn with_inference(mut self) -> Self {
        self.head = "fs_infer";
        self
    }

    /// Run a 2-layer convolutional network (conv3x3 -> ReLU -> conv3x3) over each
    /// decoded frame, zero-copy, presenting its feature map. A real CNN forward pass
    /// (the defining conv + nonlinearity + depth); the layer-1 filters are edge
    /// detectors (the features a trained first layer learns), so trained weights and
    /// more layers drop into the same shader.
    pub fn with_cnn(mut self) -> Self {
        self.head = "fs_cnn";
        self
    }

    /// Count of frames rendered to the canvas. Useful in tests.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// Acquire the WebGPU adapter/device and build the render pipeline. Async
    /// (adapter/device requests are promises); called once, on the first frame.
    async fn ensure_gpu(&mut self) -> Result<(), G2gError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let err = || G2gError::Hardware(HardwareError::Other);
        let canvas = self.canvas.as_ref().ok_or(G2gError::NotConfigured)?;

        let navigator = web_sys::window().ok_or_else(err)?.navigator();
        let gpu = navigator.gpu();

        // navigator.gpu.requestAdapter() then adapter.requestDevice(): both async.
        // Ask for the high-performance adapter so a hybrid-GPU host (integrated +
        // discrete) presents on the discrete GPU rather than defaulting to
        // integrated (which on some drivers can fail the canvas swap-chain alloc).
        let opts = GpuRequestAdapterOptions::new();
        opts.set_power_preference(GpuPowerPreference::HighPerformance);
        let adapter_promise: js_sys::Promise =
            gpu.request_adapter_with_options(&opts).unchecked_into();
        let adapter = JsFuture::from(adapter_promise)
            .await
            .map_err(|_| err())?
            .dyn_into::<web_sys::GpuAdapter>()
            .map_err(|_| err())?; // a null adapter (no WebGPU) fails the cast

        let device_promise: js_sys::Promise = adapter.request_device().unchecked_into();
        let device = JsFuture::from(device_promise)
            .await
            .map_err(|_| err())?
            .dyn_into::<GpuDevice>()
            .map_err(|_| err())?;
        let queue = device.queue();

        // Surface WebGPU's asynchronous errors (validation, out-of-memory, device
        // loss). They do NOT propagate as a Rust `Result` from the submit path, so
        // without this a swap-chain / allocation failure is a silent black canvas
        // (as seen with a mis-selected Vulkan ICD). Log them like the pipeline's
        // terminal-error `report` on the ingest side.
        let on_error = Closure::<dyn FnMut(GpuUncapturedErrorEvent)>::new(
            move |e: GpuUncapturedErrorEvent| {
                web_sys::console::error_1(&JsValue::from_str(&alloc::format!(
                    "WebGpuCanvasSink: uncaptured WebGPU error: {}",
                    e.error().message()
                )));
            },
        );
        device.set_onuncapturederror(Some(on_error.as_ref().unchecked_ref()));
        self._on_error = Some(on_error);

        // Configure the canvas' WebGPU context with the device's preferred format.
        let context = canvas
            .get_context("webgpu")
            .map_err(|_| err())?
            .ok_or_else(err)?
            .dyn_into::<GpuCanvasContext>()
            .map_err(|_| err())?;
        let format: GpuTextureFormat = gpu.get_preferred_canvas_format();
        let config = GpuCanvasConfiguration::new(&device, format);
        config.set_alpha_mode(GpuCanvasAlphaMode::Opaque);
        context.configure(&config).map_err(|_| err())?;

        let bind_group_layout = make_bind_group_layout(&device)?;
        let pipeline = build_pipeline(&device, format, &bind_group_layout, self.head)?;
        let sampler = device.create_sampler();

        self.gpu = Some(GpuState {
            device,
            queue,
            context,
            pipeline,
            sampler,
            bind_group_layout,
        });
        Ok(())
    }

    /// Import the frame's `VideoFrame` as a `GPUExternalTexture` and draw it to the
    /// canvas. Zero-copy: the frame never leaves the GPU.
    fn render(&mut self, frame: &Frame) -> Result<(), G2gError> {
        let err = || G2gError::Hardware(HardwareError::Other);
        let MemoryDomain::WebGPUExternalTexture(ext) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        // Recover the browser `VideoFrame` the decoder handed forward.
        let owner = ext
            .keep_alive()
            .as_any()
            .downcast_ref::<VideoFrameOwner>()
            .ok_or(G2gError::UnsupportedDomain)?;
        let video_frame = owner.frame();

        let gpu = self.gpu.as_ref().ok_or(G2gError::NotConfigured)?;

        // Import + bind the external texture. Both are per-frame: an imported
        // external texture is valid only for the current frame.
        let import = GpuExternalTextureDescriptor::new_with_video_frame(video_frame);
        let external = gpu
            .device
            .import_external_texture(&import)
            .map_err(|_| err())?;
        let entries = [
            GpuBindGroupEntry::new(0, &gpu.sampler),
            GpuBindGroupEntry::new_with_gpu_external_texture(1, &external),
        ];
        let bind_group = gpu.device.create_bind_group(&GpuBindGroupDescriptor::new(
            &entries,
            &gpu.bind_group_layout,
        ));

        // Render pass targeting the canvas' current swap-chain texture view.
        let view = gpu
            .context
            .get_current_texture()
            .map_err(|_| err())?
            .create_view()
            .map_err(|_| err())?;
        let attachment = GpuRenderPassColorAttachment::new_with_gpu_texture_view(
            GpuLoadOp::Clear,
            GpuStoreOp::Store,
            &view,
        );
        let color_attachments = [JsOption::wrap(attachment)];
        let pass_desc = GpuRenderPassDescriptor::new(&color_attachments);

        let encoder = gpu.device.create_command_encoder();
        let pass = encoder.begin_render_pass(&pass_desc).map_err(|_| err())?;
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, Some(&bind_group));
        pass.draw(3);
        pass.end();
        gpu.queue.submit(&[encoder.finish()]);

        self.presented += 1;
        Ok(())
    }
}

impl AsyncElement for WebGpuCanvasSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let err = || G2gError::Hardware(HardwareError::Other);
        let window = web_sys::window().ok_or_else(err)?;
        let document = window.document().ok_or_else(err)?;
        let element = document
            .get_element_by_id(&self.canvas_id)
            .ok_or_else(err)?;
        let canvas: HtmlCanvasElement = element.dyn_into().map_err(|_| err())?;
        self.canvas = Some(canvas);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width,
                    height,
                    ..
                }) => {
                    self.width = fixed_or_zero(&width);
                    self.height = fixed_or_zero(&height);
                    // Size the canvas' drawing buffer to the video so the WebGPU
                    // swap-chain texture matches (get_current_texture reads these).
                    if let Some(canvas) = &self.canvas {
                        if self.width != 0 && self.height != 0 {
                            canvas.set_width(self.width);
                            canvas.set_height(self.height);
                        }
                    }
                }
                PipelinePacket::CapsChanged(_) => return Err(G2gError::CapsMismatch),
                PipelinePacket::DataFrame(frame) => {
                    self.ensure_gpu().await?;
                    self.render(&frame)?;
                }
                PipelinePacket::Flush | PipelinePacket::Eos | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for WebGpuCanvasSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(rgba_any()))])
    }
}

/// The two-entry bind-group layout: a filtering sampler at binding 0 and an external
/// texture at binding 1, both visible to the fragment stage.
fn make_bind_group_layout(device: &GpuDevice) -> Result<GpuBindGroupLayout, G2gError> {
    let err = || G2gError::Hardware(HardwareError::Other);
    let frag = web_sys::gpu_shader_stage::FRAGMENT;

    let sampler_entry = GpuBindGroupLayoutEntry::new(0, frag);
    let sampler_layout = GpuSamplerBindingLayout::new();
    sampler_layout.set_type(GpuSamplerBindingType::Filtering);
    sampler_entry.set_sampler(&sampler_layout);

    let external_entry = GpuBindGroupLayoutEntry::new(1, frag);
    external_entry.set_external_texture(&GpuExternalTextureBindingLayout::new());

    let entries = [sampler_entry, external_entry];
    device
        .create_bind_group_layout(&GpuBindGroupLayoutDescriptor::new(&entries))
        .map_err(|_| err())
}

/// Build the fullscreen-triangle render pipeline that samples the external texture,
/// targeting `format` (the canvas' preferred format), using the shared `bgl`.
fn build_pipeline(
    device: &GpuDevice,
    format: GpuTextureFormat,
    bgl: &GpuBindGroupLayout,
    frag_entry: &str,
) -> Result<GpuRenderPipeline, G2gError> {
    let err = || G2gError::Hardware(HardwareError::Other);
    let module = device.create_shader_module(&GpuShaderModuleDescriptor::new(SHADER));

    let layouts = [JsOption::wrap(bgl.clone())];
    let pipeline_layout =
        device.create_pipeline_layout(&GpuPipelineLayoutDescriptor::new(&layouts));

    let vertex = GpuVertexState::new(&module);
    vertex.set_entry_point("vs");

    let target = GpuColorTargetState::new(format);
    let targets = [JsOption::wrap(target)];
    let fragment = GpuFragmentState::new(&module, &targets);
    fragment.set_entry_point(frag_entry);

    let primitive = GpuPrimitiveState::new();
    primitive.set_topology(GpuPrimitiveTopology::TriangleList);

    let desc = GpuRenderPipelineDescriptor::new(&pipeline_layout, &vertex);
    desc.set_fragment(&fragment);
    desc.set_primitive(&primitive);
    device.create_render_pipeline(&desc).map_err(|_| err())
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}
