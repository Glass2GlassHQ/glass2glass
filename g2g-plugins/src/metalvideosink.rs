//! M736: macOS Metal present sink (`CAMetalLayer`).
//!
//! `MetalVideoSink` is the macOS analog of `WaylandSink` / `D3D11Sink`: it
//! consumes NV12 `DataFrame`s and presents them to a `CAMetalLayer` drawable
//! through a fullscreen-triangle render pass (BT.601 video-range YUV -> RGB in
//! the fragment shader). Two input domains:
//!
//! - `MemoryDomain::CvPixelBuffer` (the M735 zero-copy domain): the decoder's
//!   IOSurface-backed buffer is imported plane-by-plane as `MTLTexture`s
//!   (`newTextureWithDescriptor:iosurface:plane:`), so `vtdec cv-output !
//!   metalvideosink` never copies pixels on the CPU.
//! - `MemoryDomain::System`: packed NV12 bytes staged into a texture pair via
//!   `replaceRegion`.
//!
//! By default the sink owns a standalone `CAMetalLayer` (a real swapchain, so
//! the full render + present path runs headless, which is what the macOS CI
//! runner validates). An app that wants the video on screen hands its own
//! layer over with [`MetalVideoSink::with_layer`]; AppKit window ownership
//! stays with the app (an element cannot own `NSApplication` / the main
//! thread).
//!
//! Gated `#[cfg(all(target_os = "macos", feature = "metal-sink"))]`; the macOS
//! CI job compiles and runtime-validates it like the VideoToolbox elements.

use core::ffi::c_void;
use core::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_core_foundation::{CFRetained, CGSize};
use objc2_core_video::{CVPixelBuffer, CVPixelBufferGetIOSurface};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSize,
    MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, DomainSet, G2gError, HardwareError,
    MemoryDomain, MemoryDomainKind, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat,
};

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

/// Fullscreen triangle + BT.601 video-range NV12 -> RGB fragment shader.
const SHADER_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float2 uv; };
vertex VOut vs(uint vid [[vertex_id]]) {
    float2 p[3] = { float2(-1.0,-1.0), float2(3.0,-1.0), float2(-1.0,3.0) };
    VOut o;
    o.pos = float4(p[vid], 0.0, 1.0);
    o.uv = float2((p[vid].x + 1.0) * 0.5, 1.0 - (p[vid].y + 1.0) * 0.5);
    return o;
}
fragment float4 fs(VOut in [[stage_in]],
                   texture2d<float> luma [[texture(0)]],
                   texture2d<float> chroma [[texture(1)]]) {
    constexpr sampler s(filter::linear);
    float y = (luma.sample(s, in.uv).r - 16.0/255.0) * (255.0/219.0);
    float2 uv = chroma.sample(s, in.uv).rg - 0.5;
    float3 rgb = float3(y + 1.596 * uv.y,
                        y - 0.391 * uv.x - 0.813 * uv.y,
                        y + 2.018 * uv.x);
    return float4(saturate(rgb), 1.0);
}
"#;

/// Everything the render loop needs, built at configure.
struct RenderState {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    layer: Retained<CAMetalLayer>,
    /// Staging texture pair for the packed-NV12 (System) input path; the
    /// zero-copy path imports the frame's IOSurface instead.
    staging_y: Retained<ProtocolObject<dyn MTLTexture>>,
    staging_cbcr: Retained<ProtocolObject<dyn MTLTexture>>,
    width: u32,
    height: u32,
}

impl core::fmt::Debug for RenderState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RenderState")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct MetalVideoSink {
    configured: bool,
    state: Option<RenderState>,
    /// An app-supplied `CAMetalLayer` (retained at configure); the sink builds
    /// its own standalone layer when absent.
    external_layer: Option<Retained<CAMetalLayer>>,
    /// Test hook: read each presented drawable back to CPU RGBA.
    readback: bool,
    last_rgba: Option<Vec<u8>>,
    presented: u64,
}

// SAFETY: the Metal / CoreAnimation objects are used single-threaded on the
// element's owning task (same contract as `VtDecode` / `MfDecode`); a
// standalone CAMetalLayer's drawables are explicitly usable off the main
// thread. We assert `Send` so the multi-thread runner accepts the element.
unsafe impl Send for MetalVideoSink {}

impl Default for MetalVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalVideoSink {
    pub fn new() -> Self {
        Self {
            configured: false,
            state: None,
            external_layer: None,
            readback: false,
            last_rgba: None,
            presented: 0,
        }
    }

    /// Present into an app-owned `CAMetalLayer` (a retained `CAMetalLayer*`),
    /// so the video shows in the app's window. The sink retains the layer and
    /// configures its device / pixel format / drawable size.
    ///
    /// # Safety
    ///
    /// `layer` must be a valid `CAMetalLayer` pointer; the app must not attach
    /// it to a view hierarchy it mutates concurrently with the running sink.
    pub unsafe fn with_layer(mut self, layer: NonNull<CAMetalLayer>) -> Self {
        // SAFETY: caller guarantees a valid CAMetalLayer; retain takes our +1.
        self.external_layer = Some(unsafe { Retained::retain(layer.as_ptr()) }.expect("non-null"));
        self
    }

    /// Read each presented drawable back to CPU RGBA (test hook; adds a
    /// GPU->CPU copy per frame).
    pub fn with_readback(mut self) -> Self {
        self.readback = true;
        self
    }

    /// Whether a Metal device exists (tests skip without one, like the
    /// wgpu/Vulkan suites skip without an adapter).
    pub fn device_available() -> bool {
        MTLCreateSystemDefaultDevice().is_some()
    }

    /// Count of frames presented. Useful in tests.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// The last presented frame as tight RGBA bytes (readback mode only).
    pub fn last_rgba(&self) -> Option<&[u8]> {
        self.last_rgba.as_deref()
    }

    fn hw() -> G2gError {
        G2gError::Hardware(HardwareError::Other)
    }

    fn build_state(&mut self, width: u32, height: u32) -> Result<(), G2gError> {
        let device = MTLCreateSystemDefaultDevice().ok_or_else(Self::hw)?;
        let queue = device.newCommandQueue().ok_or_else(Self::hw)?;

        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADER_SRC), None)
            .map_err(|_| Self::hw())?;
        let vs = library
            .newFunctionWithName(&NSString::from_str("vs"))
            .ok_or_else(Self::hw)?;
        let fs = library
            .newFunctionWithName(&NSString::from_str("fs"))
            .ok_or_else(Self::hw)?;
        let desc = MTLRenderPipelineDescriptor::new();
        // `&*x` explicitly: Option does not deref-coerce through the `&`.
        desc.setVertexFunction(Some(&*vs));
        desc.setFragmentFunction(Some(&*fs));
        // SAFETY: index 0 is a valid color attachment slot.
        unsafe { desc.colorAttachments().objectAtIndexedSubscript(0) }
            .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        let pipeline = device
            .newRenderPipelineStateWithDescriptor_error(&desc)
            .map_err(|_| Self::hw())?;

        let layer = match &self.external_layer {
            Some(l) => l.clone(),
            None => CAMetalLayer::layer(),
        };
        layer.setDevice(Some(&*device));
        layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        // Readable drawables (the readback blit); presenting still works.
        layer.setFramebufferOnly(false);
        layer.setDrawableSize(CGSize {
            width: width as f64,
            height: height as f64,
        });

        let staging_y = make_plane_texture(&device, MTLPixelFormat::R8Unorm, width, height)?;
        let staging_cbcr =
            make_plane_texture(&device, MTLPixelFormat::RG8Unorm, width / 2, height / 2)?;

        self.state = Some(RenderState {
            device,
            queue,
            pipeline,
            layer,
            staging_y,
            staging_cbcr,
            width,
            height,
        });
        Ok(())
    }

    /// Render one NV12 texture pair to the next drawable and present it.
    fn present(
        &mut self,
        y: &ProtocolObject<dyn MTLTexture>,
        cbcr: &ProtocolObject<dyn MTLTexture>,
    ) -> Result<(), G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        let drawable = st.layer.nextDrawable().ok_or_else(Self::hw)?;
        let target = drawable.texture();

        let cmd = st.queue.commandBuffer().ok_or_else(Self::hw)?;
        let pass = MTLRenderPassDescriptor::new();
        // The fullscreen triangle covers every pixel: no clear needed.
        // SAFETY: index 0 is a valid color attachment slot.
        let att = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        att.setTexture(Some(&*target));
        let enc = cmd
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or_else(Self::hw)?;
        enc.setRenderPipelineState(&st.pipeline);
        // SAFETY: texture indices 0/1 match the shader bindings; 3 vertices
        // draw the fullscreen triangle.
        unsafe {
            enc.setFragmentTexture_atIndex(Some(y), 0);
            enc.setFragmentTexture_atIndex(Some(cbcr), 1);
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.endEncoding();

        // Readback (test hook): blit the rendered drawable into a shared
        // buffer before it is presented.
        let read_buf = if self.readback {
            let bytes_per_row = (st.width as usize) * 4;
            let buf = st
                .device
                .newBufferWithLength_options(
                    bytes_per_row * st.height as usize,
                    MTLResourceOptions::empty(),
                )
                .ok_or_else(Self::hw)?;
            let blit = cmd.blitCommandEncoder().ok_or_else(Self::hw)?;
            // SAFETY: the region is within the drawable texture; the buffer is
            // sized for the full copy.
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &target,
                    0,
                    0,
                    MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize {
                        width: st.width as usize,
                        height: st.height as usize,
                        depth: 1,
                    },
                    &buf,
                    0,
                    bytes_per_row,
                    bytes_per_row * st.height as usize,
                );
            }
            blit.endEncoding();
            Some(buf)
        } else {
            None
        };

        cmd.presentDrawable(ProtocolObject::from_ref(&*drawable));
        cmd.commit();
        // The sink is the pipeline tail: wait so back-pressure is real and the
        // readback buffer is complete.
        cmd.waitUntilCompleted();

        if let Some(buf) = read_buf {
            let len = (st.width * st.height * 4) as usize;
            // SAFETY: shared-storage buffer of exactly `len` bytes, GPU work
            // completed above.
            let bytes =
                unsafe { core::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len) };
            // Drawables are BGRA; swizzle to RGBA for the accessor.
            let mut rgba = bytes.to_vec();
            for px in rgba.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            self.last_rgba = Some(rgba);
        }
        self.presented += 1;
        Ok(())
    }

    /// Import a decoded `CVPixelBuffer`'s IOSurface planes as textures
    /// (zero-copy) and present them.
    fn present_cv(&mut self, buf: &g2g_core::OwnedCvPixelBuffer) -> Result<(), G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        if (buf.width, buf.height) != (st.width, st.height) {
            return Err(G2gError::CapsMismatch);
        }
        // SAFETY: the frame's keep-alive pins the CVPixelBufferRef for this call.
        let pb = unsafe { &*(buf.pixel_buffer as *const CVPixelBuffer) };
        let surface = CVPixelBufferGetIOSurface(Some(pb)).ok_or_else(Self::hw)?;
        let y = iosurface_plane_texture(
            &st.device,
            &surface,
            MTLPixelFormat::R8Unorm,
            st.width,
            st.height,
            0,
        )?;
        let cbcr = iosurface_plane_texture(
            &st.device,
            &surface,
            MTLPixelFormat::RG8Unorm,
            st.width / 2,
            st.height / 2,
            1,
        )?;
        self.present(&y, &cbcr)
    }

    /// Stage packed NV12 bytes into the texture pair and present them.
    fn present_system(&mut self, nv12: &[u8]) -> Result<(), G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        let (w, h) = (st.width as usize, st.height as usize);
        if nv12.len() < w * h * 3 / 2 {
            return Err(G2gError::CapsMismatch);
        }
        let region_y = region(w, h);
        let region_c = region(w / 2, h / 2);
        // SAFETY: the slices cover the regions at the given strides.
        unsafe {
            st.staging_y
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region_y,
                    0,
                    NonNull::new(nv12.as_ptr() as *mut c_void).ok_or_else(Self::hw)?,
                    w,
                );
            st.staging_cbcr
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region_c,
                    0,
                    NonNull::new(nv12[w * h..].as_ptr() as *mut c_void).ok_or_else(Self::hw)?,
                    w,
                );
        }
        let (y, cbcr) = (st.staging_y.clone(), st.staging_cbcr.clone());
        self.present(&y, &cbcr)
    }
}

fn region(w: usize, h: usize) -> objc2_metal::MTLRegion {
    objc2_metal::MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
    }
}

fn make_plane_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    format: MTLPixelFormat,
    w: u32,
    h: u32,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, G2gError> {
    // SAFETY: plain descriptor construction.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, w as usize, h as usize, false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    desc.setStorageMode(MTLStorageMode::Shared);
    device
        .newTextureWithDescriptor(&desc)
        .ok_or(G2gError::Hardware(HardwareError::Other))
}

fn iosurface_plane_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    surface: &CFRetained<objc2_io_surface::IOSurfaceRef>,
    format: MTLPixelFormat,
    w: u32,
    h: u32,
    plane: usize,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, G2gError> {
    // SAFETY: plain descriptor construction.
    let desc = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            format, w as usize, h as usize, false,
        )
    };
    desc.setUsage(MTLTextureUsage::ShaderRead);
    // The IOSurface is retained and its plane geometry matches the descriptor
    // (the decoder produced it at these dims); the binding is safe.
    device
        .newTextureWithDescriptor_iosurface_plane(&desc, surface, plane)
        .ok_or(G2gError::Hardware(HardwareError::Other))
}

impl AsyncElement for MetalVideoSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        })
    }

    /// Accepts packed System NV12 or the M735 zero-copy CvPixelBuffer domain.
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::CvPixelBuffer)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } if *w % 2 == 0 && *h % 2 == 0 && *w > 0 && *h > 0 => (*w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        self.build_state(w, h)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
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
                PipelinePacket::DataFrame(frame) => match &frame.domain {
                    MemoryDomain::System(slice) => self.present_system(slice.as_slice())?,
                    MemoryDomain::CvPixelBuffer(buf) => self.present_cv(buf)?,
                    _ => return Err(G2gError::UnsupportedDomain),
                },
                PipelinePacket::CapsChanged(c) => {
                    // A geometry change rebuilds the layer + textures; anything
                    // that is not our NV12 input shape is rejected loud. (A
                    // sink has no downstream, so there is no pre-fixed output
                    // caps case here.)
                    match &c {
                        Caps::RawVideo {
                            format: RawVideoFormat::Nv12,
                            width: Dim::Fixed(w),
                            height: Dim::Fixed(h),
                            ..
                        } if *w % 2 == 0 && *h % 2 == 0 && *w > 0 && *h > 0 => {
                            let dims_changed = self
                                .state
                                .as_ref()
                                .is_some_and(|st| (st.width, st.height) != (*w, *h));
                            if dims_changed {
                                self.build_state(*w, *h)?;
                            }
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Eos | PipelinePacket::Flush => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for MetalVideoSink {
    /// Sink-only: NV12 at any geometry (domain is not in caps).
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}
