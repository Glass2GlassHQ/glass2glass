#![cfg(feature = "wgpu")]
//! §5.1: `WgpuPreprocess` converts an NV12 frame to a normalized f32 NCHW RGB
//! tensor in a wgpu compute shader. The test runs a known NV12 frame (distinct
//! luma, one neutral and one coloured chroma block) through the element on the
//! real GPU and asserts the read-back tensor matches the host BT.601 reference
//! within float tolerance, with the tensor caps emitted once. Skips when no
//! wgpu adapter is present, like the other hardware-gated elements.

use g2g_core::element::PushOutcome;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, Rate, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::wgpupreprocess::{
    gpu_available, nv12_to_gpu_texture, nv12_to_rgb_tensor, nv12_to_rgb_tensor_tagged,
    WgpuBufferOwner, WgpuNv12Texture, WgpuPreprocess,
};

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame_f32(f: &Frame) -> Vec<f32> {
    let Some(slice) = f.domain.as_system_slice() else {
        panic!("tensor frame must be System memory");
    };
    slice
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn nv12_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

// parallel per-test device creation intermittently segfaults in the NVIDIA
// driver (the recorded wgpu gotcha), so the GPU tests take one lock for their
// whole body.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn gpu_nv12_to_rgb_tensor_matches_cpu_reference() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }

    let (w, h) = (4usize, 2usize);
    // Y plane (w*h), then interleaved Cb,Cr per 2x2 block (two blocks).
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200]; // block 0 neutral, block 1 coloured
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();
    assert_eq!(nv12.len(), w * h * 3 / 2);

    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12 geometry");

    let mut out = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 4242, 7)),
            &mut out,
        )
        .await
        .expect("gpu preprocess frame 1");
    // a second frame: caps must not re-emit.
    element
        .process(
            PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 9001, 8)),
            &mut out,
        )
        .await
        .expect("gpu preprocess frame 2");

    let caps_changes: Vec<&Caps> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::CapsChanged(c) => Some(c),
            _ => None,
        })
        .collect();
    let frames: Vec<&Frame> = out
        .packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect();

    assert_eq!(caps_changes.len(), 1, "tensor caps emitted exactly once");
    assert_eq!(
        *caps_changes[0],
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, h as u32, w as u32]),
            layout: TensorLayout::Nchw,
        }
    );
    assert_eq!(frames.len(), 2);

    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    let got = frame_f32(frames[0]);
    assert_eq!(got.len(), 3 * w * h, "NCHW RGB tensor length");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: gpu {g} vs cpu reference {e}"
        );
    }
    // a coloured chroma block must produce non-equal R/G/B somewhere (proves
    // the chroma path ran, not just luma).
    let area = w * h;
    let colored = (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3);
    assert!(colored, "coloured block should break the grayscale R==G==B");

    // timing is inherited from the source frame.
    assert_eq!(frames[0].timing.pts_ns, 4242);
    assert_eq!(frames[1].timing.pts_ns, 9001);
    assert_eq!(element.emitted(), 2);
}

/// M215: in GPU-output mode the tensor stays GPU-resident
/// (`MemoryDomain::WgpuBuffer`, no read-back in the element), and reading it back
/// off-element yields the same values as the system-memory variant.
#[tokio::test]
async fn gpu_output_keeps_tensor_on_device_and_matches_reference() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let (w, h) = (4usize, 2usize);
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200];
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();

    let mut element = WgpuPreprocess::new().with_gpu_output();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12");

    let mut out = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 4242, 7)),
            &mut out,
        )
        .await
        .expect("gpu-output preprocess");

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a tensor frame");

    // The tensor is GPU-resident: NOT a System read-back.
    let MemoryDomain::WgpuBuffer(owned) = &frame.domain else {
        panic!(
            "gpu-output mode must emit MemoryDomain::WgpuBuffer, got {:?}",
            frame.domain.kind()
        );
    };
    assert_eq!(owned.len, 3 * w * h * 4, "buffer holds the f32 NCHW tensor");

    // Recover the owner and read the buffer back: the deferred GPU->CPU copy a
    // CPU consumer pays, which the element no longer pays per frame.
    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("recover the wgpu buffer owner");
    let bytes = owner.read_back().expect("read tensor back");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    assert_eq!(got.len(), 3 * w * h, "NCHW RGB tensor length");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: gpu-resident {g} vs cpu reference {e}"
        );
    }
}

/// M217: surface-import. The NV12 frame arrives already on the GPU as a
/// `MemoryDomain::WgpuTexture`; the element samples it straight into the compute
/// pass (no CPU upload) and the result matches the system-memory / BT.601 CPU
/// reference exactly. Reuses the same NV12 frame as the system-input test so the
/// two paths are proven to produce identical output.
#[tokio::test]
async fn surface_import_samples_gpu_texture_and_matches_reference() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let (w, h) = (4usize, 2usize);
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200];
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();

    // Stand in for a GPU NV12 decoder: the frame is GPU-resident before it ever
    // reaches the element.
    let domain = nv12_to_gpu_texture(&nv12, w as u32, h as u32)
        .await
        .expect("upload nv12 texture");
    assert!(
        matches!(domain, MemoryDomain::WgpuTexture(_)),
        "input is a GPU texture, not System"
    );

    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12");

    let mut out = Collect::default();
    let frame = Frame {
        domain,
        timing: FrameTiming {
            pts_ns: 555,
            dts_ns: 555,
            ..FrameTiming::default()
        },
        sequence: 3,
        meta: Default::default(),
    };
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("surface-import");

    let tensor = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a tensor frame");
    let got = frame_f32(tensor);

    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    assert_eq!(got.len(), 3 * w * h, "NCHW RGB tensor length");
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: surface-import {g} vs cpu reference {e}"
        );
    }
    // chroma path ran (coloured block breaks grayscale R==G==B).
    let area = w * h;
    assert!(
        (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3),
        "coloured block should break grayscale"
    );
    assert_eq!(
        tensor.timing.pts_ns, 555,
        "timing flows through surface-import"
    );
}

/// M217 + M215: surface-import in **and** GPU-resident tensor out, so the
/// preprocess stage touches the CPU at neither end. Read the result back only at
/// the end and compare to the CPU reference.
#[tokio::test]
async fn surface_import_with_gpu_output_stays_resident() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let (w, h) = (4usize, 2usize);
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [128u8, 128, 90, 200];
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();

    let domain = nv12_to_gpu_texture(&nv12, w as u32, h as u32)
        .await
        .expect("upload nv12 texture");
    let mut element = WgpuPreprocess::new().with_gpu_output();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12");

    let mut out = Collect::default();
    let frame = Frame {
        domain,
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    element
        .process(PipelinePacket::DataFrame(frame), &mut out)
        .await
        .expect("surface-import gpu");

    let tensor = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a tensor frame");
    let MemoryDomain::WgpuBuffer(owned) = &tensor.domain else {
        panic!(
            "texture in + gpu_output must stay resident, got {:?}",
            tensor.domain.kind()
        );
    };
    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("recover the buffer owner");
    let bytes = owner.read_back().expect("read tensor back");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();

    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    assert_eq!(got.len(), 3 * w * h);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: resident {g} vs cpu reference {e}"
        );
    }
}

/// M1127: the shader takes its matrix from the caps colorimetry, not a baked-in
/// constant. The same NV12 bytes tagged BT.709 must land on the BT.709 host
/// reference and away from the BT.601 one.
#[tokio::test]
async fn gpu_matrix_follows_the_caps_colorimetry() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }

    let (w, h) = (4usize, 2usize);
    // Off-centre chroma in both blocks: neutral chroma converts identically
    // under every matrix, so it would prove nothing here.
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [90u8, 200, 170, 60];
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();

    let mut caps = nv12_caps(w as u32, h as u32);
    let Caps::RawVideo {
        ref mut colorimetry,
        ..
    } = caps
    else {
        panic!("raw video caps");
    };
    *colorimetry = g2g_core::Colorimetry::BT709;

    let mut element = WgpuPreprocess::new();
    element.configure_pipeline(&caps).expect("configure BT.709");
    let mut out = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(nv12_frame(nv12.clone(), 0, 0)),
            &mut out,
        )
        .await
        .expect("gpu preprocess a tagged frame");

    let frame = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("one tensor out");
    let got = frame_f32(frame);
    let expected = nv12_to_rgb_tensor_tagged(&nv12, w, h, g2g_core::Colorimetry::BT709);
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "element {i}: gpu {g} vs BT.709 reference {e}"
        );
    }
    // And the BT.601 reference is a different picture, so the tag reached the
    // shader rather than the two references coinciding.
    let bt601 = nv12_to_rgb_tensor(&nv12, w, h);
    let apart = got
        .iter()
        .zip(&bt601)
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max);
    assert!(apart > 1e-2, "BT.601 and BT.709 differ by only {apart}");
}

/// Bytes per texel of an NV12 chroma plane, Cb and Cr interleaved.
const CHROMA_BYTES_PER_TEXEL: u32 = 2;

/// The two-plane counterpart of `nv12_to_gpu_texture`: the same NV12 bytes in a
/// `TextureFormat::NV12` texture of `width x height`, the layout a GPU decoder
/// hands out. `None` when the adapter cannot make one.
async fn nv12_to_two_plane_texture(nv12: &[u8], width: u32, height: u32) -> Option<MemoryDomain> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .ok()?;
    if !adapter
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        return None;
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("two-plane-nv12"),
            required_features: wgpu::Features::TEXTURE_FORMAT_NV12,
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .ok()?;

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("two-plane-nv12"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::NV12,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // Each plane is written on its own aspect: a whole-texture write has no
    // single texel size to describe.
    let luma_bytes = (width * height) as usize;
    for (aspect, bytes, plane_width, plane_height, bytes_per_row) in [
        (
            wgpu::TextureAspect::Plane0,
            &nv12[..luma_bytes],
            width,
            height,
            width,
        ),
        (
            wgpu::TextureAspect::Plane1,
            &nv12[luma_bytes..],
            width / 2,
            height / 2,
            (width / 2) * CHROMA_BYTES_PER_TEXEL,
        ),
    ] {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(plane_height),
            },
            wgpu::Extent3d {
                width: plane_width,
                height: plane_height,
                depth_or_array_layers: 1,
            },
        );
    }

    let owner = WgpuNv12Texture::new(device, queue, texture);
    Some(MemoryDomain::WgpuTexture(g2g_core::OwnedWgpuTexture::new(
        width,
        height,
        std::sync::Arc::new(owner),
    )))
}

/// The tensor `element` produces for one GPU-texture frame.
async fn preprocess_texture(element: &mut WgpuPreprocess, domain: MemoryDomain) -> Vec<f32> {
    let mut out = Collect::default();
    element
        .process(
            PipelinePacket::DataFrame(Frame {
                domain,
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            }),
            &mut out,
        )
        .await
        .expect("surface-import");
    let tensor = out
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .expect("a tensor frame");
    frame_f32(tensor)
}

/// M1157: the surface import also reads the two-plane `TextureFormat::NV12` a
/// GPU decoder hands out, through its luma and chroma views. One configured
/// element takes both NV12 texture layouts and must build the same tensor from
/// the same bytes.
#[tokio::test]
async fn two_plane_nv12_texture_matches_the_packed_plane() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let (w, h) = (4usize, 2usize);
    let y_plane = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let uv_plane = [90u8, 200, 170, 60];
    let nv12: Vec<u8> = y_plane.iter().chain(&uv_plane).copied().collect();

    let Some(two_plane) = nv12_to_two_plane_texture(&nv12, w as u32, h as u32).await else {
        eprintln!("skipping: no adapter with TEXTURE_FORMAT_NV12 on this host");
        return;
    };
    let packed = nv12_to_gpu_texture(&nv12, w as u32, h as u32)
        .await
        .expect("upload the packed nv12 plane");

    let mut element = WgpuPreprocess::new();
    element
        .configure_pipeline(&nv12_caps(w as u32, h as u32))
        .expect("configure NV12");
    let from_planes = preprocess_texture(&mut element, two_plane).await;
    let from_packed = preprocess_texture(&mut element, packed).await;

    assert_eq!(from_planes.len(), 3 * w * h, "NCHW RGB tensor length");
    // The plane views hand the shader `byte / 255` where the packed R8Uint plane
    // hands it the integer, so the two agree to the float roundtrip, not bitwise.
    for (i, (planes, packed)) in from_planes.iter().zip(&from_packed).enumerate() {
        assert!(
            (planes - packed).abs() < 1e-5,
            "element {i}: two-plane {planes} vs packed {packed}"
        );
    }
    // And that tensor is the host reference, so neither GPU path is wrong in the
    // same way.
    let expected = nv12_to_rgb_tensor(&nv12, w, h);
    for (i, (got, want)) in from_planes.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "element {i}: two-plane {got} vs cpu reference {want}"
        );
    }
}
