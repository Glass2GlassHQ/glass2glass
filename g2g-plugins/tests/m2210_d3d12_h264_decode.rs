// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "d3d12-decode"))]

#[path = "common/annexb.rs"]
mod annexb;
mod decode_ref;
#[path = "common/h264_hardware_decode.rs"]
mod h264_hardware_decode;

#[cfg(feature = "wgpu-sink")]
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::block_on;
use g2g_core::MemoryDomain;
#[cfg(feature = "wgpu-sink")]
use g2g_core::{AsyncElement, PipelinePacket};
use g2g_plugins::d3d12videodec::{
    open_d3d12_decode_device, D3D12Backend, D3D12H264Dec, D3D12H264Decoder, D3D12VideoError,
};
use g2g_plugins::gpu::{
    read_texture_plane, texture_holds_picture, texture_layout, whole_texture, WgpuNv12Texture,
};
use g2g_plugins::h264parse::extract_h264_parameter_sets;
#[cfg(feature = "wgpu-sink")]
use g2g_plugins::wgpusink::WgpuSink;
#[cfg(feature = "wgpu-sink")]
use h264_hardware_decode::Collect;
use h264_hardware_decode::{
    decode, h264_caps, system_frames, Fixture, BFRAMES, BOTTOM_PADDED, CROPPED_TOP_LEFT,
};

// the render target quantizes the shader's float result differently per NV12 layout
#[cfg(feature = "wgpu-sink")]
const MAX_CHANNEL_DELTA: i32 = 1;

// libtest runs these in parallel and concurrent device creation crashes some drivers
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn skip_reason() -> Option<String> {
    let device = match block_on(open_d3d12_decode_device()) {
        Ok(device) => device,
        Err(D3D12VideoError::NoD3D12Adapter) => {
            return Some("no D3D12 adapter with NV12 textures".into())
        }
        Err(error) => panic!("open D3D12 decode device: {error:?}"),
    };
    let sets = extract_h264_parameter_sets(BFRAMES.stream).expect("parse SPS and PPS");
    match D3D12H264Decoder::new(device, &sets.sps) {
        Ok(_) => None,
        Err(D3D12VideoError::DecodeUnsupported(missing)) => {
            Some(format!("the adapter has no {missing}"))
        }
        Err(error) => panic!("build D3D12 H.264 decoder: {error:?}"),
    }
}

fn on_a_d3d12_gpu(test: fn()) {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    test();
}

#[test]
fn system_output_is_bit_exact_in_display_order() {
    on_a_d3d12_gpu(
        h264_hardware_decode::system_output_is_bit_exact_in_display_order::<D3D12Backend>,
    );
}

#[test]
fn latency_counts_the_pictures_the_reorder_holds() {
    on_a_d3d12_gpu(
        h264_hardware_decode::latency_counts_the_pictures_the_reorder_holds::<D3D12Backend>,
    );
}

#[test]
fn a_flush_resumes_bit_exact_from_the_next_keyframe() {
    on_a_d3d12_gpu(
        h264_hardware_decode::a_flush_resumes_bit_exact_from_the_next_keyframe::<D3D12Backend>,
    );
}

#[test]
fn high_bit_depth_stream_is_refused() {
    on_a_d3d12_gpu(h264_hardware_decode::high_bit_depth_stream_is_refused::<D3D12Backend>);
}

#[test]
fn a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one() {
    on_a_d3d12_gpu(h264_hardware_decode::a_damaged_access_unit_drops_pictures_to_the_next_keyframe_and_asks_for_one::<D3D12Backend>);
}

#[test]
fn a_flush_resumes_bit_exact_from_a_recovery_point_i_picture() {
    on_a_d3d12_gpu(
        h264_hardware_decode::a_flush_resumes_bit_exact_from_a_recovery_point_i_picture::<
            D3D12Backend,
        >,
    );
}

fn texture_sizes_matching_system_output(fixture: &Fixture) -> Vec<(u32, u32)> {
    let expected = system_frames::<D3D12Backend>(fixture);
    let (_decoder, collect, decoded) =
        decode::<D3D12Backend>(fixture, MemoryDomainKind::WgpuTexture);
    decoded.expect("decode stream");
    let frames = collect.into_frames();
    assert_eq!(frames.len(), expected.len());
    let (width, height) = fixture.picture;
    let mut sizes = Vec::new();
    for (index, (frame, expected)) in frames.iter().zip(&expected).enumerate() {
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!(
                "frame {index}: expected a WgpuTexture frame, got {:?}",
                frame.domain
            );
        };
        assert_eq!((owned.width, owned.height), fixture.picture);
        let keep = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<WgpuNv12Texture>()
            .expect("the keep-alive is a WgpuNv12Texture");
        let texture = keep.texture();
        let layout = texture_layout(texture).expect("an NV12 layout");
        assert!(texture_holds_picture(texture, layout, width, height));
        let read = |aspect, region, bytes_per_texel| {
            read_texture_plane(
                keep.device(),
                keep.queue(),
                texture,
                aspect,
                region,
                bytes_per_texel,
            )
            .expect("read plane")
        };
        let mut bytes = read(wgpu::TextureAspect::Plane0, whole_texture(width, height), 1);
        bytes.extend(read(
            wgpu::TextureAspect::Plane1,
            whole_texture(width.div_ceil(2), height.div_ceil(2)),
            2,
        ));
        assert!(
            &bytes == expected,
            "{} frame {index}: texture differs from the system output",
            fixture.name
        );
        sizes.push((texture.width(), texture.height()));
    }
    sizes
}

#[test]
fn texture_output_shares_the_padded_texture_without_a_copy() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let (_, height) = BOTTOM_PADDED.picture;
    for (index, (_, texture_height)) in texture_sizes_matching_system_output(&BOTTOM_PADDED)
        .into_iter()
        .enumerate()
    {
        assert!(
            texture_height > height,
            "frame {index}: the texture is the padded decode target"
        );
    }
}

#[test]
fn texture_output_copies_a_picture_cropped_from_the_top_left() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let (width, height) = CROPPED_TOP_LEFT.picture;
    for size in texture_sizes_matching_system_output(&CROPPED_TOP_LEFT) {
        assert_eq!(
            size,
            (width.next_multiple_of(2), height.next_multiple_of(2))
        );
    }
}

#[cfg(feature = "wgpu-sink")]
fn render(sink: &mut WgpuSink, frame: Frame) -> Vec<u8> {
    block_on(sink.process(PipelinePacket::DataFrame(frame), &mut Collect::default()))
        .expect("present");
    sink.read_target().expect("read offscreen target")
}

#[cfg(feature = "wgpu-sink")]
#[test]
fn wgpusink_renders_the_decoded_textures_on_the_decode_device() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    for fixture in [BOTTOM_PADDED, CROPPED_TOP_LEFT] {
        render_matches_upload(&fixture);
    }
}

#[cfg(feature = "wgpu-sink")]
fn render_matches_upload(fixture: &Fixture) {
    let (_, system, decoded) = decode::<D3D12Backend>(fixture, MemoryDomainKind::System);
    decoded.expect("decode to system memory");
    let (decoder, textures, decoded) =
        decode::<D3D12Backend>(fixture, MemoryDomainKind::WgpuTexture);
    decoded.expect("decode to textures");
    let context = decoder.gpu_context().expect("device open").clone();
    let (width, height) = fixture.picture;
    let caps = textures.caps_changes()[0].clone();
    let mut sink = WgpuSink::offscreen(context, width, height);
    sink.configure_pipeline(&caps)
        .expect("the sink takes the decoder's caps");
    let system_frames = system.into_frames();
    let texture_frames = textures.into_frames();
    assert_eq!(system_frames.len(), texture_frames.len());
    for (index, (system_frame, texture_frame)) in
        system_frames.into_iter().zip(texture_frames).enumerate()
    {
        let uploaded = render(&mut sink, system_frame);
        let shared = render(&mut sink, texture_frame);
        let delta = uploaded
            .iter()
            .zip(&shared)
            .map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs())
            .max()
            .expect("a rendered picture");
        assert!(
            delta <= MAX_CHANNEL_DELTA,
            "{} frame {index}: the texture render differs from the upload render by {delta}",
            fixture.name
        );
    }
}

#[test]
fn decodebin_offers_it_for_h264_on_a_d3d12_gpu() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    assert!(D3D12H264Dec::accepts_input(&h264_caps(BFRAMES.picture)));
}

#[test]
fn decode_order_timestamps_leave_in_rising_order() {
    on_a_d3d12_gpu(
        h264_hardware_decode::decode_order_timestamps_leave_in_rising_order::<D3D12Backend>,
    );
}
