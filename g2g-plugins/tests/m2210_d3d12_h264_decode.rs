// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "d3d12-decode"))]

mod decode_ref;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, BitstreamAlignment, Caps, Colorimetry, Dim, FrameTiming,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, PixelAspect, PushOutcome, Rate,
    RawVideoFormat, VideoCodec,
};
use g2g_plugins::d3d12videodec::{
    open_d3d12_decode_device, D3D12H264Dec, D3D12H264Decoder, D3D12VideoError,
};
use g2g_plugins::gpu::{
    read_texture_plane, texture_holds_picture, texture_layout, whole_texture, WgpuNv12Texture,
};
use g2g_plugins::h264parse::extract_h264_parameter_sets;
#[cfg(feature = "wgpu-sink")]
use g2g_plugins::wgpusink::WgpuSink;

struct Fixture {
    name: &'static str,
    stream: &'static [u8],
    picture: (u32, u32),
}

const BFRAMES: Fixture = Fixture {
    name: "h264_640x480_bframes.h264",
    stream: include_bytes!("fixtures/h264_640x480_bframes.h264"),
    picture: (640, 480),
};
// decodes into a texture taller than the picture
const BOTTOM_PADDED: Fixture = Fixture {
    name: "h264_320x180.h264",
    stream: include_bytes!("fixtures/h264_320x180.h264"),
    picture: (320, 180),
};
// crops six columns from the left and four rows from the top
const CROPPED_TOP_LEFT: Fixture = Fixture {
    name: "h264_304x180_crop.h264",
    stream: include_bytes!("fixtures/h264_304x180_crop.h264"),
    picture: (304, 180),
};
const HIGH10: Fixture = Fixture {
    name: "h264_high10_320x240.h264",
    stream: include_bytes!("fixtures/h264_high10_320x240.h264"),
    picture: (320, 240),
};
const EIGHT_BIT: u8 = 8;
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
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn into_frames(self) -> Vec<Frame> {
        self.packets
            .into_iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }

    fn caps_changes(&self) -> Vec<&Caps> {
        self.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::CapsChanged(caps) => Some(caps),
                _ => None,
            })
            .collect()
    }
}

fn h264_caps((width, height): (u32, u32)) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Any,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: PixelAspect::Any,
        alignment: BitstreamAlignment::Unspecified,
    }
}

fn system_frame(bytes: &[u8]) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

// the whole stream as one buffer, so every picture shares one timestamp
fn decode(
    fixture: &Fixture,
    domain: MemoryDomainKind,
) -> (D3D12H264Dec, Collect, Result<(), G2gError>) {
    let mut decoder = D3D12H264Dec::new();
    decoder.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain,
        accepts: DomainSet::only(domain),
        ..Default::default()
    });
    decoder
        .configure_pipeline(&h264_caps(fixture.picture))
        .expect("configure opens the decode device");
    let mut collect = Collect::default();
    let decoded = block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(fixture.stream)),
        &mut collect,
    ));
    if decoded.is_ok() {
        block_on(decoder.process(PipelinePacket::Eos, &mut collect))
            .expect("eos drains the reorder");
    }
    (decoder, collect, decoded)
}

fn system_frames(fixture: &Fixture) -> Vec<Vec<u8>> {
    let (_, collect, decoded) = decode(fixture, MemoryDomainKind::System);
    decoded.expect("decode stream");
    let (width, height) = fixture.picture;
    let caps = collect.caps_changes();
    assert!(
        matches!(
            caps.as_slice(),
            [Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(caps_width),
                height: Dim::Fixed(caps_height),
                ..
            }] if (*caps_width, *caps_height) == (width, height)
        ),
        "{}: one NV12 caps event at the visible size, got {caps:?}",
        fixture.name
    );
    collect
        .into_frames()
        .iter()
        .map(|frame| {
            frame
                .domain
                .as_system_slice()
                .expect("system frame")
                .to_vec()
        })
        .collect()
}

#[test]
fn system_output_is_bit_exact_in_display_order() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    for fixture in [BFRAMES, BOTTOM_PADDED, CROPPED_TOP_LEFT] {
        let frames: Vec<Vec<u8>> = system_frames(&fixture)
            .iter()
            .map(|nv12| decode_ref::two_plane_to_planar(nv12, EIGHT_BIT, fixture.picture))
            .collect();
        decode_ref::assert_frames_match_reference(
            &frames,
            RawVideoFormat::I420,
            fixture.picture,
            fixture.name,
        );
    }
}

#[test]
fn a_flush_resumes_bit_exact_from_the_next_keyframe() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let (mut decoder, mut collect, decoded) = decode(&BFRAMES, MemoryDomainKind::System);
    decoded.expect("decode stream");
    block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(BFRAMES.stream)),
        &mut collect,
    ))
    .expect("decode the stream again before the seek");
    block_on(decoder.process(PipelinePacket::Flush, &mut collect)).expect("flush");
    let flushed_at = collect.packets.len();
    block_on(decoder.process(
        PipelinePacket::DataFrame(system_frame(BFRAMES.stream)),
        &mut collect,
    ))
    .expect("decode after the flush");
    block_on(decoder.process(PipelinePacket::Eos, &mut collect)).expect("eos drains the reorder");
    let frames: Vec<Vec<u8>> = collect.packets[flushed_at..]
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice(),
            _ => None,
        })
        .map(|nv12| decode_ref::two_plane_to_planar(nv12, EIGHT_BIT, BFRAMES.picture))
        .collect();
    decode_ref::assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420,
        BFRAMES.picture,
        BFRAMES.name,
    );
}

fn texture_sizes_matching_system_output(fixture: &Fixture) -> Vec<(u32, u32)> {
    let expected = system_frames(fixture);
    let (_decoder, collect, decoded) = decode(fixture, MemoryDomainKind::WgpuTexture);
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

#[test]
fn high_bit_depth_stream_is_refused() {
    let _gpu = gpu_lock();
    if let Some(reason) = skip_reason() {
        eprintln!("skipping: {reason}");
        return;
    }
    let (_, collect, decoded) = decode(&HIGH10, MemoryDomainKind::System);
    assert_eq!(decoded, Err(G2gError::CapsMismatch));
    assert!(collect.into_frames().is_empty());
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
    let (_, system, decoded) = decode(fixture, MemoryDomainKind::System);
    decoded.expect("decode to system memory");
    let (decoder, textures, decoded) = decode(fixture, MemoryDomainKind::WgpuTexture);
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
