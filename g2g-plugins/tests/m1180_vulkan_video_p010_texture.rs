//! M1180: the `VulkanVideoDec` two-plane outputs at 10 bits and for AV1 film
//! grain.
//!
//! A 10-bit stream's decode output is the Vulkan `G10X6` two-plane format, which
//! is byte-for-byte `RawVideoFormat::P010` and imports as a
//! `wgpu::TextureFormat::P010` texture. The system path announces `P010` caps for
//! it, the `WgpuTexture` path hands out a P010 texture when the caps pin it, and
//! the raw `VulkanTexture` domain hands out the `G10X6` image itself. Bit depth is
//! not known when the caps solve, so both `Nv12` and `P010` are offered on the
//! source pad and a pin the samples cannot carry (`Nv12` on a 10-bit stream,
//! `P010` on an 8-bit one) fails the decode rather than mislabelling bytes.
//!
//! AV1 film grain is synthesized on the CPU (the hardware reconstruction is
//! grain-free), so a grain frame is uploaded rather than copied; with the
//! two-plane output pinned that upload is a `TextureFormat::NV12` texture instead
//! of RGBA.
//!
//! Everything runs in ONE test function, sequentially: creating Vulkan devices in
//! parallel SIGSEGVs on this driver (see m504 / m573). Runs on the RTX 3060; each
//! part skips with a printed reason when the host lacks the decode profile, a
//! compute queue, or the wgpu two-plane texture feature.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::block_on;
use g2g_core::{Caps, Dim, G2gError, MemoryDomain, PipelinePacket, RawVideoFormat, VideoCodec};
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_av1_decode_device, open_h264_decode_device,
    open_h265_decode_device, to_std_h265_params, VulkanImageOwner, VulkanVideoError,
};

mod vulkan_nv12_common;
use vulkan_nv12_common::{
    decode_stream, raw_caps, read_two_plane, system_bytes, two_plane_frame, H, H264_CLIP, W,
};

const HEVC_10BIT: &[u8] = include_bytes!("fixtures/h265_640x480_main10.hevc");
const AV1_10BIT: &[u8] = include_bytes!("fixtures/av1_640x480_10bit.obu");
const AV1_FILMGRAIN: &[u8] = include_bytes!("fixtures/av1_640x480_filmgrain.obu");

/// The two-plane texture behind a `WgpuTexture` frame, with the device and queue
/// it lives on.
/// The wgpu features a decode device for `codec` reports on this host, or the
/// reason one could not be opened.
fn device_features(codec: VideoCodec) -> Result<wgpu::Features, &'static str> {
    let opened = match codec {
        VideoCodec::H264 => block_on(open_h264_decode_device()),
        VideoCodec::H265 => block_on(open_h265_decode_device()),
        VideoCodec::Av1 => block_on(open_av1_decode_device()),
        other => panic!("no device opener for {other:?}"),
    };
    match opened {
        Ok(dev) => Ok(dev.wgpu_device.features()),
        Err(VulkanVideoError::NoVulkanAdapter) => Err("no Vulkan adapter"),
        Err(VulkanVideoError::NoDecodeQueue) => Err("no decode queue for this codec"),
        Err(VulkanVideoError::ExtensionUnsupported) => Err("decode extensions unsupported"),
        Err(e) => panic!("unexpected device open failure: {e:?}"),
    }
}

/// The HEVC Main 10 clip decoded through the decoder API (`decode_all`), each
/// frame's planes concatenated: the reference the element's system output and both
/// texture outputs are compared against.
fn hevc_decoder_api_frames() -> Vec<Vec<u8>> {
    let ps = extract_h265_parameter_sets(HEVC_10BIT).expect("the fixture carries vps/sps/pps");
    assert_eq!(
        ps.sps.bit_depth_luma_minus8, 2,
        "fixture is not 10-bit; feature untested"
    );
    let device = block_on(open_h265_decode_device()).expect("h265 decode device");
    let std = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std, W, H).expect("session");
    let mut dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("10-bit HEVC decoder");
    dec.decode_all(HEVC_10BIT)
        .expect("decode the 10-bit stream")
        .into_iter()
        .map(|f| {
            assert_eq!(f.bit_depth, 10, "the fixture decodes 10-bit");
            let mut bytes = f.luma;
            bytes.extend_from_slice(&f.chroma);
            bytes
        })
        .collect()
}

/// The system path announces `P010` and carries the 10-bit samples the decoder API
/// produces, and a pin the bit depth cannot carry fails the decode.
fn system_p010(reference: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let (_, collect, decoded) =
        decode_stream(VideoCodec::H265, HEVC_10BIT, MemoryDomainKind::System, None);
    decoded.expect("10-bit system decode");
    let first_frame = collect
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::DataFrame(_)))
        .expect("the clip decoded at least one frame");
    let first_caps = collect
        .packets
        .iter()
        .position(|p| matches!(p, PipelinePacket::CapsChanged(_)))
        .expect("the system path announces its caps");
    assert!(
        first_caps < first_frame,
        "the P010 caps must arrive before the first frame"
    );
    let caps = collect.caps_changes();
    let Caps::RawVideo {
        format,
        width,
        height,
        ..
    } = caps[0]
    else {
        panic!("expected raw video caps, got {:?}", caps[0]);
    };
    assert_eq!(
        *format,
        RawVideoFormat::P010,
        "a 10-bit stream announces P010, not NV12"
    );
    assert_eq!((width, height), (&Dim::Fixed(W), &Dim::Fixed(H)));

    let frames = system_bytes(&collect);
    assert_eq!(
        frames.len(),
        reference.len(),
        "one frame per coded picture, same as the decoder API"
    );
    let p010_len: u64 = (0..RawVideoFormat::P010.plane_count())
        .map(|plane| {
            RawVideoFormat::P010
                .plane_bytes(plane, W, H)
                .expect("P010 plane size")
        })
        .sum();
    assert_eq!(
        p010_len,
        u64::from(W) * u64::from(H) * 3,
        "P010 holds NV12's sample counts at two bytes each"
    );
    for (i, bytes) in frames.iter().enumerate() {
        assert_eq!(
            bytes.len() as u64,
            p010_len,
            "frame {i} is a full P010 buffer"
        );
        assert!(
            bytes == &reference[i],
            "frame {i} differs from the decoder-API 10-bit decode"
        );
    }
    frames
}

/// A two-plane pin the stream's bit depth cannot carry is refused.
fn pin_bit_depth_mismatch() {
    let (_, _, decoded) = decode_stream(
        VideoCodec::H265,
        HEVC_10BIT,
        MemoryDomainKind::System,
        Some(raw_caps(RawVideoFormat::Nv12)),
    );
    assert_eq!(
        decoded,
        Err(G2gError::CapsMismatch),
        "NV12 pinned on a 10-bit stream must fail, not truncate"
    );

    let (_, _, decoded) = decode_stream(
        VideoCodec::H264,
        H264_CLIP,
        MemoryDomainKind::System,
        Some(raw_caps(RawVideoFormat::P010)),
    );
    assert_eq!(
        decoded,
        Err(G2gError::CapsMismatch),
        "P010 pinned on an 8-bit stream must fail, not mislabel"
    );
}

/// Pinning `P010` on `WgpuTexture` hands out P010 textures whose planes are the
/// system decode's bytes.
fn wgpu_p010_textures(reference: &[Vec<u8>]) {
    let (dec, collect, decoded) = decode_stream(
        VideoCodec::H265,
        HEVC_10BIT,
        MemoryDomainKind::WgpuTexture,
        Some(raw_caps(RawVideoFormat::P010)),
    );
    decoded.expect("10-bit texture decode");
    let frames = collect.frames();
    assert_eq!(frames.len(), reference.len());
    for (i, frame) in frames.iter().enumerate() {
        let owner = two_plane_frame(frame);
        assert_eq!(
            owner.texture().format(),
            wgpu::TextureFormat::P010,
            "frame {i} is not a P010 texture"
        );
        assert_eq!(
            (owner.texture().width(), owner.texture().height()),
            (W, H),
            "frame {i} dims"
        );
        let bytes = read_two_plane(owner.device(), owner.queue(), owner.texture());
        assert!(
            bytes == reference[i],
            "frame {i}: the P010 texture differs from the system P010 decode"
        );
    }
    let caps = collect.caps_changes();
    assert!(
        caps.iter().all(|c| matches!(
            c,
            Caps::RawVideo {
                format: RawVideoFormat::P010,
                ..
            }
        )),
        "every announced caps on the pinned path is P010"
    );
    drop(dec);
}

/// The `VulkanTexture` domain hands out the raw `G10X6` image under `P010` caps.
fn vulkan_texture_g10x6(reference: &[Vec<u8>]) {
    let (dec, collect, decoded) = decode_stream(
        VideoCodec::H265,
        HEVC_10BIT,
        MemoryDomainKind::VulkanTexture,
        None,
    );
    decoded.expect("10-bit raw-image decode");
    let ctx = dec.gpu_context().expect("device open");
    let frames = collect.frames();
    assert_eq!(frames.len(), reference.len());
    for (i, frame) in frames.iter().enumerate() {
        let MemoryDomain::VulkanTexture(owned) = &frame.domain else {
            panic!("frame {i}: expected a VulkanTexture frame");
        };
        assert_eq!((owned.width, owned.height), (W, H));
        assert_eq!(
            owned.format,
            ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16.as_raw(),
            "frame {i}: the raw image is the decoder's 10-bit two-plane format"
        );
        let owner = owned
            .keep_alive()
            .as_any()
            .downcast_ref::<VulkanImageOwner>()
            .expect("the keep-alive is the decoder's VulkanImageOwner");
        let bytes = read_two_plane(&ctx.device, &ctx.queue, owner.texture());
        assert!(
            bytes == reference[i],
            "frame {i}: the raw G10X6 image differs from the system P010 decode"
        );
    }
    let caps = collect.caps_changes();
    assert!(
        caps.iter().all(|c| matches!(
            c,
            Caps::RawVideo {
                format: RawVideoFormat::P010,
                ..
            }
        )),
        "the VulkanTexture domain announces P010 for a 10-bit stream"
    );
}

/// Pinning `Nv12` on `WgpuTexture` for a film-grain AV1 clip uploads the grained
/// planes into a `TextureFormat::NV12` texture instead of converting to RGBA.
fn av1_filmgrain_two_plane() {
    let (_, collect, decoded) = decode_stream(
        VideoCodec::Av1,
        AV1_FILMGRAIN,
        MemoryDomainKind::System,
        None,
    );
    decoded.expect("film-grain system decode");
    let reference = system_bytes(&collect);
    assert!(!reference.is_empty(), "the grain clip decoded no frames");

    let (_, collect, decoded) = decode_stream(
        VideoCodec::Av1,
        AV1_FILMGRAIN,
        MemoryDomainKind::WgpuTexture,
        Some(raw_caps(RawVideoFormat::Nv12)),
    );
    decoded.expect("film-grain texture decode");
    let frames = collect.frames();
    assert_eq!(frames.len(), reference.len());
    for (i, frame) in frames.iter().enumerate() {
        let owner = two_plane_frame(frame);
        assert_eq!(
            owner.texture().format(),
            wgpu::TextureFormat::NV12,
            "frame {i}: a grain frame with NV12 pinned is not an RGBA texture"
        );
        let bytes = read_two_plane(owner.device(), owner.queue(), owner.texture());
        assert!(
            bytes == reference[i],
            "frame {i}: the grained NV12 texture differs from the grained system decode"
        );
    }
}

/// A 10-bit AV1 clip with `P010` pinned produces P010 textures of the right size.
/// Content is not asserted: this driver's AV1 decode is run-to-run
/// nondeterministic (see m573).
fn av1_10bit_p010_textures() {
    let (_, collect, decoded) = decode_stream(
        VideoCodec::Av1,
        AV1_10BIT,
        MemoryDomainKind::WgpuTexture,
        Some(raw_caps(RawVideoFormat::P010)),
    );
    decoded.expect("10-bit AV1 texture decode");
    let frames = collect.frames();
    assert!(!frames.is_empty(), "the 10-bit AV1 clip decoded no frames");
    for (i, frame) in frames.iter().enumerate() {
        let owner = two_plane_frame(frame);
        assert_eq!(
            owner.texture().format(),
            wgpu::TextureFormat::P010,
            "frame {i} is not a P010 texture"
        );
        assert_eq!(
            (owner.texture().width(), owner.texture().height()),
            (W, H),
            "frame {i} dims"
        );
    }
}

#[test]
fn two_plane_output_covers_10bit_and_film_grain() {
    match device_features(VideoCodec::H265) {
        Err(reason) => eprintln!("skipping the HEVC Main 10 parts: {reason}"),
        Ok(features) => {
            let reference = hevc_decoder_api_frames();
            let system = system_p010(&reference);
            pin_bit_depth_mismatch();
            if features.contains(wgpu::Features::TEXTURE_FORMAT_P010) {
                wgpu_p010_textures(&system);
                vulkan_texture_g10x6(&system);
            } else {
                eprintln!("skipping the P010 texture parts: device lacks TEXTURE_FORMAT_P010");
            }
        }
    }

    match device_features(VideoCodec::Av1) {
        Err(reason) => eprintln!("skipping the AV1 parts: {reason}"),
        Ok(features) => {
            if features.contains(wgpu::Features::TEXTURE_FORMAT_NV12) {
                av1_filmgrain_two_plane();
            } else {
                eprintln!("skipping the film-grain part: device lacks TEXTURE_FORMAT_NV12");
            }
            if features.contains(wgpu::Features::TEXTURE_FORMAT_P010) {
                av1_10bit_p010_textures();
            } else {
                eprintln!("skipping the 10-bit AV1 part: device lacks TEXTURE_FORMAT_P010");
            }
        }
    }
}
