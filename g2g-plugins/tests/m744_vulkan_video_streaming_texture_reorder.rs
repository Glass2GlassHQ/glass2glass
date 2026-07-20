//! M744: `VulkanVideoDec` streaming display order for AV1 and for the
//! GPU-texture path.
//!
//! Two gaps closed. (1) The element's AV1 system path used the pipelined
//! `decode_push`, which walks only coded frames: an alt-ref stream's
//! `show_existing_frame` displays never emitted and film grain never
//! synthesized. It now op-walks each AU (`decode_all` on the temporal unit, the
//! DPB persisting across calls), which IS display order for AV1 (its display
//! order is the bitstream's op order, not a POC sort). (2) The GPU-texture path
//! called `decode_all_to_textures` per AU, whose POC indexing resets decoder
//! state, so it only worked fed whole-stream; H.264 / H.265 now stream through
//! `decode_push_to_textures` (state intact across AUs) plus a texture
//! `ReorderBuffer`, the texture analog of the M586 system path.
//!
//! Oracles: the whole-stream `decode_all` / `decode_all_to_textures` on a fresh
//! decoder (display order, validated bit-exact vs software decoders in
//! M565/M569). Deterministic decode + convert, so a correct element reproduces
//! them byte-for-byte.
//!
//! Runs on the RTX 3060; skips with no adapter / no decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use std::future::Future;
use std::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, DomainSet, FrameTiming, G2gError, MemoryDomain,
    MemoryDomainKind, OutputSink, PipelinePacket, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::gpu::{read_rgba_texture, texture_of};
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, extract_h264_parameter_sets, open_av1_decode_device,
    open_h264_decode_device, to_std_av1_seq_header, Nv12Frame, VulkanVideoDec, VulkanVideoError,
};

const H264: &[u8] = include_bytes!("fixtures/h264_640x480_bframes.h264");
const AV1: &[u8] = include_bytes!("fixtures/av1_640x480_showexisting.obu");

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Byte offsets of each Annex-B NAL payload (just past its start code).
fn start_code_offsets(data: &[u8]) -> Vec<usize> {
    let mut offs = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                offs.push(i + 3);
                i += 3;
                continue;
            }
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                offs.push(i + 4);
                i += 4;
                continue;
            }
        }
        i += 1;
    }
    offs
}

/// Split an Annex-B stream into per-picture access units (see M586).
fn split_access_units(stream: &[u8], is_vcl: impl Fn(&[u8]) -> bool) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        if is_vcl(nal) {
            units.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
}

/// `(start, payload_start, end, obu_type)` of every OBU in a low-overhead AV1
/// stream (payload = the bytes after the header + leb128 size field).
fn obu_bounds(stream: &[u8]) -> Vec<(usize, usize, usize, u8)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < stream.len() {
        let start = i;
        let hdr = stream[i];
        let obu_type = (hdr >> 3) & 0xF;
        let ext = (hdr >> 2) & 1;
        let has_size = (hdr >> 1) & 1;
        assert_eq!(has_size, 1, "fixture OBUs carry size fields");
        i += 1 + ext as usize;
        let mut size = 0usize;
        let mut shift = 0;
        loop {
            let b = stream[i];
            i += 1;
            size |= ((b & 0x7f) as usize) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let payload = i;
        i += size;
        out.push((start, payload, i, obu_type));
    }
    out
}

/// Split an AV1 stream into per-op temporal units: each frame-carrying OBU
/// (`OBU_FRAME` = 6 or a `show_existing` `OBU_FRAME_HEADER` = 3) closes a unit,
/// carrying any preceding TD / sequence-header / metadata OBUs with it.
fn split_temporal_units(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    for (s, _, e, t) in obu_bounds(stream) {
        cur.extend_from_slice(&stream[s..e]);
        if t == 6 || t == 3 {
            units.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        if let Some(last) = units.last_mut() {
            last.extend_from_slice(&cur);
        }
    }
    units
}

fn au_frame(bytes: Vec<u8>, seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: seq * 33_000_000,
            ..Default::default()
        },
        sequence: seq,
        meta: Default::default(),
    }
}

fn planes(f: &Nv12Frame) -> Vec<u8> {
    let mut v = Vec::with_capacity(f.luma.len() + f.chroma.len());
    v.extend_from_slice(&f.luma);
    v.extend_from_slice(&f.chroma);
    v
}

/// Drive the element AU-by-AU. `gpu == true` resolves the output domain to
/// `WgpuTexture` first (the zero-copy path); the returned buffers are then the
/// RGBA texture read-backs, else the NV12 system buffers.
fn element_streaming_output(codec: VideoCodec, aus: Vec<Vec<u8>>, gpu: bool) -> Vec<Vec<u8>> {
    let mut dec = VulkanVideoDec::new();
    if gpu {
        dec.configure_allocation(&AllocationParams {
            domain: MemoryDomainKind::WgpuTexture,
            accepts: DomainSet::only(MemoryDomainKind::WgpuTexture),
            ..Default::default()
        });
    }
    let in_caps = Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    dec.configure_pipeline(&in_caps)
        .expect("configure opens the decode device");

    let mut sink = RecordingSink::default();
    for (i, au) in aus.into_iter().enumerate() {
        block_on(dec.process(PipelinePacket::DataFrame(au_frame(au, i as u64)), &mut sink))
            .expect("decode access unit");
    }
    block_on(dec.process(PipelinePacket::Eos, &mut sink)).expect("flush at eos");

    let ctx = dec.gpu_context().expect("device opened");
    sink.packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => match &f.domain {
                MemoryDomain::System(s) => {
                    assert!(!gpu, "GPU path emitted a system frame");
                    Some(s.as_slice().to_vec())
                }
                MemoryDomain::WgpuTexture(owned) => {
                    assert!(gpu, "system path emitted a texture");
                    let tex = texture_of(owned).expect("element texture keep-alive");
                    Some(read_rgba_texture(&ctx, tex))
                }
                _ => panic!("unexpected output domain"),
            },
            _ => None,
        })
        .collect()
}

#[test]
fn h264_gpu_texture_streaming_emits_display_order() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m744 h264: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open h264 device: {e:?}"),
    };
    let ps = extract_h264_parameter_sets(H264).expect("sps/pps");
    let session = device.create_h264_session(&ps, 640, 480).expect("session");
    let mut oracle_dec = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m744 h264: no compute queue for the texture path");
            return;
        }
        Err(e) => panic!("gpu decoder: {e:?}"),
    };

    // The fixture must actually reorder, else a coding-order element would pass.
    let pocs: Vec<i32> = oracle_dec
        .index_pictures(H264)
        .expect("index")
        .iter()
        .map(|m| m.poc)
        .collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    // Oracle: whole-stream decode_all_to_textures (display order, M569/M535).
    let oracle_ctx = device.gpu_context();
    let oracle: Vec<Vec<u8>> = oracle_dec
        .decode_all_to_textures(H264)
        .expect("decode_all_to_textures")
        .iter()
        .map(|t| read_rgba_texture(&oracle_ctx, t))
        .collect();

    let aus = split_access_units(H264, |nal| {
        matches!(nal.first().map(|b| b & 0x1F), Some(1..=5))
    });
    assert!(aus.len() >= oracle.len(), "one AU per coded picture");
    let got = element_streaming_output(VideoCodec::H264, aus, true);

    assert_eq!(
        got.len(),
        oracle.len(),
        "streamed texture count == whole-stream oracle"
    );
    for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
        assert_eq!(
            g, o,
            "display frame {i} differs from the whole-stream texture oracle"
        );
    }
    eprintln!(
        "m744 h264: {} GPU textures streamed AU-by-AU in display order",
        got.len()
    );
}

#[test]
fn av1_streaming_emits_show_existing_display_order() {
    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m744 av1: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open av1 device: {e:?}"),
    };
    // The fixture must genuinely use show_existing_frame (an OBU_FRAME_HEADER
    // op), else a coded-frames-only element would pass.
    let show_existing = obu_bounds(AV1)
        .iter()
        .filter(|(_, _, _, t)| *t == 3)
        .count();
    assert!(
        show_existing > 0,
        "fixture has no show_existing_frame headers"
    );

    let seq = extract_av1_sequence_header(AV1).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device.create_av1_session(&std, 640, 480).expect("session");
    let mut oracle_dec = device
        .create_av1_dpb_decoder(&session, &seq)
        .expect("decoder");
    let oracle: Vec<Vec<u8>> = oracle_dec
        .decode_all(AV1)
        .expect("decode_all")
        .iter()
        .map(planes)
        .collect();

    let aus = split_temporal_units(AV1);
    let got = element_streaming_output(VideoCodec::Av1, aus.clone(), false);
    assert_eq!(
        got.len(),
        oracle.len(),
        "streamed frame count == whole-stream oracle (show_existing displays included)"
    );
    for (i, (g, o)) in got.iter().zip(&oracle).enumerate() {
        assert_eq!(g, o, "display frame {i} differs from decode_all oracle");
    }
    eprintln!(
        "m744 av1 system: {} frames streamed AU-by-AU in display order",
        got.len()
    );

    // The GPU-texture path: per-AU op-walk, same display order, zero-copy.
    let mut oracle_gpu = match device.create_av1_dpb_decoder_gpu(&session, &seq) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m744 av1 gpu half: no compute queue");
            return;
        }
        Err(e) => panic!("gpu decoder: {e:?}"),
    };
    let oracle_ctx = device.gpu_context();
    let oracle_tex: Vec<Vec<u8>> = oracle_gpu
        .decode_all_to_textures(AV1)
        .expect("decode_all_to_textures")
        .iter()
        .map(|t| read_rgba_texture(&oracle_ctx, t))
        .collect();
    let got_tex = element_streaming_output(VideoCodec::Av1, aus, true);
    assert_eq!(got_tex.len(), oracle_tex.len(), "texture count matches");
    for (i, (g, o)) in got_tex.iter().zip(&oracle_tex).enumerate() {
        assert_eq!(g, o, "display texture {i} differs from the oracle");
    }
    eprintln!(
        "m744 av1 gpu: {} textures streamed AU-by-AU in display order",
        got_tex.len()
    );
}
