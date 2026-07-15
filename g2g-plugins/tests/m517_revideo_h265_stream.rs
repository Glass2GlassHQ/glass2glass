//! M517: the re_video adapter ([`g2g_plugins::revideo::VulkanStreamDecoder`])
//! streams H.265, not just AV1 (m508) and H.264 (m512 / m513).
//!
//! Drives the adapter the way a Rerun `AsyncDecoder` backend would: one coded
//! picture per `submit_chunk`, DPB reference state carried across calls, output
//! packed I420. H.265 decode on this driver is byte-exact (unlike AV1, see
//! m508), so this asserts the streaming contract strictly: 10 real I420 frames,
//! consecutive frames differ (inter prediction ran against carried references),
//! and `reset()` + re-decode of the keyframe reproduces frame 0 bit-for-bit.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.265 decode support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::revideo::{VideoCodec, VideoPixelLayout, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{open_h265_decode_device, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");
const W: usize = 640;
const H: usize = 480;

/// Byte offsets of each NAL payload (just past its start code).
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

/// Split an H.265 Annex-B stream into per-picture chunks: each VCL NAL (HEVC NAL
/// type 0..=31) closes a picture, carrying any preceding VPS/SPS/PPS/SEI with it
/// (the fixture is single-slice, so one VCL NAL == one coded picture). This is
/// the per-sample chunking a demuxer feeds an `AsyncDecoder`.
fn split_pictures(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut units = Vec::new();
    let mut cur = Vec::new();
    let starts = start_code_offsets(stream);
    for (k, &begin) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        let nal = &stream[begin..end];
        cur.extend_from_slice(&[0, 0, 0, 1]);
        cur.extend_from_slice(nal);
        let nal_type = nal.first().map(|b| (b >> 1) & 0x3f).unwrap_or(63);
        if nal_type <= 31 {
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

#[test]
fn revideo_adapter_streams_h265_i420_frames() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m517: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open H.265 decode device: {e:?}"),
    };

    let mut dec =
        VulkanStreamDecoder::new(device, VideoCodec::H265, CLIP).expect("build re_video adapter");
    assert_eq!(dec.width(), W as u32);
    assert_eq!(dec.height(), H as u32);

    let chunks = split_pictures(CLIP);
    assert_eq!(chunks.len(), 10, "fixture is two GOPs of a keyframe + four P frames");

    // Feed one coded picture at a time; DPB reference state must carry across
    // calls for the P frames to decode against their references.
    let mut frames = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let out = dec.submit_chunk(chunk, i == 0).expect("submit chunk");
        frames.extend(out);
    }
    assert_eq!(frames.len(), 10, "one I420 frame per coded picture");

    let i420_len = W * H + 2 * (W / 2) * (H / 2);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.layout, VideoPixelLayout::Y_U_V420);
        assert_eq!(f.data.len(), i420_len, "frame {i} packed I420 size");
        let y = &f.data[..W * H];
        let min = *y.iter().min().unwrap();
        let max = *y.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max})");
    }

    // Consecutive frames must differ (motion decoded against carried references,
    // not a frozen or reused picture).
    for i in 1..frames.len() {
        assert!(frames[i].data != frames[i - 1].data, "frame {i} == {}", i - 1);
    }

    // reset() then re-decode the keyframe reproduces frame 0 BIT-EXACTLY: H.265
    // decode is deterministic on this driver, so a correct seek path rebuilds a
    // clean DPB and re-decodes the identical picture (a broken reset would return
    // a stale or garbage frame).
    dec.reset().expect("reset");
    let f0_again = dec.submit_chunk(&chunks[0], true).expect("re-decode frame 0");
    assert_eq!(f0_again.len(), 1);
    assert_eq!(f0_again[0].data, frames[0].data, "reset + re-decode must reproduce frame 0");
}
