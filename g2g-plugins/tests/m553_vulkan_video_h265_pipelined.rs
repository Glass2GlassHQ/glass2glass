//! M553: the pipelined streaming decode path for H.265, closing the codec gap in
//! m539 (which guarded only H.264).
//!
//! The ring split ([`VulkanStreamDecoder::submit_chunk_push`] + `flush`, backed
//! by the shared `DpbCore` decode ring) is codec-independent, but the per-codec
//! `decode_push` / `decode_flush` dispatch and the decode-order/timing pairing
//! were only validated for H.264. This guards that feeding H.265 one coded
//! picture at a time WITHOUT draining the ring (the low-latency Rerun
//! `AsyncDecoder` shape) decodes bit-exactly against the whole-clip
//! [`submit_chunk`](VulkanStreamDecoder::submit_chunk) golden. H.265 decode on
//! this driver is byte-exact (unlike AV1, see m508), so the assertion is strict.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.265 decode adapter.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::revideo::{VideoCodec, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{open_h265_decode_device, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

/// Byte offsets just past each Annex-B start code.
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

/// One coded picture per chunk: a VCL NAL (HEVC type 0..=31) closes a picture,
/// carrying its preceding VPS/SPS/PPS/SEI (the fixture is single-slice). Mirrors
/// m517's `split_pictures`.
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

fn have_adapter() -> bool {
    match block_on(open_h265_decode_device()) {
        Ok(_) => true,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => false,
        Err(e) => panic!("open H.265 decode device: {e:?}"),
    }
}

/// Whole-clip decode (dense submission) as the bit-exact golden.
fn decode_whole_clip() -> Vec<Vec<u8>> {
    let device = block_on(open_h265_decode_device()).expect("open H.265 device");
    let mut dec = VulkanStreamDecoder::new(device, VideoCodec::H265, CLIP).expect("build decoder");
    dec.submit_chunk(CLIP, true).expect("whole-clip decode").into_iter().map(|f| f.data).collect()
}

/// Per-picture pipelined decode: push each picture without draining, drain the
/// tail with flush.
fn decode_incremental(pics: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let device = block_on(open_h265_decode_device()).expect("open H.265 device");
    let mut dec = VulkanStreamDecoder::new(device, VideoCodec::H265, CLIP).expect("build decoder");
    let mut out = Vec::new();
    for pic in pics {
        for f in dec.submit_chunk_push(pic, false).expect("submit_chunk_push") {
            out.push(f.data);
        }
    }
    for f in dec.flush().expect("flush") {
        out.push(f.data);
    }
    out
}

#[test]
fn h265_pipelined_decode_bit_exact_solo() {
    if !have_adapter() {
        eprintln!("skip m553: no Vulkan H.265 decode adapter");
        return;
    }
    let golden = decode_whole_clip();
    let pics = split_pictures(CLIP);
    assert_eq!(golden.len(), pics.len(), "golden vs picture count");
    assert!(golden.len() >= 2, "fixture must have an IRAP + at least one inter frame");

    // A few runs: the pipelined path must be bit-exact every time (the ring's
    // decode-order output must pair with the right timing and match dense decode).
    for run in 0..3 {
        let inc = decode_incremental(&pics);
        assert_eq!(inc.len(), golden.len(), "run {run}: pipelined frame count");
        let diverged = golden.iter().zip(&inc).filter(|(g, i)| g != i).count();
        assert_eq!(diverged, 0, "run {run}: H.265 pipelined decode diverged from golden");
    }
    eprintln!("m553: H.265 pipelined push/flush bit-exact vs whole-clip golden ({} frames)", golden.len());
}
