//! M513: real-container ingestion on the re_video adapter.
//!
//! Rerun demuxes MP4/CMAF, so its `re_video` decoder receives the container's
//! stored form: parameter sets out of band in the sample-entry box (`avcC`) and
//! samples length-prefixed (AVCC), NOT the raw Annex-B elementary stream the
//! earlier tests fed. This drives [`VulkanStreamDecoder::from_config`] exactly
//! that way and proves it decodes **bit-identically** to the Annex-B path: build
//! an `avcC` from the fixture's SPS/PPS, split the stream into per-frame AVCC
//! samples (params stripped, VCL NALs length-prefixed), feed them one per
//! `submit_chunk`, and compare every I420 frame to the reference decode.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 decode adapter.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::revideo::{CodecConfig, VideoCodec, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const W: u32 = 640;
const H: u32 = 480;

/// Split an Annex-B stream into its NAL payloads (start codes stripped). Handles
/// both 3- and 4-byte start codes; enough for the well-formed fixture.
fn split_annexb_nals(data: &[u8]) -> Vec<&[u8]> {
    let mut payload_starts = Vec::new();
    let mut j = 0;
    while j + 3 <= data.len() {
        if data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 1 {
            payload_starts.push(j + 3);
            j += 3;
        } else {
            j += 1;
        }
    }
    let mut nals = Vec::new();
    for (k, &start) in payload_starts.iter().enumerate() {
        let end = if k + 1 < payload_starts.len() {
            let mut e = payload_starts[k + 1] - 3; // back over the next `00 00 01`
            if e > start && data[e - 1] == 0 {
                e -= 1; // and its leading zero if it was a 4-byte code
            }
            e
        } else {
            data.len()
        };
        if end > start {
            nals.push(&data[start..end]);
        }
    }
    nals
}

/// Build an `avcC` record from one SPS and one PPS (4-byte NAL length prefixes).
fn build_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut v = vec![1u8, sps[1], sps[2], sps[3]]; // version + profile/compat/level
    v.push(0xFF); // 111111 + lengthSizeMinusOne = 3
    v.push(0xE1); // 111 + numSPS = 1
    v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    v.extend_from_slice(sps);
    v.push(1); // numPPS
    v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    v.extend_from_slice(pps);
    v
}

/// One AVCC sample = a VCL NAL preceded by its 4-byte big-endian length.
fn avcc_sample(nal: &[u8]) -> Vec<u8> {
    let mut s = Vec::with_capacity(nal.len() + 4);
    s.extend_from_slice(&(nal.len() as u32).to_be_bytes());
    s.extend_from_slice(nal);
    s
}

#[test]
fn from_config_decodes_container_samples_bit_identically() {
    // Reference: the Annex-B path (already bit-exact vs ffmpeg, m508).
    let dev_ref = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m513: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open H.264 decode device: {e:?}"),
    };
    let mut ref_dec =
        VulkanStreamDecoder::new(dev_ref, VideoCodec::H264, CLIP).expect("build Annex-B adapter");
    let ref_frames = ref_dec.submit_chunk(CLIP, true).expect("Annex-B decode");
    assert_eq!(ref_frames.len(), 10, "fixture is 10 frames");

    // Container form: avcC out-of-band params + per-frame AVCC samples.
    let nals = split_annexb_nals(CLIP);
    let mut sps = None;
    let mut pps = None;
    let mut samples = Vec::new();
    for nal in nals {
        match nal[0] & 0x1F {
            7 => sps = Some(nal),
            8 => pps = Some(nal),
            1 | 5 => samples.push(avcc_sample(nal)), // VCL -> one sample
            _ => {}                                  // SEI / AUD not needed for decode
        }
    }
    let sps = sps.expect("fixture carries an SPS");
    let pps = pps.expect("fixture carries a PPS");
    assert_eq!(samples.len(), 10, "one AVCC sample per frame");
    let avcc = build_avcc(sps, pps);

    let dev = block_on(open_h264_decode_device()).expect("re-open decode device");
    let mut dec = VulkanStreamDecoder::from_config(dev, CodecConfig::Avcc(&avcc), false)
        .expect("build from avcC config");
    // Geometry comes from the config's SPS, not from any in-band sample.
    assert_eq!(dec.width(), W);
    assert_eq!(dec.height(), H);

    let mut got = Vec::new();
    for (i, s) in samples.iter().enumerate() {
        got.extend(
            dec.submit_chunk(s, i == 0)
                .expect("container-sample decode"),
        );
    }
    assert_eq!(got.len(), ref_frames.len());

    // The whole point: out-of-band params + AVCC reframing + per-sample feeding
    // yields exactly the Annex-B result, frame for frame, byte for byte.
    for (i, (g, r)) in got.iter().zip(ref_frames.iter()).enumerate() {
        assert_eq!(g.width, W);
        assert_eq!(g.height, H);
        assert_eq!(g.data, r.data, "frame {i} differs from the Annex-B decode");
    }
    eprintln!("m513: 10 frames from avcC + AVCC samples, bit-identical to Annex-B decode");
}
